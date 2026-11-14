"""End-to-end rounds on CPU: the controller, the round runner, the journal and the store together."""

from __future__ import annotations

import functools
import json
from dataclasses import replace

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import manifest_digest
from sn120_v2.cli import main
from sn120_v2.errors import ConfigError, ContractError, FencedError, HeadMismatchError, InfraError, LeaseError, V2Error
from sn120_v2.gpu import plan
from sn120_v2.journal import read_journal
from sn120_v2.params import GenesisSpec, Params, Recipe
from sn120_v2.payout import weights
from sn120_v2.public import TEXT_LIMIT, public_text, snapshot
from sn120_v2.queue import waiting
from sn120_v2.sim import JOURNAL_FILE, SIM_IMAGE, STORE_DIR, SimAdmission, SimWorld, sim_params
from sn120_v2.state import HEAD_APPLIED, HEAD_FOLDED, LEASE_FAILED, ROUND_CLOSED, ROUND_OPENED, ROUND_SETTLED, load
from sn120_v2.store import LocalStore, put_record
from sn120_v2.types import ChainSnapshot, HeadManifest, King, Registration, V1Import


@pytest.fixture
def world(tmp_path) -> SimWorld:
    return SimWorld(tmp_path, sim_params())


def kinds(world: SimWorld) -> list[str]:
    return [record.kind for record in read_journal(world.root / JOURNAL_FILE)]


def seats(record) -> list[int]:
    return [entry.reg.uid for lease in record.leases for entry in lease.entries]


def test_a_round_applies_the_winner_folds_the_head_and_names_a_king(world):
    for uid, quality in ((1, 0.3), (2, 0.0), (3, -0.25)):
        world.register(uid, quality)
    record = world.round()
    state, store = world.controller.state, world.store
    assert record.outcome == "applied" and record.decision.winner.uid == 1
    genesis, applied, folded = state.heads
    assert applied.parent_version_id == genesis.version_id and [ref.miner_uid for ref in applied.loras] == [1]
    assert folded.parent_version_id == applied.version_id and folded.folded and state.head == folded
    assert [(king.uid, king.head_version_id) for king in state.kings] == [(1, applied.version_id)]
    statuses = {entry.uid: (entry.status, entry.eliminated_at_look) for entry in record.decision.entries}
    assert statuses == {1: ("winner", None), 2: ("lost", None), 3: ("lost", 0)}, "uid 3 was dropped at the first look"
    assert state.queue == {} and state.served_hotkeys == {"hk1", "hk2", "hk3"}, "every scored seat is used up"
    lora_hash = applied.loras[0].lora_hash
    assert store.list("public/loras/") == store.list(layout.public_lora_prefix(lora_hash)) != []
    assert store.get(layout.winner_batch(record.round_id)) == store.get(layout.miner_object("hk1"))
    assert store.list("pending/") == [], "losing LoRAs are deleted and never leave pending/"
    assert HeadManifest.from_json(store.get(layout.CURRENT)).head == folded
    assert store.exists(layout.irt_snapshot(state.snapshot_id)) and store.exists(layout.round_file(record.round_id))
    assert world.publisher.snapshots[-1]["head"]["head"]["version_id"] == folded.version_id
    assert world.trainer.closed == 1 and world.evaluator.closed == 1


def test_a_null_round_keeps_the_head_and_names_no_king(world):
    world.register(1, 0.0)
    world.register(2, -0.25)
    head = world.controller.state.head
    record = world.round()
    state = world.controller.state
    assert record.outcome == "null" and state.head == head and state.kings == []
    assert state.queue == {}, "scored entries had a fair tournament: their seats are used up"
    assert world.store.list("public/loras/") == [] and not world.store.exists(layout.winner_batch(record.round_id))


def test_a_short_queue_runs_at_once_and_an_empty_queue_starts_nothing(world):
    assert world.round() is None and ROUND_OPENED not in kinds(world)
    world.register(1, 0.3)
    record = world.round()
    assert seats(record) == [1] and record.outcome == "applied"


def test_a_round_takes_at_most_the_batch_cap_in_queue_order(world):
    for uid in (6, 5, 4, 3, 2, 1):
        world.register(uid, 0.0)
    assert seats(world.round()) == [1, 2, 3, 4]
    assert seats(world.round()) == [5, 6]


def test_a_seat_without_a_valid_file_is_skipped_not_charged_and_keeps_its_place(world):
    world.register(1, None)
    world.register(2, 0.0, valid=False)
    world.register(3, 0.0)
    record = world.round()
    queue = world.controller.state.queue
    assert seats(record) == [3] and record.skipped == ((1, 1001), (2, 1002))
    assert set(queue) == {(1, 1001), (2, 1002)} and not any(seat.front or seat.leased for seat in queue.values())
    world.admission.upload("hk1", 0.0)
    assert seats(world.round()) == [1]


def test_a_training_failure_is_no_decision_for_that_seat_only_and_repeats_hold_it(world):
    world.register(1, 0.3)
    world.register(2, 0.0)
    world.trainer.fail_hotkeys = {"hk2"}
    assert world.round().outcome == "applied"
    seat = world.controller.state.queue[(2, 1002)]
    assert (seat.front, seat.leased, seat.infra_attempts, seat.held) == (True, False, 1, False)
    assert "hk2" not in world.controller.state.served_hotkeys, "not charged"
    for uid in (3, 4):  # a failure is counted only in a round in which another batch trained
        world.register(uid, 0.0)
        world.round()
    seat = world.controller.state.queue[(2, 1002)]
    assert (seat.infra_attempts, seat.held) == (3, True)
    world.trainer.fail_hotkeys = set()
    assert world.round() is None, "a held seat is passed over until the operator releases it"
    world.controller.release_seat((2, 1002), world.clock())
    assert seats(world.round()) == [2]


def test_a_round_in_which_no_training_finished_fails_its_lease_and_counts_nothing_for_a_seat(world, capsys):
    """A fault of the box fails every training. Without this rule a few such rounds, minutes apart, hold the whole queue."""
    for uid in (1, 2, 3):
        world.register(uid, 0.0)
    world.trainer.fail_hotkeys, world.trainer.reject_hotkeys = {"hk1", "hk3"}, {"hk2"}  # a skip in the round changes nothing
    cause = "InfraError: no training finished in this round: 2 of 3 batches failed. The first: training failed 3 times: simulated trainer crash"
    for failures in (1, 2, 3, 4):
        record = world.round()
        state = world.controller.state
        assert record.outcome == "no_decision" and record.decision is None and [lease.status for lease in record.leases] == ["failed"]
        assert all((seat.front, seat.leased, seat.infra_attempts, seat.held) == (True, False, 0, False) for seat in state.queue.values())
        assert not state.served_regs and state.failed_leases == failures, "nobody is charged"
        assert world.round() is None, "the next round waits apply_retry_s, as after any failed lease"
        world.now += world.params.apply_retry_s
    closed = [record.data for record in read_journal(world.root / JOURNAL_FILE) if record.kind == ROUND_CLOSED]
    assert all(data["reason"] == cause and not data["attempted"] and not data["held"] for data in closed) and len(closed) == 4
    assert main(["round", "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack"]) == 0
    assert f"no winner: {cause}" in capsys.readouterr().out, "the round summary names the cause"
    world.reopen()
    world.trainer.fail_hotkeys = {"hk3"}  # the box works again; one batch still fails: that failure is counted, as before
    world.trainer.reject_hotkeys = set()
    record = world.round()
    queue = world.controller.state.queue
    assert record.outcome == "null" and world.controller.state.failed_leases == 0 and sorted(seats(record)) == [1, 2, 3]
    assert list(queue) == [(3, 1003)] and (queue[(3, 1003)].infra_attempts, queue[(3, 1003)].front) == (1, True)


def test_a_text_from_an_exception_is_made_safe_before_it_goes_into_a_public_file(world, monkeypatch, capsys):
    """The box holds credentials. A text from an exception can hold a signed link, a secret of the environment or an address."""
    secret, link = "v2TestSecretValue0123456789", "https://store.example/bucket/private/x.json"
    monkeypatch.setenv("SN120_R2_SECRET_ACCESS_KEY", secret)
    crash = f"403 for url: {link}?X-Amz-Credential=AKID%2F2026&X-Amz-Signature=0123abcd from 203.0.113.7:443 and 2001:db8::7 with {secret}"
    safe = f"403 for url: {link}?[removed-query] from [removed-address]:443 and [removed-address] with [removed-secret]"
    private = (secret, "X-Amz", "0123abcd", "203.0.113.7", "2001:db8::7")
    fail: set[str] = set()

    def train(request, real=world.trainer.train):
        if request.entry.reg.hotkey in fail:
            raise InfraError(crash)
        return real(request)

    monkeypatch.setattr(world.trainer, "train", train)
    world.register(1, 0.0)
    world.register(2, 0.0)
    fail.add("hk1")
    one = world.round()  # one batch of two fails: the text is the reason of that row
    fail.add("hk3")
    world.register(3, 0.0)
    both = world.round()  # no training finished: the text is in the note of the round
    assert (one.outcome, both.outcome) == ("null", "no_decision") and seats(both) == [1, 3]
    first, second = (json.loads(world.store.get(layout.round_file(record.round_id))) for record in (one, both))
    assert [entry["reason"] for entry in first["entries"] if entry["uid"] == 1] == [f"training failed 3 times: {safe}"]
    note = f"InfraError: no training finished in this round: 2 of 2 batches failed. The first: training failed 3 times: {safe}"
    assert second["note"] == note, "the error class and the message stay"
    public = [path for path in (world.root / STORE_DIR / "public").rglob("*") if path.is_file()]
    shown = [path.read_text() for path in public] + [json.dumps(world.publisher.snapshots)]
    assert len(public) > 2 and not [word for word in private for text in shown if word in text], "nothing public holds a private part"
    journal = read_journal(world.root / JOURNAL_FILE)
    kept = [record.data["reason"] for record in journal if record.kind in (LEASE_FAILED, ROUND_CLOSED) and record.data["reason"]]
    assert len(kept) == 2 and all(crash in reason for reason in kept), "the journal keeps the text as it was"
    assert capsys.readouterr().err.count(crash) == 2, "and the log of the operator has it, once for each round file"
    assert main(["round", "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack"]) == 0
    assert f"no winner: {note}" in capsys.readouterr().out
    # The rule itself: a login in a URL, a named value, a long text; a time of day, a version and a test name stay.
    env = {"HF_TOKEN": "hf_0123456789abcdef", "TOKENIZERS_PARALLELISM": "false", "SN120_WORK_DIR": "/work/sn120"}
    text = "wss://user:pw@node.example:9944/x?k refused; Authorization: Bearer abc.def, token=hf_0123456789abcdef; false at 12:34:56 in /work/sn120, torch 2.15.0, t.py::test"
    assert public_text(text, env) == (
        "wss://[removed-login]@node.example:9944/x?[removed-query] refused; Authorization: Bearer [removed-secret], token=[removed-secret];"
        " false at 12:34:56 in /work/sn120, torch 2.15.0, t.py::test"
    )
    assert public_text("hf_0123456789abcdef\n at ::1", env) == "[removed-secret] at [removed-address]"
    assert public_text("x" * (TEXT_LIMIT + 40), env) == "x" * TEXT_LIMIT + " [cut: 40 more characters]"


def test_release_seat_undoes_the_holds_and_the_counts_that_a_box_fault_left(world, capsys):
    """The command line for ``Controller.release_seat``: after a fault of the box, no seat stays held or one fault from a hold."""
    world.register(1, 0.0)
    world.register(2, 0.0)
    world.register(4, None)  # a seat with no file: it takes no turn, so it has no count
    common = ["--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack"]
    assert main(["release-seat", *common, "--all"]) == 2 and "refused: V2Error: no seat is held or has a counted failure" in capsys.readouterr().err
    world.reopen()  # the command took the journal over, as set-params does
    world.trainer.fail_hotkeys = {"hk1", "hk2", "hk3"}  # a fault of the box that lets one batch train in each round
    for uid in (10, 11):
        world.register(uid, 0.0)
        world.round()
    world.register(3, 0.0)
    world.register(12, 0.0)
    world.round()
    queue = world.controller.state.queue
    assert sorted((seat.reg.uid, seat.infra_attempts, seat.held) for seat in queue.values()) == [(1, 3, True), (2, 3, True), (3, 1, False), (4, 0, False)]
    assert main(["status", "--state-dir", str(world.root)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert (status["held_seats"], status["seats_with_counted_failures"]) == ([1, 2], 1), "status shows what there is to release"
    assert main(["release-seat", *common, "--uid", "4"]) == 2 and "refused: V2Error: no seat of UID 4 is held or has a counted failure" in capsys.readouterr().err
    assert main(["release-seat", *common, "--uid", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["released"] == [{"uid": 2, "registration_block": 1002, "hotkey": "hk2", "was_held": True, "counted_failures": 3}]
    assert main(["release-seat", *common, "--all"]) == 0
    assert [(seat["uid"], seat["was_held"], seat["counted_failures"]) for seat in json.loads(capsys.readouterr().out)["released"]] == [(1, True, 3), (3, False, 1)]
    queue = world.reopen().state.queue
    assert all(not seat.held and seat.infra_attempts == 0 for seat in queue.values())
    assert [seat.reg.uid for seat in queue.values() if seat.front] == [1, 2, 3], "still first: a release changes no turn"
    assert main(["status", "--state-dir", str(world.root)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert (status["held_seats"], status["seats_with_counted_failures"]) == ([], 0)
    world.trainer.fail_hotkeys = set()
    assert sorted(seats(world.round())) == [1, 2, 3]


def test_a_batch_rejected_at_train_time_is_a_skip(world):
    world.register(1, 0.0)
    world.trainer.reject_hotkeys = {"hk1"}
    record = world.round()
    seat = world.controller.state.queue[(1, 1001)]
    assert record.decision.entries[0].status == "skip"
    assert (seat.front, seat.leased, seat.infra_attempts) == (False, False, 0)


class LateRunner:
    """A box that answers after its lease ran out."""

    def __init__(self, world: SimWorld) -> None:
        self.world = world

    def run(self, ticket):
        result = self.world.runner.run(ticket)
        self.world.now = ticket.expires_at
        return result


def test_a_result_after_the_lease_expired_is_never_used(world):
    world.register(1, 0.3)
    world.register(2, 0.0)
    head = world.controller.state.head
    record = world.controller.run_round(LateRunner(world), world.clock, "entropy")
    state = world.controller.state
    assert record.outcome == "no_decision" and record.decision is None
    assert state.head == head and state.kings == [] and not state.served_hotkeys, "a timeout never promotes"
    assert all(seat.front and not seat.leased and seat.infra_attempts == 0 for seat in state.queue.values())
    assert world.round() is None, "the controller waits before it leases again"
    world.now += world.params.apply_retry_s
    record = world.round()
    assert seats(record) == [1, 2] and record.outcome == "applied", "the same seats run first, in their original order"


def test_a_box_failure_is_no_decision(world):
    world.register(1, 0.3)
    world.evaluator.fail_looks = {0}
    record = world.round()
    assert record.outcome == "no_decision" and world.controller.state.queue[(1, 1001)].front
    assert world.trainer.closed == 1 and world.evaluator.closed == 1


def test_a_result_for_another_head_or_ticket_is_refused_and_a_result_is_used_once(world):
    world.register(1, 0.3)
    controller = world.controller
    (ticket,) = controller.open_round(world.clock(), "entropy")
    result = world.runner.run(ticket)
    with pytest.raises(HeadMismatchError):
        controller.submit_result(replace(result, head_version_id="0" * 64), world.clock())
    with pytest.raises(ContractError):
        controller.submit_result(replace(result, ticket_digest="0" * 64), world.clock())
    with pytest.raises(LeaseError):
        controller.submit_result(replace(result, lease_id="r000001-l7"), world.clock())
    assert controller.state.round.decision is None and controller.state.round.leases[0].status == "open"
    controller.submit_result(result, world.clock())
    assert controller.state.rounds[-1].outcome == "applied"
    with pytest.raises(LeaseError):
        controller.submit_result(result, world.clock())
    assert kinds(world).count(HEAD_APPLIED) == 1


def test_a_winner_whose_files_do_not_verify_is_never_applied(world):
    world.register(1, 0.3)
    world.register(2, 0.0)
    controller = world.controller
    (ticket,) = controller.open_round(world.clock(), "entropy")
    result = world.runner.run(ticket)
    world.store.put(f"{result.entries[0].lora.prefix}/adapter_model.bin", b"corrupted")
    head = controller.state.head
    controller.submit_result(result, world.clock())
    state = controller.state
    assert state.rounds[-1].outcome == "abandoned" and state.head == head and state.kings == []
    assert world.store.list("public/loras/") == [] and not state.served_hotkeys
    assert all(seat.front and seat.infra_attempts == 0 for seat in state.queue.values()), "nobody is charged"


def test_a_store_outage_during_apply_defers_then_applies_or_gives_up(tmp_path, monkeypatch):
    world = SimWorld(tmp_path, sim_params(apply_max_attempts=3))
    world.register(1, 0.3)
    outage = {"left": 1}
    real_copy = world.store.copy

    def copy(src, dst):
        if outage["left"] > 0:
            outage["left"] -= 1
            raise InfraError("store unavailable")
        return real_copy(src, dst)

    monkeypatch.setattr(world.store, "copy", copy)
    assert world.round() is None, "the round stays open: decided, not applied"
    state = world.controller.state
    assert state.round.decision.winner.uid == 1 and not state.round.applied and state.kings == []
    world.controller.tick(world.clock())
    assert world.controller.state.round is not None, "it waits for the retry interval"
    world.now += world.params.apply_retry_s
    world.controller.tick(world.clock())
    assert world.controller.state.rounds[-1].outcome == "applied" and len(world.controller.state.kings) == 1

    world.register(2, 0.3)
    outage["left"] = 99
    world.round()
    for _ in range(3):
        world.now += world.params.apply_retry_s
        world.controller.tick(world.clock())
    state = world.controller.state
    assert state.rounds[-1].outcome == "abandoned" and len(state.kings) == 1 and state.queue[(2, 1002)].front


def test_no_round_opens_before_the_fold_has_landed(world):
    world.register(1, 0.3)
    world.folder.fail = True
    assert world.round() is None
    state = world.controller.state
    assert state.round.applied and not state.head.folded and len(state.kings) == 1
    world.register(2, 0.0)
    with pytest.raises(V2Error):
        world.controller.open_round(world.clock(), "entropy")
    world.folder.fail = False
    world.now += world.params.apply_retry_s
    world.controller.tick(world.clock())
    state = world.controller.state
    assert state.head.folded and state.rounds[-1].outcome == "applied"
    assert world.round().head_version_id == state.head.version_id, "the next round runs on the folded head"


class Crash(BaseException):
    """The controller process dies."""


def crash_at(world: SimWorld, kind: str, before: bool) -> None:
    journal, dead = world.controller.journal, []
    real_append = journal.append

    def append(record_kind, data, at):
        if dead or (record_kind == kind and before):
            dead.append(True)
            raise Crash(record_kind)
        record = real_append(record_kind, data, at)
        if record_kind == kind:
            dead.append(True)
            raise Crash(record_kind)
        return record

    journal.append = append


@pytest.mark.parametrize("before", [False, True], ids=["after-write", "before-write"])
@pytest.mark.parametrize("kind", [ROUND_OPENED, ROUND_SETTLED, HEAD_APPLIED, HEAD_FOLDED, ROUND_CLOSED])
def test_a_controller_that_dies_at_any_step_is_replaced_with_no_second_apply_and_no_lost_seat(tmp_path, kind, before):
    world = SimWorld(tmp_path, sim_params())
    for uid, quality in ((1, 0.3), (2, 0.0), (3, None)):
        world.register(uid, quality)
    crash_at(world, kind, before)
    with pytest.raises(Crash):
        world.round()
    world.reopen()
    world.round()
    state = world.controller.state
    assert [head.folded for head in state.heads] == [True, False, True]
    assert [king.uid for king in state.kings] == [1] and kinds(world).count(HEAD_APPLIED) == 1
    assert state.served_hotkeys == {"hk1", "hk2"} and state.round is None
    assert [(key, seat.front, seat.leased) for key, seat in state.queue.items()] == [((3, 1003), False, False)]
    assert [record.outcome for record in state.rounds] == ["applied"]
    assert [kinds(world).count(k) for k in (ROUND_SETTLED, HEAD_FOLDED, ROUND_CLOSED)] == [1, 1, 1]


def test_replaying_the_journal_rebuilds_the_same_state_and_every_head_id(world):
    world.register(1, 0.3)
    world.register(2, 0.0)
    world.round()
    live, again = world.controller.state, load(world.root / JOURNAL_FILE)
    assert [head.version_id for head in again.heads] == [head.version_id for head in live.heads]
    assert all(head.version_id == head.expected_id() for head in again.heads)
    assert snapshot(again, world.params, 5) == snapshot(live, world.params, 5)


def test_a_second_controller_fences_the_first(world):
    world.register(1, 0.3)
    old = world.controller
    world.reopen()
    with pytest.raises(FencedError):
        old.open_round(world.clock(), "entropy")
    assert world.round().outcome == "applied" and kinds(world).count(ROUND_OPENED) == 1


def test_a_restart_gets_the_task_table_snapshot_back(world):
    world.register(1, 0.0)
    world.round()
    world.reopen()
    assert world.restored_snapshot == dict(world.sampler.public_snapshot())


def test_two_boxes_take_disjoint_seats_and_one_winner_is_promoted(world):
    for uid, quality in ((1, 0.0), (2, 0.2), (3, 0.35), (4, 0.0)):
        world.register(uid, quality)
    controller = world.controller
    tickets = controller.open_round(world.clock(), "entropy", boxes=("box-a", "box-b"))
    assert [[entry.reg.uid for entry in ticket.entries] for ticket in tickets] == [[1, 2], [3, 4]]
    controller.submit_result(world.runner.run(tickets[0]), world.clock())
    assert controller.state.round.decision is None, "the round waits for the other box"
    controller.submit_result(world.runner.run(tickets[1]), world.clock())
    record = controller.state.rounds[-1]
    assert record.decision.winner.uid == 3 and [king.uid for king in controller.state.kings] == [3]
    assert controller.state.queue == {}, "the other box's winner lost its entry; it is not requeued"


def test_public_data_never_shows_a_losers_file_or_train_values(world):
    world.register(1, 0.3)
    world.register(2, 0.0, train={"rank": 32})
    record = world.round()
    report = json.loads(world.store.get(layout.round_file(record.round_id)))
    text = json.dumps([world.publisher.snapshots, report])
    assert "private/" not in text and '"train"' not in text and '"uri"' not in text
    loser = next(entry for entry in world.publisher.snapshots[-1]["history"][-1]["entries"] if entry["uid"] == 2)
    assert set(loser) == {"queue_pos", "uid", "hotkey", "registration_block", "batch_sha256", "status", "eliminated_at_look", "delta", "se", "z", "cap_rate", "n_tasks"}
    assert report["winner_outcomes"] and list(report["null_outcomes"]) == ["box-0"]
    assert world.store.list("public/rounds/") == [layout.winner_batch(record.round_id), layout.round_file(record.round_id)]


def test_the_first_start_carries_v1_kings_and_served_seats_and_happens_once(tmp_path):
    v1 = V1Import(
        "k1.json", 9500000, kings=(King(7, "hk7", "v1-e1211", "", 5),), served_regs=((1, 1001),), served_hotkeys=("hk2",),
        demoted=((3, 1003),), front=((4, 1004),),
    )
    world = SimWorld(tmp_path, sim_params(), v1=v1)
    for uid in (1, 2, 3, 4, 5):
        world.register(uid, 0.0)
    state = world.controller.state
    assert [king.uid for king in state.kings] == [7]
    assert set(state.queue) == {(3, 1003), (4, 1004), (5, 1005)}, "a served seat and a served hotkey never enter the queue"
    assert [seat.reg.uid for seat in waiting(state.queue)] == [4, 5, 3], "front first, demoted last"
    assert weights(state.kings, {7: "hk7"}, world.params) == {7: 1.0}
    with pytest.raises(V2Error):
        world.controller.bootstrap(state.genesis, state.manifest.base_files, state.manifest.master_files, world.clock())


def test_no_round_opens_without_a_pinned_worker_image(tmp_path):
    world = SimWorld(tmp_path, sim_params(recipe=Recipe()))
    world.register(1, 0.3)
    with pytest.raises(ConfigError):
        world.controller.open_round(world.clock(), "entropy")


# ------------------------------------------------------------ a box that differs from the pinned digests

OLD_CODE, NEW_CODE = "code-before-the-fix", "code-after-the-fix"
NOTHING_COUNTED = r"\. The difference: the training code on this box differs from recipe\.code_digest\. Nothing is counted for a seat\. Next: "


@pytest.fixture
def fixed_box(tmp_path, monkeypatch) -> SimWorld:
    """The parameters pin the code before a fix; the box has the code after it. The check is the one of the trainer."""
    world = SimWorld(tmp_path, sim_params(recipe=Recipe(image_digests=(SIM_IMAGE,), code_digest=OLD_CODE)))
    monkeypatch.setattr(plan, "code_digest", lambda: NEW_CODE)
    world.controller.pins = functools.partial(plan.check_pins, SIM_IMAGE)
    return world


def store_a_result(world: SimWorld) -> None:
    """Open a round, run it and store the box result with no judging, as a ``run`` that died there on the old code."""
    (ticket,) = world.controller.open_round(world.clock(), "entropy", clock=world.clock)
    put_record(world.store, layout.sealed_result(ticket.round_id, ticket.lease_id), world.runner.run(ticket))


def pin_the_new_code(world: SimWorld, **changes) -> None:
    params = world.controller.params
    world.controller.set_params(replace(params, recipe=replace(params.recipe, code_digest=NEW_CODE), **changes), world.clock())


def test_a_box_that_differs_from_the_pins_judges_a_stored_result_and_then_opens_no_round(fixed_box):
    world = fixed_box
    for uid in range(1, 9):
        world.register(uid, 0.0 if uid <= 4 else 0.1)
    store_a_result(world)
    trainings = world.trainer.closed
    for _ in range(4):  # without the stop, 3 such starts hold every seat of the next round
        with pytest.raises(ConfigError, match="so no round was opened" + NOTHING_COUNTED + "the digest block, then set-params"):
            world.round()
    state = world.controller.state
    assert state.round is None and [record.outcome for record in state.rounds] == ["null"], "the result that the old digests made is judged"
    assert kinds(world).count(ROUND_OPENED) == 1 and world.trainer.closed == trainings, "no round opened and no training started"
    assert sorted(uid for uid, _ in state.queue) == [5, 6, 7, 8] and all(seat.infra_attempts == 0 and not seat.held for seat in state.queue.values())
    pin_the_new_code(world, n_tasks=32)  # the round is closed: every parameter can change
    record = world.round()
    assert seats(record) == [5, 6, 7, 8] and record.decision is not None and record.n_tasks == 32


def test_a_box_that_differs_from_the_pins_starts_no_fold_and_folds_when_the_new_digests_are_pinned(fixed_box):
    world = fixed_box
    world.register(1, 0.3)
    store_a_result(world)
    folds, fold = [], world.folder.fold
    world.folder.fold = lambda *args: folds.append(args) or fold(*args)
    for _ in range(2):
        with pytest.raises(ConfigError, match="so the fold was not started" + NOTHING_COUNTED + "the digest block"):
            world.round()
    state = world.controller.state
    assert state.round.decision.winner.uid == 1 and state.round.applied and not state.head.folded, "judged and promoted; the fold is left"
    assert not folds and [king.uid for king in state.kings] == [1]
    with pytest.raises(V2Error, match="a decided round takes new digests only"):
        pin_the_new_code(world, n_tasks=32)
    pin_the_new_code(world)
    assert world.round() is None and len(folds) == 1, "the same start finishes the fold; no seat waits, so no round opens"
    state = world.controller.state
    assert state.round is None and state.head.folded and state.rounds[-1].outcome == "applied"


def test_a_box_that_differs_from_the_pins_does_not_run_an_open_round_again(fixed_box):
    world = fixed_box
    world.register(1, 0.3)
    (ticket,) = world.controller.open_round(world.clock(), "entropy", clock=world.clock)
    before = kinds(world)
    with pytest.raises(ConfigError, match="so the open round was not run" + NOTHING_COUNTED + r"end-round \(the round pins the old digests\), the digest block"):
        world.round()
    state = world.controller.state
    assert kinds(world) == before and world.trainer.closed == 0 and state.round.leases[0].status == "open", "nothing started, nothing written"
    with pytest.raises(V2Error, match="a round is open"):
        pin_the_new_code(world)
    world.controller.fail_lease(ticket.lease_id, "ended by the operator", world.clock())  # what end-round does
    pin_the_new_code(world)
    seat = world.controller.state.queue[(1, 1001)]
    assert seat.front and seat.infra_attempts == 0 and not seat.held
    world.now += world.params.apply_retry_s
    assert world.round().outcome == "applied"


def test_the_command_line_runs_the_cpu_simulation(tmp_path, capsys):
    assert main(["sim", "--state-dir", str(tmp_path), "--rounds", "3", "--miners", "3", "--seed", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["replay_matches"] is True


def test_the_command_line_bootstraps_runs_a_round_and_reports_status(tmp_path, capsys, build):
    state_dir, write = tmp_path / "state", lambda name, text: (tmp_path / name).write_text(text if isinstance(text, str) else text.decode())
    base_files, master_files = {"model.bin": build.h("base")}, {"master.bin": build.h("master")}
    write("genesis.json", GenesisSpec("org/model", "rev", manifest_digest(base_files), build.h("digest"), "5Validator").to_json())
    write("genesis_files.json", json.dumps({"base_files": base_files, "master_files": master_files}))
    write("v1_import.json", V1Import("k1.json", 1, kings=(King(9, "hk9", "v1-e1", "", 5),), served_regs=(), served_hotkeys=()).to_json())
    chain = ChainSnapshot(120, 50, 60, (Registration(0, "owner", 1), *(Registration(uid, f"hk{uid}", 100 + uid, 1) for uid in (1, 2, 3))))
    write("chain.json", chain.to_json())
    write("params.json", sim_params().to_json())
    admission = SimAdmission(LocalStore(state_dir / "store"))
    admission.upload("hk1", 0.3)
    admission.upload("hk2", 0.0)
    common = ["--state-dir", str(state_dir), "--params", str(tmp_path / "params.json")]
    stack = [*common, "--stack", "sn120_v2.sim:stack"]
    files = [f"--{name}={tmp_path / file}" for name, file in (("genesis", "genesis.json"), ("genesis-files", "genesis_files.json"), ("v1-import", "v1_import.json"), ("chain", "chain.json"))]
    store = LocalStore(state_dir / "store")
    store.put(layout.public_master_file(build.h("digest"), "master.bin"), b"master")
    assert main(["bootstrap", *stack, *files]) == 2, "a genesis file that is not in the store stops the first start"
    assert not load(state_dir / JOURNAL_FILE).heads
    store.put(layout.public_base_file(manifest_digest(base_files), "model.bin"), b"base")
    assert main(["bootstrap", *stack, *files]) == 0
    assert main(["bootstrap", *stack, *files]) == 2, "a second bootstrap is refused"
    assert main(["run", *stack, "--rounds", "1", "--max-idle", "1", "--idle-sleep", "0"]) == 0
    state = load(state_dir / JOURNAL_FILE)
    assert [king.uid for king in state.kings] == [9, 1] and state.head.folded and set(state.queue) == {(3, 103)}
    capsys.readouterr()
    assert main(["status", *common]) == 0
    assert json.loads(capsys.readouterr().out)["kings"] == [1, 9]
    SimAdmission(store).upload("hk3", 0.0, valid=False)
    assert main(["status", *stack]) == 0
    assert json.loads(capsys.readouterr().out)["queued_files"] == {"sim_invalid": 1}, "status counts the queued files per verdict rule"
    assert main(["round", *stack]) == 0
    text = capsys.readouterr().out
    assert text.startswith("SN120 v2 round r000001: APPLIED\nwinner: UID 1 (hk1)") and "wall time: train " in text
    assert main(["publish", *stack]) == 0
    assert len(store.list(layout.JOURNAL)) == len(read_journal(state_dir / JOURNAL_FILE)), "status, round and publish never write to the journal"
    other = sim_params(n_tasks=32)
    write("other.json", other.to_json())
    assert main(["run", "--state-dir", str(state_dir), "--params", str(tmp_path / "other.json"), "--stack", "sn120_v2.sim:stack", "--max-idle", "1"]) == 2
    set_params = lambda name: ["set-params", "--state-dir", str(state_dir), "--params", str(tmp_path / name), "--stack", "sn120_v2.sim:stack"]
    capsys.readouterr()
    assert main(set_params("other.json")) == 0
    assert load(state_dir / JOURNAL_FILE).params == other, "new parameters are pinned by a journal record, not by a file on one machine"
    shown = json.loads(capsys.readouterr().out)
    assert shown == {"params_hash_before": sim_params().params_hash, "params_hash": other.params_hash, "changed": {"n_tasks": {"before": 64, "after": 32}}}
    write("short.json", json.dumps({"recipe": {"image_digests": [SIM_IMAGE], "steps": 2}}))
    assert main(set_params("short.json")) == 0
    assert json.loads(capsys.readouterr().out)["changed"] == {
        "batch_cap": {"before": 4, "after": Params().batch_cap},
        "n_tasks": {"before": 32, "after": Params().n_tasks},
        "looks": {"before": [0.5, 1.0], "after": list(Params().looks)},
        "min_scored_tasks": {"before": 16, "after": Params().min_scored_tasks},
        "recipe.steps": {"before": 1, "after": 2},
    }, "a key that the file lacks is back at its default, and the output shows it"
    assert main(set_params("other.json")) == 0
    again = json.loads(capsys.readouterr().out)
    assert again["params_hash"] == other.params_hash and len(again["changed"]) == 5 and load(state_dir / JOURNAL_FILE).params == other
    assert main(set_params("other.json")) == 0 and json.loads(capsys.readouterr().out)["changed"] == {}
