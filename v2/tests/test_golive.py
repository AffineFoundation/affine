"""Go-live rules: the journal copy in the store, the round time limit, pinned parameters, the batch rule, the round summary."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import replace

import pytest

from sn120_v2 import layout, sim
from sn120_v2.canonical import sha256_hex
from sn120_v2.errors import FencedError, InfraError, JournalCorrupt, V2Error
from sn120_v2.journal import JOURNAL_FILE, Journal, Mirror, pull, read_journal
from sn120_v2.params import Params
from sn120_v2.public import round_text
from sn120_v2.queue import waiting
from sn120_v2.sim import STORE_DIR, SimWorld, sim_params
from sn120_v2.state import load_from_store, replay
from sn120_v2.store import LocalStore
from sn120_v2.weights_writer import load_kings


def frozen(record) -> dict[int, str]:
    return {entry.reg.uid: entry.batch.batch_sha256 for lease in record.leases for entry in lease.entries}


def stored_sha(world: SimWorld, hotkey: str) -> str:
    return sha256_hex(world.store.get(layout.miner_object(hotkey)))


# ------------------------------------------------------------ the journal copy in the store


def test_every_record_is_in_the_store_before_it_is_in_the_local_file(tmp_path):
    store = LocalStore(tmp_path / "store")
    journal = Journal(tmp_path / "a" / JOURNAL_FILE, "box-a", 1, Mirror(store))
    journal.append("note", {"n": 1}, 2)
    lines = (tmp_path / "a" / JOURNAL_FILE).read_bytes().splitlines()
    assert [store.get(layout.journal_record(seq)) for seq in range(2)] == lines

    def down(key, data):
        raise InfraError("the store does not answer")

    store.put = down
    with pytest.raises(InfraError):
        journal.append("note", {"n": 2}, 3)
    assert (tmp_path / "a" / JOURNAL_FILE).read_bytes().splitlines() == lines, "a record that is not durable never takes effect"


def test_a_new_machine_continues_from_the_store_and_fences_the_old_one(tmp_path):
    store = LocalStore(tmp_path / "store")
    old = Journal(tmp_path / "a" / JOURNAL_FILE, "box-a", 1, Mirror(store))
    old.append("note", {"n": 1}, 2)
    new = Journal(tmp_path / "b" / JOURNAL_FILE, "box-b", 3, Mirror(store))  # an empty disk
    assert [record.kind for record in new.records] == ["leader", "note", "leader"] and new.fence == 2
    with pytest.raises(FencedError):
        old.append("note", {"n": 2}, 4)
    new.append("note", {"n": 3}, 5)
    assert pull(store, tmp_path / "c" / JOURNAL_FILE) == new.records == read_journal(tmp_path / "b" / JOURNAL_FILE)
    assert len(store.list(layout.JOURNAL)) == 4, "the fenced write left nothing behind"


def test_a_journal_without_a_copy_is_copied_and_a_different_history_is_refused(tmp_path):
    store, path = LocalStore(tmp_path / "store"), tmp_path / "a" / JOURNAL_FILE
    Journal(path, "box-a", 1).append("note", {"n": 1}, 2)  # written before the store kept a copy
    mirrored = Journal(path, "box-a", 3, Mirror(store))
    assert len(store.list(layout.JOURNAL)) == len(mirrored.records) == 3
    Journal(tmp_path / "other" / JOURNAL_FILE, "box-x", 1).append("note", {"n": 9}, 2)
    with pytest.raises(JournalCorrupt):
        pull(store, tmp_path / "other" / JOURNAL_FILE)
    store.delete(layout.journal_record(1))
    with pytest.raises(JournalCorrupt):
        pull(store, tmp_path / "fresh" / JOURNAL_FILE)


def test_the_weights_host_reads_the_kings_from_the_store(tmp_path):
    world = SimWorld(tmp_path / "box", sim_params())
    world.controller.journal.mirror = Mirror(world.store)
    with open(world.root / JOURNAL_FILE, "rb") as handle:
        Mirror(world.store).catch_up(handle, backfill=True)
    world.register(1, 0.3)
    assert world.round().outcome == "applied"
    host = tmp_path / "AffineValidator"  # a host with no journal of its own
    keys = world.store.list("")
    state = load_from_store(world.store, host / JOURNAL_FILE)
    assert state.kings == world.controller.state.kings and world.store.list("") == keys, "the read never writes to the store"
    kings, validator_hotkey = load_kings(host)  # the weights writer's own reader, on the pulled file
    assert kings == list(state.kings) and validator_hotkey == state.genesis.validator_hotkey

    def down(prefix):
        raise InfraError("the store does not answer")

    world.store.list = down
    with pytest.raises(InfraError):
        load_from_store(world.store, host / JOURNAL_FILE)
    assert load_kings(host)[0] == kings, "the last valid kings stay when the store is down"


# ------------------------------------------------------------ the batch rule (Jacob's rule 6)


def test_a_waiting_seat_may_replace_its_file_until_it_is_evaluated(tmp_path):
    world = SimWorld(tmp_path, sim_params())
    world.register(1, 0.0)
    world.register(2, 0.0)
    world.register(3, 0.0)
    place = lambda uid: [seat.reg.uid for seat in sorted(world.controller.state.queue.values(), key=lambda s: (not s.front, s.reg.registration_block))].index(uid)
    # Before the lease: the file may change any number of times; the round takes the one that is there at the lease.
    world.admission.upload("hk1", 0.1)
    world.admission.upload("hk1", 0.3)
    world.admission.upload("hk3", 0.0, valid=False)  # a replacement is checked again: this one is invalid
    world.trainer.fail_hotkeys = {"hk1"}  # our failure: no_decision for seat 1
    first = world.round()
    assert frozen(first) == {1: stored_sha(world, "hk1"), 2: stored_sha(world, "hk2")} and first.skipped == ((3, 1003),)
    queue = world.controller.state.queue
    assert set(queue) == {(1, 1001), (3, 1003)} and "hk1" not in world.controller.state.served_hotkeys, "not evaluated, so not charged"
    assert place(1) == 0, "a seat that comes back after no_decision keeps its place at the front"
    # After no_decision and after a skip: the seat still waits, so its file may change again.
    world.trainer.fail_hotkeys = set()
    world.admission.upload("hk1", 0.25)
    world.admission.upload("hk3", 0.0)
    second = world.round()
    assert frozen(second) == {1: stored_sha(world, "hk1"), 3: stored_sha(world, "hk3")} and frozen(second)[1] != frozen(first)[1]
    assert [entry.reg.uid for lease in second.leases for entry in lease.entries] == [1, 3]
    # Evaluated: the turn is used. A newer file gets no second turn.
    world.admission.upload("hk1", 0.5)
    assert world.controller.state.queue == {} and world.round() is None


# ------------------------------------------------------------ the round time limit


def limited(tmp_path, stop_after_chunks: int, fail_next: bool = False) -> SimWorld:
    """A world whose clock jumps to the round's time limit right after ``stop_after_chunks`` scored chunks."""
    world = SimWorld(tmp_path, sim_params(eval_chunk_tasks=16))  # 64 tasks, looks at 32 and 64, so 4 chunks
    real, calls = world.evaluator.evaluate, []

    def evaluate(request):
        if fail_next and len(calls) == stop_after_chunks:
            world.now += 1
            raise InfraError("a rollout was still running at the deadline")
        chunk = real(request)
        calls.append(request)
        if len(calls) == stop_after_chunks:
            world.now += world.params.round_time_limit_s - world.params.promote_reserve_s
        return chunk

    world.evaluator.evaluate = evaluate
    world.calls = calls
    return world


@pytest.mark.parametrize("fail_next", [False, True])
def test_at_the_time_limit_a_live_lora_is_judged_on_the_whole_chunks_scored_so_far(tmp_path, fail_next):
    world = limited(tmp_path, 3, fail_next)
    world.register(1, 0.4)
    world.register(2, 0.0)
    record = world.round()
    params = world.params
    limit = record.opened_at + params.round_time_limit_s - params.promote_reserve_s
    assert [request.deadline for request in world.calls] == [limit] * 3, "the limit cuts every look, the first one too"
    assert {entry.uid: entry.score.n_tasks for entry in record.decision.entries} == {1: 48, 2: 48}, "the same prefix for every LoRA and for null M"
    assert record.outcome == "applied" and record.decision.winner.uid == 1, "the final rule, z > 1, on what was scored"
    assert world.controller.state.served_hotkeys == {"hk1", "hk2"} and world.controller.state.queue == {}
    report = json.loads(world.store.get(layout.round_file(record.round_id)))
    assert report["stopped_at_limit"] and "(stopped at the time limit)" in round_text(report)
    assert report["wall_s"]["lease_to_promotion"] <= params.round_time_limit_s


def test_the_limit_cuts_the_first_look_too_and_those_loras_are_not_charged_and_never_put_on_hold(tmp_path):
    world = limited(tmp_path, 1)  # the round's time limit passes inside the first look, after 16 of its 32 tasks
    world.register(1, 0.4)
    world.register(2, 0.0)
    for _ in range(world.params.max_infra_attempts + 1):
        world.calls.clear()
        record = world.round()
        assert len(world.calls) == 1, "no block of tasks starts after the limit"
        assert record.outcome == "no_decision" and {entry.status for entry in record.decision.entries} == {"not_scored"}
        assert "after 16 tasks, before the first look of 32" in record.decision.entries[0].reason
        queue = world.controller.state.queue
        assert set(queue) == {(1, 1001), (2, 1002)}, "both keep their place"
        assert all(seat.front and not seat.leased and not seat.held and seat.infra_attempts == 0 for seat in queue.values())
    assert world.controller.state.served_hotkeys == set() and len(world.controller.state.heads) == 1 and world.store.list("pending/") == []
    report = json.loads(world.store.get(layout.round_file(record.round_id)))
    assert report["wall_s"]["total"] <= world.params.round_time_limit_s
    assert "not scored: 2 of 2 batches" in round_text(report) and "(stopped at the time limit)" in round_text(report)
    world.register(7, 0.0, block=900)  # registered earlier than both, and it never had a turn
    order = [seat.reg.uid for seat in waiting(world.controller.state.queue)]
    assert order == [1, 2, 7], "the not_scored seats keep their order among themselves, ahead of the rest of the queue"


def slow_first_job(world: SimWorld) -> None:
    """Each training takes the whole training time of the round."""
    real = world.trainer.train

    def train(request):
        artifact = real(request)
        world.now += world.params.train_time_budget_s
        return artifact

    world.trainer.train = train


def test_a_batch_that_was_not_trained_for_lack_of_time_is_first_next_round_and_is_never_put_on_hold(tmp_path):
    world = SimWorld(tmp_path, sim_params(n_gpus=1, batch_cap=5))  # one GPU: one batch trains at a time
    slow_first_job(world)
    for uid in range(1, 6):
        world.register(uid, 0.0)
    for trained, waiting_seats in ((1, [2, 3, 4, 5]), (2, [3, 4, 5]), (3, [4, 5])):
        record = world.round()
        statuses = {entry.uid: entry.status for entry in record.decision.entries}
        assert statuses == {trained: "lost"} | {uid: "not_trained" for uid in waiting_seats}, "the trained LoRA got its full test"
        assert record.decision.entries[0].score.n_tasks == 64 and record.outcome == "null"
        queue = world.controller.state.queue
        assert sorted(key[0] for key in queue) == waiting_seats and world.controller.state.served_hotkeys == {f"hk{uid}" for uid in range(1, trained + 1)}
        assert all(seat.front and not seat.leased and not seat.held and seat.infra_attempts == 0 for seat in queue.values())
    assert world.params.max_infra_attempts == 3, "seats 4 and 5 were not trained three times, and that is not a failure of theirs or ours"
    report = json.loads(world.store.get(layout.round_file(record.round_id)))
    assert "not trained: 2 of 3 batches" in round_text(report) and "the training time of the round ended before this batch could start" in round_text(report)
    world.register(6, 0.0, block=1)  # an older registration does not pass the seats that wait at the front
    assert [entry.reg.uid for lease in world.round().leases for entry in lease.entries] == [4, 5, 6]


def test_at_the_end_of_the_training_time_a_running_job_is_stopped_and_the_scoring_starts(tmp_path):
    world = SimWorld(tmp_path, sim_params(n_gpus=2))  # two slots: A and B start at the lease, C starts when A is done
    world.runner.poll_s = 0.01
    real, close, stopped, started = world.trainer.train, world.trainer.close, threading.Event(), []

    def train(request):
        hotkey = request.entry.reg.hotkey
        started.append(hotkey)
        if hotkey == "hk1":
            return real(request)
        if hotkey == "hk3":
            world.now += world.params.train_time_budget_s  # the training time ends while B and C still run
        assert stopped.wait(10), "the runner must stop a job that runs past the training time"
        raise InfraError("the training job was stopped")

    world.trainer.train = train
    world.trainer.close = lambda: (stopped.set(), close())
    for uid in (1, 2, 3):
        world.register(uid, 0.3)
    record = world.round()
    by_uid = {entry.uid: entry for entry in record.decision.entries}
    assert sorted(started) == ["hk1", "hk2", "hk3"], "a stopped job is not started again"
    assert (by_uid[1].status, by_uid[1].score.n_tasks, record.outcome) == ("winner", 64, "applied"), "the trained LoRA had its whole test"
    assert by_uid[2].status == "no_decision" and "had the whole training time" in by_uid[2].reason
    assert by_uid[3].status == "not_trained" and "while this batch trained" in by_uid[3].reason
    queue = world.controller.state.queue
    assert {key[0]: (seat.front, seat.infra_attempts) for key, seat in queue.items()} == {2: (True, 1), 3: (True, 0)}
    assert world.controller.state.served_hotkeys == {"hk1"} and world.store.list("pending/") == []


def test_a_box_that_starts_a_round_after_its_training_time_charges_nobody(tmp_path):
    world = SimWorld(tmp_path, sim_params())
    world.register(1, 0.3)
    world.register(2, 0.0)
    ticket = world.controller.open_round(world.clock(), "entropy")[0]
    world.now += world.params.train_time_budget_s  # for example a box that started again in the middle of the round
    world.controller.submit_result(world.runner.run(ticket), world.clock())
    world.controller.tick(world.clock())
    record = world.controller.state.rounds[-1]
    assert record.outcome == "no_decision" and {entry.status for entry in record.decision.entries} == {"not_trained"}
    queue = world.controller.state.queue
    assert len(queue) == 2 and all(seat.front and seat.infra_attempts == 0 and not seat.leased for seat in queue.values())


def test_the_launch_limits_are_32_batches_100_minutes_and_temperature_1():
    params = Params()
    assert (params.batch_cap, params.round_time_limit_s, params.temperature) == (32, 6000, 1.0)
    assert params.train_time_budget_s < params.round_time_limit_s - params.promote_reserve_s < params.round_time_limit_s <= params.lease_timeout_s <= 2 * 3600
    assert params.eval_chunk_tasks == params.look_sizes(params.n_tasks)[0], "the first look is one chunk"
    with pytest.raises(V2Error):
        Params(train_time_budget_s=5400)  # no time left for the scoring


def test_batches_train_at_the_same_time_on_separate_gpu_slots(tmp_path):
    world = SimWorld(tmp_path, sim_params(batch_cap=8))
    real, slots = world.trainer.train, []

    def train(request):
        slots.append(request.gpu_ids)
        return real(request)

    world.trainer.train = train
    for uid in range(1, 9):
        world.register(uid, 0.0)
    world.round()
    assert sorted(slots) == [(gpu,) for gpu in range(8)], "8 batches on 8 GPUs: one GPU each, no GPU twice"


# ------------------------------------------------------------ pinned parameters


def test_the_parameters_are_pinned_in_the_journal_at_the_first_start(tmp_path):
    capped = sim_params()
    capped = replace(capped, recipe=replace(capped.recipe, global_norm_cap=2.5))
    world = SimWorld(tmp_path, capped)
    world.params = sim_params()  # a later process is started with other parameters
    controller = world.reopen()
    assert controller.params == capped and controller.params.recipe.global_norm_cap == 2.5, "the journal's set is the one in force"
    world.register(1, 0.3)
    tickets = controller.open_round(world.clock(), "entropy")
    assert tickets[0].params == capped
    with pytest.raises(V2Error):
        controller.set_params(sim_params(), world.clock())
    assert replay(read_journal(tmp_path / JOURNAL_FILE)).params == capped


def test_a_decided_round_whose_fold_fails_takes_new_digests_and_nothing_else(tmp_path):
    world = SimWorld(tmp_path, sim_params())
    world.register(1, 0.3)
    world.folder.fail = True  # the fold code has a fault; its fix changes the code digest
    assert world.round() is None and world.controller.state.round.applied
    fixed = replace(world.params, recipe=replace(world.params.recipe, image_digests=("image-after-the-fix",), code_digest="c" * 64))
    with pytest.raises(V2Error):
        world.controller.set_params(replace(fixed, z_bar=0.0), world.clock())
    world.controller.set_params(fixed, world.clock())
    world.folder.fail = False
    world.now += world.params.apply_retry_s
    world.controller.tick(world.clock())
    state = world.controller.state
    assert state.round is None and state.head.folded and state.rounds[-1].outcome == "applied" and state.params == fixed
    assert replay(read_journal(tmp_path / JOURNAL_FILE)).params == fixed


# ------------------------------------------------------------ the round summary


def test_the_round_summary_has_every_lora_the_winner_and_the_wall_times(tmp_path):
    world = SimWorld(tmp_path, sim_params())
    for uid, quality in ((1, 0.3), (2, 0.0), (3, -0.25)):
        world.register(uid, quality)
    record = world.round()
    report = json.loads(world.store.get(layout.round_file(record.round_id)))
    assert [(e["uid"], e["status"]) for e in report["entries"]] == [(1, "winner"), (2, "lost"), (3, "lost")]
    assert all({"delta", "z", "cap_rate", "reason", "step_norm"} <= set(e) for e in report["entries"])
    assert {"training", "evaluation", "promote", "fold", "lease_to_promotion", "total"} <= set(report["wall_s"])
    assert report["new_head_version_id"] == world.controller.state.head.version_id and report["head_version_id"] == world.controller.state.heads[0].version_id
    lines = round_text(report).splitlines()
    assert lines[0] == f"SN120 v2 round {record.round_id}: APPLIED" and lines[1].startswith("winner: UID 1 (hk1), delta +")
    assert "dropped at look 0" in lines[-2] and lines[-1].startswith("wall time: train ")
    kings = list(world.controller.state.kings)
    world.register(4, 0.0)
    null = world.round()
    assert null.outcome == "null" and world.controller.state.kings == kings, "a round with no winner leaves the kings as they are"
    assert "no winner: null round" in round_text(json.loads(world.store.get(layout.round_file(null.round_id))))


# ------------------------------------------------------------ the file verdicts in the public snapshot


def shown(world: SimWorld) -> dict[str, bool]:
    return {seat["hotkey"]: seat["file_valid"] for seat in world.publisher.snapshots[-1]["queue"]["next"]}


def test_the_public_snapshot_shows_admissions_verdict_on_each_waiting_file(tmp_path):
    world = SimWorld(tmp_path, sim_params())
    world.register(1, None)  # no file
    world.register(2, 0.0, valid=False)
    assert world.round() is None, "no valid file, so no round"
    assert world.controller.publish(world.clock(), only_changes=True) and shown(world) == {"hk1": False, "hk2": False}
    count = len(world.publisher.snapshots)
    assert world.round() is None and not world.controller.publish(world.clock(), only_changes=True)
    assert len(world.publisher.snapshots) == count, "nothing changed, so nothing is published again"
    world.admission.upload("hk1", 0.0)
    assert world.controller.publish(world.clock(), check_files=True, only_changes=True)
    assert shown(world) == {"hk1": True, "hk2": False}, "a new file shows as valid before any round takes it"
    checked = world.publisher.snapshots[-1]["queue"]["files_checked_at"]
    assert checked is not None and checked <= world.now


def test_the_waiting_files_are_read_before_the_lease_starts_and_never_inside_a_round(tmp_path):
    world = SimWorld(tmp_path, sim_params())
    world.register(1, 0.3)
    world.register(2, None)
    real, calls = world.admission.verdicts, []

    def slow(hotkeys):
        calls.append(world.now)
        world.now += 1000  # reading and checking every waiting file takes time
        return real(hotkeys)

    world.admission.verdicts = slow
    record = world.round()
    assert record.outcome == "applied" and len(calls) == 1, "one read of the waiting files for the whole round"
    assert record.opened_at > calls[0] + 1000, "the lease, and so the round's time limit, starts after the files are read"
    assert all(lease.expires_at == record.opened_at + world.params.lease_timeout_s for lease in record.leases)
    assert shown(world) == {"hk2": False} and world.publisher.snapshots[-1]["queue"]["files_checked_at"] == calls[0]
    world.register(3, 0.0)
    world.controller.open_round(world.clock(), "entropy")
    with pytest.raises(V2Error, match="round is open"):
        world.controller.publish(world.clock(), check_files=True)


# ------------------------------------------------------------ a bucket shared with v1

V1_OBJECTS = {  # keys that v1's source names for its bucket (CUTOVER.md, "Store"), and the file of a hotkey that does not wait
    "public/current.json": b"an old mock of v1",
    "public/streams/nonpayable-live-reward-math-v1-/current.json": b"v1's live pointer",
    "public/checkpoints/000123/model.safetensors": b"v1 weights",
    "public/epoch-evidence/000123.json": b"v1 evidence",
    "private/ff/v1/turns/000123/hkA/batch.json": b"a turn of v1",
    "private/ff/v1/miners/hkGone/batch.json": b"the file of a hotkey that left",
}
STORE_CALLS = ("put", "put_file", "get", "get_file", "exists", "size", "sha256", "copy", "delete", "list")
NARROW_LIST = re.compile(r"private/journal|pending/(rounds|folds)/[^/]+")


class WatchedStore(LocalStore):
    """Notes every call as (method, key or prefix). ``copy`` notes both keys."""

    def __init__(self, root) -> None:
        self.calls: list[tuple[str, str]] = []
        super().__init__(root)

    def __getattribute__(self, name):
        method = super().__getattribute__(name)
        if name not in STORE_CALLS:
            return method

        def noted(*args, **kwargs):
            self.calls.extend((name, key) for key in args[: 2 if name == "copy" else 1])
            return method(*args, **kwargs)

        return noted


@pytest.mark.skipif(layout.CURRENT in V1_OBJECTS, reason="the head pointer still has a key of v1; this test starts when layout.CURRENT moves")
def test_on_a_bucket_shared_with_v1_nothing_lists_a_broad_prefix_or_touches_an_object_of_v1(tmp_path, monkeypatch):
    seed = LocalStore(tmp_path / STORE_DIR)
    for key, data in V1_OBJECTS.items():
        seed.put(key, data)
    monkeypatch.setattr(sim, "LocalStore", WatchedStore)
    world = SimWorld(tmp_path, sim_params())  # bootstraps on the shared bucket

    def mirror() -> None:
        world.controller.journal.mirror = Mirror(world.store)
        with open(world.root / JOURNAL_FILE, "rb") as handle:
            Mirror(world.store).catch_up(handle, backfill=True)

    mirror()
    for uid, quality in ((1, 0.3), (2, 0.0), (3, None)):
        world.register(uid, quality)
    assert world.round().outcome == "applied" and len(world.controller.state.heads) > 1, "a winner: promotion, a fold, a new head"
    world.reopen()  # a restart reads the journal copy and the task table back
    mirror()
    world.register(4, 0.25)
    assert world.round().outcome == "applied"
    load_from_store(world.store, tmp_path / "AffineValidator" / JOURNAL_FILE)
    calls = world.store.calls
    assert {key: seed.get(key) for key in V1_OBJECTS} == V1_OBJECTS
    assert [call for call in calls if call[1] in V1_OBJECTS] == [], "no probe, read, write or delete of an object of v1"
    lists = {prefix for name, prefix in calls if name == "list"}
    assert lists and all(NARROW_LIST.fullmatch(prefix) for prefix in lists), "a list names the journal, or one round or one fold in pending/"


def test_a_store_bootstrapped_with_the_old_pointer_key_gets_the_new_pointer_at_the_next_tick(tmp_path):
    """A state made before ``layout.CURRENT`` moved replays as it is. The first tick writes the pointer; the old key stays as it was."""
    world = SimWorld(tmp_path, sim_params())
    old, pointer = "public/current.json", world.store.get(layout.CURRENT)
    world.store.put(old, pointer)
    world.store.delete(layout.CURRENT)
    with open(world.root / JOURNAL_FILE, "rb") as handle:
        Mirror(world.store).catch_up(handle, backfill=True)
    head = world.controller.state.head
    assert load_from_store(world.store, tmp_path / "weights-writer" / JOURNAL_FILE).head == head, "the weights writer reads the journal, not the pointer"
    world.reopen()
    world.register(1, 0.3)
    assert world.round().outcome == "applied", "run_round ticks before it reads any file"
    assert world.store.get(layout.CURRENT) == world.controller.state.manifest.to_json() != pointer
    assert world.store.get(old) == pointer
