"""Go-live rules: the journal copy in the store, the round time limit, pinned parameters, the batch rule, the round summary."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import sha256_hex
from sn120_v2.errors import FencedError, InfraError, JournalCorrupt, V2Error
from sn120_v2.journal import JOURNAL_FILE, Journal, Mirror, pull, read_journal
from sn120_v2.params import Params
from sn120_v2.public import round_text
from sn120_v2.sim import SimWorld, sim_params
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
    """A world whose clock reaches the box deadline right after ``stop_after_chunks`` scored chunks."""
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
    assert len(world.calls) == 3 and all(request.deadline == record.opened_at + 5400 for request in world.calls)
    assert {entry.uid: entry.score.n_tasks for entry in record.decision.entries} == {1: 48, 2: 48}, "the same prefix for every LoRA and for null M"
    assert record.outcome == "applied" and record.decision.winner.uid == 1, "the final rule, z > 1, on what was scored"
    assert world.controller.state.served_hotkeys == {"hk1", "hk2"} and world.controller.state.queue == {}
    report = json.loads(world.store.get(layout.round_file(record.round_id)))
    assert report["stopped_at_limit"] and "(stopped at the time limit)" in round_text(report)
    assert report["wall_s"]["lease_to_promotion"] <= world.params.round_time_limit_s


def test_a_time_limit_before_the_first_look_charges_nobody(tmp_path):
    world = limited(tmp_path, 1)
    world.register(1, 0.4)
    world.register(2, 0.0)
    record = world.round()
    assert record.outcome == "no_decision" and len(world.controller.state.heads) == 1
    queue = world.controller.state.queue
    assert set(queue) == {(1, 1001), (2, 1002)} and all(seat.front and not seat.leased for seat in queue.values())
    assert world.controller.state.served_hotkeys == set() and world.store.list("pending/") == []


def test_a_batch_that_cannot_start_training_before_the_limit_keeps_its_place(tmp_path):
    world = SimWorld(tmp_path, sim_params(n_gpus=1))
    real = world.trainer.train

    def slow(request):
        artifact = real(request)
        world.now += world.params.round_time_limit_s - world.params.promote_reserve_s
        return artifact

    world.trainer.train = slow
    world.register(1, 0.4)
    world.register(2, 0.0)
    record = world.round()
    statuses = {entry.uid: entry.status for entry in record.decision.entries}
    assert statuses == {1: "no_decision", 2: "no_decision"} and set(world.controller.state.queue) == {(1, 1001), (2, 1002)}


def test_the_launch_limits_are_32_batches_100_minutes_and_temperature_1():
    params = Params()
    assert (params.batch_cap, params.round_time_limit_s, params.lease_timeout_s, params.temperature) == (32, 6000, 6000, 1.0)
    assert params.promote_reserve_s < params.round_time_limit_s


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
