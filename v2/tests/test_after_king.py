"""The order of the steps after a king, ``SN120_AFTER_KING=overlap`` (``afterking.py``), on a real tiny head.

Three things are proven here.

* ``overlap`` writes the journal and the store that ``serial`` writes: the same records, the same bytes.
* Each new point where a ``run`` can stop keeps the king, the head and the chain of hashes: the next ``run``
  finishes the step. A new ``run`` is a new controller, a ``Resident`` that has verified nothing, and the
  proof of the local head at the start.
* A head that does not read back as the fold wrote it: no result of a round on it is stored, the lease fails
  as a fault of the box does, ``run`` stops, and the next start rebuilds the head from the chain.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import threading
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sn120_v2.sim_head import ChainWorld  # noqa: E402 - after the skip guard

from sn120_v2 import cli, controller, headfold, layout  # noqa: E402
from sn120_v2.afterking import AHEAD_MAX_AGE_S, OVERLAP, PUBLISH, SERIAL, Beside, Overlap  # noqa: E402
from sn120_v2.chain import FakeChain, Feed, Reader  # noqa: E402
from sn120_v2.controller import PUBLISH_TRIES, Stack  # noqa: E402
from sn120_v2.errors import HeadLost, InfraError, VerifyError  # noqa: E402
from sn120_v2.gpu.resident import hash_tree  # noqa: E402
from sn120_v2.headchain import CHAIN, check_chain, checkpoints, heads_of  # noqa: E402
from sn120_v2.headfold import READ_BACK, inline  # noqa: E402
from sn120_v2.journal import JOURNAL_FILE, read_journal  # noqa: E402
from sn120_v2.sampling import make_round_seed  # noqa: E402
from sn120_v2.sim import sim_params  # noqa: E402
from sn120_v2.state import APPLY_DEFERRED, HEAD_APPLIED, HEAD_FOLDED, LEASE_FAILED, ROUND_CLOSED, ROUND_OPENED, ROUND_SETTLED  # noqa: E402
from sn120_v2.types import HeadManifest  # noqa: E402

WAIT_S = 30  # the longest a test waits for a thread


def a_world(path: Path, order: str = OVERLAP, **settings) -> ChainWorld:
    world = ChainWorld(path, sim_params(), order=order, **settings)
    world.start(CHAIN)
    return world


@pytest.fixture
def world(tmp_path) -> ChainWorld:
    return a_world(tmp_path / "world")


def records(world: ChainWorld) -> list[tuple[str, int, dict]]:
    return [(record.kind, record.at, record.data) for record in read_journal(world.root / JOURNAL_FILE)]


def kinds(world: ChainWorld) -> list[str]:
    return [kind for kind, _, _ in records(world)]


def stored(world: ChainWorld) -> dict[str, bytes]:
    return {key: world.store.get(key) for key in world.store.list("")}


def cache(world: ChainWorld) -> list[str]:
    return sorted(path.name for path in (world.box.work_dir / "cache").iterdir())


def is_head(world: ChainWorld, manifest: HeadManifest) -> bool:
    """True when the cache of the box holds exactly the base and the master of ``manifest``."""
    head = manifest.head
    return hash_tree(world.local(head.base_hash)) == manifest.base_files and hash_tree(world.local(head.base_fp32_master_digest)) == manifest.master_files


def kept(world: ChainWorld) -> None:
    """The steps that wait for a next open, done now: ``run`` does them in a pass with no round, and at its end."""
    assert world.overlap.flush(world.now) == []


def new_run(world: ChainWorld) -> None:
    """The process of ``run`` died and the loop started a new one: nothing is verified in its memory, and its start proves the local head."""
    world.wait()
    if world.folder._read_back is not None:  # noqa: SLF001 - a dead process leaves no thread; the test waits for it instead
        world.folder._read_back.wait()  # noqa: SLF001
    world.folder.resident._verified.clear()  # noqa: SLF001
    world.start()


def without_time(snapshot: dict) -> dict:
    return {key: value for key, value in snapshot.items() if key != "generated_at"}


class Crash(BaseException):
    """The process of ``run`` dies."""


def crash_at(world: ChainWorld, kind: str, before: bool) -> None:
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


def changed_on_disk(world: ChainWorld) -> None:
    """Each fold from now writes a master file whose bytes on the disk are not the bytes that it hashed: a fault of the disk."""

    def faulty(jobs):
        results = inline(jobs)
        target = sorted(file for file in jobs[0].out_master.iterdir() if file.is_file())[0]
        raw = bytearray(target.read_bytes())
        raw[-1] ^= 1
        target.write_bytes(bytes(raw))
        return results

    world.folder.run = faulty


def gate_read_back(monkeypatch) -> threading.Event:
    """The second sha256 of a new head waits for the returned event, as it does for minutes with 56 GB."""
    gate, real = threading.Event(), headfold.file_sha256

    def slow(path):
        if threading.current_thread().name.startswith(READ_BACK):
            assert gate.wait(WAIT_S)
        return real(path)

    monkeypatch.setattr(headfold, "file_sha256", slow)
    return gate


# ---- the same records and the same bytes as ``serial`` ----


def a_story(world: ChainWorld) -> ChainWorld:
    """Three kings, a round with no winner, a seat with no file, and a pass with no round."""
    for _ in range(2):
        world.king()
    world.register(80, 0.0)
    world.register(81, 0.0)
    assert world.round().outcome == "null"
    world.king()
    world.register(90, None)
    assert world.round() is None
    world.wait()
    return world


def test_overlap_writes_the_journal_and_the_store_that_serial_writes(tmp_path):
    serial, overlap = a_story(a_world(tmp_path / "serial", SERIAL)), a_story(a_world(tmp_path / "overlap"))
    assert serial.controller.overlap is None and overlap.controller.overlap is overlap.overlap
    kept(overlap)
    assert records(overlap) == records(serial), "the same records with the same times, in the same order"
    assert stored(overlap) == stored(serial), "each key of the store, public and private, with the same bytes"
    assert cache(overlap) == cache(serial) and is_head(overlap, overlap.head) and overlap.head == serial.head
    assert not overlap.store.list("pending/") and overlap.folder._read_back is not None  # noqa: SLF001
    overlap.folder.proven(True)  # the second sha256 of the last head passed
    # The public snapshots: ``overlap`` shows each open round as ``serial`` does. It leaves out the snapshot of a
    # close when the next open comes at once: the snapshot of that open shows the close too.
    shown, all_serial = [without_time(snapshot) for snapshot in overlap.publisher.snapshots], [without_time(snapshot) for snapshot in serial.publisher.snapshots]
    opens = [snapshot for snapshot in all_serial if snapshot["round"] is not None]
    assert [snapshot for snapshot in shown if snapshot["round"] is not None] == opens and len(opens) == 4
    assert len(shown) < len(all_serial) and shown[-1]["kings"] == all_serial[-1]["kings"] and shown[-1]["head"] == all_serial[-1]["head"]


def test_serial_is_the_order_of_a_world_with_no_setting_and_it_makes_no_thread_of_this_module(tmp_path):
    world = a_world(tmp_path / "world", SERIAL)
    before = {thread.name for thread in threading.enumerate()}
    world.king()
    assert world.overlap is None and world.folder._read_back is None  # noqa: SLF001
    assert not {thread.name for thread in threading.enumerate()} - before


# ---- beside the fold ----


def test_the_fold_runs_while_the_winners_files_go_to_public_and_the_head_moves_only_after_both(world):
    """With ``serial`` this test cannot end: the public copy waits for the fold to start, and that fold waits for the copy."""
    folding, order = threading.Event(), []

    def fold(jobs):
        folding.set()
        order.append(("fold", bool(world.store.list("public/loras/")), kinds(world).count(HEAD_APPLIED)))
        return inline(jobs)

    def wait_for_the_fold(key):
        if key.startswith("public/loras/"):
            assert folding.wait(WAIT_S), "the fold did not start beside the public copy"
        return False

    world.folder.run, world.store.refuse = fold, wait_for_the_fold
    manifest = world.king()
    assert order == [("fold", False, 0)], "one fold, started before a file of the LoRA was public and before HEAD_APPLIED"
    stacked = world.state.manifests[manifest.head.parent_version_id].head
    assert len(world.store.list(layout.public_lora_prefix(stacked.loras[0].lora_hash))) == 2 and world.store.exists(layout.winner_batch(stacked.loras[0].round_id))
    assert kinds(world)[-4:] == [ROUND_SETTLED, HEAD_APPLIED, HEAD_FOLDED, ROUND_CLOSED] and is_head(world, manifest)


@pytest.mark.parametrize("fault", ["store", "bytes"])
def test_when_the_winners_files_do_not_reach_public_the_fold_is_dropped_and_the_head_does_not_move(world, fault):
    start, before = world.king(), cache(world)
    folds = []
    world.folder.run = lambda jobs: folds.append(1) or inline(jobs)
    if fault == "store":
        world.store.refuse = lambda key: key.startswith("public/loras/")
    else:  # the frozen batch of the winner is gone: the batch cannot become public, so the king is abandoned as with ``serial``
        run = world.runner.run

        def lose_the_batches(ticket):
            result = run(ticket)
            for entry in ticket.entries:
                world.store.delete(entry.batch.uri)
            return result

        world.runner.run = lose_the_batches
    world.register(90, 0.3)
    world.register(91, 0.0)
    record = world.round()
    assert folds == [1], "the fold ran beside the copy"
    assert world.state.head == start.head and HEAD_APPLIED not in kinds(world)[kinds(world).index(ROUND_SETTLED, -4) :]
    assert set(cache(world)) == {start.head.base_hash, start.head.base_fp32_master_digest} < set(before) and is_head(world, start), "no file of the dropped fold is in the cache, and the head is whole"
    world.folder.proven(True)  # no second sha256 of a dropped fold is asked for
    if fault == "bytes":
        assert record.outcome == "abandoned" and len(world.state.kings) == 1 and json.loads(world.store.get(layout.round_file(record.round_id)))["note"] == "the winner's frozen batch is missing"
        kept(world)
        assert not world.store.list("pending/")
        world.runner.run = run
        folded = world.king().head
        assert world.state.heads[-2].version_id == folded.parent_version_id and world.state.heads[-2].parent_version_id == start.head.version_id, "the next king folds on the head that stayed"
        return
    assert record is None and kinds(world)[-1] == APPLY_DEFERRED and world.state.round is not None
    world.store.refuse = None
    world.now += world.params.apply_retry_s
    assert world.round() is None and world.state.rounds[-1].outcome == "applied" and folds == [1, 1], "the fold is made again for the head of the second try"
    assert [king.uid for king in world.state.kings][-1] == 90 and world.state.head.folded and is_head(world, world.head)
    kept(world)
    assert not world.store.list("pending/")
    check_chain(world.chain_file())


def test_a_fold_beside_that_fails_is_made_again_after_the_wait_as_a_fold_of_serial(world):
    start, real = world.king(), world.folder.run

    def faulty(jobs):
        raise InfraError("the fold process died")

    world.folder.run = faulty
    world.register(90, 0.3)
    assert world.round() is None
    stacked = world.state.head
    assert not stacked.folded and kinds(world)[-1] == HEAD_APPLIED and "fold failed: the fold process died" in world.controller.last_error
    assert world.store.list(layout.public_lora_prefix(stacked.loras[0].lora_hash)), "the winner's files are public: the king is in the journal"
    world.folder.run = real
    world.now += world.params.apply_retry_s
    assert world.round() is None, "the pass closed the round of the king, and no seat waits for a next one"
    assert world.state.rounds[-1].outcome == "applied" and world.state.round is None and world.state.head.parent_version_id == stacked.version_id
    assert is_head(world, world.head) and world.head.head.parent_version_id == stacked.version_id and world.head != start


# ---- a ``run`` that dies ----


@pytest.mark.parametrize("before", [False, True], ids=["after-write", "before-write"])
@pytest.mark.parametrize("kind", [ROUND_OPENED, ROUND_SETTLED, HEAD_APPLIED, HEAD_FOLDED, ROUND_CLOSED])
def test_a_run_that_dies_at_any_record_is_replaced_with_one_king_a_whole_head_and_a_whole_chain(world, kind, before):
    old = world.king()
    world.register(90, 0.3)
    world.register(91, 0.0)
    world.register(92, None)
    crash_at(world, kind, before)
    with pytest.raises(Crash):
        world.round()
    assert is_head(world, old), "the files of the last recorded head are on the box at each of these points"
    new_run(world)
    assert is_head(world, world.head), "the start of the new run found or made the files of the journal's head"
    world.round()
    state = world.state
    assert [king.uid for king in state.kings][1:] == [90] and kinds(world).count(HEAD_APPLIED) == 2 and kinds(world).count(HEAD_FOLDED) == 2
    assert state.round is None and state.head.folded and [record.outcome for record in state.rounds] == ["applied", "applied"]
    assert [(key, seat.leased) for key, seat in state.queue.items()] == [((92, 1092), False)]
    assert is_head(world, world.head) and world.head.head.parent_version_id != old.head.version_id
    world.folder.proven(True)
    kept(world)
    world.wait()
    doc = world.chain_file()
    check_chain(doc)
    assert [head["version_id"] for head in heads_of(doc)] == [head.version_id for head in state.heads if head.folded]
    assert not world.store.list("pending/"), "the staged files of the round and of its fold are deleted, whichever run made them"
    last = world.publisher.snapshots[-1]
    assert 90 in [king["uid"] for king in last["kings"]] and last["heads"][-1] == {"version_id": state.head.version_id, "round_id": state.head.round_id, "applied_at": state.head.applied_at, "folded": True}, "and the public files show the king"


def test_a_run_that_dies_in_the_fold_beside_has_recorded_no_head_and_the_next_run_names_the_king(world):
    old = world.king()

    def dies(jobs):
        raise Crash("in the fold")

    world.folder.run = dies
    world.register(90, 0.3)
    with pytest.raises(Crash):
        world.round()
    assert kinds(world)[-1] == ROUND_SETTLED and world.state.head == old.head and is_head(world, old)
    new_run(world)
    assert world.round() is None and world.state.rounds[-1].outcome == "applied" and [king.uid for king in world.state.kings][1:] == [90]
    assert kinds(world).count(HEAD_APPLIED) == 2 and is_head(world, world.head)
    kept(world)
    assert not world.store.list("pending/")
    check_chain(world.chain_file())


def test_a_run_that_dies_before_the_second_sha256_ended_finds_a_changed_head_at_its_next_start(world, monkeypatch, capsys):
    """The read back is memory of one process. The proof at the start of ``run`` is the same hash, so a new ``run`` cannot miss the fault."""
    world.king()
    gate = gate_read_back(monkeypatch)
    changed_on_disk(world)
    manifest = world.king()  # HEAD_FOLDED and ROUND_CLOSED are written; the second sha256 has not ended
    assert kinds(world)[-1] == ROUND_CLOSED and not is_head(world, manifest)
    world.folder.proven(False)  # a read that still runs passes
    gate.set()
    capsys.readouterr()
    new_run(world)
    assert f"the local files of head {manifest.head.version_id[:12]} are not there" in capsys.readouterr().err
    assert is_head(world, manifest) and world.head == manifest, "rebuilt from the chain, with the hashes of the journal"
    world.folder.run = inline
    assert world.king().head.parent_version_id != manifest.head.version_id and len(world.state.kings) == 3


# ---- the second sha256 of the new head ----


def test_a_round_opens_before_the_second_sha256_ended_and_its_result_waits_for_it(world, monkeypatch):
    gate = gate_read_back(monkeypatch)
    manifest = world.king()
    assert world.state.head == manifest.head and kinds(world)[-1] == ROUND_CLOSED, "the king is closed while the read still waits"
    run, seen = world.runner.run, []

    def box(ticket):
        seen.append((ticket.head.version_id, world.folder._read_back._work.done()))  # noqa: SLF001
        result = run(ticket)
        threading.Timer(0.2, gate.set).start()
        return result

    world.runner.run = box
    world.register(90, 0.0)
    started = time.monotonic()
    assert world.round().outcome == "null"
    assert seen == [(manifest.head.version_id, False)], "the round ran on the new head while its files were read"
    assert time.monotonic() - started >= 0.2 and world.folder._read_back._work.done()  # noqa: SLF001 - the result was stored after the read


def test_a_head_that_does_not_read_back_fails_the_lease_stops_run_and_the_next_start_rebuilds_it(world, monkeypatch, capsys):
    """Decision: no new way to end a round. The lease fails as a fault of the box does, ``run`` stops, and its start rebuilds the head."""
    world.king()
    gate = gate_read_back(monkeypatch)
    changed_on_disk(world)
    manifest = world.king()
    assert world.state.head == manifest.head and [king.uid for king in world.state.kings] == [2, 4]
    run = world.runner.run

    def box(ticket):
        result = run(ticket)
        gate.set()  # the read ends, with a fault, while the box works
        return result

    world.runner.run, world.folder.run = box, inline
    world.register(90, 0.3)
    world.register(91, 0.0)
    record = world.round()
    reason = f"HeadLost: the local files of {manifest.head.base_fp32_master_digest[:16]} do not read back as the fold wrote them"
    failed = [data for kind, _, data in records(world) if kind == LEASE_FAILED]
    assert record.outcome == "no_decision" and len(failed) == 1 and failed[0]["reason"].startswith(reason)
    assert not world.store.exists(layout.sealed_result(record.round_id, failed[0]["lease_id"])), "no result that was made on these files is stored"
    state = world.state
    assert [king.uid for king in state.kings] == [2, 4] and state.head == manifest.head and not state.served_hotkeys & {"hk90", "hk91"}
    assert [(key[0], seat.front) for key, seat in state.queue.items()] == [(90, True), (91, True)], "nobody is charged: both seats are back at the front"
    with pytest.raises(HeadLost, match="Start run again: the start rebuilds the head from the chain"):
        world.round()
    assert kinds(world)[-1] == ROUND_CLOSED and world.state.round is None, "the stop opened no round and wrote no record"
    # ``run`` (cli.cmd_run) stops with the exit code that the run loop takes as a crash, so the loop starts it again.
    # The start of this ``run`` is past, so the stack of the test has no proof of the start.
    stack = Stack(world.store, world.admission, world.sampler, runner=world.runner, head=dataclasses.replace(world.controller.head, prepare=None), overlap=True)
    monkeypatch.setattr(cli, "open_controller", lambda args, command: (world.controller, stack))
    monkeypatch.setattr(cli, "now", world.clock)
    capsys.readouterr()
    assert cli.cmd_run(argparse.Namespace(rounds=None, max_idle=1, idle_sleep=0)) == cli.HEAD_LOST_EXIT == 75
    assert "stopped: the local files of" in capsys.readouterr().err and kinds(world)[-1] == ROUND_CLOSED
    new_run(world)
    assert f"the local files of head {manifest.head.version_id[:12]} are not there" in capsys.readouterr().err
    assert is_head(world, manifest) and world.head == manifest, "the head of the journal, rebuilt from the chain: the king is kept"
    check_chain(world.chain_file())
    world.now += world.params.apply_retry_s  # the wait after a failed lease, as after each fault of the box
    assert world.round().outcome == "applied" and [king.uid for king in world.state.kings] == [2, 4, 90]
    assert is_head(world, world.head) and world.head.head.parent_version_id != manifest.head.version_id
    world.folder.proven(True)


def test_a_head_that_does_not_read_back_is_never_uploaded_as_a_checkpoint(tmp_path, monkeypatch, capsys):
    world = a_world(tmp_path / "world", every=1)
    world.king()
    world.wait()
    gate = gate_read_back(monkeypatch)
    good = world.king()
    time.sleep(0.2)
    assert not any(key.startswith(layout.public_master_prefix(good.head.base_fp32_master_digest)) for key in world.store.wrote), "the upload waits for the read"
    gate.set()
    world.wait()
    assert good.head.version_id in checkpoints(world.store)
    changed_on_disk(world)
    bad = world.king()
    world.wait()
    assert "stopped by HeadLost" in capsys.readouterr().err
    assert bad.head.version_id not in checkpoints(world.store) and not world.store.list(layout.public_master_prefix(bad.head.base_fp32_master_digest))


def test_the_next_fold_waits_for_the_second_sha256_of_the_head_that_it_reads(world, monkeypatch):
    world.king()
    gate = gate_read_back(monkeypatch)
    changed_on_disk(world)
    manifest = world.king()
    gate.set()
    before, stacked = cache(world), world.state.heads[-2]
    assert stacked.loras and not stacked.folded
    for _ in range(2):  # the fault stays: each later ask gets it
        with pytest.raises(HeadLost):
            world.folder.fold(stacked, world.params, "pending/never")
        with pytest.raises(HeadLost):
            world.folder.proven(True)
    assert not world.store.list("pending/never") and cache(world) == before and world.state.head == manifest.head, "the fold stopped before it changed the cache or the store"


# ---- the check of the waiting files stays at the open; the read of the chain is made beside the fold ----


class CountedCheck:
    def __init__(self, world: ChainWorld) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.real = world.admission.verdicts
        world.admission.verdicts = self

    def __call__(self, hotkeys):
        self.calls.append((threading.current_thread().name, tuple(hotkeys)))
        return self.real(hotkeys)


def test_the_open_after_a_king_checks_the_file_of_each_waiting_seat_itself(tmp_path):
    world = ChainWorld(tmp_path / "world", sim_params(batch_cap=2), order=OVERLAP)
    world.start(CHAIN)
    for uid, quality in ((2, 0.3), (3, 0.0), (4, 0.0), (5, 0.3), (6, None)):
        world.register(uid, quality)
    check = CountedCheck(world)
    assert world.round().outcome == "applied"
    world.register(7, 0.0)
    check.calls.clear()
    opened_before = world.now
    record = world.round()
    assert check.calls == [("MainThread", ("hk4", "hk5", "hk6", "hk7"))], "one check, in the thread of run, of each seat that waits at the open"
    assert [entry.reg.uid for lease in record.leases for entry in lease.entries] == [4, 5] and record.outcome == "applied"
    kept(world)
    assert world.publisher.snapshots[-1]["queue"]["files_checked_at"] > opened_before, "the public files give the time of the check at the open"


def during_the_fold(world: ChainWorld, act) -> None:
    """Call ``act`` when the fold of a king starts: ``fold_ahead`` with ``overlap``, ``fold`` with ``serial``."""
    for name in ("fold_ahead", "fold"):
        real = getattr(world.folder, name)
        setattr(world.folder, name, lambda *args, real=real, **more: (act(), real(*args, **more))[1])


def test_a_file_that_a_miner_stores_while_the_fold_runs_keeps_his_old_seat_as_with_serial(tmp_path):
    """The squat rule moves an old seat with no valid file to the back. It must see the files of the open, not those of the decision."""

    def story(order):
        world = ChainWorld(tmp_path / order, sim_params(batch_cap=2, squat_min_wait_s=1), order=order)
        world.start(CHAIN)
        for uid, quality in ((2, 0.3), (3, 0.0), (4, None), (5, 0.0), (6, 0.0)):
            world.register(uid, quality)
        during_the_fold(world, lambda: world.admission.upload("hk4", 0.0))
        assert world.round().outcome == "applied"
        return world, world.round()

    (serial, _), (overlap, record) = story(SERIAL), story(OVERLAP)
    assert "seats_demoted" not in kinds(overlap), "hk4 has a valid file at the open: his seat is old, and it stays"
    assert [entry.reg.uid for lease in record.leases for entry in lease.entries] == [4, 5]
    kept(overlap)
    assert records(overlap) == records(serial) and stored(overlap) == stored(serial)


def test_a_file_that_is_not_valid_at_the_open_is_not_picked_as_with_serial(tmp_path):
    def story(order):
        world = ChainWorld(tmp_path / order, sim_params(batch_cap=2), order=order)
        world.start(CHAIN)
        for uid, quality in ((2, 0.3), (3, 0.0), (4, 0.0), (5, 0.0), (6, 0.0)):
            world.register(uid, quality)
        during_the_fold(world, lambda: world.admission.upload("hk4", 0.0, valid=False))
        world.round()
        return world, world.round()

    (serial, _), (overlap, record) = story(SERIAL), story(OVERLAP)
    assert [entry.reg.uid for lease in record.leases for entry in lease.entries] == [5, 6]
    kept(overlap)
    assert records(overlap) == records(serial) and stored(overlap) == stored(serial)
    assert overlap.controller._verdicts == serial.controller._verdicts  # noqa: SLF001


def test_a_chain_read_ahead_is_used_one_time_and_an_old_or_a_failed_one_is_made_again():
    snapshots = iter(["chain-1", "chain-2"])
    overlap = Overlap(chain=lambda: next(snapshots))
    assert overlap.snapshot(100) is None, "nothing was started"
    overlap.look_ahead(100)
    overlap.look_ahead(101)
    assert overlap.snapshot(100 + AHEAD_MAX_AGE_S) == "chain-1", "a second try of the same king starts no second read"
    assert overlap.snapshot(102) is None, "one use"
    overlap.look_ahead(200)
    assert overlap.snapshot(201 + AHEAD_MAX_AGE_S) is None, "older than the limit: not used"

    def node_is_down():
        raise InfraError("the node did not answer")

    failing = Overlap(chain=node_is_down)
    failing.look_ahead(300)
    assert failing.snapshot(301) is None, "the caller makes the read itself, and gets the error itself if it fails again"


def test_run_takes_the_chain_read_of_the_look_ahead_and_reads_the_seed_at_the_open(world, monkeypatch, capsys):
    """``cli.cmd_run`` with ``overlap``: the registrations of the pass after a king come from the read beside the fold."""
    reads, seeds = [], []

    def seed():
        seeds.append(len(world.store.list("private/rounds/")))
        return "the-newest-block"

    for uid in (90, 91):
        world.admission.upload(f"hk{uid}", 0.3 if uid == 90 else 0.0)
    world.register(90, 0.3)
    world.register(91, 0.0)
    monkeypatch.setattr(world.controller, "sync_registrations", lambda snapshot, now, force=False: reads.append(("synced", snapshot)))
    stack = Stack(world.store, world.admission, world.sampler, runner=world.runner, head=world.controller.head, overlap=True, chain=lambda: reads.append(threading.current_thread().name) or "a-snapshot", entropy=lambda: pytest.fail("the block of the last snapshot is not the seed with overlap"), entropy_now=seed)
    monkeypatch.setattr(cli, "open_controller", lambda args, command: (world.controller, stack))
    monkeypatch.setattr(cli, "now", world.clock)
    monkeypatch.setattr(cli, "print_round", lambda stack, round_id: None)
    assert cli.cmd_run(argparse.Namespace(rounds=1, max_idle=1, idle_sleep=0)) == 0
    assert world.controller.overlap.snapshot(world.now) == "a-snapshot", "the next pass of this run has its registrations: no read in the line"
    assert reads == ["MainThread", ("synced", "a-snapshot"), "sn120-v2-chain-read-ahead"], "one read in the line before the round, and one beside the fold of its king"
    record = world.state.rounds[-1]
    assert seeds == [seeds[0]] and seeds[0] > 0, "one read of the seed, after the batches of the round were frozen"
    assert record.round_seed == make_round_seed(record.round_id, record.head_version_id, "the-newest-block")
    assert not world.store.list("pending/"), "the end of run did the steps that waited for a next open"


def test_the_seed_is_not_read_when_no_round_can_open(world):
    world.register(90, None)
    assert world.controller.run_round(world.runner, world.clock, lambda: pytest.fail("no round opens, so no block is read")) is None


def test_the_feed_gives_the_hash_of_the_newest_block_in_one_call_and_refuses_a_node_that_is_behind():
    class Line(FakeChain):
        def __init__(self) -> None:
            super().__init__({uid: f"hk{uid}" for uid in range(3)}, "hk0", now=int(time.time()))
            self.calls: list[str] = []
            self.in_head = False

        def head(self):
            self.calls.append("head")
            self.in_head = True
            try:
                return super().head()
            finally:
                self.in_head = False

        def block_hash(self, block):
            if not self.in_head:  # the fake head asks itself for the hash; a real node gives it in the answer
                self.calls.append("block_hash")
            return super().block_hash(block)

        def uid_count(self, block):
            self.calls.append("uid_count")
            return super().uid_count(block)

    line = Line()
    feed = Feed(120, lambda: line)
    newest = feed.entropy_now()
    assert line.calls == ["head"] and newest == line.head()[1], "no snapshot is needed before it, and it is one call"
    line.advance(2)
    assert feed.entropy_now() == line.head()[1] != newest, "each call reads the block of now"
    feed.snapshot()
    assert feed.entropy() == feed.entropy_now(), "at the time of a snapshot, both ways give the same block"
    with pytest.raises(InfraError, match="the node is behind"):
        Reader(line, 120, clock=lambda: time.time() + 10_000).head_hash()


# ---- the steps that wait for the next open ----


def test_the_kept_steps_run_in_one_thread_after_the_open_and_a_step_that_fails_stops_nothing():
    overlap, done = Overlap(), []
    overlap.keep("the delete", lambda now: done.append(("delete", now, threading.current_thread().name)))
    overlap.keep("the files of the close", lambda now: done.append(("close", now)), PUBLISH)
    overlap.keep("a step that fails", lambda now: (_ for _ in ()).throw(InfraError("the store did not answer")))
    overlap.keep("the files of the open", lambda now: done.append(("open", now, threading.current_thread().name)), PUBLISH)
    overlap.release(7)
    assert overlap.settle() == ["a step that fails: InfraError: the store did not answer"]
    assert done == [("delete", 7, "sn120-v2-after-the-open"), ("open", 7, "sn120-v2-after-the-open")], "in the order kept; the later snapshot took the place of the earlier one"
    assert overlap.settle() == [] and overlap.flush(8) == [] and len(done) == 2
    overlap.keep("the files of a close", lambda now: done.append(("close", now, threading.current_thread().name)), PUBLISH)
    assert overlap.flush(9) == [] and done[-1] == ("close", 9, "MainThread")


def test_the_public_files_of_an_open_round_are_written_while_the_box_works_and_before_its_result_is_stored(world):
    world.king()
    shown_before, run, seen = len(world.publisher.snapshots), world.runner.run, []

    def box(ticket):
        seen.append(len(world.publisher.snapshots) - shown_before)
        result = run(ticket)
        return result

    world.runner.run = box
    real = world.controller.submit_result

    def submit(result, now):
        seen.append((len(world.publisher.snapshots) - shown_before, bool(world.store.list("pending/rounds/r000001"))))
        return real(result, now)

    world.controller.submit_result = submit
    world.register(90, 0.0)
    record = world.round()
    assert seen[1] == (1, False), "at the result: the round is shown, and the staged files of the round before are deleted"
    assert seen[0] in (0, 1), "the box started with no wait for those two steps"
    shown = world.publisher.snapshots[-1]
    assert shown["round"]["round_id"] == record.round_id and shown["generated_at"] == world.state.rounds[-1].opened_at


def test_a_publish_that_fails_after_the_open_is_a_problem_in_the_log_and_the_round_goes_on(world, capsys):
    world.king()

    class Down:
        def publish(self, snapshot):
            raise RuntimeError("the site store is down")

    world.controller.publisher = Down()
    world.register(90, 0.3)
    world.register(91, 0.0)
    assert world.round().outcome == "applied"
    assert "after the open: open: the public files of round r000002: RuntimeError: the site store is down" in capsys.readouterr().err
    assert world.state.head.folded and len(world.state.kings) == 2


def a_store_that_fails(world: ChainWorld, times: int, slept: list[float], monkeypatch) -> list[bool]:
    """The publisher of ``world`` fails ``times`` times, then it works. Returns the list of the failures so far."""
    real, failed = world.publisher.publish, []

    def publish(snapshot):
        if len(failed) < times:
            failed.append(True)
            raise InfraError("the site store is down")
        real(snapshot)

    world.publisher.publish = publish
    monkeypatch.setattr(controller, "PUBLISH_PAUSE_S", 7.0)
    monkeypatch.setattr(controller.time, "sleep", slept.append)
    return failed


@pytest.mark.parametrize("order", [SERIAL, OVERLAP])
def test_a_publish_that_fails_two_times_is_written_by_the_third_try_with_no_word_of_a_problem(tmp_path, monkeypatch, capsys, order):
    world, slept = a_world(tmp_path / "world", order), []
    world.king()
    shown = len(world.publisher.snapshots)
    failed = a_store_that_fails(world, PUBLISH_TRIES - 1, slept, monkeypatch)
    world.register(90, 0.0)
    record = world.round()
    world.controller.close_in_public(world.clock())
    assert failed == [True] * (PUBLISH_TRIES - 1) and slept == [7.0] * (PUBLISH_TRIES - 1)
    assert world.publisher.snapshots[shown]["round"]["round_id"] == record.round_id, "the publish at the open of the round is there"
    assert world.controller.last_error == "" and "PUBLISH FAILED" not in capsys.readouterr().err


@pytest.mark.parametrize("order", [SERIAL, OVERLAP])
def test_a_publish_with_no_good_try_is_one_line_in_the_log_and_the_next_good_publish_ends_the_last_problem(tmp_path, monkeypatch, capsys, order):
    world, slept = a_world(tmp_path / "world", order), []
    world.king()
    world.controller.close_in_public(world.clock())
    shown = len(world.publisher.snapshots)
    failed = a_store_that_fails(world, PUBLISH_TRIES, slept, monkeypatch)
    capsys.readouterr()
    assert world.controller.publish(world.clock())
    line = [text for text in capsys.readouterr().err.splitlines() if text.startswith("PUBLISH FAILED: ")]
    assert line == ["PUBLISH FAILED: 3 tries, and the store took none: the site store is down. The public files (mining.json, llms.txt) still show the last good publish. run goes on, and the next publish writes them again."]
    assert failed == [True] * 3 and slept == [7.0] * 2, "3 tries, and no pause after the last try"
    assert world.controller.last_error == "publish failed: the site store is down" and len(world.publisher.snapshots) == shown
    assert world.controller.publish(world.clock(), only_changes=True), "the queue did not change, but the publish before failed: the next pass writes again"
    assert world.controller.last_error == "" and len(world.publisher.snapshots) == shown + 1
    assert not world.controller.publish(world.clock(), only_changes=True), "nothing changed and nothing failed: no write"
    assert "PUBLISH FAILED" not in capsys.readouterr().err


def test_a_good_publish_keeps_a_last_problem_that_is_not_of_a_publish(world):
    world.controller.last_error = "fold failed: the fold process died"
    world.register(90, 0.0)
    assert world.controller.publish(world.clock())
    assert world.controller.last_error == "fold failed: the fold process died"


def test_beside_gives_the_result_or_the_error_of_its_call_to_the_thread_that_asks():
    assert Beside(lambda: 5, "five").result() == 5
    failing = Beside(lambda: (_ for _ in ()).throw(VerifyError("no")), "fails")
    failing.wait()
    assert failing.done()
    with pytest.raises(VerifyError, match="no"):
        failing.result()
