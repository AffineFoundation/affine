"""The order ``overlap`` together with the queue rules of the record ``seats_restored`` and of the owner coldkey.

Words. The back: the last zone of the queue. The squat rule moves a seat there when it is old and has no valid
file. ``seats_restored``: the record that ``open_round`` writes when a seat at the back has a valid file at the
open. The seat then leaves the back. The record comes before ``round_opened``, or with no round after it when
too few seats are ready. Owner coldkey: the account that owns a hotkey on the chain. A round takes one seat for
each coldkey.

Each part was tested alone. Here they run together, on a real tiny head, in the mode ``chain``:

* ``overlap`` writes the records and the bytes of ``serial`` when a seat leaves the back.
* A ``run`` that dies at the record, and a request to stop after the round (``stopafter.py``) that comes
  before it, at it, or after it.
* The chain file and its checkpoints while a round with such a seat is open.
* One seat for each coldkey when the registrations of the open were read beside the fold, and the round opens
  while the second sha256 of the folded head still runs.
* Each query of a read of the chain names one block. With ``overlap`` the read of an open can be older than
  the head of the chain at that open, and llms.txt says no more than that.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

torch = pytest.importorskip("torch")

from sn120_v2.sim_head import ChainWorld  # noqa: E402 - after the skip guard

from sn120_v2 import cli, headfold, layout, publisher  # noqa: E402
from sn120_v2.afterking import OVERLAP, SERIAL  # noqa: E402
from sn120_v2.chain import FakeChain, Reader, SubtensorClient  # noqa: E402
from sn120_v2.controller import Stack  # noqa: E402
from sn120_v2.headchain import CHAIN, check_chain, checkpoints, heads_of, tail  # noqa: E402
from sn120_v2.headfold import READ_BACK  # noqa: E402
from sn120_v2.journal import JOURNAL_FILE, LEADER, read_journal  # noqa: E402
from sn120_v2.queue import waiting  # noqa: E402
from sn120_v2.sim import sim_params  # noqa: E402
from sn120_v2.state import HEAD_FOLDED, REGISTRATIONS_SYNCED, ROUND_CLOSED, ROUND_OPENED, SEATS_DEMOTED, SEATS_RESTORED  # noqa: E402
from sn120_v2.stopafter import LOOP_ENV, OBEYED, STOP_FILE, WORK_ENV, stamp  # noqa: E402

ORDERS = [SERIAL, OVERLAP]
WAIT_S = 30  # the longest a test waits for a thread
ASKED_AT = 1_700_000_000  # the time of the request file in these tests
READ_AHEAD = "sn120-v2-chain-read-ahead"
# (uid, quality of the file). UID 4 has no file at first. Rounds of two seats: (2, 3), then (5, 6), then (4, 7). UID 8 stays.
SEATS = ((2, 0.3), (3, 0.0), (4, None), (5, 0.3), (6, 0.0), (7, 0.3), (8, 0.0))
SEAT = [4, 1004]  # the seat that goes to the back and leaves it: (uid, registration block)


def a_world(path: Path, order: str, seats=SEATS, every: int = 0, **changes) -> ChainWorld:
    """The squat rule is on (``squat_min_wait_s=1``), and hk4 stores a valid file while the second king is folded."""
    world = ChainWorld(path, sim_params(batch_cap=2, squat_min_wait_s=1, **changes), order=order)
    world.start(CHAIN, every=every)
    for uid, quality in seats:
        world.register(uid, quality)

    def stores_his_file() -> None:
        if len(world.state.rounds) == 1:  # one round is closed: this fold is the fold of the second king
            world.admission.upload("hk4", 0.0)

    for name in ("fold_ahead", "fold"):
        real = getattr(world.folder, name)
        setattr(world.folder, name, lambda *args, real=real, **more: (stores_his_file(), real(*args, **more))[1])
    return world


def two_rounds(path: Path, order: str, **settings) -> ChainWorld:
    """Two kings are closed. The open of the second round sent the seat of UID 4 to the back, and hk4 has a valid file now."""
    world = a_world(path, order, **settings)
    assert [world.round().decision.winner.uid for _ in range(2)] == [2, 5]
    assert kinds(world).count(SEATS_DEMOTED) == 1 and SEATS_RESTORED not in kinds(world)
    assert [(seat.reg.uid, seat.demoted) for seat in waiting(world.state.queue)][-1] == (4, True)
    return world


def records(world: ChainWorld) -> list[tuple[str, int, dict]]:
    return [(record.kind, record.at, record.data) for record in read_journal(world.root / JOURNAL_FILE)]


def kinds(world: ChainWorld) -> list[str]:
    return [kind for kind, _, _ in records(world)]


def stored(world: ChainWorld) -> dict[str, bytes]:
    return {key: world.store.get(key) for key in world.store.list("")}


def shapes(world: ChainWorld) -> list[str]:
    """The keys of the store, with each sha256 in a name replaced."""
    return sorted(re.sub(r"[0-9a-f]{64}", "<sha256>", key) for key in world.store.list(""))


def seats_of(record) -> list[int]:
    return [entry.reg.uid for lease in record.leases for entry in lease.entries]


def line(world: ChainWorld) -> list[tuple]:
    """The seats that wait, in turn order, with what decides the turn."""
    return [(seat.reg.uid, seat.front, seat.demoted, seat.held, seat.reg.coldkey) for seat in waiting(world.state.queue)]


def shown(world: ChainWorld, uid: int) -> dict:
    """What the last public snapshot shows of the waiting seat of ``uid``."""
    return next(seat for seat in world.publisher.snapshots[-1]["queue"]["next"] if seat["uid"] == uid)


def kept(world: ChainWorld) -> None:
    """The steps that wait for a next open, done now: ``run`` does them in a pass with no round, and at its end."""
    if world.overlap is not None:
        assert world.overlap.flush(world.now) == []


def read_back_ended(world: ChainWorld) -> bool:
    """False while the second sha256 of the last folded head runs (``overlap`` only)."""
    read = getattr(world.folder, "_read_back", None)
    return read is None or read._work.done()  # noqa: SLF001


def new_run(world: ChainWorld) -> None:
    """The process of ``run`` died and the loop started a new one: nothing is verified in its memory, and its start proves the local head."""
    world.wait()
    if getattr(world.folder, "_read_back", None) is not None:
        world.folder._read_back.wait()  # noqa: SLF001 - a dead process leaves no thread; the test waits for it instead
    world.folder.resident._verified.clear()  # noqa: SLF001
    world.start()


def until(what: Callable[[], bool]) -> None:
    end = time.monotonic() + WAIT_S
    while not what():
        assert time.monotonic() < end, "waited too long"
        time.sleep(0.005)


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


def at_record(world: ChainWorld, kind: str, act: Callable[[], None]) -> None:
    """Do ``act`` one time, right after the next record of ``kind`` is in the journal."""
    journal, done = world.controller.journal, []
    real_append = journal.append

    def append(record_kind, data, at):
        record = real_append(record_kind, data, at)
        if record_kind == kind and not done:
            done.append(True)
            act()
        return record

    journal.append = append


def in_the_box(world: ChainWorld, act: Callable[[], None]) -> None:
    """Do ``act`` one time while the box works on the next round, after the thread of the open ended."""
    real, done = world.runner.run, []

    def box(ticket):
        if not done:
            done.append(True)
            if world.controller.overlap is not None:
                until(lambda: world.controller.overlap._worker is None or world.controller.overlap._worker.done())  # noqa: SLF001
            act()
        return real(ticket)

    world.runner.run = box


def gate_read_back(monkeypatch) -> threading.Event:
    """The second sha256 of a new head waits while the returned event is not set, as it does for minutes with 56 GB."""
    gate, real = threading.Event(), headfold.file_sha256

    def slow(path):
        if threading.current_thread().name.startswith(READ_BACK):
            assert gate.wait(WAIT_S)
        return real(path)

    monkeypatch.setattr(headfold, "file_sha256", slow)
    return gate


# ---- ``python -m sn120_v2 run`` and the request to stop after the round


def request_file(world: ChainWorld) -> Path:
    return world.box.work_dir / "logs" / STOP_FILE


def ask(world: ChainWorld) -> None:
    """The operator's one line: ``touch "$SN120_WORK_DIR/logs/run-loop.stop-after-round"``."""
    path = request_file(world)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    os.utime(path, (ASKED_AT, ASKED_AT))


def run(world: ChainWorld, monkeypatch, rounds: int | None = None, **parts) -> int:
    """``python -m sn120_v2 run`` on this world, started by the run loop, with SN120_WORK_DIR set. ``parts``: more of the stack."""
    more = {"entropy": lambda: "a-block"} | parts
    stack = Stack(world.store, world.admission, world.sampler, runner=world.runner, head=world.controller.head, overlap=world.order == OVERLAP, **more)
    monkeypatch.setattr(cli, "open_controller", lambda args, command: (world.controller, stack))
    monkeypatch.setattr(cli, "now", world.clock)
    monkeypatch.setattr(cli, "print_round", lambda stack, round_id: None)
    monkeypatch.setenv(WORK_ENV, str(world.box.work_dir))
    monkeypatch.setenv(LOOP_ENV, str(world.root))
    return cli.cmd_run(argparse.Namespace(rounds=rounds, max_idle=1, idle_sleep=0))


def obeyed(world: ChainWorld, capsys, last_round: str) -> None:
    """The last line of ``run`` says that it stopped as asked, after ``last_round``, and the store shows that close."""
    lines = [text for text in capsys.readouterr().err.splitlines() if not text.startswith("time: ")]
    assert lines[-1].startswith(f"{OBEYED} Asked at {stamp(ASKED_AT)}. Round {last_round} is closed (applied). No round is open. All that a close writes is in the store.")
    assert (" The second sha256 of the local head passed." in lines[-1]) == (world.order == OVERLAP)
    state = world.state
    assert state.round is None and state.rounds[-1].round_id == last_round and not any(seat.leased for seat in state.queue.values())
    assert world.store.get(layout.CURRENT) == state.manifest.to_json(), "the head pointer names the head of the journal"
    assert not world.store.list(layout.pending_round_prefix(last_round)), "the staged files of the last round are deleted"
    public = world.publisher.snapshots[-1]
    assert public["round"] is None and public["heads"][-1]["version_id"] == state.head.version_id
    assert [(seat["uid"], seat["demoted"]) for seat in public["queue"]["next"]] == [(uid, demoted) for uid, _, demoted, _, _ in line(world)], "the public files show the queue of the journal"
    whole(world)


def whole(world: ChainWorld) -> None:
    """The chain file is a chain of the folded heads of the journal, and mining.json shows its end."""
    doc = world.chain_file()
    check_chain(doc)
    assert [head["version_id"] for head in heads_of(doc)] == [head.version_id for head in world.state.heads if head.folded]
    assert world.publisher.snapshots[-1]["head_chain"] == tail(doc)


# ---- the same records and the same bytes as ``serial`` ----


def test_overlap_writes_the_records_and_the_bytes_of_serial_when_a_seat_leaves_the_back(tmp_path):
    def story(order):
        world = two_rounds(tmp_path / order, order)
        third = world.round()
        assert world.round().outcome == "null" and world.round() is None  # UID 8 alone, then a pass with no round
        world.wait()
        kept(world)
        return world, third

    (serial, _), (overlap, third) = story(SERIAL), story(OVERLAP)
    names = kinds(overlap)
    at = names.index(SEATS_RESTORED)
    assert names.count(SEATS_RESTORED) == 1 and names[at - 1 : at + 2] == [ROUND_CLOSED, SEATS_RESTORED, ROUND_OPENED], "one record, between the close of the second king and the open of the next round"
    assert records(overlap)[at][2] == {"keys": [SEAT]}
    assert seats_of(third) == [4, 7] and third.decision.winner.uid == 7, "the seat is in its old place, before UID 7"
    assert records(overlap) == records(serial), "the same records with the same times, in the same order"
    assert stored(overlap) == stored(serial), "each key of the store, public and private, with the same bytes"
    for world in (serial, overlap):
        assert line(world) == [] and not any(seat.demoted for seat in world.state.queue.values())
        whole(world)


def test_the_thread_of_run_writes_the_record_before_the_steps_that_wait_for_the_open_and_while_the_second_sha256_runs(tmp_path, monkeypatch):
    gate = gate_read_back(monkeypatch)
    gate.set()
    world = a_world(tmp_path / "world", OVERLAP)
    assert world.round().decision.winner.uid == 2
    in_the_box(world, gate.clear)  # the second sha256 of the head of the second king waits from here
    assert world.round().decision.winner.uid == 5 and not read_back_ended(world)
    journal, seen = world.controller.journal, []
    real_append, publishes = journal.append, len(world.publisher.snapshots)

    def append(kind, data, at):
        seen.append((kind, threading.current_thread().name, len(world.overlap._kept) > 0, world.overlap._worker is None, read_back_ended(world), len(world.publisher.snapshots) - publishes))  # noqa: SLF001
        return real_append(kind, data, at)

    journal.append = append
    in_the_box(world, lambda: threading.Timer(0.2, gate.set).start())
    third = world.round()
    assert seen[:2] == [
        (SEATS_RESTORED, "MainThread", True, True, False, 0),  # the steps of the last close wait, no thread of an open runs, the head is not read back, nothing was published
        (ROUND_OPENED, "MainThread", True, True, False, 0),
    ]
    assert seats_of(third) == [4, 7] and third.decision.winner.uid == 7 and read_back_ended(world)
    after_the_open = world.publisher.snapshots[publishes]
    assert [entry["uid"] for entry in after_the_open["round"]["entries"]] == [4, 7], "the first public files after the record show the seat in its round"
    assert [seat["uid"] for seat in after_the_open["queue"]["next"]] == [8]


@pytest.mark.parametrize("before", [False, True], ids=["after-write", "before-write"])
@pytest.mark.parametrize("order", ORDERS)
def test_a_run_that_dies_at_the_record_of_a_seat_that_leaves_the_back_is_replaced_by_one_that_opens_the_same_round(tmp_path, order, before):
    calm = two_rounds(tmp_path / "calm", order)
    calm.round()
    world = two_rounds(tmp_path / "world", order)
    crash_at(world, SEATS_RESTORED, before)
    with pytest.raises(Crash):
        world.round()
    assert (kinds(world)[-1] == SEATS_RESTORED) != before and world.state.round is None
    new_run(world)
    third = world.round()
    assert seats_of(third) == [4, 7] and third.decision.winner.uid == 7
    assert [kind for kind in kinds(world) if kind != LEADER] == [kind for kind in kinds(calm) if kind != LEADER], "one record of the seat, and each other record one time"
    assert kinds(world).count(LEADER) == kinds(calm).count(LEADER) + 1, "and the record of the start of the new run"
    assert line(world) == line(calm) == [(8, False, False, False, "")]
    for each in (world, calm):
        each.wait()
        kept(each)
        whole(each)
    assert shapes(world) == shapes(calm), "the store has the same files; the names of the last head differ, because its time differs"


# ---- the request to stop after the round ----


@pytest.mark.parametrize("order", ORDERS)
def test_a_request_before_the_open_writes_no_record_and_the_seat_stays_at_the_back_until_run_goes_on(tmp_path, monkeypatch, capsys, order):
    world = two_rounds(tmp_path / "world", order)
    before = records(world)
    ask(world)
    capsys.readouterr()
    assert run(world, monkeypatch) == 0
    obeyed(world, capsys, "r000002")
    assert records(world) == before, "no file was checked and no seat was moved: run opens no round, so it picks no seats"
    assert shown(world, 4)["demoted"] is True
    request_file(world).unlink()
    new_run(world)
    assert run(world, monkeypatch) == 0 and OBEYED not in capsys.readouterr().err
    names = kinds(world)
    assert names.count(SEATS_RESTORED) == 1 and names[len(before) :][:3] == [LEADER, SEATS_RESTORED, ROUND_OPENED], "the new run starts, and its first open writes the record"
    assert [seats_of(record) for record in world.state.rounds[2:]] == [[4, 7], [8]]


@pytest.mark.parametrize("order", ORDERS)
def test_a_request_between_the_record_and_the_open_stops_run_after_that_round(tmp_path, monkeypatch, capsys, order):
    world = two_rounds(tmp_path / "world", order)
    seen = []
    at_record(world, SEATS_RESTORED, lambda: (ask(world), seen.append(kinds(world)[-1])))
    capsys.readouterr()
    assert run(world, monkeypatch) == 0
    assert seen == [SEATS_RESTORED]
    obeyed(world, capsys, "r000003")
    names = kinds(world)
    at = names.index(SEATS_RESTORED)
    assert names[at + 1] == ROUND_OPENED and names[-1] == ROUND_CLOSED and names.count(ROUND_OPENED) == 3, "the round of the record was opened and was not cut, and no round follows it"
    assert seats_of(world.state.rounds[-1]) == [4, 7] and line(world) == [(8, False, False, False, "")]
    assert request_file(world).exists(), "the loop removes it, not run"


@pytest.mark.parametrize("order", ORDERS)
def test_a_run_that_dies_after_the_record_and_then_finds_a_request_stops_with_no_round_and_shows_the_seat_out_of_the_back(tmp_path, monkeypatch, capsys, order):
    world = two_rounds(tmp_path / "world", order)
    crash_at(world, SEATS_RESTORED, before=False)
    with pytest.raises(Crash):
        run(world, monkeypatch)
    ask(world)
    new_run(world)
    capsys.readouterr()
    assert run(world, monkeypatch) == 0
    obeyed(world, capsys, "r000002")
    assert kinds(world)[-2:] == [SEATS_RESTORED, LEADER], "the record has no round after it: the run that wrote it died, and the new run (its start is the last record) opens none"
    assert shown(world, 4)["demoted"] is False and [seat[0] for seat in line(world)] == [4, 7, 8]
    request_file(world).unlink()
    new_run(world)
    assert run(world, monkeypatch) == 0 and OBEYED not in capsys.readouterr().err
    assert kinds(world).count(SEATS_RESTORED) == 1 and [seats_of(record) for record in world.state.rounds[2:]] == [[4, 7], [8]]


@pytest.mark.parametrize("order", ORDERS)
def test_a_record_with_no_round_after_it_is_in_the_public_files_before_run_says_that_it_stopped(tmp_path, monkeypatch, capsys, order):
    """Too few seats are ready (``n_min_start=2``): the open writes the record and opens no round. A request comes at that record."""
    world = two_rounds(tmp_path / "world", order, seats=SEATS[:5], n_min_start=2)
    at_record(world, SEATS_RESTORED, lambda: ask(world))
    capsys.readouterr()
    assert run(world, monkeypatch) == 0
    obeyed(world, capsys, "r000002")
    assert kinds(world)[-1] == SEATS_RESTORED and kinds(world).count(ROUND_OPENED) == 2
    assert line(world) == [(4, False, False, False, "")] and shown(world, 4)["demoted"] is False
    assert world.controller._public_is_now()  # noqa: SLF001 - the last good publish has each record of the journal


# ---- the chain file ----


@pytest.mark.parametrize("order", ORDERS)
def test_the_chain_file_takes_the_checkpoints_of_two_heads_while_a_round_with_a_seat_from_the_back_is_open(tmp_path, order):
    world = a_world(tmp_path / "world", order, every=1)
    world.store.gate = threading.Event()  # the upload of each checkpoint waits
    heads = []
    for winner in (2, 5):
        assert world.round().decision.winner.uid == winner
        heads.append(world.head.head.version_id)
    assert checkpoints(world.store) == set()
    writes, put = [], world.store.put

    def watched(key: str, data: bytes) -> None:
        if key == layout.HEAD_CHAIN:
            doc = json.loads(data)
            writes.append((threading.current_thread().name, doc["head_version_id"], {step["version_id"] for step in doc["steps"] if step["checkpoint"]}, kinds(world)[-1]))
        put(key, data)

    world.store.put = watched

    def the_uploads_run() -> None:
        world.store.gate.set()
        until(lambda: checkpoints(world.store) == set(heads))
        world.wait()

    in_the_box(world, the_uploads_run)
    third = world.round()
    assert seats_of(third) == [4, 7] and SEATS_RESTORED in kinds(world)
    heads.append(world.head.head.version_id)
    world.wait()
    kept(world)
    world.controller.publish(world.clock())
    in_the_round = [write for write in writes if write[3] == ROUND_OPENED]
    assert in_the_round and in_the_round[-1][:3] == ("head-checkpoint", heads[1], set(heads[:2])), "the checkpoint thread wrote the two flags while the round was open"
    marked = set()
    for _, _, flags, _ in writes:
        assert marked <= flags, "a write of the chain file took a checkpoint back"
        marked = flags
    assert [step["checkpoint"] for step in world.chain_file()["steps"]] == [True, True, True] and world.chain_file()["start"]["version_id"] == heads[2]
    whole(world)


@pytest.mark.parametrize("order", ORDERS)
def test_the_record_of_a_seat_that_leaves_the_back_changes_no_byte_of_the_chain_file(tmp_path, monkeypatch, capsys, order):
    world = two_rounds(tmp_path / "world", order, seats=SEATS[:5], n_min_start=2)
    kept(world)
    assert world.controller.publish(world.clock(), check_files=True) is True
    assert (shown(world, 4)["demoted"], shown(world, 4)["file_valid"]) == (True, True), "the public files show the valid file while the seat is at the back"
    before, heads = world.store.get(layout.HEAD_CHAIN), len(world.state.heads)
    world.store.wrote.clear()
    assert world.round() is None and kinds(world)[-1] == SEATS_RESTORED and len(world.state.heads) == heads
    kept(world)
    assert world.controller.publish(world.clock(), only_changes=True) is True, "the seat left the back and nothing else changed: that is a change of the queue, so the pass with no round publishes"
    assert shown(world, 4)["demoted"] is False
    assert layout.HEAD_CHAIN not in world.store.wrote and world.store.get(layout.HEAD_CHAIN) == before, "the record makes no head, and the writer puts no file that the store has"
    whole(world)


# ---- one seat for each coldkey ----


@pytest.mark.parametrize("order", ORDERS)
def test_a_round_that_opens_while_the_second_sha256_runs_takes_one_seat_for_each_coldkey_of_the_chain_read_beside_the_fold(tmp_path, monkeypatch, order):
    """The journal has seats with no coldkey, as one that older code wrote. While the first round runs, the owner of hk3 gets hk4 on the chain."""
    gate = gate_read_back(monkeypatch)
    gate.set()
    world = ChainWorld(tmp_path / "world", sim_params(batch_cap=2), order=order)
    world.start(CHAIN)
    for uid, quality in ((1, 0.3), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0)):
        world.register(uid, quality)
    assert {seat.reg.coldkey for seat in world.state.queue.values()} == {""}
    fake = FakeChain({0: "owner"} | {uid: f"hk{uid}" for uid in range(1, 6)}, "owner", block=2000)
    reader, reads = Reader(fake, world.params.netuid, clock=lambda: fake.now), []

    def chain():
        reads.append(threading.current_thread().name)
        return reader.snapshot()

    real, boxes, opens = world.runner.run, [], []

    def box(ticket):
        boxes.append(ticket.round_id)
        if len(boxes) == 1:
            fake.owner[4] = fake.owner[3]  # after the read of the first pass, before the read beside the fold
            gate.clear()  # the second sha256 of the head that this round makes waits
        if len(boxes) == 2:
            opens.append(read_back_ended(world))
            threading.Timer(0.2, gate.set).start()
        return real(ticket)

    world.runner.run = box
    assert run(world, monkeypatch, rounds=3, chain=chain) == 0
    first, second, third = world.state.rounds
    assert first.decision.winner.uid == 1 and seats_of(first) == [1, 2]
    assert seats_of(second) == [3, 5], "hk3 and hk4 have one owner now: the round takes the seat of hk3, and the seat of hk4 waits"
    assert seats_of(third) == [4], "the seat of hk4 has its turn in the next round, when the seat of hk3 has had its turn"
    synced = [data for kind, _, data in records(world) if kind == REGISTRATIONS_SYNCED and data.get("refreshed")]
    assert [[(reg["uid"], reg["coldkey"]) for reg in data["refreshed"]] for data in synced] == [[(uid, f"ck{uid}") for uid in range(1, 6)], [(4, "ck3")]]
    names = kinds(world)
    assert names.index(HEAD_FOLDED, names.index(ROUND_OPENED)) < [at for at, name in enumerate(names) if name == REGISTRATIONS_SYNCED][-1] < [at for at, name in enumerate(names) if name == ROUND_OPENED][1]
    if order == OVERLAP:
        assert reads == ["MainThread", READ_AHEAD, "MainThread"], "the registrations of the second open were read beside the fold; the round after a round with no king reads them in the line"
        assert opens == [False], "the second round ran while the second sha256 of the new head ran"
    else:
        assert reads == ["MainThread"] * 3 and opens == [True]
    world.wait()
    whole(world)


# ---- the block of each chain query, and the read that an open uses ----

READ = ("SubnetworkN", "Keys", "BlockAtRegistration", "Owner")  # the storage items of one read of the registrations
NAMES = SimpleNamespace(storage=SimpleNamespace(SubtensorModule=SimpleNamespace(**{name: name for name in READ})))
READ_TIME = 1_800_000_000


class MovingChain:
    """The SDK calls of ``SubtensorClient`` on a chain that moves. It keeps the thread and the block of each query.

    The head moves one block while a read of the registrations runs, and the test moves it more. ``owner`` has the owner
    coldkey of a hotkey from a block on. So the answer of an Owner query shows the block that the query named.
    """

    def __init__(self, uids: range, block: int) -> None:
        self.block, self.uids, self.asked = block, uids, []
        self.owner: dict[str, tuple[int, str]] = {}

    def block_info(self, block):
        return SimpleNamespace(number=block, hash=f"0xhash{block}", timestamp=datetime.fromtimestamp(READ_TIME, timezone.utc))

    def query(self, item, params, *, block=None):
        self.asked.append((threading.current_thread().name, item, block))
        if item == "SubnetworkN":
            return len(self.uids)
        since, coldkey = self.owner.get(params[0], (0, ""))
        return coldkey if coldkey and block >= since else params[0].replace("hk", "ck")

    def query_map(self, item, params, *, block=None):
        self.asked.append((threading.current_thread().name, item, block))
        self.block += item == "BlockAtRegistration"  # the chain makes a block before the Owner queries of this read
        return [(uid, {"Keys": f"hk{uid}", "BlockAtRegistration": 1000 + uid}[item]) for uid in self.uids]

    def reads(self) -> list[tuple[str, int]]:
        """(thread, block) of each read of the registrations. A read is one UID count, the two maps, and one Owner query for each hotkey, at one block."""
        size = 3 + len(self.uids)
        parts = [self.asked[at : at + size] for at in range(0, len(self.asked), size)]
        for part in parts:
            assert [item for _, item, _ in part] == [*READ[:3], *["Owner"] * len(self.uids)]
            assert len({(thread, block) for thread, _, block in part}) == 1, "one thread and one block for all the queries of a read"
        return [(part[0][0], part[0][2]) for part in parts]


@pytest.mark.parametrize("order", ORDERS)
def test_each_owner_query_names_the_block_of_the_registration_read_and_an_open_uses_one_such_read(tmp_path, monkeypatch, order):
    """After the first king is decided the chain moves 40 blocks, and in the new block hk3 owns hk4 and hk5 owns hk6."""
    world = ChainWorld(tmp_path / "world", sim_params(batch_cap=2), order=order)
    world.start(CHAIN)
    for uid, quality in ((1, 0.3), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0), (6, 0.0)):
        world.register(uid, quality)
    chain = MovingChain(range(7), block=2000)
    reader = Reader(SubtensorClient(world.params.netuid, chain, sdk=NAMES), world.params.netuid, clock=lambda: READ_TIME)

    def move():
        for thread in threading.enumerate():
            if thread.name == READ_AHEAD:
                thread.join(WAIT_S)
        chain.block = 2040
        chain.owner = {"hk4": (2040, "ck3"), "hk6": (2040, "ck5")}

    at_record(world, HEAD_FOLDED, move)
    heads = []
    real = world.runner.run
    world.runner.run = lambda ticket: (heads.append(chain.block), real(ticket))[1]
    assert run(world, monkeypatch, rounds=3, chain=reader.snapshot) == 0
    world.wait()
    reads, rounds = chain.reads(), [seats_of(record) for record in world.state.rounds]
    synced = [(data["block"], [(reg["uid"], reg["coldkey"]) for reg in data["refreshed"]]) for kind, _, data in records(world) if kind == REGISTRATIONS_SYNCED and data.get("refreshed")]
    every = [(uid, f"ck{uid}") for uid in range(1, 7)]
    if order == OVERLAP:
        assert reads == [("MainThread", 2000), (READ_AHEAD, 2001), ("MainThread", 2040)]
        assert heads == [2001, 2040, 2041], "the head of the chain when the box got each round: the second open used a read that was 39 blocks old"
        assert rounds == [[1, 2], [3, 4], [5]], "at block 2001 hk3 and hk4 had two owners, so the second round took the two"
        assert synced == [(2000, every), (2040, [(6, "ck5")])], "the read beside the fold changed nothing, so it has no record"
    else:
        assert reads == [("MainThread", 2000), ("MainThread", 2040), ("MainThread", 2041)]
        assert heads == [2001, 2041, 2042], "each open used the read of its pass"
        assert rounds == [[1, 2], [3, 5], [4, 6]], "at block 2040 hk3 owns hk4 and hk5 owns hk6, so the second round took one seat of each owner"
        assert synced == [(2000, every), (2040, [(4, "ck3"), (6, "ck5")])]
    flat = " ".join(publisher.render(json.loads(json.dumps(world.publisher.snapshots[-1])), "https://site.example")[1].decode().split())
    assert "The validator reads the chain before each round." in flat
    assert "The owner coldkey is the chain's `SubtensorModule.Owner` of the seat's hotkey, read at the same block as the registrations, in the read of the chain before the round." in flat
    assert "round-open" not in flat, "with ``overlap`` the read of an open is made beside the fold of the last king, not at the open"
    whole(world)
