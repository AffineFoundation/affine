"""The order ``overlap`` together with the queue rules: the record ``seats_restored``, and one seat for a hotkey in a round.

Words. The back: the last zone of the queue. The squat rule moves a seat there when it is old and has no valid
file. ``seats_restored``: the record that ``open_round`` writes when a seat at the back has a valid file at the
open. The seat then leaves the back. The record comes before ``round_opened``, or with no round after it when
too few seats are ready. Coldkey: the account that owns a hotkey on the chain. It is not used: the validator
does not read it from the chain, and no seat waits for it. A walk: one call of ``select_round``, which goes
through the queue in turn order one time and gives the seats of a round.

Each part was tested alone. Here they run together, on a real tiny head, in the mode ``chain``:

* ``overlap`` writes the records and the bytes of ``serial`` when a seat leaves the back.
* A ``run`` that dies at the record, and a request to stop after the round (``stopafter.py``) that comes
  before it, at it, or after it.
* The chain file and its checkpoints while a round with such a seat is open.
* A round limits the seats by hotkey only. A read of the chain makes no query for an owner, and a chain that
  answers no such query stops no sync and no open. The seats of a round come from one walk.
* A hotkey that has a new registration on the chain while its seat is in the round that folds.
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

from sn120_v2 import cli, controller, headfold, layout, publisher  # noqa: E402
from sn120_v2.afterking import OVERLAP, SERIAL  # noqa: E402
from sn120_v2.chain import Reader, SubtensorClient  # noqa: E402
from sn120_v2.controller import Stack  # noqa: E402
from sn120_v2.headchain import CHAIN, check_chain, checkpoints, heads_of, tail  # noqa: E402
from sn120_v2.headfold import READ_BACK  # noqa: E402
from sn120_v2.journal import JOURNAL_FILE, LEADER, read_journal  # noqa: E402
from sn120_v2.queue import waiting  # noqa: E402
from sn120_v2.sim import sim_params  # noqa: E402
from sn120_v2.state import HEAD_FOLDED, REGISTRATIONS_SYNCED, ROUND_CLOSED, ROUND_OPENED, SEATS_DEMOTED, SEATS_RESTORED  # noqa: E402
from sn120_v2.stopafter import LOOP_ENV, OBEYED, STOP_FILE, WORK_ENV, stamp  # noqa: E402
from sn120_v2.types import ChainSnapshot, Registration  # noqa: E402

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


# ---- one seat for a hotkey in a round; the coldkey is not used ----

READ = ("SubnetworkN", "Keys", "BlockAtRegistration")  # the storage items of one read of the registrations
NAMES = SimpleNamespace(storage=SimpleNamespace(SubtensorModule=SimpleNamespace(**{name: name for name in READ})))  # the SDK of the test has no item ``Owner``
READ_TIME = 1_800_000_000
ONE_OWNER = "ck-of-hk3-and-hk4"


class MovingChain:
    """The SDK calls of ``SubtensorClient`` on a chain that moves. It keeps the thread, the item and the block of each query.

    It answers the three items of a read of the registrations. A query for any other item, as one for the owner of a hotkey,
    fails. The head moves one block while a read runs, and a test moves it more. ``hotkey`` and ``born`` change what the chain
    says of a UID: its hotkey, and the block of its registration.
    """

    def __init__(self, uids: range, block: int) -> None:
        self.block, self.uids, self.asked = block, uids, []
        self.hotkey: dict[int, str] = {}
        self.born: dict[int, int] = {}

    def block_info(self, block):
        return SimpleNamespace(number=block, hash=f"0xhash{block}", timestamp=datetime.fromtimestamp(READ_TIME, timezone.utc))

    def query(self, item, params, *, block=None):
        self.asked.append((threading.current_thread().name, item, block))
        if item != "SubnetworkN":
            raise TimeoutError(f"the chain of the test answers no query for {item}")
        return len(self.uids)

    def query_map(self, item, params, *, block=None):
        self.asked.append((threading.current_thread().name, item, block))
        self.block += item == "Keys"  # the chain makes a block in the middle of this read
        if item == "Keys":
            return [(uid, self.hotkey.get(uid, f"hk{uid}")) for uid in self.uids]
        return [(uid, self.born.get(uid, 1000 + uid)) for uid in self.uids]

    def reads(self) -> list[tuple[str, int]]:
        """(thread, block) of each read of the registrations. A read is the UID count and the two maps at one block, and no other query."""
        parts = [self.asked[at : at + 3] for at in range(0, len(self.asked), 3)]
        for part in parts:
            assert [item for _, item, _ in part] == list(READ)
            assert len({(thread, block) for thread, _, block in part}) == 1, "one thread and one block for all the queries of a read"
        return [(part[0][0], part[0][2]) for part in parts]


def seated(world: ChainWorld, seats: dict[int, tuple[str, str]], block: int = 1999) -> None:
    """Seats of the journal: UID -> (hotkey, coldkey). The king of the first round has UID 1. A coldkey here is one that other code wrote to the journal."""
    for hotkey in {hotkey for hotkey, _ in seats.values()}:
        world.admission.upload(hotkey, 0.3 if hotkey == "hk1" else 0.0)
    regs = [Registration(0, "hk0", 1, 1)] + [Registration(uid, hotkey, 1000 + uid, 1, coldkey) for uid, (hotkey, coldkey) in seats.items()]
    world.controller.sync_registrations(ChainSnapshot(world.params.netuid, block, world.clock(), tuple(regs)), world.clock(), force=True)


def walks_of(world: ChainWorld, monkeypatch) -> list:
    """Each walk of the queue (its thread, its seats) and each ``round_opened`` record (its seats), in the order of the time."""
    seen, real, journal = [], controller.select_round, world.controller.journal
    real_append = journal.append

    def walk(queue, verdicts, freeze, cap):
        picked = real(queue, verdicts, freeze, cap)
        seen.append(("walk", threading.current_thread().name, [entry.reg.uid for entry in picked.entries]))
        return picked

    def append(kind, data, at):
        record = real_append(kind, data, at)
        if kind == ROUND_OPENED:
            seen.append((ROUND_OPENED, [entry["reg"]["uid"] for lease in data["round"]["leases"] for entry in lease["entries"]]))
        return record

    monkeypatch.setattr(controller, "select_round", walk)
    journal.append = append
    return seen


def on_a_chain(world: ChainWorld, uids: int) -> tuple[MovingChain, Callable[[], ChainSnapshot]]:
    chain = MovingChain(range(uids), block=2000)
    return chain, Reader(SubtensorClient(world.params.netuid, chain, sdk=NAMES), world.params.netuid, clock=lambda: READ_TIME).snapshot


@pytest.mark.parametrize("order", ORDERS)
def test_one_coldkey_with_two_hotkeys_has_two_seats_in_one_round_and_one_hotkey_with_two_seats_has_one(tmp_path, monkeypatch, order):
    """The journal says that hk3 and hk4 have one coldkey. On the chain, UID 5 and UID 6 have the hotkey hk5."""
    world = ChainWorld(tmp_path / "world", sim_params(batch_cap=2), order=order)
    world.start(CHAIN)
    seated(world, {1: ("hk1", ""), 2: ("hk2", ""), 3: ("hk3", ONE_OWNER), 4: ("hk4", ONE_OWNER), 5: ("hk5", ""), 6: ("hk5", "")})
    chain, read = on_a_chain(world, 7)
    chain.hotkey = {6: "hk5"}
    seen = walks_of(world, monkeypatch)
    assert run(world, monkeypatch, rounds=3, chain=read) == 0
    world.wait()
    assert [seats_of(record) for record in world.state.rounds] == [[1, 2], [3, 4], [5]]
    assert [(seat.reg.uid, seat.reg.coldkey) for record in world.state.rounds[1:2] for lease in record.leases for seat in lease.entries] == [(3, ONE_OWNER), (4, ONE_OWNER)], "one round, two seats of one coldkey"
    assert [(seat.reg.uid, seat.reg.hotkey) for seat in waiting(world.state.queue)] == [(6, "hk5")], "the second seat of hk5 had no place in the round of the first one, with room for it"
    walks = [("walk", "MainThread", seats) for seats in ([1, 2], [3, 4], [5])]
    assert seen == [step for walk in walks for step in (walk, (ROUND_OPENED, walk[2]))], "one walk in the thread of run for each round, and the round has the seats of that walk"
    assert [data for kind, _, data in records(world) if kind == REGISTRATIONS_SYNCED][1:] == [], "the reads of run changed nothing: the coldkeys of the journal stay as they are"
    whole(world)


@pytest.mark.parametrize("order", ORDERS)
def test_a_chain_that_answers_no_owner_query_stops_no_sync_and_no_open(tmp_path, monkeypatch, order):
    """A read of the registrations asks for the UID count, the hotkeys and the registration blocks at one block, and for nothing more."""
    world = ChainWorld(tmp_path / "world", sim_params(batch_cap=2), order=order)
    world.start(CHAIN)
    seated(world, {1: ("hk1", ""), 2: ("hk2", "")})
    for uid in range(3, 7):
        world.admission.upload(f"hk{uid}")
    chain, read = on_a_chain(world, 7)
    with pytest.raises(TimeoutError):
        chain.query("Owner", ["hk3"], block=2000)
    chain.asked.clear()
    at_record(world, HEAD_FOLDED, lambda: setattr(chain, "block", chain.block + 40))
    assert run(world, monkeypatch, rounds=3, chain=read) == 0
    world.wait()
    assert [seats_of(record) for record in world.state.rounds] == [[1, 2], [3, 4], [5, 6]], "each sync added the seats of the chain and each round opened"
    assert {item for _, item, _ in chain.asked} == set(READ)
    if order == OVERLAP:
        assert chain.reads() == [("MainThread", 2000), (READ_AHEAD, 2001), ("MainThread", 2042)], "the read of the second open was made beside the fold"
    else:
        assert chain.reads() == [("MainThread", 2000), ("MainThread", 2041), ("MainThread", 2042)]
    synced = [(data["block"], sorted(data), [reg["uid"] for reg in data["added"]], {reg["coldkey"] for reg in data["added"]}) for kind, _, data in records(world) if kind == REGISTRATIONS_SYNCED][1:]
    assert synced == [(2000, ["added", "block", "rehotkeyed", "removed"], [3, 4, 5, 6], {""})], "the record has the keys that it had before, and no seat gets a coldkey"
    assert {seat.reg.coldkey for seat in world.state.queue.values()} <= {""}
    flat = " ".join(publisher.render(json.loads(json.dumps(world.publisher.snapshots[-1])), "https://site.example")[1].decode().split())
    assert "- One seat for a hotkey in a round. A second seat of the same hotkey waits for a later round. The coldkey is not used: one coldkey can have several seats, each with its own hotkey, in the same round." in flat
    assert flat.lower().count("coldkey") == 2 and "SubtensorModule.Owner" not in flat, "no other sentence of llms.txt names a coldkey"
    whole(world)


@pytest.mark.parametrize("order", ORDERS)
def test_a_hotkey_with_a_new_registration_on_the_chain_while_its_round_folds_gets_no_seat_in_the_next_round(tmp_path, monkeypatch, order):
    """Round 1 has hk1 (the king) and hk2. While the box works on it, the chain gets a new registration of hk1 at UID 1 and one of hk2 at UID 2.

    With ``overlap`` the read beside the fold sees them while the two old seats are in the round that is not closed yet.
    """
    gate = gate_read_back(monkeypatch)
    gate.set()
    world = ChainWorld(tmp_path / "world", sim_params(batch_cap=2), order=order)
    world.start(CHAIN)
    seated(world, {uid: (f"hk{uid}", "") for uid in range(1, 7)})
    chain, read = on_a_chain(world, 7)
    real, boxes, opens, journal = world.runner.run, [], [], world.controller.journal
    real_append = journal.append

    def box(ticket):
        boxes.append(ticket.round_id)
        if len(boxes) == 1:
            chain.born = {1: 1998, 2: 1999}  # after the read of the first pass, before the read beside the fold
            gate.clear()  # the second sha256 of the head that this round makes waits
        if len(boxes) == 2:
            threading.Timer(0.2, gate.set).start()
        return real(ticket)

    def append(kind, data, at):
        if kind == ROUND_OPENED:
            opens.append((world.state.round is None, read_back_ended(world), sorted(world.state.served_hotkeys), [seat.reg.key for seat in world.state.queue.values() if seat.reg.hotkey in ("hk1", "hk2")]))
        return real_append(kind, data, at)

    world.runner.run, journal.append = box, append
    assert run(world, monkeypatch, rounds=3, chain=read) == 0
    world.wait()
    assert [seats_of(record) for record in world.state.rounds] == [[1, 2], [3, 4], [5, 6]]
    names = kinds(world)
    second = [at for at, name in enumerate(names) if name == ROUND_OPENED][1]
    assert names.index(HEAD_FOLDED) < names.index(ROUND_CLOSED) < second, "the round that folded is closed in the journal before the next round opens"
    assert opens[0] == (True, True, [], [(1, 1001), (2, 1002)])
    assert opens[1] == (True, order == SERIAL, ["hk1", "hk2"], []), "at the second open the two hotkeys had their turn: no seat of them is in the queue, the old one or the new one"
    assert [(data["added"], data["removed"]) for kind, _, data in records(world) if kind == REGISTRATIONS_SYNCED][1:] == [], "no sync added a seat for the new registrations"
    if order == OVERLAP:
        assert chain.reads() == [("MainThread", 2000), (READ_AHEAD, 2001), ("MainThread", 2002)], "the read beside the fold started while the seats of hk1 and hk2 were in the open round"
    assert not any(seat.reg.hotkey in ("hk1", "hk2") for seat in world.state.queue.values())
    whole(world)
