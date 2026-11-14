"""A stop of ``run`` after the round, asked for with a file (``stopafter.py``), on a real tiny head.

What is proven here, with ``cli.cmd_run`` itself, in both orders of the steps after a king and in both head modes.

* With no request ``run`` writes the journal and the store that it writes when no request can be made.
* A request that is made before the round, in the round, at the decision, in the fold or in the close stops
  ``run`` before the next open, and only there. The close of the last round is complete in the store first.
* A request that is taken back stops nothing.
* A ``run`` that dies at any record after a request, or in the stop itself, is replaced by one that finishes
  the round and then stops: one king, a whole head, a whole chain, no round more.

The run loop's part (who removes the file, the last line of ``run.log``) is in ``test_run_loop.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sn120_v2.sim_head import ChainWorld  # noqa: E402 - after the skip guard

from sn120_v2 import cli, layout  # noqa: E402
from sn120_v2.afterking import OVERLAP, SERIAL  # noqa: E402
from sn120_v2.cli import HEAD_LOST_EXIT  # noqa: E402
from sn120_v2.controller import Stack  # noqa: E402
from sn120_v2.errors import InfraError  # noqa: E402
from sn120_v2.gpu.resident import hash_tree  # noqa: E402
from sn120_v2.headchain import CHAIN, FULL, check_chain, heads_of  # noqa: E402
from sn120_v2.headfold import inline  # noqa: E402
from sn120_v2.journal import JOURNAL_FILE, read_journal  # noqa: E402
from sn120_v2.queue import waiting  # noqa: E402
from sn120_v2.sim import sim_params  # noqa: E402
from sn120_v2.state import HEAD_APPLIED, HEAD_FOLDED, ROUND_CLOSED, ROUND_OPENED, ROUND_SETTLED  # noqa: E402
from sn120_v2.stopafter import LOOP_ENV, OBEYED, SAID, STOP_FILE, WORK_ENV, StopRequest, stamp  # noqa: E402

WORLDS = [(CHAIN, SERIAL), (CHAIN, OVERLAP), (FULL, SERIAL)]  # (SN120_HEAD_PUBLISH, SN120_AFTER_KING); ``overlap`` needs ``chain``
ORDERS = [SERIAL, OVERLAP]
ROUNDS = 3  # the seats that wait fill this many rounds, and each has one winner
ASKED_AT = 1_700_000_000  # the time of the request file in these tests: 2023-11-14T22:13:20Z


def a_world(path: Path, order: str = SERIAL, mode: str = CHAIN) -> ChainWorld:
    world = ChainWorld(path, sim_params(), order=order)
    world.start(mode)
    world.king()  # a head with one fold: the chain file has a link
    for number in range(ROUNDS * world.params.batch_cap):
        world.register(100 + number, 0.3 if number % world.params.batch_cap == 0 else 0.0)
    return world


def winners(count: int) -> list[int]:
    return [100 + round_number * 4 for round_number in range(count)]


def request_file(world: ChainWorld) -> Path:
    return world.box.work_dir / "logs" / STOP_FILE


def ask(world: ChainWorld) -> None:
    """The operator's one line: ``touch "$SN120_WORK_DIR/logs/run-loop.stop-after-round"``."""
    path = request_file(world)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    os.utime(path, (ASKED_AT, ASKED_AT))


def take_back(world: ChainWorld) -> None:
    request_file(world).unlink()


def run(world: ChainWorld, monkeypatch, loop: bool = True, work: bool = True, max_idle: int = 1) -> int:
    """``python -m sn120_v2 run`` on this world. ``loop``: the run loop started it. ``work``: SN120_WORK_DIR is set."""
    stack = Stack(world.store, world.admission, world.sampler, runner=world.runner, head=world.controller.head, overlap=world.order == OVERLAP, entropy=lambda: "a-block")
    monkeypatch.setattr(cli, "open_controller", lambda args, command: (world.controller, stack))
    monkeypatch.setattr(cli, "now", world.clock)
    monkeypatch.setattr(cli, "print_round", lambda stack, round_id: None)
    for name, value in ((WORK_ENV, str(world.box.work_dir) if work else None), (LOOP_ENV, str(world.root) if loop else None)):
        monkeypatch.delenv(name, raising=False) if value is None else monkeypatch.setenv(name, value)
    return cli.cmd_run(argparse.Namespace(rounds=None, max_idle=max_idle, idle_sleep=0))


def new_run(world: ChainWorld) -> None:
    """The process of ``run`` died and the loop started a new one: nothing is verified in its memory, and its start proves the local head."""
    world.wait()
    if getattr(world.folder, "_read_back", None) is not None:
        world.folder._read_back.wait()  # noqa: SLF001 - a dead process leaves no thread; the test waits for it instead
    world.folder.resident._verified.clear()  # noqa: SLF001
    world.start()


def records(world: ChainWorld) -> list[tuple[str, int, dict]]:
    return [(record.kind, record.at, record.data) for record in read_journal(world.root / JOURNAL_FILE)]


def kinds(world: ChainWorld) -> list[str]:
    return [kind for kind, _, _ in records(world)]


def stored(world: ChainWorld) -> dict[str, bytes]:
    return {key: world.store.get(key) for key in world.store.list("")}


def is_head(world: ChainWorld) -> bool:
    """True when the cache of the box holds exactly the base and the master of the journal's last folded head."""
    manifest = world.head
    return hash_tree(world.local(manifest.head.base_hash)) == manifest.base_files and hash_tree(world.local(manifest.head.base_fp32_master_digest)) == manifest.master_files


class Crash(BaseException):
    """The process of ``run`` dies."""


def at_record(world: ChainWorld, kind: str, act) -> None:
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


def in_the_round(world: ChainWorld, act) -> None:
    """Do ``act`` one time while the box works on the next round."""
    real, done = world.runner.run, []

    def box(ticket):
        if not done:
            done.append(True)
            act()
        return real(ticket)

    world.runner.run = box


# Where the operator's line lands in a round. Each is a point at which ``run`` must not stop.
POINTS = {
    "in the round": lambda world, act: in_the_round(world, act),
    "at the decision": lambda world, act: at_record(world, ROUND_SETTLED, act),
    "in the fold": lambda world, act: at_record(world, HEAD_APPLIED, act),
    "in the close": lambda world, act: at_record(world, ROUND_CLOSED, act),  # the record is written; the pointer and the public files are not
}


def closed_in_public(world: ChainWorld, kings: int) -> None:
    """``kings`` rounds of this test are closed, each with its king, nothing of them waits, and the store shows it."""
    state = world.state
    assert state.round is None and state.head.folded and is_head(world)
    assert [record.outcome for record in state.rounds] == ["applied"] * (1 + kings) and [record.decision.winner.uid for record in state.rounds][1:] == winners(kings)
    assert kinds(world).count(ROUND_OPENED) == kinds(world).count(ROUND_CLOSED) == kinds(world).count(HEAD_APPLIED) == kinds(world).count(HEAD_FOLDED) == 1 + kings
    assert len(waiting(state.queue)) == (ROUNDS - kings) * world.params.batch_cap, "the seats of the next rounds wait"
    assert not any(seat.leased for seat in state.queue.values())
    assert world.store.get(layout.CURRENT) == state.manifest.to_json(), "the head pointer names the head of the journal"
    assert world.store.exists(layout.round_file(state.rounds[-1].round_id))
    assert not world.store.list(layout.pending_round_prefix(state.rounds[-1].round_id)), "the staged files of the last round are deleted"
    shown = world.publisher.snapshots[-1]
    assert shown["round"] is None and shown["kings"][0]["uid"] == state.kings[-1].uid == state.rounds[-1].decision.winner.uid, "the public files show the last king, and no open round"
    assert shown["heads"][-1] == {"version_id": state.head.version_id, "round_id": state.head.round_id, "applied_at": state.head.applied_at, "folded": True}
    if world.mode == CHAIN:
        doc = world.chain_file()
        check_chain(doc)
        assert [head["version_id"] for head in heads_of(doc)] == [head.version_id for head in state.heads if head.folded]


def last_line(capsys) -> str:
    lines = [line for line in capsys.readouterr().err.splitlines() if not line.startswith("time: ")]
    return lines[-1] if lines else ""


def said_obeyed(world: ChainWorld, line: str, last_round: str) -> None:
    assert line.startswith(f"{OBEYED} Asked at {stamp(ASKED_AT)}. Round {last_round} is closed (applied). No round is open. All that a close writes is in the store.")
    assert (" The second sha256 of the local head passed." in line) == (world.order == OVERLAP) and " run ends with exit code 0." in line


# ---- no request


@pytest.mark.parametrize("mode, order", WORLDS)
def test_with_no_request_run_writes_what_it_writes_when_no_request_can_be_made(tmp_path, monkeypatch, capsys, mode, order):
    plain, watched = a_world(tmp_path / "plain", order, mode), a_world(tmp_path / "watched", order, mode)
    capsys.readouterr()
    assert run(plain, monkeypatch, work=False) == 0
    assert SAID not in capsys.readouterr().err, "no work directory: no request can be made, and run says nothing of it"
    assert run(watched, monkeypatch) == 0
    said = [line for line in capsys.readouterr().err.splitlines() if line.startswith(SAID)]
    assert said == [f"{SAID}run looks for the file {request_file(watched)} each time before it opens a round (make it: touch, take it back: rm). Now no request is there."]
    for world in (plain, watched):
        world.wait()
        closed_in_public(world, ROUNDS)
    assert records(watched) == records(plain), "the same records with the same times, in the same order"
    assert stored(watched) == stored(plain), "each key of the store, public and private, with the same bytes"
    assert watched.publisher.snapshots == plain.publisher.snapshots, "and each public snapshot, in the same order"
    assert not request_file(watched).exists(), "run makes no request file"


# ---- a request, and where run obeys it


@pytest.mark.parametrize("mode, order", WORLDS)
def test_a_request_before_the_round_opens_no_round_and_under_the_loop_the_file_stays_for_the_loop(tmp_path, monkeypatch, capsys, mode, order):
    world = a_world(tmp_path / "world", order, mode)
    before, public = records(world), len(world.publisher.snapshots)
    ask(world)
    capsys.readouterr()
    assert run(world, monkeypatch) == 0
    text = capsys.readouterr().err
    assert f"Now a request of {stamp(ASKED_AT)} is there." in text
    line = [line for line in text.splitlines() if not line.startswith("time: ")][-1]
    said_obeyed(world, line, world.state.rounds[-1].round_id)
    assert line.endswith(" The run loop removes the request file.") and request_file(world).exists()
    assert records(world) == before, "no record: no round was opened, and nothing was synced"
    closed_in_public(world, 0)
    assert len(world.publisher.snapshots) == public + 1, "this run had published nothing yet, so it wrote the public files one time"
    # The run was killed between its line and its exit, so the loop starts one more: it stops in the same way.
    new_run(world)
    assert run(world, monkeypatch) == 0 and records(world)[: len(before)] == before and ROUND_OPENED not in kinds(world)[len(before) :]
    said_obeyed(world, last_line(capsys), world.state.rounds[-1].round_id)
    # The loop removed the file, and the next start runs on.
    take_back(world)
    new_run(world)
    assert run(world, monkeypatch) == 0 and OBEYED not in capsys.readouterr().err
    closed_in_public(world, ROUNDS)


@pytest.mark.parametrize("order", ORDERS)
def test_a_run_with_no_loop_over_it_removes_the_request_itself_and_says_so_in_its_last_line(tmp_path, monkeypatch, capsys, order):
    world = a_world(tmp_path / "world", order)
    in_the_round(world, lambda: ask(world))
    capsys.readouterr()
    assert run(world, monkeypatch, loop=False) == 0
    line = last_line(capsys)
    said_obeyed(world, line, world.state.rounds[-1].round_id)
    assert line.endswith(" No run loop is over this run, so run removed the request file: the next start runs on.") and not request_file(world).exists()
    closed_in_public(world, 1)
    new_run(world)
    assert run(world, monkeypatch, loop=False) == 0 and OBEYED not in capsys.readouterr().err
    closed_in_public(world, ROUNDS)


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("mode, order", WORLDS)
def test_a_request_at_any_point_of_a_round_stops_run_after_the_close_of_that_round_and_before_the_next_open(tmp_path, monkeypatch, capsys, mode, order, point):
    world = a_world(tmp_path / "world", order, mode)
    seen = []
    POINTS[point](world, lambda: (ask(world), seen.append(kinds(world)[-1])))
    capsys.readouterr()
    assert run(world, monkeypatch) == 0
    said_obeyed(world, last_line(capsys), world.state.rounds[-1].round_id)
    assert seen == [{"in the round": ROUND_OPENED, "at the decision": ROUND_SETTLED, "in the fold": HEAD_APPLIED, "in the close": ROUND_CLOSED}[point]]
    assert kinds(world)[-1] == ROUND_CLOSED, "the round of the request was not cut: its last record is its close, and no record follows"
    closed_in_public(world, 1)
    assert request_file(world).exists(), "the loop removes it, not run"
    assert [shown["round"] is None for shown in world.publisher.snapshots[-2:]] == [False, True], "the public files of the close were written one time, not two"
    if order == OVERLAP:
        world.folder.proven(False)  # no second sha256 is in work, and none failed


@pytest.mark.parametrize("order", ORDERS)
def test_run_finishes_a_round_that_it_finds_open_and_then_stops(tmp_path, monkeypatch, capsys, order):
    world = a_world(tmp_path / "world", order)
    crash_at(world, ROUND_SETTLED, before=False)
    with pytest.raises(Crash):
        run(world, monkeypatch)
    ask(world)
    new_run(world)
    capsys.readouterr()
    assert run(world, monkeypatch) == 0
    text = capsys.readouterr().err
    open_round = world.state.rounds[-1].round_id
    assert f"{SAID}asked at {stamp(ASKED_AT)}. Round {open_round} is open (phase APPLY): run stops when it is closed." in text
    said_obeyed(world, [line for line in text.splitlines() if not line.startswith("time: ")][-1], open_round)
    closed_in_public(world, 1)


# ---- a request that is taken back


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("order", ORDERS)
def test_a_request_that_is_taken_back_before_the_next_open_stops_nothing(tmp_path, monkeypatch, capsys, order, point):
    plain, world = a_world(tmp_path / "plain", order), a_world(tmp_path / "world", order)
    in_the_round(world, lambda: ask(world))
    POINTS[point](world, lambda: take_back(world)) if point != "in the round" else at_record(world, ROUND_SETTLED, lambda: take_back(world))
    assert run(plain, monkeypatch) == 0
    capsys.readouterr()
    assert run(world, monkeypatch) == 0
    assert OBEYED not in capsys.readouterr().err
    for each in (plain, world):
        each.wait()
    closed_in_public(world, ROUNDS)
    assert records(world) == records(plain) and stored(world) == stored(plain), "as a run that never had a request"


@pytest.mark.parametrize("order", ORDERS)
def test_a_request_that_is_taken_back_while_run_finishes_an_open_round_stops_nothing(tmp_path, monkeypatch, capsys, order):
    world = a_world(tmp_path / "world", order)
    crash_at(world, ROUND_SETTLED, before=False)
    with pytest.raises(Crash):
        run(world, monkeypatch)
    ask(world)
    new_run(world)
    at_record(world, HEAD_FOLDED, lambda: take_back(world))
    capsys.readouterr()
    assert run(world, monkeypatch) == 0
    text = capsys.readouterr().err
    assert "run stops when it is closed." in text and OBEYED not in text
    closed_in_public(world, ROUNDS)


# ---- a ``run`` that dies after a request


EVENTS = ["in the round", (ROUND_SETTLED, True), "at the decision", (HEAD_APPLIED, True), "in the fold", (HEAD_FOLDED, True), (HEAD_FOLDED, False), (ROUND_CLOSED, True), "in the close"]
AFTER = {"at the decision": (ROUND_SETTLED, False), "in the fold": (HEAD_APPLIED, False), "in the close": (ROUND_CLOSED, False)}
KILLS = [(point, *(AFTER[event] if isinstance(event, str) else event)) for point in POINTS for event in EVENTS[EVENTS.index(point) + 1 :] + [point] if event != "in the round"]


@pytest.mark.parametrize("point, kind, before", KILLS, ids=[f"asked {point}, killed {'before' if before else 'after'} {kind}" for point, kind, before in KILLS])
@pytest.mark.parametrize("order", ORDERS)
def test_a_run_that_dies_after_a_request_is_replaced_by_one_that_finishes_the_round_and_stops(tmp_path, monkeypatch, capsys, order, point, kind, before):
    world = a_world(tmp_path / "world", order)
    POINTS[point](world, lambda: ask(world))  # first, so a kill at the record of the request comes after the request
    crash_at(world, kind, before)
    with pytest.raises(Crash):
        run(world, monkeypatch)
    assert request_file(world).exists(), "the request is a file: it is there for the next run"
    new_run(world)
    assert is_head(world), "the start of the new run found or made the files of the journal's head"
    capsys.readouterr()
    assert run(world, monkeypatch) == 0
    said_obeyed(world, last_line(capsys), world.state.rounds[-1].round_id)
    world.wait()
    closed_in_public(world, 1)
    assert request_file(world).exists()


@pytest.mark.parametrize("order", ORDERS)
def test_a_run_that_dies_while_it_writes_the_public_files_of_the_last_close_is_replaced_by_one_that_writes_them_and_stops(tmp_path, monkeypatch, capsys, order):
    """``serial`` writes them in the close. ``overlap`` keeps them for the next open, so it writes them in the stop itself."""
    world = a_world(tmp_path / "world", order)
    in_the_round(world, lambda: ask(world))
    real, died = world.publisher.publish, []

    def dies(snapshot):
        if request_file(world).exists() and snapshot["round"] is None:
            died.append(kinds(world)[-1])
            raise Crash("in the public files")
        real(snapshot)

    world.publisher.publish = dies
    with pytest.raises(Crash):
        run(world, monkeypatch)
    assert died == [ROUND_CLOSED] and world.publisher.snapshots[-1]["round"] is not None, "the round is closed in the journal, and the public files show it as open"
    world.publisher.publish = real
    new_run(world)
    capsys.readouterr()
    assert run(world, monkeypatch) == 0
    said_obeyed(world, last_line(capsys), world.state.rounds[-1].round_id)
    closed_in_public(world, 1)


# ---- the close is not complete in the store, or the head is not proven


@pytest.mark.parametrize("order", ORDERS)
def test_run_does_not_say_obeyed_before_the_public_files_show_the_close(tmp_path, monkeypatch, capsys, order):
    """The store fails one time for the public files of the last close. ``run`` writes them again before it stops."""
    world = a_world(tmp_path / "world", order)
    in_the_round(world, lambda: ask(world))
    real, failed = world.publisher.publish, []

    def fails_one_time(snapshot):
        if not failed and request_file(world).exists() and snapshot["round"] is None:
            failed.append(True)
            raise InfraError("the store is down")
        real(snapshot)

    world.publisher.publish = fails_one_time
    capsys.readouterr()
    assert run(world, monkeypatch, max_idle=3) == 0
    text = capsys.readouterr().err
    said_obeyed(world, [line for line in text.splitlines() if not line.startswith("time: ")][-1], world.state.rounds[-1].round_id)
    assert failed == [True] and text.count(OBEYED) == 1
    if order == OVERLAP:  # the write was in the stop: the first try says what waits, opens no round, and the next try writes it
        assert f"{SAID}asked at {stamp(ASKED_AT)}. No round is open, and run opens none. Not all that a close writes is in the store yet: publish failed: the store is down. The next try is in 0 s." in text
    closed_in_public(world, 1)


@pytest.mark.parametrize("order", ORDERS)
def test_a_store_that_stays_down_ends_run_with_no_new_round_and_without_the_word_obeyed(tmp_path, monkeypatch, capsys, order):
    world = a_world(tmp_path / "world", order)
    in_the_round(world, lambda: ask(world))
    real = world.publisher.publish

    def down(snapshot):
        if request_file(world).exists() and snapshot["round"] is None:
            raise InfraError("the store is down")
        real(snapshot)

    world.publisher.publish = down
    capsys.readouterr()
    assert run(world, monkeypatch, max_idle=2) == 0, "--max-idle counts the tries"
    text = capsys.readouterr().err
    assert OBEYED not in text and text.count("Not all that a close writes is in the store yet: publish failed: the store is down.") == 2
    assert kinds(world).count(ROUND_OPENED) == 2 and world.state.round is None and request_file(world).exists()


def test_a_head_that_does_not_read_back_is_not_a_stop_as_asked_and_the_next_run_rebuilds_it_and_stops(tmp_path, monkeypatch, capsys):
    """``overlap``: the second sha256 of the last head can end after the close. The stop waits for it."""
    world = a_world(tmp_path / "world", OVERLAP)
    in_the_round(world, lambda: ask(world))

    def faulty(jobs):
        results = inline(jobs)
        target = sorted(file for file in jobs[0].out_master.iterdir() if file.is_file())[0]
        raw = bytearray(target.read_bytes())
        raw[-1] ^= 1
        target.write_bytes(bytes(raw))
        return results

    world.folder.run = faulty
    capsys.readouterr()
    assert run(world, monkeypatch) == HEAD_LOST_EXIT == 75, "the code that the run loop takes as a crash: it starts run again"
    text = capsys.readouterr().err
    assert OBEYED not in text and f"{SAID}not obeyed yet, the request of {stamp(ASKED_AT)} stays. The next start of run rebuilds the local head and then stops for it." in text
    assert kinds(world)[-1] == ROUND_CLOSED and request_file(world).exists() and not is_head(world)
    world.folder.run = inline
    new_run(world)
    assert is_head(world), "rebuilt from the chain"
    assert run(world, monkeypatch) == 0
    said_obeyed(world, last_line(capsys), world.state.rounds[-1].round_id)
    closed_in_public(world, 1)


# ---- the request file


def test_the_request_is_the_file_in_the_logs_of_the_work_directory_and_its_time_is_the_time_of_the_request(tmp_path):
    assert StopRequest.from_env({}).path is None and StopRequest.from_env({}).asked() is None and StopRequest.from_env({WORK_ENV: " "}).path is None
    request = StopRequest.from_env({WORK_ENV: str(tmp_path), LOOP_ENV: "a state directory"})
    assert request.path == tmp_path / "logs" / "run-loop.stop-after-round" and request.under_loop and not StopRequest.from_env({WORK_ENV: str(tmp_path)}).under_loop
    assert request.asked() is None, "no logs directory, no file: no request"
    request.path.parent.mkdir()
    request.path.touch()
    os.utime(request.path, (ASKED_AT, ASKED_AT))
    assert request.asked() == ASKED_AT and stamp(request.asked()) == "2023-11-14T22:13:20Z"
    request.path.unlink()
    assert request.asked() is None
