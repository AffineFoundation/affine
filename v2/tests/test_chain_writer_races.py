"""The chain file (``public/head_chain.json``) with more than one writer, on a real tiny head.

A checkpoint is a head whose full weight files are in the store. The thread ``head-checkpoint`` uploads them and
writes a marker. The chain file shows a checkpoint as a flag of its step, and ``start`` names the newest one.
A reader builds the head from ``start`` and the LoRAs after it.

Three things are proven here, each in the orders ``serial`` and ``overlap`` where the order matters.

* No write of the chain file takes back a checkpoint that an earlier write showed.
* A reader with no cache gets the head at each moment. It has mining.json, the chain file, and the links file
  that the site makes from the chain file. So the chain file and mining.json name the same ``start``, or ``start``
  of the chain file is the last folded head: the links file then has the plain files of the head, and ``start``
  of mining.json is not used.
* The writer compares its record with the chain file of the store, not with what it wrote last. So ``run`` writes
  its record again when another process wrote the file, and a process that starts writes no file that the store
  has. ``publish`` by hand only reads the journal: when ``run`` made a new head after that read, it writes no
  public file. ``mark`` writes nothing before the first publish of its process.
  ``run`` owns the journal: it writes its record over a later one too, and says so in one line. A read of the
  chain file that the store fails is handled as a write that fails: this is proven with ``R2Store`` on an S3 fake.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, NamedTuple

import boto3
import pytest
from botocore.exceptions import ClientError
from moto.server import ThreadedMotoServer

torch = pytest.importorskip("torch")

from sn120_v2.sim_head import ChainWorld  # noqa: E402 - after the skip guard

from sn120_v2 import controller, layout  # noqa: E402
from sn120_v2.afterking import OVERLAP, SERIAL  # noqa: E402
from sn120_v2.controller import HeadPublish  # noqa: E402
from sn120_v2.errors import HeadMismatchError, V2Error  # noqa: E402
from sn120_v2.gpu.resident import hash_tree  # noqa: E402
from sn120_v2.headchain import CHAIN, ChainWriter, chain_links, checkpoints, heads_of, promote_fold, tail  # noqa: E402
from sn120_v2.headfold import HeadKeeper, LinkSource, rebuild  # noqa: E402
from sn120_v2.journal import JOURNAL_FILE, read_journal  # noqa: E402
from sn120_v2.r2 import R2Store  # noqa: E402
from sn120_v2.sim import sim_params  # noqa: E402
from sn120_v2.state import State, replay  # noqa: E402
from sn120_v2.store import LocalStore  # noqa: E402

WAIT_S = 30  # the longest a test waits for a thread
MARKERS = layout.head_checkpoint("x").rpartition("/")[0]  # what ``headchain.checkpoints`` lists
AFTER_THE_OPEN = "sn120-v2-after-the-open"
COUNTED = ("stopped: ", "PUBLISH FAILED", "run ended with exit code", "do not hash to the record of the chain")  # the words that the operator counts in the log of ``run``
PLAIN, BY_CHAIN = "the plain files of the head", "the newest checkpoint and the folds after it"  # the two ways of ``replicate.fetch``


class Write(NamedTuple):
    thread: str
    head: str
    marked: frozenset[str]  # the heads that this write shows as a checkpoint


def a_world(path: Path, order: str) -> ChainWorld:
    """``run`` in the mode ``chain`` with a checkpoint for each folded head. The upload of a weight file waits for ``world.store.gate``."""
    world = ChainWorld(path, sim_params(), order=order)
    world.start(CHAIN, every=1)
    world.store.gate = threading.Event()
    return world


def until(what: Callable[[], bool]) -> None:
    end = time.monotonic() + WAIT_S
    while not what():
        assert time.monotonic() < end, "waited too long"
        time.sleep(0.005)


def writes_of(world: ChainWorld) -> list[Write]:
    """Each write of the chain file from now on, in order."""
    writes, put = [], world.store.put

    def watched(key: str, data: bytes) -> None:
        if key == layout.HEAD_CHAIN:
            doc = json.loads(data)
            writes.append(Write(threading.current_thread().name, doc["head_version_id"], frozenset(step["version_id"] for step in doc["steps"] if step["checkpoint"])))
        put(key, data)

    world.store.put = watched
    return writes


def went_back(writes: list[Write]) -> list[Write]:
    """The writes that show no checkpoint for a head that an earlier write showed as one."""
    shown, back = set(), []
    for write in writes:
        if shown - write.marked:
            back.append(write)
        shown |= write.marked
    return back


def a_marker_after_the_list_of(world: ChainWorld, thread: str) -> list[bool]:
    """The next list of the markers by ``thread`` is the list of a write of the chain file. Before that list returns, the checkpoint thread writes its marker and calls ``mark``.

    The list returns what the store had before the marker. Without the lock around the list and the write,
    ``mark`` writes its checkpoint here, and the write that follows takes it back. With the lock, ``mark``
    waits for it: the wait for the checkpoint thread below then ends by its time, and ``mark`` comes after the write.
    """
    listed, fired, before = world.store.list, [], checkpoints(world.store)

    def held(prefix: str) -> list[str]:
        found = listed(prefix)
        if not fired and prefix == MARKERS and threading.current_thread().name == thread:
            fired.append(True)
            world.store.gate.set()
            until(lambda: checkpoints(world.store) != before)
            world.keeper.wait(2)
        return found

    world.store.list = held
    return fired


def idle_pass(world: ChainWorld) -> bool:
    """One pass of ``run`` with no round, as ``cli.cmd_run`` makes it. True when it wrote the public files."""
    assert world.round() is None
    return world.controller.publish(world.clock(), only_changes=True)


class Uploads:
    """The checkpoint upload of each head waits here until the test lets it go. ``marked``: the heads whose ``mark`` returned."""

    def __init__(self, world: ChainWorld) -> None:
        self.world, self.marked, self._go, self._all = world, [], {}, threading.Event()
        upload, mark = world.keeper.upload, world.writer.mark

        def held(manifest) -> None:
            until(lambda: self._all.is_set() or self._event(manifest.head.version_id).is_set())
            upload(manifest)

        def told(version_id: str) -> None:
            mark(version_id)
            self.marked.append(version_id)

        world.keeper.upload, world.writer.mark, world.store.gate = held, told, None

    def _event(self, version_id: str) -> threading.Event:
        return self._go.setdefault(version_id, threading.Event())

    def let(self, version_id: str) -> None:
        """The upload of this head runs now: its files, its marker, and ``mark``. Returns after ``mark``."""
        self._event(version_id).set()
        until(lambda: version_id in self.marked)

    def end(self) -> None:
        self._all.set()
        self.world.wait()


class Moment(NamedTuple):
    thread: str
    wrote: str  # "chain" or "mining"
    chain: dict  # the chain file as the store has it after the write
    mining: dict  # ``head_chain`` of the mining.json of then
    half: bool  # the chain file of a publish is written and its mining.json is not yet: the next write of the thread is that one


def moments_of(world: ChainWorld, monkeypatch: pytest.MonkeyPatch) -> list[Moment]:
    """From now on, after each write of the chain file and of mining.json: the two as a reader can get them then."""
    moments, put, publish, snapshot_of, publishing = [], world.store.put, world.publisher.publish, controller.public_snapshot, set()

    def now(wrote: str) -> None:
        thread = threading.current_thread().name
        moments.append(Moment(thread, wrote, world.chain_file(), world.publisher.snapshots[-1]["head_chain"], wrote == "chain" and thread in publishing))

    def put_and_look(key: str, data: bytes) -> None:
        put(key, data)
        if key == layout.HEAD_CHAIN:
            now("chain")

    def snapshot_and_note(*args, **kwargs):
        publishing.add(threading.current_thread().name)
        return snapshot_of(*args, **kwargs)

    def publish_and_look(snapshot) -> None:
        publish(snapshot)
        publishing.discard(threading.current_thread().name)
        now("mining")

    world.store.put, world.publisher.publish = put_and_look, publish_and_look
    monkeypatch.setattr(controller, "public_snapshot", snapshot_and_note)
    return moments


def links_file(world: ChainWorld, chain: dict) -> dict:
    """The links file as ``read_links`` of ``site/pull.py`` makes it from this chain file.

    It has the newest checkpoint and the LoRAs of the folds after it. ``base`` has the plain files of the head
    only when that checkpoint is the last folded head.
    """
    last = heads_of(chain)[-1]
    links = {"expires_at": int(time.time()) + 600, "head_version_id": chain["head_version_id"]} | chain_links(world.store, chain, 600, target=last["version_id"])
    point = links["checkpoints"][0]
    return links | {"base": point["base"] if point["base"]["id"] == last["base_hash"] else None}


def read(world: ChainWorld, chain: dict, mining: dict, work: Path) -> str:
    """A reader with no cache gets the head with these two files, as ``replicate.fetch`` does. Returns the way it took. An error of the reader is raised."""
    links, head = links_file(world, chain), heads_of(mining)[-1]
    assert links["head_version_id"] == mining["head_version_id"], "the two files name two heads: fetch says so, and the reader fetches both again"
    if links["base"] is not None:  # ``fetch`` looks at ``base`` first: ``start`` of mining.json is not used
        assert links["base"]["id"] == head["base_hash"]
        return PLAIN
    built = rebuild(mining, work, LinkSource(links))
    assert built["version_id"] == head["version_id"] and hash_tree(Path(built["base_dir"])) == world.state.manifests[head["version_id"]].base_files
    return BY_CHAIN


def no_way(world: ChainWorld, moments: list[Moment], root: Path) -> list[str]:
    """The moments at which a reader with no cache did not get the head.

    Two kinds of moment are not read. The two files name two heads: ``fetch`` says so. A publish is at its half:
    it writes the chain file and then mining.json, and a store has no write of two files at once.
    """
    found, seen = [], {}
    for moment in moments:
        if moment.half or moment.chain["head_version_id"] != moment.mining["head_version_id"]:
            continue
        key = json.dumps([moment.chain["start"], moment.mining["start"], moment.chain["head_version_id"]], sort_keys=True)
        if key not in seen:
            try:
                seen[key] = read(world, moment.chain, moment.mining, root / f"reader-{len(seen)}") and ""
            except V2Error as exc:
                seen[key] = f"{type(exc).__name__}: {exc}"
        if seen[key]:
            found.append(f"after a write of {moment.wrote} by {moment.thread}: start {moment.chain['start']['version_id'][:8]} in the chain file, {moment.mining['start']['version_id'][:8]} in mining.json: {seen[key]}")
    return found


def starts(world: ChainWorld) -> tuple[str, str]:
    """``start`` of the chain file and of mining.json, as they are now."""
    return world.chain_file()["start"]["version_id"], world.publisher.snapshots[-1]["head_chain"]["start"]["version_id"]


def flags(world: ChainWorld) -> list[bool]:
    return [step["checkpoint"] for step in world.chain_file()["steps"]]


def two_kings(path: Path, order: str) -> tuple[ChainWorld, Uploads, str, str, str]:
    """Two kings whose checkpoint uploads wait, and all that their closes write is in the store. Returns the launch head and the two heads."""
    world = a_world(path, order)
    uploads = Uploads(world)
    first, second = world.king().head.version_id, world.king().head.version_id
    assert world.controller.close_in_public(world.clock()) == ""  # ``overlap``: the public files of the last close, which wait for a next open
    idle_pass(world)  # the first pass after a round publishes: the files of the seats that left are no longer shown
    genesis = world.chain_file()["genesis"]["version_id"]
    assert starts(world) == (genesis, genesis) and flags(world) == [False, False] and world.chain_file()["steps"][-1]["version_id"] == second
    return world, uploads, genesis, first, second


def a_round_with_no_king(world: ChainWorld, in_the_box: Callable[[], None] | None = None) -> None:
    """One round that makes no head. ``in_the_box`` runs while the box has the round, after the thread of the open."""
    run = world.runner.run

    def box(ticket):
        if world.overlap is not None:
            until(lambda: world.overlap._worker is None or world.overlap._worker.done())
        in_the_box()
        return run(ticket)

    if in_the_box is not None:
        world.runner.run = box
    world.register(world.uid, 0.0)
    world.uid += 1
    try:
        record = world.round()
    finally:
        world.runner.run = run
    assert record is not None and record.outcome != "applied"


# ---- no write takes a checkpoint back ----


@pytest.mark.parametrize(
    "order, publish, thread",
    [(SERIAL, "a publish", "MainThread"), (OVERLAP, "the publish at the end of run", "MainThread"), (OVERLAP, "the publish after the open", AFTER_THE_OPEN)],
)
def test_a_marker_that_comes_between_the_list_and_the_write_of_a_publish_is_not_taken_back(tmp_path, order, publish, thread):
    world = a_world(tmp_path / "world", order)
    head = world.king().head.version_id  # its upload waits
    assert checkpoints(world.store) == set() and world.chain_file()["steps"][-1]["checkpoint"] is False
    writes, fired = writes_of(world), a_marker_after_the_list_of(world, thread)
    if publish == "a publish":
        world.controller.publish(world.clock())
    elif publish == "the publish at the end of run":
        assert world.overlap.flush(world.clock()) == []  # ``cli.cmd_run`` after its loop
    else:
        world.register(world.uid, 0.0)
        assert world.round().outcome != "applied"  # a round with no king: its open starts the thread that publishes
    world.wait()
    assert fired and checkpoints(world.store) == {head}
    assert went_back(writes) == [], "a write of the chain file took a checkpoint back"
    idle_pass(world)
    assert world.chain_file()["steps"][0]["checkpoint"] is True and world.chain_file()["start"]["version_id"] == head
    assert went_back(writes) == []


# ---- the chain file and mining.json move ``start`` in one publish ----


@pytest.mark.parametrize("order", [SERIAL, OVERLAP])
def test_the_checkpoint_of_a_head_that_is_not_the_last_one_shows_with_the_next_publish_in_the_two_files(tmp_path, monkeypatch, order):
    world, uploads, _, first, second = two_kings(tmp_path / "world", order)
    before, moments, count = world.store.get(layout.HEAD_CHAIN), moments_of(world, monkeypatch), len(world.publisher.snapshots)
    uploads.let(first)  # the marker of head 1 comes when head 2 is the head
    assert checkpoints(world.store) == {first}
    assert world.store.get(layout.HEAD_CHAIN) == before, "``mark`` moved ``start`` to head 1 in the chain file alone: mining.json starts at the launch head, and the links file has no file of it"
    assert no_way(world, moments, tmp_path) == []
    assert idle_pass(world) is False and world.store.get(layout.HEAD_CHAIN) == before, "the pass with no round does not publish for a marker"
    world.controller.publish(world.clock())
    assert starts(world) == (first, first) and flags(world) == [True, False] and len(world.publisher.snapshots) == count + 1
    assert read(world, world.chain_file(), world.publisher.snapshots[-1]["head_chain"], tmp_path / "from-head-1") == BY_CHAIN
    uploads.let(second)  # the marker of the head itself: the chain file shows it at once, as before
    assert starts(world) == (second, first) and flags(world) == [True, True] and len(world.publisher.snapshots) == count + 1
    assert read(world, world.chain_file(), world.publisher.snapshots[-1]["head_chain"], tmp_path / "plain") == PLAIN
    uploads.end()
    assert no_way(world, moments, tmp_path) == []
    assert [moment.thread for moment in moments if moment.wrote == "chain"] == ["MainThread", "head-checkpoint"]


@pytest.mark.parametrize("order", [SERIAL, OVERLAP])
def test_a_mark_that_writes_gives_the_file_each_flag_that_the_store_has(tmp_path, monkeypatch, order):
    world, uploads, genesis, first, second = two_kings(tmp_path / "world", order)
    before, moments = world.store.get(layout.HEAD_CHAIN), moments_of(world, monkeypatch)
    uploads.let(first)
    assert world.store.get(layout.HEAD_CHAIN) == before
    uploads.let(second)  # no publish came between the two
    assert flags(world) == [True, True] and starts(world) == (second, genesis), "the flag of head 1 waited for no publish"
    marked = world.store.get(layout.HEAD_CHAIN)
    world.controller.publish(world.clock())
    assert world.store.get(layout.HEAD_CHAIN) == marked and starts(world) == (second, second), "the write of ``mark`` is the write that a publish makes"
    uploads.end()
    assert [(moment.thread, moment.wrote) for moment in moments] == [("head-checkpoint", "chain"), ("MainThread", "mining")]
    assert no_way(world, moments, tmp_path) == []


@pytest.mark.parametrize("order", [SERIAL, OVERLAP])
@pytest.mark.parametrize("which", ["a head before the last one", "the last head"])
def test_a_mark_between_the_chain_file_and_the_mining_json_of_one_publish(tmp_path, monkeypatch, order, which):
    world, uploads, genesis, first, second = two_kings(tmp_path / "world", order)
    moments, publish = moments_of(world, monkeypatch), world.publisher.publish
    if which == "the last head":
        uploads.let(first)  # the uploads go oldest first. This one writes no chain file: the publish below shows it
    head, start = (first, genesis) if which == "a head before the last one" else (second, first)

    def after_the_chain_file(snapshot) -> None:
        world.publisher.publish = publish
        assert starts(world)[0] == snapshot["head_chain"]["start"]["version_id"] == start  # the chain file of this publish is written
        uploads.let(head)
        publish(snapshot)

    world.publisher.publish = after_the_chain_file
    world.controller.publish(world.clock())
    assert world.publisher.publish == publish and head in uploads.marked
    # Head 1: ``mark`` wrote nothing, the two files have the ``start`` of this publish. Head 2: ``start`` of the chain file is the head.
    assert starts(world) == ((genesis, genesis) if head == first else (second, first))
    assert no_way(world, moments, tmp_path) == []
    world.controller.publish(world.clock())
    assert starts(world) == (head, head) and no_way(world, moments, tmp_path) == []
    uploads.end()
    assert no_way(world, moments, tmp_path) == []


THEN = {
    "the next open": lambda world: a_round_with_no_king(world),
    "the end of run": lambda world: world.overlap.flush(world.clock()),  # ``cli.cmd_run`` after its loop
    "the stop after the round": lambda world: world.controller.close_in_public(world.clock()),
}


@pytest.mark.parametrize("order, then", [(SERIAL, ""), *((OVERLAP, then) for then in THEN)])
def test_a_mark_in_a_round_with_no_king_moves_start_in_the_two_files_at_once_or_in_none(tmp_path, monkeypatch, order, then):
    world, uploads, genesis, first, _ = two_kings(tmp_path / "world", order)
    before, moments = world.store.get(layout.HEAD_CHAIN), moments_of(world, monkeypatch)
    a_round_with_no_king(world, in_the_box=lambda: uploads.let(first))
    assert first in uploads.marked and no_way(world, moments, tmp_path) == []
    if order == SERIAL:  # the close publishes
        assert starts(world) == (first, first)
    else:  # the close writes no chain file for a round with no new head: the public files wait for the next open
        assert world.store.get(layout.HEAD_CHAIN) == before and starts(world) == (genesis, genesis), "the close moved ``start`` in the chain file alone"
        THEN[then](world)
        world.overlap.settle()  # the thread of the open, when there was one
        assert starts(world) == (first, first) and flags(world) == [True, False], f"{then} did not write the two files"
        wrote = [(moment.thread, moment.wrote) for moment in moments if moment.chain["start"]["version_id"] == first][:2]
        assert wrote == [(AFTER_THE_OPEN if then == "the next open" else "MainThread", "chain"), (wrote[0][0], "mining")]
    uploads.end()
    assert no_way(world, moments, tmp_path) == []


# ---- the writer reads the chain file of the store; a command by hand writes no record of an older head ----


def a_run(path: Path, order: str, kings: int, every: int = 1) -> tuple[ChainWorld, list[str]]:
    """``run`` after ``kings`` rounds with a king. All that their closes write is in the store, and no upload waits. Returns the heads."""
    world = ChainWorld(path, sim_params(), order=order)
    world.start(CHAIN, every=every)
    world.controller.publish(world.clock())  # the chain file of the launch head
    heads = [world.king().head.version_id for _ in range(kings)]
    world.wait()
    assert world.controller.close_in_public(world.clock()) == ""
    idle_pass(world)  # the first pass after a round publishes
    world.controller.publish(world.clock())
    assert world.chain_file()["head_version_id"] == world.state.head.version_id
    return world, heads


def by_hand(world: ChainWorld, state: State, owner: bool = False) -> dict:
    """What ``cli.cmd_publish`` does with the state that it read: a new writer that does not own the journal, the chain file, then mining.json. With ``owner``: the first publish of a process that took the journal."""
    head = HeadPublish(promote_fold, ChainWriter(world.store, owner=owner))
    snapshot = controller.public_snapshot(head, state, world.params, world.clock())
    world.publisher.publish(snapshot)
    return snapshot


def another_process_wrote(world: ChainWorld, data: bytes) -> None:
    """The chain file of the store is ``data`` now, and the writer of ``world`` was not told."""
    LocalStore.put(world.store, layout.HEAD_CHAIN, data)


def states_of(world: ChainWorld) -> list[State]:
    """The state of the journal at each count of heads, oldest first: what a command read that started then."""
    records, states = read_journal(world.root / JOURNAL_FILE), []
    for count in range(1, len(records) + 1):
        state = replay(records[:count])
        if len(state.heads) > (len(states[-1].heads) if states else 0):
            states.append(state)
    return states


@pytest.mark.parametrize("order", [SERIAL, OVERLAP])
def test_run_writes_its_chain_file_again_when_the_store_holds_another_one(tmp_path, order):
    world, _ = a_run(tmp_path / "world", order, kings=0)
    older = world.store.get(layout.HEAD_CHAIN)
    head = world.king().head.version_id
    world.wait()
    world.controller.publish(world.clock())
    mine = world.store.get(layout.HEAD_CHAIN)
    assert json.loads(mine)["head_version_id"] == head != json.loads(older)["head_version_id"]
    writes = writes_of(world)
    another_process_wrote(world, older)  # the record of an older state
    world.controller.publish(world.clock())
    assert world.store.get(layout.HEAD_CHAIN) == mine, "run compared with what it wrote last: the chain file of the older head stays"
    world.controller.publish(world.clock())
    assert [write.head for write in writes] == [head], "the store has these bytes: the second publish writes no chain file"


@pytest.mark.parametrize("order", [SERIAL, OVERLAP])
def test_the_first_publish_after_a_restart_of_run_reads_the_chain_file_of_the_store(tmp_path, order):
    world, (head,) = a_run(tmp_path / "world", order, kings=1)
    before, writes, count = world.store.get(layout.HEAD_CHAIN), writes_of(world), len(world.publisher.snapshots)
    world.start()  # ``run`` starts again: a new writer, which has written nothing
    world.controller.publish(world.clock())
    assert writes == [] and world.store.get(layout.HEAD_CHAIN) == before, "the store had the record of this state: there is nothing to write"
    assert len(world.publisher.snapshots) == count + 1 and starts(world) == (head, head)
    another_process_wrote(world, b"not a chain file")
    world.start()
    world.controller.publish(world.clock())
    assert world.store.get(layout.HEAD_CHAIN) == before and [write.head for write in writes] == [head]


@pytest.mark.parametrize("order", [SERIAL, OVERLAP])
def test_a_mark_before_the_first_publish_of_a_process_writes_nothing_and_that_publish_shows_the_marker(tmp_path, order):
    world, (head,) = a_run(tmp_path / "world", order, kings=1, every=0)  # no checkpoint is due
    before, writes = world.store.get(layout.HEAD_CHAIN), writes_of(world)
    assert checkpoints(world.store) == set() and starts(world)[0] != head
    world.start(every=1)  # ``run`` starts again, and now the head is due: the upload and its ``mark`` come before a publish of this process
    world.wait()
    assert checkpoints(world.store) == {head}
    assert writes == [] and world.store.get(layout.HEAD_CHAIN) == before, "a writer that wrote no file has no record to mark"
    world.controller.publish(world.clock())
    assert starts(world) == (head, head) and [(write.thread, write.marked) for write in writes] == [("MainThread", {head})]


def test_the_checkpoint_command_by_hand_writes_no_chain_file_and_the_next_publish_of_run_shows_its_marker(tmp_path):
    world, (head,) = a_run(tmp_path / "world", SERIAL, kings=1, every=0)
    before, writes = world.store.get(layout.HEAD_CHAIN), writes_of(world)
    keeper = HeadKeeper(world.store, world.box.work_dir, ChainWriter(world.store, owner=False), 0, world.settings, world.run)  # as ``assembly.stack`` builds it for ``checkpoint``
    assert keeper.checkpoint(world.state)["uploaded"] is True and checkpoints(world.store) == {head}
    assert writes == [] and world.store.get(layout.HEAD_CHAIN) == before
    assert idle_pass(world) is False and writes == [], "the pass with no round does not publish for a marker"
    world.controller.publish(world.clock())
    assert starts(world) == (head, head) and [write.marked for write in writes] == [{head}]


def test_a_command_that_only_reads_the_journal_writes_no_public_file_of_an_older_state(tmp_path):
    world, heads = a_run(tmp_path / "world", SERIAL, kings=2)
    states = states_of(world)  # the launch head; then for each king the head with its LoRA, and the folded head
    assert [len(state.heads) for state in states] == [1, 2, 3, 4, 5] and [states[2].head.version_id, states[4].head.version_id] == heads
    records = []
    for state in reversed(states):  # the owner of the journal writes its record over any other, also over a later one
        ChainWriter(world.store)(state)
        records.insert(0, world.store.get(layout.HEAD_CHAIN))
        assert json.loads(records[0])["head_version_id"] == state.head.version_id
    assert len(set(records)) == len(states)
    last, launch = json.loads(records[-1]), states[0].head.version_id
    assert (last["start"]["version_id"], last["start"]["steps_to_head"]) == (heads[1], 0) and heads[1] not in {head.version_id for state in states[:4] for head in state.heads}
    assert tail(last)["steps"] == [], "the part that mining.json shows has no head of the older states"
    # ``start`` of the last record is past each head of the older states. The whole file still lists each fold from the launch head,
    # so the code knows it as a later record of this journal, and the loop below proves that each older state is refused.
    assert last["genesis"]["version_id"] == launch and [step["version_id"] for step in last["steps"]] == heads
    for there, stored in enumerate(records):
        for here, state in enumerate(states):
            another_process_wrote(world, stored)
            count = len(world.publisher.snapshots)
            if here < there:  # ``run`` made a head after the command read the journal
                with pytest.raises(HeadMismatchError, match="a later head"):
                    by_hand(world, state)
                assert world.store.get(layout.HEAD_CHAIN) == stored and len(world.publisher.snapshots) == count, f"the state with {here + 1} heads was written over the record with {there + 1}"
            else:
                assert by_hand(world, state)["head_chain"]["head_version_id"] == state.head.version_id
                assert world.store.get(layout.HEAD_CHAIN) == records[here] and len(world.publisher.snapshots) == count + 1
    other = json.loads(records[-1])
    other["genesis"]["version_id"] = "0" * 64  # a record that is not of this journal
    for stored in (b"not a chain file", b"{}", (json.dumps(other) + "\n").encode()):
        another_process_wrote(world, stored)
        by_hand(world, states[0])
        assert world.store.get(layout.HEAD_CHAIN) == records[0], "only a later record of the same journal stops the command"


def test_run_says_one_line_when_it_writes_its_record_over_the_record_of_a_later_state_of_its_journal(tmp_path, capsys):
    world, heads = a_run(tmp_path / "world", SERIAL, kings=2)
    states, later = states_of(world), world.store.get(layout.HEAD_CHAIN)
    assert "chain file:" not in capsys.readouterr().err, "run with one journal and one writer has no such line"
    by_hand(world, states[2], owner=True)  # ``run`` starts with a journal that was put back by hand to the state after the first king
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("chain file:")]
    assert len(lines) == 1 and f"head {heads[1][:12]}, 2 head(s) more" in lines[0] and f"(head {heads[0][:12]})" in lines[0]
    assert not [words for words in COUNTED if words in lines[0]]
    assert world.chain_file()["head_version_id"] == heads[0]
    by_hand(world, states[2], owner=True)  # the store has the record of this state now
    other = json.loads(later)
    other["genesis"]["version_id"] = "0" * 64
    for stored in (b"not a chain file", (json.dumps(other) + "\n").encode(), ChainWriter(world.store).doc(states[1])):  # no later record of this journal
        another_process_wrote(world, stored if isinstance(stored, bytes) else (json.dumps(stored) + "\n").encode())
        by_hand(world, states[2], owner=True)
        assert world.chain_file()["head_version_id"] == heads[0]
    assert "chain file:" not in capsys.readouterr().err


# ---- the read before the write, with ``R2Store`` on an S3 fake ----


@pytest.fixture(scope="module")
def endpoint():
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def bucket(endpoint) -> R2Store:
    """A new bucket with no object."""
    env = {"SN120_R2_ENDPOINT": endpoint, "SN120_R2_BUCKET": f"b-{uuid.uuid4().hex}", "SN120_R2_ACCESS_KEY_ID": "fake-id", "SN120_R2_SECRET_ACCESS_KEY": "fake-secret"}
    raw = boto3.client("s3", endpoint_url=endpoint, region_name="auto", aws_access_key_id="fake-id", aws_secret_access_key="fake-secret")
    raw.create_bucket(Bucket=env["SN120_R2_BUCKET"], CreateBucketConfiguration={"LocationConstraint": "auto"})
    return R2Store.from_env(env, attempts=1)


def on(bucket: R2Store, world: ChainWorld) -> None:
    """From now on the chain writer of ``run`` reads and writes the bucket: the publish, the close, and ``mark`` from the checkpoint thread.

    The bucket gets each object that the store of the world has now. The other parts of ``run`` keep the store
    of the world, so a marker that the checkpoint thread writes later is not in the bucket.
    """
    for key in world.store.list(""):
        bucket.put(key, world.store.get(key))
    world.writer.store = bucket


def the_store_fails(patch: pytest.MonkeyPatch, bucket: R2Store, call: str) -> list[str]:
    """Each ``call`` of the S3 client for the chain file gets the answer ``InternalError`` from now on. Returns the list of these calls."""
    calls, real = [], getattr(bucket._s3, call)

    def failed(**request):
        if request["Key"] != layout.HEAD_CHAIN:
            return real(**request)
        calls.append(call)
        raise ClientError({"Error": {"Code": "InternalError"}}, call)

    patch.setattr(bucket._s3, call, failed)
    return calls


def in_the(bucket: R2Store) -> dict:
    return json.loads(bucket.get(layout.HEAD_CHAIN))


FAILS = {"get_object": "r2 get 'public/head_chain.json' failed: InternalError", "put_object": "r2 put 'public/head_chain.json' failed: InternalError"}  # the read, and the write as the base to compare with


def test_the_first_chain_publish_of_run_writes_the_chain_file_into_a_bucket_that_has_none(tmp_path, bucket):
    world = ChainWorld(tmp_path / "world", sim_params(), order=SERIAL)
    world.start(CHAIN, every=1)
    world.writer.store, puts, put = bucket, [], bucket.put
    bucket.put = lambda key, data: (puts.append(key), put(key, data))[1]
    assert world.writer._doc is None and bucket.list("") == []
    with pytest.raises(KeyError):
        bucket.get(layout.HEAD_CHAIN)
    world.controller.publish(world.clock())
    assert world.controller.last_error == "" and puts == [layout.HEAD_CHAIN]
    assert in_the(bucket)["head_version_id"] == world.state.head.version_id == world.publisher.snapshots[-1]["head_chain"]["head_version_id"]
    world.controller.publish(world.clock())
    assert puts == [layout.HEAD_CHAIN], "the bucket has these bytes: no second write"


@pytest.mark.parametrize("order", [SERIAL, OVERLAP])
@pytest.mark.parametrize("call", FAILS)
def test_a_publish_of_run_handles_a_read_of_the_chain_file_that_fails_as_it_handles_a_write_that_fails(tmp_path, monkeypatch, capsys, bucket, order, call):
    world, _ = a_run(tmp_path / "world", order, kings=0)
    on(bucket, world)
    older = bucket.get(layout.HEAD_CHAIN)
    head = world.king().head.version_id
    world.wait()
    assert world.controller.close_in_public(world.clock()) == ""
    world.controller.publish(world.clock())
    mine, count = bucket.get(layout.HEAD_CHAIN), len(world.publisher.snapshots)
    assert json.loads(mine)["head_version_id"] == head and world.controller.last_error == ""
    bucket.put(layout.HEAD_CHAIN, older)  # another process wrote: the publish has a file to read and a file to write
    capsys.readouterr()
    with monkeypatch.context() as patch:
        calls = the_store_fails(patch, bucket, call)
        world.controller.publish(world.clock())  # it does not raise
        said = capsys.readouterr().err
        assert calls == [call] * controller.PUBLISH_TRIES and world.controller.last_error == controller.PUBLISH_PROBLEM + FAILS[call]
        assert f"PUBLISH FAILED: {controller.PUBLISH_TRIES} tries, and the store took none: {FAILS[call]}." in said
    assert bucket.get(layout.HEAD_CHAIN) == older and len(world.publisher.snapshots) == count, "no mining.json without its chain file"
    world.controller.publish(world.clock())
    assert bucket.get(layout.HEAD_CHAIN) == mine and len(world.publisher.snapshots) == count + 1 and world.controller.last_error == ""


@pytest.mark.parametrize("call", FAILS)
def test_the_close_of_a_king_round_with_overlap_handles_a_read_of_the_chain_file_that_fails_as_it_handles_a_write_that_fails(tmp_path, monkeypatch, bucket, call):
    world, _ = a_run(tmp_path / "world", OVERLAP, kings=0)
    on(bucket, world)
    older = bucket.get(layout.HEAD_CHAIN)
    with monkeypatch.context() as patch:
        calls = the_store_fails(patch, bucket, call)
        head = world.king().head.version_id  # the round is applied: the chain write of its close does not stop it
        assert calls and world.controller.last_error == controller.PUBLISH_PROBLEM + FAILS[call]
    assert bucket.get(layout.HEAD_CHAIN) == older
    world.wait()
    assert world.controller.close_in_public(world.clock()) == ""
    world.controller.publish(world.clock())
    assert in_the(bucket)["head_version_id"] == head and world.controller.last_error == ""


@pytest.mark.parametrize("call", FAILS)
def test_the_checkpoint_thread_handles_a_read_of_the_chain_file_that_fails_in_mark_as_it_handles_a_write_that_fails(tmp_path, monkeypatch, capsys, bucket, call):
    world, uploads, _, first, second = two_kings(tmp_path / "world", SERIAL)
    on(bucket, world)
    before = bucket.get(layout.HEAD_CHAIN)
    assert [step["checkpoint"] for step in json.loads(before)["steps"]] == [False, False]
    capsys.readouterr()
    with monkeypatch.context() as patch:
        calls = the_store_fails(patch, bucket, call)
        uploads.end()  # the two uploads run. ``mark`` of the first head writes nothing; ``mark`` of the last head reads the chain file, then writes it
        said = capsys.readouterr().err
        assert calls == [call] and uploads.marked == [first], "mark of the last head did not return"
        assert f"checkpoint of {second[:12]}: stopped by InfraError: {FAILS[call]}; it starts again at the next pass of run" in said
    assert checkpoints(world.store) == {first, second} and bucket.get(layout.HEAD_CHAIN) == before, "the marker is in the store; the chain file waits"
    assert world.round() is None  # the next pass of ``run``: the thread starts again, and ``mark`` of the last head writes
    world.wait()
    assert uploads.marked == [first, second] and [step["checkpoint"] for step in in_the(bucket)["steps"]] == [False, True] and in_the(bucket)["start"]["version_id"] == second
    for key in world.store.list(MARKERS):
        bucket.put(key, world.store.get(key))  # the markers, which the thread wrote into the store of the world
    world.controller.publish(world.clock())
    assert [step["checkpoint"] for step in in_the(bucket)["steps"]] == [True, True] and world.controller.last_error == ""
