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
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, NamedTuple

import pytest

torch = pytest.importorskip("torch")

from sn120_v2.sim_head import ChainWorld  # noqa: E402 - after the skip guard

from sn120_v2 import controller, layout  # noqa: E402
from sn120_v2.afterking import OVERLAP, SERIAL  # noqa: E402
from sn120_v2.errors import V2Error  # noqa: E402
from sn120_v2.gpu.resident import hash_tree  # noqa: E402
from sn120_v2.headchain import CHAIN, chain_links, checkpoints, heads_of  # noqa: E402
from sn120_v2.headfold import LinkSource, rebuild  # noqa: E402
from sn120_v2.sim import sim_params  # noqa: E402

WAIT_S = 30  # the longest a test waits for a thread
MARKERS = layout.head_checkpoint("x").rpartition("/")[0]  # what ``headchain.checkpoints`` lists
AFTER_THE_OPEN = "sn120-v2-after-the-open"
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
    world, uploads, genesis, first, second = two_kings(tmp_path / "world", order)
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
    world, uploads, genesis, first, second = two_kings(tmp_path / "world", order)
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
