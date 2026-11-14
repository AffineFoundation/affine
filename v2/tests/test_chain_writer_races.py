"""The chain file (``public/head_chain.json``) with more than one writer, on a real tiny head.

A checkpoint is a head whose full weight files are in the store. The thread ``head-checkpoint`` uploads them and
writes a marker. The chain file shows a checkpoint as a flag of its step, and ``start`` names the newest one.
A reader builds the head from ``start`` and the LoRAs after it.

Three things are proven here, each in the orders ``serial`` and ``overlap`` where the order matters.

* No write of the chain file takes back a checkpoint that an earlier write showed.
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

from sn120_v2 import layout  # noqa: E402
from sn120_v2.afterking import OVERLAP, SERIAL  # noqa: E402
from sn120_v2.headchain import CHAIN, checkpoints  # noqa: E402
from sn120_v2.sim import sim_params  # noqa: E402

WAIT_S = 30  # the longest a test waits for a thread
MARKERS = layout.head_checkpoint("x").rpartition("/")[0]  # what ``headchain.checkpoints`` lists
AFTER_THE_OPEN = "sn120-v2-after-the-open"


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
