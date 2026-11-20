"""The chain feed of the round runner, on fake connections: a failed or a silent read costs one pass, not the run."""

from __future__ import annotations

import threading
import time

import pytest

from sn120_v2.chain import FakeChain, Feed
from sn120_v2.errors import InfraError

WAIT_S = 5  # the longest a test waits for a background thread


class Line(FakeChain):
    """One fake connection. It can stop answering, and it records that it was closed."""

    def __init__(self) -> None:
        super().__init__({uid: f"hk{uid}" for uid in range(3)}, "hk0", now=int(time.time()))
        self.silent = False
        self.answer, self.closed = threading.Event(), threading.Event()

    def head(self) -> tuple[int, str, int]:
        if self.silent:
            self.answer.wait(WAIT_S)
        return super().head()

    def close(self) -> None:
        self.closed.set()
        self.answer.set()  # as the SDK does: a close ends the calls that still wait


class Lines:
    """The ``connect`` of a feed: every call opens a new fake connection."""

    def __init__(self) -> None:
        self.opened: list[Line] = []
        self.refuse = ""
        self.gate: threading.Event | None = None  # when set, a connection attempt waits for it

    def __call__(self) -> Line:
        if self.gate is not None:
            self.gate.wait(WAIT_S)
        if self.refuse:
            raise InfraError(f"no connection to the chain: {self.refuse}")
        self.opened.append(Line())
        return self.opened[-1]


def test_one_connection_serves_every_read_while_it_works():
    lines = Lines()
    feed = Feed(120, lines)
    assert not lines.opened, "nothing connects before the first read"
    with pytest.raises(InfraError, match="no chain snapshot yet"):
        feed.entropy()
    first = feed.snapshot()
    lines.opened[0].advance(3)
    second = feed.snapshot()
    assert (first.netuid, second.block) == (120, first.block + 3) and len(first.registrations) == 3
    assert feed.entropy() == lines.opened[0].block_hash(second.block), "the hash of the block of the last snapshot"
    assert len(lines.opened) == 1 and not lines.opened[0].closed.is_set()


def test_after_a_failed_read_the_next_read_opens_a_new_connection():
    lines = Lines()
    feed = Feed(120, lines)
    block = feed.snapshot().block
    lines.opened[0].down = "connection reset"
    with pytest.raises(InfraError, match="connection reset"):
        feed.snapshot()
    assert lines.opened[0].closed.wait(WAIT_S), "the failed connection is closed"
    assert len(lines.opened) == 1, "no second attempt in the same read: the next pass makes it"
    assert feed.snapshot().block == block and len(lines.opened) == 2
    lines.opened[1].down = "connection reset"
    with pytest.raises(InfraError, match="connection reset"):
        feed.entropy()
    assert lines.opened[1].closed.wait(WAIT_S)
    assert feed.entropy() == lines.opened[2].block_hash(block), "the block of the last good snapshot is kept"


def test_a_failed_connection_attempt_is_tried_again_at_the_next_read():
    lines = Lines()
    feed = Feed(120, lines)
    lines.refuse = "name not resolved"
    for _ in range(2):
        with pytest.raises(InfraError, match="name not resolved"):
            feed.snapshot()
    lines.refuse = ""
    assert feed.snapshot().block and len(lines.opened) == 1


def test_a_read_that_never_answers_is_a_failure_within_the_bound_and_not_a_hang():
    lines = Lines()
    feed = Feed(120, lines, timeout_s=0.05)
    feed.snapshot()
    dead = lines.opened[0]
    dead.silent = True
    start = time.monotonic()
    with pytest.raises(InfraError, match="did not end in 0.05 s"):
        feed.snapshot()
    assert time.monotonic() - start < WAIT_S / 2
    assert dead.closed.wait(WAIT_S), "the silent connection is closed, which ends the read that still waits"
    assert feed.snapshot().block and len(lines.opened) == 2 and not lines.opened[1].closed.is_set()


def test_a_connection_that_opens_after_the_bound_is_closed_and_never_used():
    lines = Lines()
    lines.gate = threading.Event()
    feed = Feed(120, lines, timeout_s=0.05)
    with pytest.raises(InfraError, match="did not end in 0.05 s"):
        feed.snapshot()
    lines.gate.set()
    deadline = time.monotonic() + WAIT_S
    while not lines.opened and time.monotonic() < deadline:
        time.sleep(0.01)
    late = lines.opened[0]
    assert late.closed.wait(WAIT_S), "the late connection is closed by the read that was given up"
    assert feed.snapshot().block and lines.opened[1] is not late and len(lines.opened) == 2
