"""Bounded, crash-isolated process checks of admission files.

The controller owns all verdict caches and clocks. Workers only judge the bytes
sent to them. Results are returned in submission order; a worker that dies
invalidates only its current file as an unproven machine failure. Spawned
workers have their own process group, so their grader children can be reaped.
"""

from __future__ import annotations

import multiprocessing
import os
import queue
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterable, Iterator, Protocol, TypeVar

from .admission import CHECK_FAULT, TRANSIENT, Invalid, _judge, _verdict
from .canonical import sha256_hex
from .params import Params
from .types import Verdict

T = TypeVar("T")
_READY, _RESULT, _START_FAILED = 0, 1, 2


@dataclass(frozen=True)
class Outcome:
    """One compact check result; no prompt, rollout or other batch content."""

    file_id: str  # sha256 of the exact checked bytes, including an empty file
    verdict: Verdict
    fault: tuple[str, str, bool] | None = None  # transient rule, public reason, proven; exception pickling is not assumed
    crash: bool = False  # the process died: preserve an earlier proven-fault sequence


def checked(raw: bytes, hotkey: str, params: Params, tokens: Any, grader: Any) -> Outcome:
    """Check bytes with a worker's private dependencies and discard the content."""
    result = _judge(raw, hotkey, params, tokens, grader)
    return Outcome(
        sha256_hex(bytes(raw)),
        _verdict(hotkey, raw, result, 0),
        (result.rule, str(result), result.proven) if isinstance(result, Invalid) and result.rule in TRANSIENT else None,
    )


class Worker(Protocol):
    """A picklable recipe for a worker's private tokenizer and grader."""

    def start(self, params: Params) -> Callable[[bytes, str], Outcome]:
        """Build private dependencies once. The returned callable checks one file."""
        ...


class InputWorker:
    """The default recipe, for picklable test and miner tokenizer/grader inputs."""

    def __init__(self, tokens: Any, grader: Any) -> None:
        self.tokens, self.grader = tokens, grader

    def start(self, params: Params) -> Callable[[bytes, str], Outcome]:
        return lambda raw, hotkey: checked(raw, hotkey, params, self.tokens, self.grader)


def crashed(hotkey: str, file_id: str, detail: str) -> Outcome:
    """A machine-caused check failure: it must never consume a fault try."""
    fault = Invalid(CHECK_FAULT, f"the file-check process stopped on this machine ({detail})", proven=False)
    return Outcome(file_id, Verdict(hotkey, False, str(fault), batch_sha256=file_id, rule=fault.rule), (fault.rule, str(fault), fault.proven), crash=True)


def _entry(worker: Worker, params: Params, tasks: Any, results: Any, identity: int) -> None:
    """One persistent child. Its first action separates its process group."""
    try:
        os.setsid()
        results.put((identity, _READY, None))
        check = worker.start(params)
    except BaseException as exc:  # a failed child must not take down the pool
        results.put((identity, _START_FAILED, type(exc).__name__))
        return
    while True:
        item = tasks.get()
        if item is None:
            return
        index, raw, hotkey = item
        try:
            outcome = check(raw, hotkey)
            results.put((identity, _RESULT, (index, outcome)))
        except BaseException as exc:  # infra exceptions retain their type for the controller
            try:
                results.put((identity, _RESULT, (index, exc)))
            except BaseException:
                results.put((identity, _START_FAILED, type(exc).__name__))
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise


@dataclass
class _Child:
    identity: int
    process: multiprocessing.Process
    tasks: Any
    index: int | None = None
    ready: bool = False
    stopped: bool = False
    own_group: bool = False


@dataclass
class _Pending(Generic[T]):
    value: T
    started: float
    file_id: str = field(repr=False)
    outcome: Any = None
    done: bool = False
    seconds: float = 0.0


class CheckPool:
    """A bounded pool whose results are consumed in input order."""

    def __init__(self, worker: Worker, params: Params, workers: int, timer: Callable[[], float]) -> None:
        self.context = multiprocessing.get_context("spawn")
        self.worker, self.params, self.workers, self.timer = worker, params, workers, timer
        self.results = self.context.Queue()
        self.parent_pid = os.getpid()
        self.children: dict[int, _Child] = {}
        self.pending: dict[int, _Pending[Any]] = {}
        self.finished: dict[int, _Child] = {}
        self.identity = 0
        self.closed = False

    def _stop(self, child: _Child) -> None:
        if child.stopped:
            return
        child.stopped = True
        process = child.process
        if process.is_alive():
            if not child.own_group:
                try:
                    # A live child may still be in multiprocessing bootstrap
                    # before _entry executes. Its inherited group is ours:
                    # never signal a negative PID until the kernel confirms
                    # that the child leads a different process group.
                    child.own_group = os.getpgid(process.pid) == process.pid and os.getpgid(process.pid) != os.getpgrp()
                except ProcessLookupError:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL) if child.own_group else process.kill()
            except ProcessLookupError:
                pass
        process.join(timeout=10)
        process.close()
        child.tasks.close()
        child.tasks.join_thread()
        self.children.pop(child.identity, None)
        self.finished.pop(child.identity, None)

    def submit(self, index: int, pending: _Pending[tuple[str, bytes, Any]]) -> None:
        # A result may already be queued, or a child may have died since the
        # last wait. Neither is allowed to make a new file share its identity.
        while self._receive():
            pass
        self._reap()
        child = next((one for one in self.children.values() if one.index is None and not one.stopped), None)
        if child is None:
            if self.finished:
                _, stopped = self.finished.popitem()
                self._stop(stopped)
            self.identity += 1
            identity = self.identity
            tasks = self.context.Queue(maxsize=1)
            process = self.context.Process(
                target=_entry, args=(self.worker, self.params, tasks, self.results, identity),
                name=f"file-check-{identity}",
            )
            process.start()
            child = _Child(identity, process, tasks)
            self.children[identity] = child
            # A spawn child becomes a session leader atomically in preexec.
            # Waiting for its first message also avoids submitting to a
            # process which failed before it could receive any task.
            while not child.ready:
                if not self._receive(timeout=0.05):
                    self._reap()
                if child.identity not in self.children:
                    # Startup is machine-caused; wait() returns this file's
                    # crash result, and the next file gets a fresh identity.
                    if index in self.pending and not self.pending[index].done:
                        self._finish(pending, crashed(pending.value[0], pending.file_id, "worker died before startup"))
                    return
        hotkey, raw, _ = pending.value
        child.index = index
        child.tasks.put((index, raw, hotkey))

    def _receive(self, timeout: float = 0.0) -> bool:
        try:
            identity, kind, payload = self.results.get(timeout=timeout)
        except queue.Empty:
            return False
        child = self.children.get(identity)
        if child is None:
            return True
        if kind == _READY:
            child.ready = True
            return True
        if kind == _START_FAILED:
            index = child.index
            self.finished[identity] = self.children.pop(identity)
            self._stop(child)
            if index is not None and index in self.pending:
                pending = self.pending[index]
                self._finish(pending, crashed(pending.value[0], pending.file_id, f"worker startup failed: {payload}"))
            return True
        index, outcome = payload
        child.index = None
        pending = self.pending.get(index)
        if pending is not None:
            self._finish(pending, outcome)
        return True

    def _finish(self, pending: _Pending[Any], outcome: Any) -> None:
        pending.outcome, pending.done = outcome, True
        pending.seconds = max(0.0, self.timer() - pending.started)

    def _reap(self) -> None:
        for child in list(self.children.values()):
            if child.process.exitcode is None:
                continue
            index = child.index
            self.finished[child.identity] = self.children.pop(child.identity)
            self._stop(child)
            if index is not None and index in self.pending and not self.pending[index].done:
                pending = self.pending[index]
                self._finish(pending, crashed(pending.value[0], pending.file_id, "worker died"))

    def wait(self, index: int) -> tuple[Any, float]:
        pending = self.pending[index]
        while not pending.done:
            received = self._receive(timeout=0.05)
            self._reap()
            if received:
                # Drain an already queued final result before concluding that
                # a different child can no longer complete its own file.
                continue

        outcome, seconds = pending.outcome, pending.seconds
        del self.pending[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome, seconds

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for child in list(self.children.values()):
            self._stop(child)
        self.results.close()
        self.results.join_thread()


def judged(
    worker: Worker,
    params: Params,
    workers: int,
    jobs: Iterable[tuple[str, bytes | None, T]],
    timer: Callable[[], float] = time.monotonic,
    deferred: Callable[[T], tuple[str, bytes | None, T]] | None = None,
) -> Iterator[tuple[T, Outcome | None, float]]:
    """Check ``(hotkey, raw, metadata)`` jobs with bounded memory.

    ``raw=None`` means that no worker check is needed. At most ``workers``
    raw files are in flight; each was bounded by the caller to
    ``params.max_batch_bytes``. A later result may finish first, but only its
    compact verdict is held until every earlier file has been recorded.
    """
    pool = CheckPool(worker, params, workers, timer)
    todo = iter(jobs)
    ahead: deque[tuple[int, _Pending[tuple[str, bytes | None, T]]]] = deque()
    index = 0
    try:
        while True:
            # One earlier worker result is yielded before the parent prepares
            # any later file. Raw reads may run ahead in _ahead, but a later
            # store error or hold decision cannot jump this queue position.
            if ahead:
                index, pending = ahead.popleft()
                if getattr(pending.value[2], "needs_check", False) and deferred is not None:
                    pending.value = deferred(pending.value[2])
                    if pending.value[1] is not None:
                        pool.pending[index] = pending
                        pool.submit(index, pending)  # type: ignore[arg-type]
                if pending.value[1] is not None:
                    outcome, seconds = pool.wait(index)
                else:
                    outcome, seconds = None, 0.0
                yield pending.value[2], outcome, seconds
            else:
                try:
                    job = next(todo)
                except StopIteration:
                    return
                pending = _Pending(job, timer(), job[2].file_id)
                if job[1] is not None:
                    pool.pending[index] = pending
                    pool.submit(index, pending)  # type: ignore[arg-type]
                else:
                    pending.done = True
                ahead.append((index, pending))
                index += 1
                while len(ahead) < workers:
                    try:
                        job = next(todo)
                    except StopIteration:
                        break
                    pending = _Pending(job, timer(), job[2].file_id)
                    if job[1] is not None:
                        pool.pending[index] = pending
                        pool.submit(index, pending)  # type: ignore[arg-type]
                    else:
                        pending.done = True
                    ahead.append((index, pending))
                    index += 1
    finally:
        pool.close()
