"""The order of the steps after a king: ``SN120_AFTER_KING``.

A king is a miner's LoRA that won a round. After the decision, the next round cannot open before the head is
folded. With ``serial`` (the default) every other step waits in the same line, in the one thread of ``run``:
the winner's files to ``public/``, the second sha256 of the folded head, the deletes in ``pending/``, the
public files, the read of the chain and the check of the waiting files.

``overlap`` takes those steps out of the line. Only ``run`` uses it, and only in the mode ``chain``:

* beside the fold: the winner's files go to ``public/`` (``Controller._apply``), the waiting files are
  checked and the registrations are read from the chain (``Overlap.look_ahead``)
* after the open, while the box trains: the deletes in ``pending/`` and the public files (``Overlap.keep``)
* the second sha256 of the folded head runs beside the close and the open (``headfold.ReadBack``). No box
  result is stored or judged before it passed

The thread of ``run`` alone writes the journal: the same records in the same order as ``serial``. Everything
here is the memory of one process. A process that stops loses it, and the next one does each step as
``serial`` does, so nothing here is needed to finish a king.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Callable, Mapping, Sequence

from .errors import ConfigError, V2Error
from .store import timed
from .types import ChainSnapshot, Verdict

ORDER_ENV, SERIAL, OVERLAP = "SN120_AFTER_KING", "serial", "overlap"
AHEAD_MAX_AGE_S = 600  # a check or a chain read that was made beside a fold is used by an open that comes this soon; later it is made again
PUBLISH = "publish"  # the kind of a kept step that shows the whole state: a later one replaces it


def after_king(environ: Mapping[str, str] = os.environ) -> str:
    """``serial``: each step after a king waits for the one before it (the default). ``overlap``: see the module text."""
    value = environ.get(ORDER_ENV, "").strip() or SERIAL
    if value not in (SERIAL, OVERLAP):
        raise ConfigError(f"{ORDER_ENV} must be {SERIAL!r} or {OVERLAP!r}")
    return value


def say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


class Beside:
    """One call in its own thread. ``result`` waits for it: it gives what the call returned, or raises what it raised."""

    def __init__(self, call: Callable[[], Any], name: str) -> None:
        self._out: Any = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(call,), name=name, daemon=True)
        self._thread.start()

    def _run(self, call: Callable[[], Any]) -> None:
        try:
            self._out = call()
        except BaseException as exc:  # noqa: BLE001 - the thread that asks for the result gets the error
            self._error = exc

    def done(self) -> bool:
        return not self._thread.is_alive()

    def wait(self) -> None:
        self._thread.join()

    def result(self) -> Any:
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._out


Step = tuple[str, str, Callable[[int], None]]  # (kind, name for the log, the work; it gets the time at which it is done)


class Overlap:
    """What one ``run`` keeps in memory for the order ``overlap``.

    ``chain`` reads the registrations (``Stack.chain``). ``max_age_s``: see ``AHEAD_MAX_AGE_S``.
    """

    def __init__(self, chain: Callable[[], ChainSnapshot] | None = None, max_age_s: int = AHEAD_MAX_AGE_S) -> None:
        self.chain, self.max_age_s = chain, max_age_s
        self._kept: list[Step] = []
        self._worker: Beside | None = None
        self._files: tuple[int, tuple[str, ...], Beside] | None = None
        self._chain: tuple[int, Beside] | None = None
        self._one_reader = threading.Lock()

    # ---- the steps that the next open does not need

    def keep(self, step: str, work: Callable[[int], None], kind: str = "") -> None:
        """Keep a step for after the next open. With a ``kind``, it takes the place of a kept step of that kind.

        ``work`` gets the time that ``release`` or ``flush`` got: a public snapshot shows the state of that time, so it carries that time.
        """
        if kind:
            self._kept = [item for item in self._kept if item[0] != kind]
        self._kept.append((kind, step, work))

    def release(self, now: int) -> None:
        """The round is open and the box starts: do the kept steps now, in one thread, in the order they were kept."""
        steps, self._kept = self._kept, []
        if steps:
            self._worker = Beside(lambda: self._do(steps, now), "sn120-v2-after-the-open")

    def settle(self) -> list[str]:
        """Wait for the thread of ``release``. The thread of ``run`` calls it before it writes its next record. Returns the problems."""
        worker, self._worker = self._worker, None
        return worker.result() if worker is not None else []

    def flush(self, now: int) -> list[str]:
        """No round opened, or ``run`` ends: do the kept steps here, now. Returns the problems."""
        steps, self._kept = self._kept, []
        return self.settle() + self._do(steps, now)

    @staticmethod
    def _do(steps: Sequence[Step], now: int) -> list[str]:
        """A step that fails says so and stops nothing: it is a delete of staged files or a public snapshot, and a later one does it again."""
        problems = []
        for _, step, work in steps:
            try:
                with timed(f"after the open: {step}"):
                    work(now)
            except Exception as exc:  # noqa: BLE001 - a thread that dies must say why
                problems.append(f"{step}: {type(exc).__name__}: {exc}")
                say(f"after the open: {problems[-1]}")
        return problems

    # ---- the steps that the next open needs: made beside the fold

    def look_ahead(self, now: int, hotkeys: Sequence[str], check: Callable[[Sequence[str]], Mapping[str, Verdict]]) -> None:
        """A king is decided at ``now``: check the files of ``hotkeys`` and read the chain, each in its own thread.

        A look that is still in memory and not old is kept: a second try of the same king starts no second one.
        """
        if self._files is None or now - self._files[0] > self.max_age_s:
            asked = tuple(hotkeys)
            self._files = (now, asked, Beside(lambda: dict(check(asked)), "sn120-v2-file-check-ahead"))
        if self.chain is not None and (self._chain is None or now - self._chain[0] > self.max_age_s):
            self._chain = (now, Beside(lambda: self.on_chain(self.chain), "sn120-v2-chain-read-ahead"))

    def on_chain(self, read: Callable[[], Any]) -> Any:
        """One read of the chain at a time: the connection of the feed is made for one thread. ``run`` makes each of its reads through here."""
        with self._one_reader:
            return read()

    def wait(self) -> None:
        """Wait for each thread of a look ahead. ``run`` has no need of it: each use of a look waits for its own thread."""
        for held in (self._files, self._chain):
            if held is not None:
                held[-1].wait()

    def _take(self, held: tuple[Any, ...] | None, now: int, what: str) -> Any:
        """The result of one look, or None: there is none, it is too old, or it failed (the caller then does the step itself)."""
        if held is None:
            return None
        try:
            found = held[-1].result()
        except V2Error as exc:
            say(f"{what} beside the fold failed, so it is made again now: {type(exc).__name__}: {exc}")
            return None
        if now - held[0] > self.max_age_s:
            say(f"{what} beside the fold is {now - held[0]} s old, so it is made again now")
            return None
        return found

    def verdicts(self, now: int) -> tuple[Mapping[str, Verdict], tuple[str, ...], int] | None:
        """The check beside the fold, one time: the verdicts, the hotkeys that it asked for, and when it started."""
        held, self._files = self._files, None
        found = self._take(held, now, "the check of the waiting files")
        return None if found is None else (found, held[1], held[0])

    def snapshot(self, now: int) -> ChainSnapshot | None:
        """The chain read beside the fold, one time."""
        held, self._chain = self._chain, None
        return self._take(held, now, "the read of the chain")
