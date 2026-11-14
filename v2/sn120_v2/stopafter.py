"""A stop of ``run`` after the round: the request of the operator, made with a file.

The file is ``<SN120_WORK_DIR>/logs/run-loop.stop-after-round``, beside the other files of the run loop. The
operator makes it from a second shell (``touch``) and takes the request back with ``rm``. The time of the file is
the time of the request. With no file, ``run`` does what it does with no such code. Only a file is a request. A
directory with that name is none (``rm -f`` cannot remove it, so it would stop each later start): ``run`` goes
on and says at its start that the thing is there.

``run`` looks for the file at one point only: before it opens a round. Two places of the code are that point:
the start of each pass of its loop, and ``Controller.run_round`` after it finished a round that it found open.
With no round open there, it opens none. It makes sure that all that the close of a round writes is in the
store, waits for the second sha256 of the local head (``overlap``), and ends with exit code 0. Its last line
starts with ``OBEYED``. With a round open there (a start that found a stored box result, a king whose fold was
put off), it goes on, and stops before the pass after the close of that round.

Who removes the file. Under the run loop (``SN120_LOOP_MARK`` is in the environment) the loop does, when it sees
the exit code 0 and the line: so a ``run`` that dies between its line and its exit loses no request, and the
``run`` that the loop starts in its place finds the file and stops in the same way. With no loop over it,
``run`` removes the file itself, before its last line.

The request is not in the journal and not in the store: it is the matter of one box. No published file knows it.
"""

from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path
from typing import Mapping

STOP_FILE = "run-loop.stop-after-round"  # in <SN120_WORK_DIR>/logs; systemd/sn120-v2-run-loop.sh has the same name
WORK_ENV, LOOP_ENV = "SN120_WORK_DIR", "SN120_LOOP_MARK"
SAID = "stop after the round: "
OBEYED = SAID + "obeyed."  # the run loop looks for a line of ``run`` that starts with this text


def say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def stamp(at: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(at))


class StopRequest:
    """The request file of one ``run``. ``path`` is None when the process has no work directory: then no request can be made."""

    def __init__(self, path: Path | None, under_loop: bool = False) -> None:
        self.path, self.under_loop = path, under_loop

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> StopRequest:
        work = environ.get(WORK_ENV, "").strip()
        return cls(Path(work) / "logs" / STOP_FILE if work else None, LOOP_ENV in environ)

    def asked(self) -> float | None:
        """The time of the request, or None when there is none. A file that cannot be read, or a directory, is no request: ``run`` goes on, as with no file."""
        if self.path is None:
            return None
        try:
            found = self.path.stat()
        except OSError:
            return None
        return found.st_mtime if stat.S_ISREG(found.st_mode) else None

    def at_start(self) -> str:
        """One line for the start of ``run``. It shows that this code knows the request, and where the file is."""
        asked = self.asked()
        now = "no request is there" if asked is None else f"a request of {stamp(asked)} is there"
        if asked is None and os.path.lexists(self.path):
            now = "something with that name is there, but it is not a file, so it is NO request and run does not stop for it. Remove it by hand"
        return f"{SAID}run looks for the file {self.path} each time before it opens a round (make it: touch, take it back: rm). Now {now}."

    def obeyed(self, asked: float, text: str) -> None:
        """The last line of a ``run`` that stops for the request of ``asked``. With no loop over this ``run``, the request ends here."""
        if self.under_loop:
            say(f"{OBEYED} Asked at {stamp(asked)}. {text} The run loop removes the request file.")
            return
        try:
            self.path.unlink(missing_ok=True)
            removed = "run removed the request file: the next start runs on."
        except OSError as exc:
            removed = f"run could NOT remove the request file ({exc}). Remove it before the next start: rm {self.path}"
        say(f"{OBEYED} Asked at {stamp(asked)}. {text} No run loop is over this run, so {removed}")
