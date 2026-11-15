"""Reuse of rollouts: what makes two rollouts the same, and the rollouts of every published winner file.

The file of a round winner is public (``layout.winner_batch``). The admission refuses a file in which too large a
share of the rollouts are copies of rollouts in those files (rule ``public_reuse``), or of rollouts of a seat ahead
in the same round (rule ``seat_reuse``). This module gives both rules their inputs.

It uses the standard library and plain Python modules of this package only. So a miner computes the same keys
from the same public files as the validator.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import unicodedata
from pathlib import Path
from typing import Callable, Collection, Iterable, Mapping, Sequence

from . import layout
from .seams import ObjectStore

KEY_BYTES = 32
TMP = ".tmp"
# The number of the normal form of a text (``normal``). It is a part of the name of each cache file, so keys of
# another form are never read as keys of this one. A change of ``normal`` needs a new number.
KEY_FORM = 2
# Unicode general categories that ``normal`` removes: white space (Zs, Zl, Zp), control and format characters (Cc, Cf:
# a tab, a line break, a zero-width space, a soft hyphen, a byte order mark), and marks with no width of their own
# (Mn, Me: an accent after its letter, a variation selector).
REMOVED = ("Zs", "Zl", "Zp", "Cc", "Cf", "Mn", "Me")
TRAILING = "PS"  # first letters of the categories that ``normal`` removes from the end: punctuation and symbols
REASON_LIMIT = 300  # characters of the reason of a winner file that was left out
GONE = "no such object, and the round has a winner"  # the reason of a winner file that the record of the rounds names and ``read`` does not find

Reader = Callable[[str], "bytes | None"]  # round id -> the bytes of the winner file of that round; None when there is no such object


class _Removal(dict):
    """A table for ``str.translate`` that removes each character of a category in ``REMOVED``. It learns a character at its first use."""

    def __missing__(self, code: int) -> int | None:
        self[code] = None if unicodedata.category(chr(code)) in REMOVED else code
        return self[code]


_REMOVAL = _Removal()


def normal(text: str) -> str:
    """The form of a prompt or an output that the reuse rules compare. Two texts are equal for them when this form is equal.

    1. Unicode compatibility decomposition (NFKD), case folding (``str.casefold``), then NFKD again. Each two
       texts that NFKC makes equal are equal after this step. So are two texts that differ in case, a full-width
       and a plain letter, a ligature and its letters.
    2. Each character of a category in ``REMOVED`` is removed: all white space, the control and format characters,
       and the marks with no width. A letter loses its accent here. This is why step 1 decomposes: a zero-width
       character between a letter and its accent would stop a composition, and the two texts would differ.
    3. Punctuation and symbols are removed from the end (categories P and S).

    Step 2 removes all white space. To make each run of white space one space would match less: a copy with a
    space put between two characters would pass.

    The categories are those of the Unicode data of this Python (``unicodedata.unidata_version``). A character
    that a later Unicode version added or moved to another category can get another form on another Python.
    """
    text = unicodedata.normalize("NFKD", unicodedata.normalize("NFKD", text).casefold()).translate(_REMOVAL)
    end = len(text)
    while end and unicodedata.category(text[end - 1])[0] in TRAILING:
        end -= 1
    return text[:end]


def rollout_key(prompt: str, output: str) -> bytes:
    """What two rollouts must share to be the same rollout: the sha256 of the normal forms of the prompt and the output.

    ``normal`` says which changes of a text keep its form: white space, case, compatibility forms, accents,
    characters with no width, and punctuation at the end. So a copy with such changes is still a copy. One zero
    byte stands between the two texts.

    Nothing else of the file is in the key. Not the hotkey, the version or ``train``: a copier must change the
    hotkey, and may change the others. Not the reward: it follows from the two texts by the official grader, and
    a copier may move it inside the tolerance. Not the place of the group or of the rollout in the file.

    Each other change of the prompt or the output gives another key. So one added letter or digit, or one changed
    character that stays visible before the end, makes a copied rollout pass for a new one: these rules do not
    look for texts that are almost equal.
    """
    return hashlib.sha256(b"\x00".join(normal(text).encode("utf-8", "surrogatepass") for text in (prompt, output))).digest()


def rollout_keys(groups: Iterable[tuple[str, Sequence[tuple[str, float]]]]) -> list[bytes]:
    """The key of each rollout of the checked groups (``admission.Group``), in file order."""
    return [rollout_key(prompt, output) for prompt, rollouts in groups for output, _ in rollouts]


def file_keys(raw: bytes) -> list[bytes]:
    """The key of each rollout of a published winner file, in file order. All of its groups count, also the groups that the caps dropped.

    Bytes that are not a batch file raise ``ValueError``. The validator publishes no such winner file: a winner
    file passed the check. A simulated store does (``sim.SimAdmission``: a JSON object with no ``groups``).
    """
    try:
        return [rollout_key(group["prompt"], rollout["output"]) for group in json.loads(raw)["groups"] for rollout in group["rollouts"]]
    except (ValueError, LookupError, TypeError, AttributeError) as exc:
        raise ValueError(f"the bytes are not a batch file ({type(exc).__name__})") from None


def round_name(number: int) -> str:
    """The id of the round with this number: the controller numbers its rounds from 1."""
    return f"r{number:06d}"


def store_reader(store: ObjectStore) -> Reader:
    """Read the winner files from the object store: the durable record of the validator."""

    def read(round_id: str) -> bytes | None:
        try:
            return store.get(layout.winner_batch(round_id))
        except KeyError:
            return None

    return read


class WinnerRollouts:
    """The rollout keys of the published winner files of rounds 1 to N, each with the first round that published it.

    A closed round never changes: it has its winner file or it never gets one. So each round is read once for
    each process, by ``read``. ``cache`` is a directory that keeps what was read: one file for each round, which
    holds the 32-byte keys of its winner file one after the other, and no byte for a round with no winner file.
    The cache only saves the reading. Without it, or without one of its files, the round is read again. A cache
    file that cannot be read or written stops nothing.

    A winner file that cannot be read stops nothing too. ``read`` failed, or the bytes are not a batch file:
    ``known`` leaves that file out and goes on, and ``left_out`` names the round and the reason. Nothing of that
    round goes to the cache, so it is never kept as a round with no winner file. Each later call of ``known``
    asks for it again, and from the call that reads it, its rollouts count.

    A winner file that is gone is such a file. ``read`` says "no such object" for a round with no winner, and
    also for a winner file that is no longer there, or is not there yet. ``known`` tells the two apart by ``won``:
    the rounds that have a winner, from the record of the rounds. For a round of ``won``, "no such object" is a
    read that failed, and a cache file with no byte is not believed. A file that was read keeps its keys when it
    is gone later: what a round published never changes.
    """

    def __init__(self, read: Reader, cache: Path | None = None) -> None:
        self.read, self.cache = read, cache
        self._rounds = 0  # each round from 1 to this number is in ``_known`` or in ``_left_out``
        self._known: dict[bytes, str] = {}
        self._left_out: dict[int, str] = {}  # round number -> why its winner file is not in ``_known``

    @property
    def left_out(self) -> tuple[tuple[str, str], ...]:
        """(round id, reason) of each winner file that the last ``known`` call could not read, in round order."""
        return tuple((round_name(number), reason) for number, reason in sorted(self._left_out.items()))

    def known(self, rounds: int, settled: int | None = None, won: Collection[str] = ()) -> Mapping[bytes, str]:
        """The keys of rounds 1 to ``rounds``. Give only closed rounds. It never raises for a winner file: see ``left_out``.

        ``won``: the ids of the rounds that have a winner. Such a round with no file is left out, and asked again.
        A round that is not in ``won`` and has no file is a round with no winner file.

        ``settled``: a round above this number that has no winner file is asked again by the next process. It is
        for a reader that can be behind the record, as a public site is for its newest rounds. Not given: every
        round is settled.
        """
        settled = rounds if settled is None else settled
        for number in (*sorted(self._left_out), *range(self._rounds + 1, rounds + 1)):
            name = round_name(number)
            try:
                keys = self._keys(name, final=number <= settled, won=name in won)
            except Exception as exc:  # every failure of one file, of any kind: the check of the waiting files goes on without that file
                self._left_out[number] = " ".join(f"{type(exc).__name__}: {exc}".split())[:REASON_LIMIT]
                continue
            self._left_out.pop(number, None)
            for key in keys:
                self._known[key] = min(self._known.get(key, name), name)
        self._rounds = max(self._rounds, rounds)
        return self._known

    def _keys(self, round_id: str, final: bool, won: bool) -> list[bytes]:
        path = None if self.cache is None else Path(self.cache) / f"{round_id}.form{KEY_FORM}.keys"
        if path is not None:
            with contextlib.suppress(OSError):
                kept = path.read_bytes()
                if len(kept) % KEY_BYTES == 0 and (kept or not won):
                    return [kept[at : at + KEY_BYTES] for at in range(0, len(kept), KEY_BYTES)]
        raw = self.read(round_id)
        if raw is None and won:
            raise LookupError(GONE)
        keys = [] if raw is None else file_keys(raw)
        if path is not None and (raw is not None or final):
            with contextlib.suppress(OSError):
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + TMP)
                partial.write_bytes(b"".join(keys))
                os.replace(partial, path)
        return keys
