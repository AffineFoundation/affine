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
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from . import layout
from .errors import InfraError
from .seams import ObjectStore

KEY_BYTES = 32
TMP = ".tmp"

Reader = Callable[[str], "bytes | None"]  # round id -> the bytes of the winner file of that round; None when the round has none


def _bare(text: str) -> bytes:
    return "".join(text.split()).encode("utf-8", "surrogatepass")


def rollout_key(prompt: str, output: str) -> bytes:
    """What two rollouts must share to be the same rollout: the sha256 of the prompt and the output.

    All white space is removed from both texts first, as for a final answer (``admission.final_answer``). So a
    copy with other line breaks or spaces is still a copy. One zero byte stands between the two texts.

    Nothing else of the file is in the key. Not the hotkey, the version or ``train``: a copier must change the
    hotkey, and may change the others. Not the reward: it follows from the two texts by the official grader, and
    a copier may move it inside the tolerance. Not the place of the group or of the rollout in the file.

    Each other change of the prompt or the output gives another key. So one changed character that is not white
    space makes a copied rollout pass for a new one: these rules do not look for texts that are almost equal.
    """
    return hashlib.sha256(_bare(prompt) + b"\x00" + _bare(output)).digest()


def rollout_keys(groups: Iterable[tuple[str, Sequence[tuple[str, float]]]]) -> list[bytes]:
    """The key of each rollout of the checked groups (``admission.Group``), in file order."""
    return [rollout_key(prompt, output) for prompt, rollouts in groups for output, _ in rollouts]


def file_keys(raw: bytes) -> list[bytes]:
    """The key of each rollout of a published winner file, in file order. All of its groups count, also the groups that the caps dropped.

    A JSON object with no ``groups`` holds no rollout, so it gives no key. The validator publishes no such file:
    a winner file passed the check. A simulated store does (``sim.SimAdmission``). Each other fault raises.
    """
    value = json.loads(raw)
    if isinstance(value, dict) and "groups" not in value:
        return []
    return [rollout_key(group["prompt"], rollout["output"]) for group in value["groups"] for rollout in group["rollouts"]]


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
    """

    def __init__(self, read: Reader, cache: Path | None = None) -> None:
        self.read, self.cache = read, cache
        self._rounds = 0  # rounds 1 to this number are in ``_known``
        self._known: dict[bytes, str] = {}

    def known(self, rounds: int, settled: int | None = None) -> Mapping[bytes, str]:
        """The keys of rounds 1 to ``rounds``. Give only closed rounds.

        ``settled``: a round above this number that has no winner file is asked again by the next process. It is
        for a reader that can be behind the record, as a public site is for its newest rounds. Not given: every
        round is settled.
        """
        settled = rounds if settled is None else settled
        while self._rounds < rounds:
            number = self._rounds + 1
            for key in self._keys(round_name(number), final=number <= settled):
                self._known.setdefault(key, round_name(number))
            self._rounds = number
        return self._known

    def _keys(self, round_id: str, final: bool) -> list[bytes]:
        path = None if self.cache is None else Path(self.cache) / f"{round_id}.keys"
        if path is not None:
            with contextlib.suppress(OSError):
                kept = path.read_bytes()
                if len(kept) % KEY_BYTES == 0:
                    return [kept[at : at + KEY_BYTES] for at in range(0, len(kept), KEY_BYTES)]
        raw = self.read(round_id)
        try:
            keys = [] if raw is None else file_keys(raw)
        except (ValueError, LookupError, TypeError, AttributeError) as exc:
            raise InfraError(f"the published winner file of round {round_id} is not a batch file: {type(exc).__name__}") from None
        if path is not None and (raw is not None or final):
            with contextlib.suppress(OSError):
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + TMP)
                partial.write_bytes(b"".join(keys))
                os.replace(partial, path)
        return keys
