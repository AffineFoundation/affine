from __future__ import annotations

import re
from typing import Any, Iterable, List, Mapping, Sequence

from .types import Verdict

_INT_RE = re.compile(r"[-+]?\d+")


def last_integer(text: str) -> int | None:
    """Return the last signed integer literal embedded in ``text``."""
    matches = _INT_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def ensure_last_integer(text: str, expected: int) -> Verdict:
    """Compare the final integer literal to ``expected``."""
    value = last_integer(text)
    if value is None:
        return Verdict(False, "no-integer-found")
    if value == expected:
        return Verdict(True, "")
    return Verdict(False, f"mismatch:{value}")


def verify_transcript(actions: Sequence[int], legal_moves: Sequence[int]) -> Verdict:
    """Check that the given actions are legal and non-repeating."""
    taken: set[int] = set()
    for move in actions:
        if move not in legal_moves:
            return Verdict(False, f"illegal-move:{move}")
        if move in taken:
            return Verdict(False, f"repeated-move:{move}")
        taken.add(move)
    return Verdict(True, "")


__all__ = ["ensure_last_integer", "last_integer", "verify_transcript"]
