"""Deterministic pre-checks of a candidate task, before any judge sees it.

Hard rejects end a candidate's admission for the current body. Soft flags are
only context for the judge: they never reject on their own. Every grade here
goes through the production grader in ``envs.math_grader``; the grader itself
is not changed. ``grade_answer`` must run in the main thread (math-verify uses
an alarm signal), so these checks are never called from a worker thread.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping

from ..canonical import canonical_json, sha256_hex
from ..envs.math_grader import _reference_parses, grade_answer
from ..envs.math_pool import Task
from ..types import PASS

from .version import CHECK_VERSION  # noqa: F401  (re-exported; part of guard_version)

MAX_PROBLEM_CHARS = 6000
MAX_KEY_CHARS = 200
KEY_LONG_CHARS = 60
NEAR_DUPLICATE_J = 0.7
SHINGLE = 5
MINHASH_PERMS = 32
MINHASH_BANDS = 16  # rows per band = MINHASH_PERMS // MINHASH_BANDS

HARD_REJECTS = ("empty", "oversized", "mc_letter_key", "key_unparseable", "self_grade_fail", "duplicate", "partial_key")
SOFT_FLAGS = ("key_prose", "key_multivalue", "key_long", "multipart", "figure_ref", "possible_truncation",
              "unbalanced_math", "near_duplicate")

_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_NOT_KEPT = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")
_MC_LETTER = re.compile(r"\s*(?:\\(?:textbf|text|mathrm)\s*\{)?\s*\(?\s*([A-E])\s*\)?\s*\}?\s*")
_TEXT_WRAPPER = re.compile(r"\\text\w*\s*\{([^{}]*)\}")
_WORD = re.compile(r"[A-Za-z\u4e00-\u9fff]+")
_MULTI_VALUE = re.compile(r"\bor\b|;|\\pm|\\mp")
_MULTIPART = re.compile(r"\(1\)[\s\S]*\(2\)|\bI\)[\s\S]*\bII\)|\(i\)[\s\S]*\(ii\)")
_FIGURE = re.compile(r"as shown|figure|diagram|\[asy\]|如图", re.IGNORECASE)
_ENDS_CLEAN = re.compile(r"[.?!。？！$)\]}\"'`*]\s*$")
_CONTROL_OPEN = "([{"
_CONTROL_CLOSE = ")]}"


def normalise(text: str) -> str:
    """NFKC, lowercase, LaTeX command names dropped, only ``[0-9a-z]`` and CJK kept, spaces collapsed."""
    folded = unicodedata.normalize("NFKC", text).lower()
    folded = _LATEX_COMMAND.sub(" ", folded)
    folded = _NOT_KEPT.sub(" ", folded)
    return " ".join(folded.split())


def _top_level_split(text: str) -> list[str]:
    """Split at commas, semicolons, ``\\qquad``, ``\\quad`` and `` and `` outside every bracket."""
    pieces: list[str] = []
    depth = 0
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in _CONTROL_OPEN:
            depth += 1
        elif char in _CONTROL_CLOSE:
            depth = max(0, depth - 1)
        elif depth == 0:
            width = 0
            if char in ",;":
                width = 1
            elif text.startswith("\\qquad", index):
                width = len("\\qquad")
            elif text.startswith("\\quad", index):
                width = len("\\quad")
            elif text[index:index + 5].lower() == " and ":
                width = 5
            if width:
                pieces.append(text[start:index])
                index += width
                start = index
                continue
        index += 1
    pieces.append(text[start:])
    return pieces


def _top_level_equals(text: str) -> int | None:
    """Index of the last ``=`` outside brackets, or None."""
    depth = 0
    found = None
    for index, char in enumerate(text):
        if char in _CONTROL_OPEN:
            depth += 1
        elif char in _CONTROL_CLOSE:
            depth = max(0, depth - 1)
        elif char == "=" and depth == 0:
            found = index
    return found


def _strip_math(text: str) -> str:
    return text.strip().strip("$").strip()


def key_components(key: str) -> list[str]:
    """Top-level parts of a key, and the right-hand side of each ``name=value``, excluding the whole key."""
    whole = _strip_math(key)
    out: list[str] = []
    seen: set[str] = set()
    for piece in _top_level_split(whole):
        piece = _strip_math(piece)
        if not piece:
            continue
        candidates = [piece]
        equals = _top_level_equals(piece)
        if equals is not None:
            candidates.append(_strip_math(piece[equals + 1:]))
        for part in candidates:
            if part and part != whole and part not in seen:
                seen.add(part)
                out.append(part)
    return out


def _boxed(answer: str) -> str:
    return "<think>x</think> \\boxed{" + answer + "}"


def _graded_pass(gold: str, problem: str, answer: str) -> bool:
    return grade_answer(gold, problem, _boxed(answer), False).code == PASS


def content_hash(problem: str, answer: str) -> str:
    """The same body hash the reserve uses for ``content_sha``."""
    return sha256_hex(canonical_json({"problem": problem, "answer": answer}))


class Universe:
    """Every task the duplicate check must see, with one owner per normalised problem.

    The caller gives the entries in priority order and the first entry with a
    given normalised problem owns it. A candidate duplicates another task when
    that other task owns its normalised problem, so the earlier entry wins.
    See ``sn120_v2.guard.__main__._universe`` for the order the guard uses.
    """

    def __init__(self, ranked: Iterable[tuple[str, str, str]]) -> None:
        # ranked: (task_id, problem, group) in priority order
        self.owner: dict[str, str] = {}
        self.problem: dict[str, str] = {}
        for task_id, problem, _group in ranked:
            self.problem[task_id] = problem
            self.owner.setdefault(normalise(problem), task_id)

    def duplicate_of(self, task_id: str, problem: str) -> str | None:
        owner = self.owner.get(normalise(problem))
        return owner if owner not in (None, task_id) else None


class NearIndex:
    """MinHash over 5-word shingles, with banding for candidate retrieval (estimate only; J >= 0.7 flags)."""

    def __init__(self, texts: Mapping[str, str]) -> None:
        self._coeffs = [self._coeff(index) for index in range(MINHASH_PERMS)]
        self._bands: dict[tuple[int, tuple[int, ...]], list[str]] = {}
        self._signature: dict[str, tuple[int, ...]] = {}
        for task_id in sorted(texts):
            signature = self.signature(texts[task_id])
            if signature is None:
                continue
            self._signature[task_id] = signature
            for key in self._band_keys(signature):
                self._bands.setdefault(key, []).append(task_id)

    @staticmethod
    def _coeff(index: int) -> tuple[int, int]:
        digest = hashlib.blake2b(f"minhash:{index}".encode(), digest_size=16).digest()
        return int.from_bytes(digest[:8], "big") | 1, int.from_bytes(digest[8:], "big")

    def shingles(self, text: str) -> set[str]:
        words = normalise(text).split()
        if len(words) < SHINGLE:
            return {" ".join(words)} if words else set()
        return {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}

    def signature(self, text: str) -> tuple[int, ...] | None:
        shingles = self.shingles(text)
        if not shingles:
            return None
        hashes = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in shingles]
        mask = (1 << 64) - 1
        return tuple(min(((a * h + b) & mask) for h in hashes) for a, b in self._coeffs)

    @staticmethod
    def _band_keys(signature: tuple[int, ...]) -> list[tuple[int, tuple[int, ...]]]:
        rows = MINHASH_PERMS // MINHASH_BANDS
        return [(band, signature[band * rows:(band + 1) * rows]) for band in range(MINHASH_BANDS)]

    def nearest(self, task_id: str, text: str) -> str | None:
        """The smallest other task id whose signature agreement estimates J >= 0.7."""
        signature = self.signature(text)
        if signature is None:
            return None
        best: str | None = None
        for key in self._band_keys(signature):
            for other in self._bands.get(key, ()):
                if other == task_id:
                    continue
                agree = sum(1 for x, y in zip(signature, self._signature[other]) if x == y) / MINHASH_PERMS
                if agree >= NEAR_DUPLICATE_J and (best is None or other < best):
                    best = other
        return best


@dataclass(frozen=True)
class CheckResult:
    task_id: str
    content_sha: str
    rejects: tuple[str, ...]  # hard rejects found, in check order; the first decides the category
    flags: tuple[str, ...]    # soft flags for the judge

    @property
    def category(self) -> str | None:
        return self.rejects[0] if self.rejects else None


def _key_prose(answer: str) -> bool:
    for wrapper in _TEXT_WRAPPER.findall(answer):
        if len(wrapper.strip()) >= 12:
            return True
    stripped = _LATEX_COMMAND.sub(" ", answer)
    return len(_WORD.findall(stripped)) >= 3


def _truncated(problem: str) -> bool:
    return not _ENDS_CLEAN.search(problem.rstrip() or " ")


def _unbalanced(text: str) -> bool:
    dollars = re.sub(r"\\\$", "", text).count("$")
    return dollars % 2 == 1 or text.count("\\begin") != text.count("\\end")


def soft_flags(problem: str, answer: str, near: str | None) -> tuple[str, ...]:
    flags = []
    if _key_prose(answer):
        flags.append("key_prose")
    if _MULTI_VALUE.search(answer) or len(_top_level_split(answer)) > 1:
        flags.append("key_multivalue")
    if len(answer) > KEY_LONG_CHARS:
        flags.append("key_long")
    if _MULTIPART.search(problem):
        flags.append("multipart")
    if _FIGURE.search(problem):
        flags.append("figure_ref")
    if _truncated(problem):
        flags.append("possible_truncation")
    if _unbalanced(problem) or _unbalanced(answer):
        flags.append("unbalanced_math")
    if near is not None:
        flags.append("near_duplicate")
    return tuple(flags)


def check_task(task: Task, universe: Universe, near: NearIndex | None = None) -> CheckResult:
    """Hard rejects first (stopping at the first), then soft flags. Main thread only."""
    sha = content_hash(task.problem, task.answer)
    done = lambda rejects, flags=(): CheckResult(task.task_id, sha, tuple(rejects), tuple(flags))  # noqa: E731
    problem, answer = task.problem, task.answer
    if not problem.strip() or not answer.strip():
        return done(["empty"])
    if len(problem) > MAX_PROBLEM_CHARS or len(answer) > MAX_KEY_CHARS:
        return done(["oversized"])
    if _MC_LETTER.fullmatch(answer):
        return done(["mc_letter_key"])
    if not _reference_parses(answer):
        return done(["key_unparseable"])
    if not _graded_pass(answer, problem, answer):
        return done(["self_grade_fail"])
    if universe.duplicate_of(task.task_id, problem) is not None:
        return done(["duplicate"])
    for component in key_components(answer):
        if _graded_pass(answer, problem, component):
            return done(["partial_key"])
    near_id = near.nearest(task.task_id, problem) if near is not None else None
    return done([], soft_flags(problem, answer, near_id))
