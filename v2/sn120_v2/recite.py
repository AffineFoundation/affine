"""Anti-recitation: how much of a batch's training targets reproduces active-pool task text.

Pure CPU and deterministic. The pool index holds hashed word 10-grams of every active task's
problem text with its reference answer appended. A batch row's target is the text the LoRA is
trained on (a GRPO rollout output or an SFT output). Its 10-grams that also occur in the row's
own prompt are never counted: restating one's own problem is fine.

Hashes are taken over tuples of interned word ids, so they do not depend on PYTHONHASHSEED.
A collision can only add a count; it is accepted.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

N = 10
_WORD = re.compile(r"\\[^\W\d_]+|\w+")
SHARED = -1  # an n-gram that occurs in more than one pool task


def words(text: str) -> list[str]:
    """Lowercase NFKC word tokens. LaTeX commands (``\\frac``) and digit runs are words; punctuation is dropped."""
    return _WORD.findall(unicodedata.normalize("NFKC", text).lower())


class PoolIndex:
    """Hashed word 10-grams of the active pool, each with the pool task it came from."""

    def __init__(self, tasks: Iterable[tuple[str, str, str]]) -> None:
        self.vocab: dict[str, int] = {}
        self.task_ids: list[str] = []
        self.grams: dict[int, int] = {}
        for task_id, problem, answer in tasks:
            index = len(self.task_ids)
            self.task_ids.append(str(task_id))
            ids = [self.vocab.setdefault(word, len(self.vocab)) for word in words(f"{problem}\n{answer}")]
            for key in _hashes(ids):
                seen = self.grams.get(key)
                self.grams[key] = index if seen is None or seen == index else SHARED

    def __len__(self) -> int:
        return len(self.task_ids)

    def ids(self, text: str) -> list[int]:
        """Word ids of ``text``; a word that no pool task holds is -1, so no 10-gram over it can match."""
        vocab = self.vocab
        return [vocab.get(word, -1) for word in words(text)]


def _hashes(ids: Sequence[int]) -> Iterator[int]:
    for start in range(len(ids) - N + 1):
        yield hash(tuple(ids[start:start + N]))


@dataclass(frozen=True)
class ReciteReport:
    recited: int
    total: int
    rows: int
    rows_with_foreign_tasks: int
    max_row_share: float

    @property
    def share(self) -> float:
        return self.recited / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"share": round(self.share, 6), "recited": self.recited, "total": self.total, "rows": self.rows,
                "rows_with_foreign_tasks": self.rows_with_foreign_tasks, "max_row_share": round(self.max_row_share, 6)}


def batch_rows(body: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(prompt, target)`` for every trained row of an affine batch file of any format (v1 to v4)."""
    from .admission import FORMATS  # local: admission imports the heavy grading path lazily, keep this module light

    if not isinstance(body, Mapping) or body.get("version") not in FORMATS:
        raise ValueError("not an affine batch file")
    if body.get("version") == "affine-batch-v4":
        stages = body.get("stages")
        if not isinstance(stages, list):
            raise ValueError("a v4 file has no stages")
        groups = [group for stage in stages if isinstance(stage, Mapping) for group in stage.get("groups") or ()]
        sft = [(trace.get("prompt"), trace.get("output")) for stage in stages if isinstance(stage, Mapping)
               for trace in stage.get("sft") or () if isinstance(trace, Mapping)]
    else:
        groups, sft = body.get("groups"), []
        if not isinstance(groups, list):
            raise ValueError("not an affine-batch-v1/v2/v3 groups file")
    rows = [(group.get("prompt"), rollout.get("output")) for group in groups if isinstance(group, Mapping)
            for rollout in group.get("rollouts") or () if isinstance(rollout, Mapping)]
    return [(prompt, output) for prompt, output in [*rows, *sft] if isinstance(prompt, str) and isinstance(output, str)]


def measure_rows(rows: Sequence[tuple[str, str]], index: PoolIndex) -> ReciteReport:
    """Share of target 10-grams found in the pool index and not in the row's own prompt."""
    grams, recited, total, foreign_rows, top = index.grams, 0, 0, 0, 0.0
    own_cache: dict[str, tuple[frozenset[int], frozenset[int]]] = {}
    for prompt, target in rows:
        own = own_cache.get(prompt)
        if own is None:
            hashes = frozenset(_hashes(index.ids(prompt)))
            own = own_cache[prompt] = (hashes, frozenset(grams[key] for key in hashes if key in grams))
        own_hashes, own_tasks = own
        ids = index.ids(target)
        count = max(0, len(ids) - N + 1)
        if not count:
            continue
        hit, tasks, unknown = 0, set(), -N  # the last -1 position, so windows over it skip the lookup
        for end in range(len(ids)):
            if ids[end] < 0:
                unknown = end
            start = end - N + 1
            if start < 0 or unknown >= start:
                continue
            key = hash(tuple(ids[start:end + 1]))
            task = grams.get(key)
            if task is None or key in own_hashes:
                continue
            hit += 1
            if task != SHARED and task not in own_tasks:
                tasks.add(task)
        recited, total = recited + hit, total + count
        top = max(top, hit / count)
        foreign_rows += len(tasks) >= 2
    return ReciteReport(recited, total, len(rows), foreign_rows, top)


def measure_batch(raw: bytes, index: PoolIndex) -> ReciteReport:
    return measure_rows(batch_rows(json.loads(raw)), index)


def fail_reason(report: ReciteReport, limit: float) -> str:
    """The gate's text when ``report`` is above ``limit`` (``limit`` 0 is off), else an empty string."""
    if limit <= 0 or report.share <= limit:
        return ""
    return (f"training targets reproduce pool task text/answers: share {report.share:.4f} > recite_max_share {limit:g} "
            f"({report.rows_with_foreign_tasks} rows cite 2+ other pool tasks)")
