"""What the current king already told us about a task, as optional judge context.

A hint is never a reject on its own, and a missing hint is not an error: the
judge simply sees no hint. Two sources, both already public:

  epoch results   ``s``/``n`` per source task id: how often the head passed it.
  gate rollouts   the extracted answers of the head's public gate rollouts,
                  which give the most common wrong answer.

The live epoch results carry rollouts as bare ``"p"``/``"f"`` codes with no
extracted answer, so the wrong-answer tally comes only from the gate rollouts.
When neither source has the task, ``hint_for`` returns None.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Hint:
    task_id: str
    passes: int | None = None          # head passes on this task, from epoch results
    attempts: int | None = None        # head attempts on this task
    common_wrong: str | None = None    # most common wrong extracted answer, from gate rollouts
    wrong_count: int = 0

    @property
    def pass_rate(self) -> float | None:
        if not self.attempts:
            return None
        return (self.passes or 0) / self.attempts

    def prompt_text(self) -> str:
        """One short block for the judge prompt. Empty when the hint says nothing."""
        lines = []
        rate = self.pass_rate
        if rate is not None:
            lines.append(f"The current head answered this task {self.passes}/{self.attempts} times correctly "
                         f"(pass rate {rate:.2f}).")
        if self.common_wrong:
            lines.append(f"Its most common wrong answer was {self.common_wrong!r} "
                         f"({self.wrong_count} of its wrong answers).")
        return "\n".join(lines)


def epoch_counts(results: Mapping[str, object]) -> dict[str, tuple[int, int]]:
    """``{source task id: (passes, attempts)}`` from one epoch results file."""
    out: dict[str, tuple[int, int]] = {}
    tasks = results.get("tasks")
    if not isinstance(tasks, list):
        return out
    for entry in tasks:
        if not isinstance(entry, dict):
            continue
        task_id = entry.get("task_id") or entry.get("id")
        if not isinstance(task_id, str):
            continue
        passes, attempts = entry.get("s"), entry.get("n")
        if not isinstance(passes, int) or not isinstance(attempts, int) or attempts <= 0:
            continue
        have = out.get(task_id, (0, 0))
        out[task_id] = (have[0] + passes, have[1] + attempts)
    return out


def wrong_answers(rollouts: Iterable[Mapping[str, object]]) -> dict[str, tuple[str, int]]:
    """``{task id: (most common wrong extracted answer, count)}`` from public gate rollout rows."""
    tally: dict[str, Counter] = {}
    for row in rollouts:
        task_id = row.get("task_id") or row.get("id")
        extracted = row.get("extracted") or row.get("answer")
        code = row.get("code") or row.get("grade")
        if not isinstance(task_id, str) or not isinstance(extracted, str) or not extracted.strip():
            continue
        if isinstance(code, str) and code.upper() == "P":
            continue
        tally.setdefault(task_id, Counter())[extracted.strip()] += 1
    out: dict[str, tuple[str, int]] = {}
    for task_id, counter in tally.items():
        answer, count = min(counter.most_common(), key=lambda item: (-item[1], item[0]))
        out[task_id] = (answer, count)
    return out


def hint_for(task_id: str,
             counts: Mapping[str, tuple[int, int]],
             wrong: Mapping[str, tuple[str, int]]) -> Hint | None:
    """The king hint for one task, or None when neither source knows it."""
    seen = counts.get(task_id)
    missed = wrong.get(task_id)
    if seen is None and missed is None:
        return None
    return Hint(task_id=task_id,
                passes=seen[0] if seen else None,
                attempts=seen[1] if seen else None,
                common_wrong=missed[0] if missed else None,
                wrong_count=missed[1] if missed else 0)
