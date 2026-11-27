"""The pool quality guard: a candidate enters the active pool only with a recorded PASS.

The guard is deliberately outside the rotation rule. ``rotate()`` stays a pure
function of published inputs, and the guard's only effect on it is the sealed
public ledger ``guard.jsonl``, which the ``info-guard`` rule reads as one more
rotation input. No LLM call and no network access happens inside ``rotate()``.

A decision has two stages:

  1. deterministic checks (``checks.py``) that can reject on their own and that
     also produce soft flags for the judge;
  2. an LLM judge (``judge.py``), given the problem, the key, the soft flags and
     an optional king hint (``hint.py``).

Only a judge verdict of OK at high or medium confidence is a PASS. Everything
else, including a judge that never returned a verdict, is not admitted: the
gate is fail-closed, so a guard that cannot decide keeps the candidate out
rather than letting it in.
"""
from __future__ import annotations

from .checks import (CHECK_VERSION, CheckResult, NearIndex, Universe, check_task, content_hash, normalise)
from .hint import Hint, epoch_counts, hint_for, wrong_answers
from .judge import Budget, BudgetExhausted, Verdict, ask, guard_version
from .store import Decision, append_ledger, append_private, public_row, record, utc_date

__all__ = [
    "CHECK_VERSION", "Budget", "BudgetExhausted", "CheckResult", "Decision", "Hint", "NearIndex", "Universe",
    "Verdict", "append_ledger", "append_private", "ask", "check_task", "content_hash", "decide", "epoch_counts",
    "guard_version", "hint_for", "normalise", "public_row", "record", "utc_date", "wrong_answers",
]


def decide(task, universe: Universe, *, near: NearIndex | None = None, hint: Hint | None = None,
           budget: Budget | None = None, ask_judge=ask) -> Decision:
    """One full decision for one candidate task: checks, then the judge if they passed.

    A hard reject never reaches the judge: it costs nothing to decide and the
    category is already exact.
    """
    version = guard_version()
    result = check_task(task, universe, near)
    if result.category is not None:
        return Decision(task_id=task.task_id, content_sha=result.content_sha, passed=False,
                        category=result.category, guard_version=version, checks=result)
    hint_text = hint.prompt_text() if hint is not None else ""
    verdict = ask_judge(task.problem, task.answer, flags=result.flags, hint=hint_text, budget=budget)
    return Decision(task_id=task.task_id, content_sha=result.content_sha, passed=verdict.passed,
                    category="ok" if verdict.passed else verdict.category, guard_version=version,
                    checks=result, verdict=verdict, hint=hint_text)
