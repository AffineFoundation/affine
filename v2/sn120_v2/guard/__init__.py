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

from typing import TYPE_CHECKING

from .hint import Hint, epoch_counts, hint_for, wrong_answers
from .judge import Budget, BudgetExhausted, Verdict, ask, guard_version
from .store import Decision, append_ledger, append_private, public_row, record, utc_date
from .version import CHECK_VERSION

if TYPE_CHECKING:
    from .checks import CheckResult, NearIndex, Universe

# ``checks.py`` needs the math grader (math-verify). It is imported on first use,
# so the judge relay (``judge-requests``) runs on a host with only this repo and
# the standard library.
_CHECKS = ("CheckResult", "NearIndex", "Universe", "check_task", "content_hash", "normalise")


def __getattr__(name: str):
    if name in _CHECKS:
        from . import checks
        return getattr(checks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CHECK_VERSION", "Budget", "BudgetExhausted", "CheckResult", "Decision", "Hint", "NearIndex", "Universe",
    "Verdict", "append_ledger", "append_private", "ask", "check_task", "content_hash", "decide", "epoch_counts",
    "finish", "guard_version", "hint_for", "normalise", "precheck", "public_row", "record", "utc_date",
    "wrong_answers",
]


def precheck(task, universe: "Universe", *, near: "NearIndex | None" = None,
             hint: Hint | None = None) -> tuple[Decision | None, "CheckResult", str]:
    """The deterministic half of a decision: a final hard-reject Decision, or None when the judge must decide.

    Also returns the check result and the hint text the judge is to see.
    """
    from .checks import check_task
    result = check_task(task, universe, near)
    if result.category is not None:
        return (Decision(task_id=task.task_id, content_sha=result.content_sha, passed=False,
                         category=result.category, guard_version=guard_version(), checks=result),
                result, "")
    return None, result, (hint.prompt_text() if hint is not None else "")


def finish(task_id: str, content_sha: str, verdict: Verdict, *, version: str,
           checks: "CheckResult | None" = None, hint: str = "", added_at: str = "") -> Decision:
    """The judged half of a decision. The public row depends only on id, body, verdict and version."""
    return Decision(task_id=task_id, content_sha=content_sha, passed=verdict.passed,
                    category="ok" if verdict.passed else verdict.category, guard_version=version,
                    checks=checks, verdict=verdict, hint=hint, added_at=added_at)


def decide(task, universe: "Universe", *, near: "NearIndex | None" = None, hint: Hint | None = None,
           budget: Budget | None = None, ask_judge=ask) -> Decision:
    """One full decision for one candidate task: checks, then the judge if they passed.

    A hard reject never reaches the judge: it costs nothing to decide and the
    category is already exact.
    """
    version = guard_version()
    final, result, hint_text = precheck(task, universe, near=near, hint=hint)
    if final is not None:
        return final
    verdict = ask_judge(task.problem, task.answer, flags=result.flags, hint=hint_text, budget=budget)
    return finish(task.task_id, result.content_sha, verdict, version=version, checks=result, hint=hint_text)
