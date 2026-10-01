"""affine-autobench-v1: Zapier AutomationBench - business workflows over a simulated SaaS workspace.

Wrapper over research-environments' `automationbench_v1` (the Environments
Hub's second most starred env, `zapier/AutomationBench`): 600 scored tasks in
six domains (sales, marketing, operations, support, finance, hr) over a
simulated workspace of ~500 API endpoints across 47 apps, driven through the
task-scoped `api_search` / `api_fetch` toolset served in-process (no
container, no external service, no key). Grading is upstream's rubric on
the final world state: deterministic assertions.

Changes for the duel corpus:

  * `tasks: list[str]` selector by upstream task name (`sales.multi_hop_
    lookup`), so the scheduler addresses rows by name; rollouts/catalog.py
    (`autobench` catalog) lists the same names through the verifiers env;
  * the grade is the Artificial Analysis rule (AutomationBench-AA, the
    Intelligence Index component): `aa_score` = fraction of OBJECTIVES the
    agent completed, 0 for the task when any GUARDRAIL (an assertion that
    passed in the initial world) is broken. `solved` = 1.0 iff `aa_score >=
    solved_threshold` (default 0.5). Until 2026-09-21 `solved` was upstream's
    all-assertions-pass bit (`partial_credit == 1.0`): the teacher cleared
    18 % of tasks under it, so most references on this env were noise and
    every partially completed workflow was a `king_fail`. `partial_credit`
    (upstream) and `aa_score` both stay as metrics. `solved_threshold = 1.0`
    restores the strict bit.

The row's system prompt ("You are a workflow automation agent... using the
available tools") carries the fold's `tool_call` marker already.
"""

from __future__ import annotations

import verifiers.v1 as vf
from automationbench.schema.world import WorldState
from automationbench_v1.taskset import (
    AutomationBenchConfig,
    AutomationBenchTask as BaseAutomationBenchTask,
    AutomationBenchTaskConfig,
    AutomationBenchTaskset as BaseAutomationBenchTaskset,
)
from automationbench_v1.common import AutomationBenchData

from affine_autobench_v1.state import AffineAutomationBenchState, aa_grade
from affine_autobench_v1.state import aa_score as compute_aa_score
from affine_autobench_v1.toolset import AffineAutomationBenchToolset


class AffineAutomationBenchTaskConfig(AutomationBenchTaskConfig):
    solved_threshold: float = 0.5
    """`solved` = 1.0 iff the AA score (objectives fraction, 0 on a broken
    guardrail) reaches this value. 1.0 = upstream's strict all-pass bit."""


class AffineAutomationBenchTask(BaseAutomationBenchTask,
                                vf.Task[AutomationBenchData, AffineAutomationBenchState,
                                        AffineAutomationBenchTaskConfig]):
    @classmethod
    def toolsets(cls, config: AffineAutomationBenchTaskConfig) -> list[vf.Toolset]:
        return [AffineAutomationBenchToolset(config.tools)]

    async def finalize(self, trace: vf.Trace) -> None:
        await super().finalize(trace)      # trace.info["partial_credit"]
        st = trace.state
        broken, passed, total = st.guardrail_broken, st.objectives_passed, st.objectives_total
        if total is None:
            # The agent never called api_fetch: grade the untouched initial
            # world (guardrails intact, objectives as the author left them).
            broken, passed, total = aa_grade(
                self.data.assertions, self.data.initial_state,
                WorldState(**self.data.initial_state))
        trace.info["guardrail_broken"] = bool(broken)
        trace.info["objectives_passed"] = int(passed or 0)
        trace.info["objectives_total"] = int(total or 0)
        trace.info["aa_score"] = compute_aa_score(broken, passed, total, trace.info["partial_credit"])

    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace) -> float:
        return float(trace.info["aa_score"] >= self.config.solved_threshold)

    @vf.metric
    async def aa_score(self, trace: vf.Trace) -> float:
        # Artificial Analysis' headline number for the task (AutomationBench-AA).
        return trace.info["aa_score"]

    @vf.metric
    async def guardrail_broken(self, trace: vf.Trace) -> float:
        return float(trace.info["guardrail_broken"])

    @vf.metric
    async def partial_credit(self, trace: vf.Trace) -> float:
        # Upstream's graded reward, kept as telemetry (counted once, as a metric).
        return trace.info["partial_credit"]

    async def task_completed_correctly(self, trace: vf.Trace) -> float:
        # Undecorated: upstream's strict bit, not counted twice.
        return float(trace.info["partial_credit"] == 1.0)


class AffineAutomationBenchConfig(AutomationBenchConfig):
    tasks: list[str] = []
    """Upstream task names to load (empty = every task of the selected domains)."""
    task: AffineAutomationBenchTaskConfig = AffineAutomationBenchTaskConfig()


class AffineAutomationBenchTaskset(BaseAutomationBenchTaskset,
                                   vf.Taskset[AffineAutomationBenchTask, AffineAutomationBenchConfig]):
    def load(self) -> list[AffineAutomationBenchTask]:
        want = set(self.config.tasks)
        out: list[AffineAutomationBenchTask] = []
        for task in super().load():
            if want and task.data.name not in want:
                continue
            out.append(AffineAutomationBenchTask(task.data, self.config.task))
        if want and not out:
            raise ValueError(f"no AutomationBench task matched {sorted(want)[:5]}...")
        return out
