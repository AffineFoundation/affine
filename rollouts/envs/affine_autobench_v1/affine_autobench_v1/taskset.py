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
  * the grade lands under `solved` = 1.0 iff every assertion passes
    (upstream's `partial_credit` in [0, 1] stays a metric - the fold's rule
    is score >= 1.0, and "failed at 0.8" would be a weak `king_fail` label).

The row's system prompt ("You are a workflow automation agent... using the
available tools") carries the fold's `tool_call` marker already.
"""

from __future__ import annotations

import verifiers.v1 as vf
from automationbench_v1.taskset import (
    AutomationBenchConfig,
    AutomationBenchTask as BaseAutomationBenchTask,
    AutomationBenchTaskConfig,
    AutomationBenchTaskset as BaseAutomationBenchTaskset,
)

from affine_autobench_v1.toolset import AffineAutomationBenchToolset


class AffineAutomationBenchTask(BaseAutomationBenchTask):
    @classmethod
    def toolsets(cls, config: AutomationBenchTaskConfig) -> list[vf.Toolset]:
        return [AffineAutomationBenchToolset(config.tools)]

    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace) -> float:
        return float(trace.info["partial_credit"] == 1.0)

    @vf.metric
    async def partial_credit(self, trace: vf.Trace) -> float:
        # Upstream's graded reward, kept as telemetry (counted once, as a metric).
        return trace.info["partial_credit"]

    async def task_completed_correctly(self, trace: vf.Trace) -> float:
        # Undecorated: same quantity as `solved`, not counted twice.
        return await self.solved(trace)


class AffineAutomationBenchConfig(AutomationBenchConfig):
    tasks: list[str] = []
    """Upstream task names to load (empty = every task of the selected domains)."""


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
