"""affine-eog-v1: EnterpriseOps-Gym, stateful enterprise tool use over live MCP services.

Wrapper over research-environments' `enterprise_ops_gym_v1` (649 public
oracle tasks across Calendar / CSM / Drive / Email / HR / ITSM / Teams and
88 two-service hybrids). Each rollout starts the row's one or two service
containers from digest-pinned Docker Hub images, seeds fresh databases,
serves the merged MCP catalog to the harness, and grades the FINAL database
state with the row's SQL verifiers. The base config already runs the
services in local Docker (`service_runtime: vf.DockerConfig`), so nothing
touches Prime compute.

Changes for the duel corpus:

  * `tasks: list[str]` = the base `task_ids` under the name every other
    source uses, so the scheduler's `--env.taskset.tasks` works unchanged
    (rollouts/catalog.py `_eog_meta` names rows by `task_id`);
  * the grade lands under `solved` (the fold reads `solved` / `correct` /
    `passed_fraction`; the base reward is `database_state`, kept as the
    undecorated method so it is counted once);
  * the row's own system prompt is kept; if it does not contain the word
    "tool" (the fold's `tool_call` marker), one sentence naming the tools is
    appended - measured on the probe, not assumed.
"""

from __future__ import annotations

import verifiers.v1 as vf
from enterprise_ops_gym_v1.taskset import (
    EnterpriseOpsTask as BaseEnterpriseOpsTask,
    EnterpriseOpsTaskConfig,
    EnterpriseOpsTaskset as BaseEnterpriseOpsTaskset,
    EnterpriseOpsTasksetConfig,
)

from affine_eog_v1.toolset import AffineEnterpriseOpsToolset

TOOL_MARKER = "tool"
TOOL_NOTE = (
    " You complete the task by calling the provided enterprise service tools; "
    "call one tool per turn and stop when the requested changes are in place."
)


class AffineEnterpriseOpsTask(BaseEnterpriseOpsTask):
    @classmethod
    def toolsets(cls, config: EnterpriseOpsTaskConfig) -> list[vf.Toolset]:
        return [AffineEnterpriseOpsToolset(config.tools)]

    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace) -> float:
        results = trace.state.verifier_results
        return float(bool(results) and all(result["passed"] for result in results))

    async def database_state(self, trace: vf.Trace) -> float:
        # Undecorated override: the base reward is counted once, under `solved`.
        return await self.solved(trace)


class AffineEnterpriseOpsConfig(EnterpriseOpsTasksetConfig):
    tasks: list[str] = []
    """Task ids to load (empty = every task of the selected domains)."""


class AffineEnterpriseOpsTaskset(BaseEnterpriseOpsTaskset,
                                 vf.Taskset[AffineEnterpriseOpsTask, AffineEnterpriseOpsConfig]):
    def load(self) -> list[AffineEnterpriseOpsTask]:
        if self.config.tasks and not self.config.task_ids:
            self.config.task_ids = list(self.config.tasks)
        out: list[AffineEnterpriseOpsTask] = []
        for task in super().load():
            data = task.data
            system = data.system_prompt or ""
            if TOOL_MARKER not in system.lower():
                data = data.model_copy(update={"system_prompt": system.rstrip() + TOOL_NOTE})
            out.append(AffineEnterpriseOpsTask(data, self.config.task))
        return out
