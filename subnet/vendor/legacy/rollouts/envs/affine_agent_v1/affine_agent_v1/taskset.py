"""affine-agent-v1: general-agent tool worlds with native tool calls (`tool_call` dialect).

Reuses general_agent_v1 (PrimeIntellect research-environments) unchanged: each
of its 4,417 tasks ships its own database (`db.json`) and tool set (`tools.py`),
served per rollout on the host; the reward replays the gold tool chain and
compares database hashes. Two things change for the duel corpus:

  * every task gets a *system* message that states the tool contract — the
    fold admits a `tool_call` turn only if the turn's first system message
    mentions tools (`affine/dialects.py` `system_marker`), and the upstream
    tasks carry a user prompt only;
  * grading stays as upstream (model-free, host-side), but resolution is
    telemetry only (Reason policy) — an unsolved task is still a valid
    rollout.

Task identity: `name = <task directory name>` (e.g. `coast_guard_t2`), the
same string rollouts/catalog.py writes as the catalog uid, so
`--taskset.tasks` selects exactly the batch.
"""

from __future__ import annotations

import verifiers.v1 as vf
from general_agent_v1.taskset import GeneralAgentSolverTaskset, GeneralAgentTask, GeneralAgentTaskConfig

from affine_agent_v1.toolset import AffineAgentToolset

SYSTEM = (
    "You operate a small organisation's records through the tools listed "
    "here. Read the task, then work through it with tool calls: call exactly "
    "one tool per turn, and before each call write one or two sentences "
    "saying what you are about to do and why. Check the current records "
    "before you change them. When the task is complete, reply with a short "
    "summary and no tool call."
)


class AffineAgentTask(GeneralAgentTask):
    """Upstream task (reward, metrics, validate) served by the hook-compatible toolset."""

    @classmethod
    def toolsets(cls, config: GeneralAgentTaskConfig) -> list[vf.Toolset]:
        return [AffineAgentToolset(config.tools)]


class AffineAgentTaskset(GeneralAgentSolverTaskset):
    """general_agent_v1's taskset with the system-message tool contract added."""

    def load(self) -> list[GeneralAgentTask]:
        tasks = super().load()
        return [
            AffineAgentTask(
                task.data.model_copy(update={"system_prompt": SYSTEM}),
                task.config,
            )
            for task in tasks
        ]


__all__ = ["AffineAgentTaskset", "SYSTEM"]
