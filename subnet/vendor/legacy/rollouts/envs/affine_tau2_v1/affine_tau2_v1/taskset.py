"""affine-tau2-v1: τ²-bench telecom with an LLM user simulator - the datagen pool.

Why: D had no interactive-user states. Every harness in D talks to a task,
never to a person, so "ask the user for the missing detail" is never the
teacher's recorded move, and the king learned to think "I should ask" while
its action slot emits a tool call with an invented argument (Alan's τ² note,
2026-09-17: 1,140 / 1,140 telecom tool-call nodes). This source puts the
teacher (and the king) in τ²'s customer-service loop - a simulated customer
answers, the domain tools mutate a database, Sierra's official evaluation
grades the final state and the required communication.

Wrapper over prime-envs' `tau2_bench_v1` (taskset + `Tau2Harness`, which runs
τ²'s own orchestrator as a program; the agent's LLM calls go through the
verifiers interception endpoint and are recorded, the user simulator is a
separate LLM - see harness.py for the Engy DeepSeek user).

Pool: telecom `full` split MINUS the `base` split. `base` (114 tasks) is what
the benchsuite runs (`tau2-telecom`, and the airline / retail `base` splits,
whose every task is in `base` - so those domains are NOT offered here). τ²'s
`train` split is a subset of `base` (74 of 114) and therefore also
benchmarked; the only disjoint τ² material is telecom `full \\ base` =
2,171 generated tasks (checked against `split_tasks.json` at revision
337326e: full 2,285 ∩ base 114 = 114 → 2,171 remain).

Changes for the duel corpus: `tasks` selector by `name = "tau2-" + τ² task id`
(τ² ids start with `[...]`, which the eval CLI would parse as a JSON list;
rollouts/catalog.py `tau2` catalog lists the same names through the
verifiers env);
`solved` = τ²'s reward (fold key; the base name `tau2_reward` is kept as the
undecorated method). Dialect: the agent's tool calls are native
(`tool_call`); its messages to the customer are prose replies followed by a
`user` message - the fold's `text` admission of those turns is the open
item documented in docs/env-wave-4.md.
"""

from __future__ import annotations

import verifiers.v1 as vf
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.run import load_tasks
from tau2_bench_v1.taskset import Tau2Data, Tau2Task as BaseTau2Task, Tau2Taskset as BaseTau2Taskset, Tau2TasksetConfig

DOMAIN = "telecom"
POOL_SPLIT = "full"
BENCHMARK_SPLIT = "base"
NAME_PREFIX = "tau2-"


def task_name(task_id: str) -> str:
    return NAME_PREFIX + task_id


class AffineTau2Task(BaseTau2Task):
    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace) -> float:
        return await BaseTau2Task.tau2_reward(self, trace)

    async def tau2_reward(self, trace: vf.Trace) -> float:
        # Undecorated override: the grade is counted once, under `solved`.
        return await self.solved(trace)


class AffineTau2Config(Tau2TasksetConfig):
    tasks: list[str] = []
    """τ² task ids to load (empty = the whole disjoint pool)."""
    domain: str = DOMAIN
    pool_split: str = POOL_SPLIT
    exclude_split: str = BENCHMARK_SPLIT
    """Tasks of this split are never loaded (the benchmarked set)."""


class AffineTau2Taskset(BaseTau2Taskset, vf.Taskset[AffineTau2Task, AffineTau2Config]):
    def load(self) -> list[AffineTau2Task]:
        cfg = self.config
        if cfg.domain != DOMAIN:
            raise ValueError(
                f"domain {cfg.domain!r}: only telecom has tasks outside the benchmarked "
                f"`{BENCHMARK_SPLIT}` split (airline / retail base = every task)")
        # The base load() bootstraps the τ² data dir and loads `base`; call
        # it for the bootstrap side effect, then load our own pool.
        base_ids = {t.data.name for t in super().load()}
        assert base_ids, "τ² base split loaded empty"
        excluded = {t.id for t in load_tasks(task_set_name=DOMAIN, task_split_name=cfg.exclude_split)}
        want = set(cfg.tasks)
        out: list[AffineTau2Task] = []
        for index, task in enumerate(load_tasks(task_set_name=DOMAIN, task_split_name=cfg.pool_split)):
            name = task_name(task.id)
            if task.id in excluded or (want and name not in want):
                continue
            out.append(AffineTau2Task(
                Tau2Data(
                    **task.model_dump(exclude={"description"}),
                    idx=index,
                    name=name,
                    description=str(task.description) if task.description else None,
                    prompt=DEFAULT_FIRST_AGENT_MESSAGE.content or "",
                    domain=DOMAIN,
                    tau_description=task.description,
                ),
                cfg.task,
            ))
        if want and not out:
            raise ValueError(f"no τ² task matched {sorted(want)[:3]}...")
        return out

