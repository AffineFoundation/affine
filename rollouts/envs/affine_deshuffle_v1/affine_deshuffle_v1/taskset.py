"""affine-deshuffle-v1: reorder a shuffled research paper from a file in the sandbox.

Wrapper over prime-envs' `deshuffle_papers` (vendored under
rollouts/vendor/prime-envs at commit c4d04dfe - the pods' research-environments
checkout `b10db76` predates it): 1,000 CC-BY papers (500 arXiv, 500 bioRxiv),
one paper's paragraphs shuffled into `paragraphs.jsonl` in the workspace; the
agent answers with the original order as a JSON permutation in `\\boxed{}`;
upstream's reward is `grouping_f1 * norm_kendall_tau` (graded 0-1).
Long-document reasoning with perfect grading; `network_allow=[]` so the
paper cannot be looked up (honoured by the docker runtime). Changes for the
duel corpus:

  * index-named tasks `deshuffle-<index:05d>` (the base names them
    `"1-papers:<idx>.<occurrence>"`) and a `tasks` selector, generated from
    the base's own deterministic `(idx, seed)` rule (rollouts/catalog.py
    `procedural`, `_deshuffle_meta`);
  * a real *system* message (the base ships none). The permutation is a
    `boxed` answer, but the rollout is a shell-harness trajectory - the shell
    harness supplies the dialect word and the final reply carries the box;
  * `solved` = 1.0 iff upstream's reward is 1.0 (exact reconstruction);
    upstream's graded `ordering_reward` stays a metric.
"""

from __future__ import annotations

import verifiers.v1 as vf
from deshuffle_papers.taskset import (
    DeshuffleConfig,
    DeshuffleTask as BaseDeshuffleTask,
    DeshuffleTaskset as BaseDeshuffleTaskset,
    analyze_ordering,
)

DEFAULT_NUM_TASKS = 3000
SYSTEM = (
    "You reconstruct a document whose paragraphs were shuffled. The file "
    "paragraphs.jsonl in the working directory holds them, one JSON line per "
    "paragraph with a numeric label. Use the shell or Python to read and "
    "compare paragraphs, reason about the document's flow, and give the final "
    "order as a JSON list of labels inside \\boxed{} in your last reply."
)


def task_name(index: int) -> str:
    return f"deshuffle-{index:05d}"


class DeshuffleTask(BaseDeshuffleTask):
    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace) -> float:
        reward, metrics = analyze_ordering(trace.last_reply or "", self.data.gold)
        trace.record_metrics({**metrics, "ordering_reward": reward})
        return 1.0 if reward >= 1.0 else 0.0

    async def ordering_reward(self, trace: vf.Trace) -> float:
        # Undecorated override: counted once, under `solved` (graded value is a metric).
        return await self.solved(trace)


class AffineDeshuffleConfig(DeshuffleConfig):
    tasks: list[str] = []
    """Task names to load (empty = the first `num_tasks` of the stream)."""
    num_tasks: int = DEFAULT_NUM_TASKS
    task_system_prompt: str = SYSTEM


class DeshuffleTaskset(BaseDeshuffleTaskset, vf.Taskset[DeshuffleTask, AffineDeshuffleConfig]):
    INFINITE = False

    def load(self) -> list[DeshuffleTask]:
        cfg = self.config
        want = set(cfg.tasks)
        tasks: list[DeshuffleTask] = []
        for index in range(cfg.num_tasks):
            name = task_name(index)
            if want and name not in want:
                continue
            base = self._make_task(index)
            data = base.data.model_copy(update={"name": name, "system_prompt": cfg.task_system_prompt})
            tasks.append(DeshuffleTask(data, cfg.task))
        return tasks
