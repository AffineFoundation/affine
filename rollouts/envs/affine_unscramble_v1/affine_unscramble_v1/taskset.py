"""affine-unscramble-v1: put scrambled sentences back in order, `text` dialect.

Wrapper over research-environments' `unscramble_v1` (10,300 rows of
`kalomaze/unscramble-mix-it2`). The prompt asks for the answer inside
`<unscrambled_text>` tags; that tag is not a dialect — the whole visible
reply is the action (`text`) and the checker extracts the block itself.

Changes for the duel corpus: a system message (the fold needs one), a
`tasks` selector keyed on the row's `problem_id`, and a binary `correct`
reward (1.0 only when the ordering is exactly right) next to the base
power-scaled `similarity`, which stays as a metric: the fold's outcome
rule is `score >= 1.0` = solved, and a partial-credit reward would make
"failed at 0.95" a failure label.

Task identity: `name = "unscr-" + problem_id`.
"""

from __future__ import annotations

import json

import verifiers.v1 as vf
from datasets import load_dataset
from unscramble_v1.taskset import (
    DATASET_NAME,
    UnscrambleData,
    UnscrambleTask,
    UnscrambleTaskConfig,
)

SPLIT = "train"

SYSTEM = (
    "You restore the original order of scrambled text blocks. Think about "
    "how the blocks connect, then give the final ordering exactly in the "
    "format the task specifies, inside one <unscrambled_text> ... "
    "</unscrambled_text> block, and nothing after it."
)


def task_name(problem_id: str) -> str:
    return f"unscr-{problem_id}"


class UnscrambleTaskWrapped(UnscrambleTask):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return 1.0 if await self.similarity_score(trace) >= 1.0 else 0.0

    @vf.metric
    async def similarity(self, trace: vf.Trace) -> float:
        return await self.similarity_score(trace)

    async def similarity_score(self, trace: vf.Trace) -> float:
        return await UnscrambleTask.similarity(self, trace)


class UnscrambleConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole split)."""
    task: UnscrambleTaskConfig = UnscrambleTaskConfig()


class UnscrambleTaskset(vf.Taskset[UnscrambleTaskWrapped, UnscrambleConfig]):
    def load(self) -> list[UnscrambleTaskWrapped]:
        want = set(self.config.tasks)
        rows = load_dataset(DATASET_NAME, split=SPLIT)
        tasks: list[UnscrambleTaskWrapped] = []
        for i, row in enumerate(rows):
            name = task_name(str(row["problem_id"]))
            if want and name not in want:
                continue
            tasks.append(UnscrambleTaskWrapped(
                UnscrambleData(
                    idx=i,
                    name=name,
                    system_prompt=SYSTEM,
                    prompt=row["prompt"],
                    answer=json.loads(row["verification_info"])["ground_truth"],
                ),
                self.config.task,
            ))
        return tasks
