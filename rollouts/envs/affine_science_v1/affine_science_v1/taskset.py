"""affine-science-v1: INTELLECT-3-RL science problems, `boxed` dialect.

Wrapper over research-environments' `i3_science_v1` (29,307 problems with a
short gold answer). Changes for the duel corpus:

  * the `\\boxed{}` mandate in a real *system* message (the fold's `boxed`
    marker must be there, as in affine_math_v1), the base INSTRUCTION line
    stays in the user prompt;
  * `correct` = math-verify ONLY. The base reward falls back to an LLM judge
    whenever math-verify says 0 — the datagen pods have no judge endpoint,
    and a 401 there marks the rollout errored (the wiki lesson). Some gold
    answers are units or expressions math-verify does not equate, so the
    measured solve rate is a lower bound;
  * a `tasks: list[str]` selector; difficulty band [0.1, 0.9] on the
    shipped `avg@16_qwen3_4b_instruct_2507` column by default
    (rollouts/catalog.py `_i3_science_meta` applies the same band).

Task identity: `name = "science-" + sha256(question)[:12]`.
"""

from __future__ import annotations

import hashlib

import verifiers.v1 as vf
from datasets import load_dataset
from i3_science_v1.taskset import INSTRUCTION, Filter, ScienceData

DATASET = "PrimeIntellect/INTELLECT-3-RL"
SUBSET = "science"
SPLIT = "train"
DIFFICULTY_MIN = 0.1
DIFFICULTY_MAX = 0.9

SYSTEM = (
    "Solve the science problem. Reason step by step in plain text, then end "
    "your response with the final answer in `\\boxed{}`. Emit exactly one "
    "`\\boxed{}` block and nothing after it."
)


def task_name(question: str) -> str:
    return "science-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]


class ScienceTask(vf.Task[ScienceData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return vf.verify_boxed_math_answer(trace.last_reply or "", self.data.answer)


class ScienceConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole filtered split)."""
    filter: Filter = Filter(min=DIFFICULTY_MIN, max=DIFFICULTY_MAX)


class ScienceTaskset(vf.Taskset[ScienceTask, ScienceConfig]):
    def load(self) -> list[ScienceTask]:
        want = set(self.config.tasks)
        flt = self.config.filter
        rows = load_dataset(DATASET, SUBSET, split=SPLIT)
        tasks: list[ScienceTask] = []
        for i, row in enumerate(rows):
            question = str(row["question"])
            name = task_name(question)
            if want and name not in want:
                continue
            if flt.column is not None:
                value = row.get(flt.column)
                if value is None or not (flt.min <= float(value) <= flt.max):
                    continue
            tasks.append(ScienceTask(
                ScienceData(
                    idx=i,
                    name=name,
                    system_prompt=SYSTEM,
                    prompt=f"{INSTRUCTION}{question}",
                    question=question,
                    answer=str(row["answer"]),
                ),
                self.config.task,
            ))
        return tasks
