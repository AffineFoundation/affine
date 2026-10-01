"""affine-i3math-v1: INTELLECT-3-RL math, single turn, `\\boxed{}` answer.

Wrapper over prime-envs' `i3_math_v1` (INTELLECT-3-RL `math` train; math-verify
runs as a uv script in the rollout runtime; reward `correct` - already a fold
key). Harder and fresher than the MATH train set behind `affine_math`, on the
same `boxed` dialect and `teacher_boxed` policy. Changes for the duel corpus:

  * the `\\boxed{}` mandate moves into a real *system* message (the fold reads
    the `boxed` marker from the system message; the base puts its instruction
    in the user turn, which stays as the bare problem here);
  * `tasks: list[str]` selector by `name = "i3math-" + sha256(question)[:12]`
    (rollouts/catalog.py `_i3_math_meta` computes the same string);
  * difficulty band [0.1, 0.9] on the shipped `avg@8_qwen3_4b_instruct_2507`
    column by default (same rule as affine_logic / affine_science).
"""

from __future__ import annotations

import hashlib

import verifiers.v1 as vf
from datasets import load_dataset
from i3_math_v1.taskset import (
    ANSWER_KEY,
    DATASET_NAME,
    DATASET_SPLIT,
    DATASET_SUBSET,
    QUESTION_KEY,
    Filter,
    I3MathConfig,
    I3MathTask,
    MathData,
)

DIFFICULTY_MIN = 0.1
DIFFICULTY_MAX = 0.9
# Same mandate as affine_math_v1 (the fold's `boxed` marker).
SYSTEM = (
    "Solve the math problem. Reason step by step in plain text, then end "
    "your response with the final answer in `\\boxed{}`. Emit exactly one "
    "`\\boxed{}` block and nothing after it."
)


def task_name(question: str) -> str:
    return "i3math-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]


class AffineI3MathConfig(I3MathConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole filtered split)."""
    filter: Filter = Filter(min=DIFFICULTY_MIN, max=DIFFICULTY_MAX)
    task_system_prompt: str = SYSTEM


class I3MathTaskset(vf.Taskset[I3MathTask, AffineI3MathConfig]):
    def load(self) -> list[I3MathTask]:
        cfg = self.config
        want = set(cfg.tasks)
        flt = cfg.filter
        rows = load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)
        tasks: list[I3MathTask] = []
        for i, row in enumerate(rows):
            question = str(row[QUESTION_KEY])
            name = task_name(question)
            if want and name not in want:
                continue
            if flt.column is not None:
                value = row.get(flt.column)
                if value is None or not (flt.min <= float(value) <= flt.max):
                    continue
            answer = row.get(ANSWER_KEY)
            if answer in (None, ""):
                continue
            tasks.append(I3MathTask(
                MathData(idx=i, name=name, system_prompt=cfg.task_system_prompt,
                         prompt=question, answer=str(answer)),
                cfg.task,
            ))
        return tasks
