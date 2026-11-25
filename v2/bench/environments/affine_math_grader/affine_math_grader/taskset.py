"""Shared rule-based grading for the public Affine math contest environments.

MathArena's HMMT, BRUMO and CMIMC packs are distributed under CC BY-NC-SA 4.0;
MathArena and the respective competitions are attributed by each environment.
BeyondAIME is distributed by ByteDance Seed under CC0 1.0.

One user turn, no system prompt. The reward is ``verify_boxed_math_answer`` on
its last boxed answer after ``</think>``. For symbolic answers, retry both gold
and prediction after rewriting plain implicit products, as ArXivMath does. This
handles latex2sympy's treatment of plain letter runs and of a standalone ``d``;
LaTeX commands are left unchanged. There is no judge and no hidden task data.
"""

import re
from collections.abc import Iterator

import verifiers.v1 as vf

INSTRUCTION = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"
_RUN = re.compile(r"(?<![\\A-Za-z])[A-Za-z]{2,}(?![A-Za-z])")
_D = re.compile(r"(?<![\\A-Za-z])d(?![A-Za-z])")


def implicit_mult(s: str) -> str:
    """Rewrite a plain letter run as separate factors; rename standalone ``d``."""
    return _D.sub(r"\\vartheta ", _RUN.sub(lambda m: " ".join(m.group(0)), s))


def check(reply: str | None, answer: str, timeout: int = 5) -> float:
    """Score the last boxed answer, with one implicit-multiplication retry."""
    s = vf.verify_boxed_math_answer(reply, answer, timeout_seconds=timeout)
    if s or not reply or ("<think>" in reply and "</think>" not in reply):
        return s
    tail = reply.rsplit("</think>", 1)[-1]
    i = tail.rfind("\\boxed")
    if i < 0:
        return 0.0
    ga, pa = implicit_mult(answer), implicit_mult(tail[i:])
    if ga == answer and pa == tail[i:]:
        return 0.0
    return vf.verify_boxed_math_answer(pa, ga, timeout_seconds=timeout)


class MathContestData(vf.TaskData):
    answer: str


class MathContestTaskConfig(vf.TaskConfig):
    math_verify_timeout: int = 5


class MathContestTask(vf.Task[MathContestData, vf.State, MathContestTaskConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return check(trace.last_reply, self.data.answer, self.config.math_verify_timeout)


class MathContestConfig(vf.TasksetConfig):
    task: MathContestTaskConfig = MathContestTaskConfig()


class MathContestTaskset(vf.Taskset[MathContestTask, MathContestConfig]):
    """Dataset metadata lives in the small taskset subclass of each environment."""

    DATASET_NAME: str
    DATASET_SPLIT: str
    DATASET_REVISION: str
    DATASET_COUNT: int
    SORT_BY_INDEX = True

    def __init__(self, config: MathContestConfig | None = None):
        super().__init__(config or MathContestConfig())

    def load(self) -> Iterator[MathContestTask]:
        from datasets import load_dataset

        rows = list(load_dataset(self.DATASET_NAME, split=self.DATASET_SPLIT, revision=self.DATASET_REVISION))
        if len(rows) != self.DATASET_COUNT:
            raise ValueError(f"{self.DATASET_NAME}@{self.DATASET_REVISION} has {len(rows)} rows, expected {self.DATASET_COUNT}")
        if self.SORT_BY_INDEX:
            rows.sort(key=lambda row: int(row["problem_idx"]))
        for i, row in enumerate(rows):
            yield MathContestTask(MathContestData(idx=i, prompt=INSTRUCTION + row["problem"], answer=str(row["answer"])),
                                  self.config.task)
