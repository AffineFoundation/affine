"""math-test: Full MATH test split (Hendrycks et al., 5,000 problems; answer = last \\boxed{} of the reference solution).

verifiers v1 taskset in the same format as primeintellect/math500 and primeintellect/aime24: one user turn, no system prompt,
reward = vf.verify_boxed_math_answer (math-verify on the last \\boxed{} after </think>), so it grades exactly like the Hub envs.
Dataset pinned by revision; task order is fixed (idx = position after the deterministic sort below)."""

from collections.abc import Iterator

import verifiers.v1 as vf

DATASET_NAME = "DigitalLearningGmbH/MATH-lighteval"
DATASET_SPLIT = "test"
DATASET_REVISION = "0530c78699ea5e8eb5530600900e1f328b48acad"
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def _num(x) -> str:
    f = float(x)
    return str(int(f)) if f == int(f) else str(x)


def _last_boxed(s: str) -> str:
    i = s.rfind("\\boxed")
    if i < 0:
        i = s.rfind("\\fbox")
    j = s.find("{", i)
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    raise ValueError("no boxed answer")


class MathTestData(vf.TaskData):
    answer: str


class MathTestTaskConfig(vf.TaskConfig):
    math_verify_timeout: int = 5


class MathTestTask(vf.Task[MathTestData, vf.State, MathTestTaskConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return vf.verify_boxed_math_answer(trace.last_reply, self.data.answer, timeout_seconds=self.config.math_verify_timeout)


class MathTestConfig(vf.TasksetConfig):
    task: MathTestTaskConfig = MathTestTaskConfig()


class MathTestTaskset(vf.Taskset[MathTestTask, MathTestConfig]):
    def load(self) -> Iterator[MathTestTask]:
        from datasets import load_dataset

        rows = load_dataset(DATASET_NAME, split=DATASET_SPLIT, revision=DATASET_REVISION)
        rows = list(rows)
        pass
        for i, row in enumerate(rows):
            yield MathTestTask(MathTestData(idx=i, prompt=row['problem'] + INSTRUCTION, answer=_last_boxed(row['solution'])), self.config.task)
