"""amc23: AMC 2023 (the 40-problem AMC12 2023 A/B set used by Qwen/DeepSeek/SimpleRL evals).

verifiers v1 taskset in the same format as primeintellect/math500 and primeintellect/aime24: one user turn, no system prompt,
reward = vf.verify_boxed_math_answer (math-verify on the last \\boxed{} after </think>), so it grades exactly like the Hub envs.
Dataset pinned by revision; task order is fixed (idx = position after the deterministic sort below)."""

from collections.abc import Iterator

import verifiers.v1 as vf

DATASET_NAME = "math-ai/amc23"
DATASET_SPLIT = "test"
DATASET_REVISION = "80815d37005feb82cd7f8fbc6901d5d3eff43057"
INSTRUCTION = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"


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


class AMC23Data(vf.TaskData):
    answer: str


class AMC23TaskConfig(vf.TaskConfig):
    math_verify_timeout: int = 5


class AMC23Task(vf.Task[AMC23Data, vf.State, AMC23TaskConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return vf.verify_boxed_math_answer(trace.last_reply, self.data.answer, timeout_seconds=self.config.math_verify_timeout)


class AMC23Config(vf.TasksetConfig):
    task: AMC23TaskConfig = AMC23TaskConfig()


class AMC23Taskset(vf.Taskset[AMC23Task, AMC23Config]):
    def load(self) -> Iterator[AMC23Task]:
        from datasets import load_dataset

        rows = load_dataset(DATASET_NAME, split=DATASET_SPLIT, revision=DATASET_REVISION)
        rows = list(rows)
        rows = sorted(rows, key=lambda r: int(r['id']))
        for i, row in enumerate(rows):
            yield AMC23Task(AMC23Data(idx=i, prompt=INSTRUCTION + row['question'], answer=_num(row['answer'])), self.config.task)
