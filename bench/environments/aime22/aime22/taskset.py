"""aime22: AIME 2022 (I+II, 30 problems).

verifiers v1 taskset in the same format as primeintellect/math500 and primeintellect/aime24: one user turn, no system prompt,
reward = vf.verify_boxed_math_answer (math-verify on the last \\boxed{} after </think>), so it grades exactly like the Hub envs.
Dataset pinned by revision; task order is fixed (idx = position after the deterministic sort below)."""

from collections.abc import Iterator

import verifiers.v1 as vf

DATASET_NAME = "di-zhang-fdu/AIME_1983_2024"
DATASET_SPLIT = "train"
DATASET_REVISION = "3e2cc86390666c5c756622afc0eeb9e6194496bc"
INSTRUCTION = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"


def _num(x) -> str:
    f = float(x)
    return str(int(f)) if f == int(f) else str(x)


def _aime(x) -> str:
    """'080 or 081 (both were accepted)' -> '80|81' (2022 II #8 had two accepted answers)."""
    import re
    return "|".join(str(int(v)) for v in re.findall(r"\d+", str(x)))


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


class AIME22Data(vf.TaskData):
    network_allow: list[str] = []
    answer: str
    """Accepted integer answer(s), '|'-separated when the MAA accepted more than one."""


class AIME22TaskConfig(vf.TaskConfig):
    math_verify_timeout: int = 5


class AIME22Task(vf.Task[AIME22Data, vf.State, AIME22TaskConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return max(vf.verify_boxed_math_answer(trace.last_reply, a, timeout_seconds=self.config.math_verify_timeout)
                   for a in self.data.answer.split("|"))


class AIME22Config(vf.TasksetConfig):
    task: AIME22TaskConfig = AIME22TaskConfig()


class AIME22Taskset(vf.Taskset[AIME22Task, AIME22Config]):
    def load(self) -> Iterator[AIME22Task]:
        from datasets import load_dataset

        rows = load_dataset(DATASET_NAME, split=DATASET_SPLIT, revision=DATASET_REVISION)
        rows = [r for r in rows if str(r['Year']) == '2022']
        rows = sorted(rows, key=lambda r: (r['Part'], int(r['Problem Number'])))
        for i, row in enumerate(rows):
            yield AIME22Task(AIME22Data(idx=i, prompt=INSTRUCTION + row['Question'], answer=_aime(row['Answer'])), self.config.task)
