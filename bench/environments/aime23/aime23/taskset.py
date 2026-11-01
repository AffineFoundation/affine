"""aime23: AIME 2023 (I+II, 30 problems).

verifiers v1 taskset in the same format as primeintellect/math500 and primeintellect/aime24: one user turn, no system prompt,
reward = vf.verify_boxed_math_answer (math-verify on the last \\boxed{} after </think>), so it grades exactly like the Hub envs.
Dataset pinned by revision; task order is fixed (idx = position after the deterministic sort below)."""

from collections.abc import Iterator

import verifiers.v1 as vf

DATASET_NAME = "di-zhang-fdu/AIME_1983_2024"
DATASET_SPLIT = "train"
DATASET_REVISION = "3e2cc86390666c5c756622afc0eeb9e6194496bc"
# 2023 AIME I problem 15 is absent from the pinned dataset; text and answer from the AoPS wiki
# (https://artofproblemsolving.com/wiki/index.php/2023_AIME_I_Problems/Problem_15).
SUPPLEMENT_2023_I_15 = {
    "ID": "2023-I-15", "Year": 2023, "Problem Number": 15, "Part": "I", "Answer": "349",
    "Question": ("Find the largest prime number $p<1000$ for which there exists a complex number $z$ satisfying\n"
                 "the real and imaginary part of $z$ are both integers;\n$|z|=\\sqrt{p},$ and\n"
                 "there exists a triangle whose three side lengths are $p,$ the real part of $z^{3},$ and the imaginary part of $z^{3}.$"),
}
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


class AIME23Data(vf.TaskData):
    network_allow: list[str] = []
    answer: str
    """Accepted integer answer(s), '|'-separated when the MAA accepted more than one."""


class AIME23TaskConfig(vf.TaskConfig):
    math_verify_timeout: int = 5


class AIME23Task(vf.Task[AIME23Data, vf.State, AIME23TaskConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return max(vf.verify_boxed_math_answer(trace.last_reply, a, timeout_seconds=self.config.math_verify_timeout)
                   for a in self.data.answer.split("|"))


class AIME23Config(vf.TasksetConfig):
    task: AIME23TaskConfig = AIME23TaskConfig()


class AIME23Taskset(vf.Taskset[AIME23Task, AIME23Config]):
    def load(self) -> Iterator[AIME23Task]:
        from datasets import load_dataset

        rows = load_dataset(DATASET_NAME, split=DATASET_SPLIT, revision=DATASET_REVISION)
        rows = [r for r in rows if str(r['Year']) == '2023']
        if not any(r['Part'] == 'I' and int(r['Problem Number']) == 15 for r in rows):
            rows.append(SUPPLEMENT_2023_I_15)   # missing from the pinned dataset revision
        rows = sorted(rows, key=lambda r: (r['Part'], int(r['Problem Number'])))
        for i, row in enumerate(rows):
            yield AIME23Task(AIME23Data(idx=i, prompt=INSTRUCTION + row['Question'], answer=_aime(row['Answer'])), self.config.task)
