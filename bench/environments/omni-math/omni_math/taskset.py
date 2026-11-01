"""omni-math: Omni-MATH (KbsdJames/Omni-MATH, test, 4,428 olympiad-level problems).

verifiers v1 taskset; prompt in the math500 style; reward = affine-math-grader on the last \\boxed{} after </think>.
Omni-MATH answers are often free-form (text, conditions, multiple values); rule-undecidable misses are flagged with the
`needs_judge` metric (the Omni-MATH authors grade with an LLM judge, Omni-Judge). Disabled in the scheduler for now."""
from collections.abc import Iterator

import verifiers.v1 as vf
from affine_math_grader import grade, last_boxed

DATASET_NAME = "KbsdJames/Omni-MATH"
DATASET_SPLIT = "test"
DATASET_REVISION = "40ba231d8f16e29ecd40e6407e2c8640145a8f62"
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."


class OmniMathData(vf.TaskData):
    answer: str
    difficulty: float | None = None
    domain: str | None = None
    source: str | None = None


class OmniMathTask(vf.Task[OmniMathData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return grade(last_boxed(trace.last_reply), self.data.answer, multiple="," in self.data.answer).score

    @vf.metric
    async def needs_judge(self, trace: vf.Trace) -> float:
        return float(grade(last_boxed(trace.last_reply), self.data.answer, multiple="," in self.data.answer).needs_judge)


class OmniMathTaskset(vf.Taskset[OmniMathTask, vf.TasksetConfig]):
    def load(self) -> Iterator[OmniMathTask]:
        from datasets import load_dataset

        rows = load_dataset(DATASET_NAME, split=DATASET_SPLIT, revision=DATASET_REVISION)
        for i, r in enumerate(rows):
            try:
                diff = float(r.get("difficulty"))
            except (TypeError, ValueError):
                diff = None
            yield OmniMathTask(OmniMathData(idx=i, prompt=r["problem"] + INSTRUCTION, answer=str(r["answer"]).strip().strip("$"),
                                            difficulty=diff, domain=str(r.get("domain")), source=r.get("source")), self.config.task)
