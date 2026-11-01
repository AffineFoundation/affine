"""olympiadbench: OlympiadBench, text-only English competition math, open-ended (OE_TO_maths_en_COMP, 674 problems).

verifiers v1 taskset; prompt in the math500 style; reward = affine-math-grader (sympy equivalence, intervals/unions,
sets/tuples, multi-part answers, the dataset's numeric tolerance `error`, unit stripping). Misses the rules cannot decide are
scored 0 and flagged by the `needs_judge` metric. Disabled in the scheduler for now (bench/config/benchmarks.toml)."""
import ast
from collections.abc import Iterator

import verifiers.v1 as vf
from affine_math_grader import grade, last_boxed

DATASET_NAME = "Hothan/OlympiadBench"
DATASET_CONFIG = "OE_TO_maths_en_COMP"
DATASET_SPLIT = "train"
DATASET_REVISION = "91184b52131e7fc9455fef848035173aea8cc01a"
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
INSTRUCTION_MULTI = "\nPlease reason step by step, and put all final answers within one \\boxed{}, separated by commas."


def _list(x):
    if isinstance(x, list):
        return [str(v) for v in x]
    try:
        v = ast.literal_eval(x)
        return [str(t) for t in v] if isinstance(v, (list, tuple)) else [str(v)]
    except Exception:
        return [str(x)]


def _opt(x):
    return None if x in (None, "", "None") else str(x)


class OlympiadBenchData(vf.TaskData):
    answers: list[str]
    multiple: bool = False
    unit: str | None = None
    tolerance: float | None = None
    answer_type: str | None = None


class OlympiadBenchTask(vf.Task[OlympiadBenchData]):
    def _grade(self, trace: vf.Trace):
        d = self.data
        gold = d.answers if (d.multiple or len(d.answers) > 1) else d.answers[0]
        return grade(last_boxed(trace.last_reply), gold, unit=d.unit, tolerance=d.tolerance, multiple=d.multiple or len(d.answers) > 1)

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return self._grade(trace).score

    @vf.metric
    async def needs_judge(self, trace: vf.Trace) -> float:
        return float(self._grade(trace).needs_judge)


class OlympiadBenchTaskset(vf.Taskset[OlympiadBenchTask, vf.TasksetConfig]):
    def load(self) -> Iterator[OlympiadBenchTask]:
        from datasets import load_dataset

        rows = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, revision=DATASET_REVISION)
        for i, r in enumerate(rows):
            multi = str(r.get("is_multiple_answer")) == "True"
            ctx = _opt(r.get("context"))
            q = (ctx + "\n\n" if ctx else "") + r["question"]
            tol = _opt(r.get("error"))
            try:
                tol = float(tol) if tol is not None else None
            except ValueError:
                tol = None
            yield OlympiadBenchTask(
                OlympiadBenchData(idx=i, prompt=q + (INSTRUCTION_MULTI if multi else INSTRUCTION), answers=[a.strip("$ ") for a in _list(r["final_answer"])],
                                  multiple=multi, unit=_opt(r.get("unit")), tolerance=tol, answer_type=_opt(r.get("answer_type"))),
                self.config.task,
            )
