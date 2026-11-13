"""arxivmath: MathArena ArXivMath (https://huggingface.co/datasets/MathArena/arxivmath), 103 questions.

Research-level questions with a single final answer (a number or a closed-form LaTeX expression), generated from arXiv papers
submitted Dec 2025 - Mar 2026 and human-curated by MathArena (ETH SRI). Same format as the AIME envs: one user turn, no system
prompt, the AIME instruction. Grading: MathArena grades ArXivMath with an LLM judge; we stay rule-based like the rest of the
suite: math-verify on the last \\boxed{} after </think> (vf.verify_boxed_math_answer), plus one retry on both sides rewritten
for two latex2sympy quirks that matter for symbolic answers: a run of plain letters is one symbol ("2dr" -> dr) and a plain "d"
reads as a differential ("d r" -> dr, "...) d" dropped). The retry splits letter runs and renames d; it is applied to gold and
prediction alike, so it can only turn a parser miss into a match. Dataset pinned by revision; task order is fixed (month, then
problem_idx)."""

import re
from collections.abc import Iterator

import verifiers.v1 as vf

DATASET_NAME = "MathArena/arxivmath"
DATASET_SPLIT = "train"
DATASET_REVISION = "f0a9cf83259855c31b37168eca646ec6f0e2598c"
INSTRUCTION = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"
MONTHS = {"arxiv/december": 0, "arxiv/january": 1, "arxiv/february": 2, "arxiv/march": 3}
_RUN = re.compile(r"(?<![\\A-Za-z])[A-Za-z]{2,}(?![A-Za-z])")  # plain letter run (not a \\command)
_D = re.compile(r"(?<![\\A-Za-z])d(?![A-Za-z])")


def implicit_mult(s: str) -> str:
    """'2dr-r^2' -> '2\\vartheta  r-r^2': letters of a plain run are separate factors, d is renamed; \\ln, \\min ... untouched."""
    return _D.sub(r"\\vartheta ", _RUN.sub(lambda m: " ".join(m.group(0)), s))


def check(reply: str | None, answer: str, timeout: int = 5) -> float:
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


class ArxivMathData(vf.TaskData):
    answer: str
    competition: str
    """MathArena month split, e.g. arxiv/january."""
    source: str
    """arXiv id of the source paper."""


class ArxivMathTaskConfig(vf.TaskConfig):
    math_verify_timeout: int = 5


class ArxivMathTask(vf.Task[ArxivMathData, vf.State, ArxivMathTaskConfig]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return check(trace.last_reply, self.data.answer, self.config.math_verify_timeout)


class ArxivMathConfig(vf.TasksetConfig):
    task: ArxivMathTaskConfig = ArxivMathTaskConfig()


class ArxivMathTaskset(vf.Taskset[ArxivMathTask, ArxivMathConfig]):
    def load(self) -> Iterator[ArxivMathTask]:
        from datasets import load_dataset

        rows = load_dataset(DATASET_NAME, split=DATASET_SPLIT, revision=DATASET_REVISION)
        rows = sorted(rows, key=lambda r: (MONTHS[r["competition"]], int(r["problem_idx"])))
        for i, row in enumerate(rows):
            yield ArxivMathTask(ArxivMathData(idx=i, prompt=INSTRUCTION + row["problem"], answer=str(row["answer"]),
                                              competition=row["competition"], source="%.5f" % row["source"]), self.config.task)
