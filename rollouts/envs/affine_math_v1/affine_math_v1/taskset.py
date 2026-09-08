"""affine-math-v1: competition math, single turn, `\\boxed{}` answer contract.

Why this exists next to the bundled gsm8k_v1: the duel's `boxed` dialect
(affine/dialects.py) needs two things gsm8k_v1 does not give it — the
format mandate in a real *system* message (the fold requires the dialect
marker there, not in the user turn) and a `\\boxed{}` answer rather than
`#### N`. The problems come from the MATH train split (7.5k, 7 subjects x 5
levels); the test split is a public benchmark and is never loaded.

Task identity: `name = "math-" + sha256(problem)[:12]`. rollouts/catalog.py
computes the same string from the same dataset row so the scheduler can
address tasks by name through `--env.taskset.tasks`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import verifiers.v1 as vf
from datasets import load_dataset

DATASET = "DigitalLearningGmbH/MATH-lighteval"
SPLIT = "train"

# The mandate the fold's `boxed` marker looks for and the phrasing the
# teacher-fluency smoke used (0/5 -> 5/5 parsable answers on Qwen3.8-27B).
SYSTEM = (
    "Solve the math problem. Reason step by step in plain text, then end "
    "your response with the final answer in `\\boxed{}`. Emit exactly one "
    "`\\boxed{}` block and nothing after it."
)
VERIFY = (Path(__file__).parent / "verify.py").read_bytes()
BOXED_OPEN = "\\boxed{"


def task_name(problem: str) -> str:
    return "math-" + hashlib.sha256(problem.encode("utf-8")).hexdigest()[:12]


def last_boxed_body(text: str) -> str | None:
    """Content of the last brace-balanced `\\boxed{...}` in `text`."""
    found = None
    start = text.find(BOXED_OPEN)
    while start != -1:
        depth = 0
        for i in range(start + len(BOXED_OPEN) - 1, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    found = text[start + len(BOXED_OPEN):i]
                    break
        start = text.find(BOXED_OPEN, start + len(BOXED_OPEN))
    return found


class MathData(vf.TaskData):
    problem: str
    answer: str
    """Bare content of the reference solution's final \\boxed{}."""
    subject: str
    level: str


class MathTask(vf.Task[MathData]):
    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.prepare_uv_script(VERIFY)

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        result = await runtime.run_uv_script(
            VERIFY, args=[self.data.answer, trace.last_reply or ""])
        if result.exit_code != 0:
            raise RuntimeError(f"verify.py failed: {result.stderr.strip()[-500:]}")
        lines = result.stdout.strip().splitlines()
        return float(lines[-1]) if lines else 0.0

    async def validate(self, runtime: vf.Runtime) -> bool:
        result = await runtime.run_uv_script(
            VERIFY, args=[self.data.answer, BOXED_OPEN + self.data.answer + "}"])
        if result.exit_code != 0:
            raise RuntimeError(f"verify.py failed: {result.stderr.strip()[-500:]}")
        lines = result.stdout.strip().splitlines()
        return bool(lines) and float(lines[-1]) == 1.0


class MathConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole split)."""


class MathTaskset(vf.Taskset[MathTask, MathConfig]):
    def load(self) -> list[MathTask]:
        want = set(self.config.tasks)
        rows = load_dataset(DATASET, "default", split=SPLIT)
        tasks: list[MathTask] = []
        for i, row in enumerate(rows):
            name = task_name(row["problem"])
            if want and name not in want:
                continue
            answer = last_boxed_body(row["solution"] or "")
            if not answer:
                continue
            tasks.append(MathTask(
                MathData(
                    idx=i,
                    name=name,
                    system_prompt=SYSTEM,
                    prompt=row["problem"],
                    problem=row["problem"],
                    answer=answer,
                    subject=str(row.get("type") or ""),
                    level=str(row.get("level") or ""),
                ),
                self.config.task,
            ))
        return tasks
