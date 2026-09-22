"""affine-scitext-v1: teacher-synthesized textbook-science problems with a code-checked answer, `boxed` dialect.

aa-gap-fill-plan §1.2 (Jacob "go", 2026-09-22 15:24 UTC; build after items
1-4 are in flight). HLE (10 % of the Index) is expert-level science; our
`affine_science` band gives depth on existing problems, this env adds
SUPPLY in the same shape: for each seed problem from an open problem set
(OlympiadBench, Apache-2.0; SciBench, MIT; TheoremQA, MIT) the teacher
writes 2 variants — changed constants, changed asked quantity — with a
solution and Python code that recomputes the answer; a variant is kept
only if the code reproduces the boxed answer and the teacher re-solves it
blind in 1-3 of 3 attempts (`generate.py`). HLE itself (`cais/hle`) is
never used, not even as a seed. Uids carry `[GEN:e<epoch>]`.

Shape = affine_science: system message with the `\\boxed{}` mandate, one
visible reply, `correct` = math-verify (`vf.verify_boxed_math_answer`),
`null` harness. Grade key `correct` is what the fold reads for this
family (PRIMARY_REWARD_KEYS: solved, correct, ...).
"""

from __future__ import annotations

from pathlib import Path

import verifiers.v1 as vf

from affine_gen_v1.store import GenTaskStore

SOURCE = "affine_scitext"
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_EPOCH = 1

SYSTEM = (
    "You solve physics, chemistry and quantitative science problems. Think the problem through step by "
    "step, keep track of units, then end your response with the final answer in `\\boxed{}` — a number "
    "(with the unit the problem asks for, or unitless if it says so) or a closed-form expression. Emit "
    "exactly one `\\boxed{}` block and nothing after it."
)


class SciTextData(vf.TaskData):
    uid: str
    answer: str
    subject: str
    seed_source: str
    teacher_pass: int


class SciTextTask(vf.Task[SciTextData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return vf.verify_boxed_math_answer(trace.last_reply or "", self.data.answer)


class SciTextConfig(vf.TasksetConfig):
    tasks: list[str] = []
    epoch: int = DEFAULT_EPOCH


def list_catalog(epoch: int = DEFAULT_EPOCH) -> list[dict]:
    store = GenTaskStore(SOURCE, PACKAGE_DIR, epoch)
    return [{"uid": r["uid"], "domain": r["subject"], "topic": r["seed_source"], "teacher_pass": r.get("teacher_pass", 0)}
            for r in store.tasks()]


class SciTextTaskset(vf.Taskset[SciTextTask, SciTextConfig]):
    def load(self) -> list[SciTextTask]:
        want = set(self.config.tasks)
        store = GenTaskStore(SOURCE, PACKAGE_DIR, self.config.epoch)
        tasks: list[SciTextTask] = []
        for i, rec in enumerate(store.tasks()):
            if want and rec["uid"] not in want:
                continue
            tasks.append(SciTextTask(
                SciTextData(idx=i, name=rec["uid"], system_prompt=SYSTEM, prompt=rec["problem"],
                            uid=rec["uid"], answer=rec["answer"], subject=rec["subject"],
                            seed_source=rec["seed_source"], teacher_pass=int(rec.get("teacher_pass") or 0)),
                self.config.task,
            ))
        if want and not tasks:
            raise ValueError(f"no scitext task matched {sorted(want)[:5]}...")
        return tasks
