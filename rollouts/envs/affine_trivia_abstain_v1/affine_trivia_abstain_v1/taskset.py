"""affine-trivia-abstain-v1: TriviaQA closed-book QA with abstention credit, `text` dialect.

Why: Artificial Analysis' AA-Omniscience (15 % of the Intelligence Index)
scores +1 for a correct answer, -1 for a wrong one and 0 for "I don't know".
Genesis and teacher both sit at a ~50 % hallucination rate there: they guess
when they do not know. The knowledge itself cannot be taught from D, the
decision to abstain can — it is one state ("do I know this?") the teacher
either gets right or not, the same shape that transferred for τ².

Same rows as affine_trivia_v1 (TriviaQA train, `rc.wikipedia.nocontext`;
validation stays a held-out card) under a system prompt that names the
abstention. A question may appear in both envs; the prefixes differ.

Grades (fold reads `solved` first, `>= 1.0` = solved):

  * `solved` = 1.0 for a correct answer OR an abstention, 0.0 for a wrong
    answer. So a teacher abstention is an admissible reference and a king
    that guesses wrong is a `king_fail` state; a king that abstains is not.
  * `omniscience` = +1 / 0 / -1 (AA's index, per task) — metric.
  * `correct`, `abstained` — metrics.
"""

from __future__ import annotations

import verifiers.v1 as vf
from datasets import load_dataset
from triviaqa_v1.taskset import (
    ANSWER_STOP_PATTERN,
    DATASET_NAME,
    DATASET_REVISION,
    TriviaQAData,
    TriviaQATask,
    normalize_answer,
)

DATASET_CONFIG = "rc.wikipedia.nocontext"
SPLIT = "train"
METRIC = "normalized_alias_exact_match"

ABSTAIN = "I don't know"
ABSTAIN_FORMS = frozenset({
    "i don t know", "i do not know", "don t know", "dont know", "unknown",
    "i m not sure", "not sure", "no idea", "i cannot answer", "i can t answer",
})

SYSTEM = (
    "You answer trivia questions from memory. Think about the question "
    "first. If you are confident, reply with the answer only: the name, "
    "term, number or short phrase asked for, on one line, with no "
    f"explanation, no full sentence and no punctuation after it. If you "
    f"are not confident you know the answer, reply exactly: {ABSTAIN}. A "
    "wrong answer counts against you; saying you do not know does not."
)


def prediction_of(reply: str) -> str:
    # lm-eval / the base task: cut at the first newline, period or comma.
    return normalize_answer(ANSWER_STOP_PATTERN.split(reply.strip(), maxsplit=1)[0])


def is_abstention(prediction: str) -> bool:
    return prediction in ABSTAIN_FORMS


class TriviaAbstainTask(TriviaQATask):
    async def finalize(self, trace: vf.Trace) -> None:
        pred = prediction_of(trace.last_reply)
        trace.info["abstained"] = is_abstention(pred)
        trace.info["correct"] = (not trace.info["abstained"]) and pred in self.data.answers

    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace) -> float:
        return float(trace.info["correct"] or trace.info["abstained"])

    @vf.metric
    async def correct(self, trace: vf.Trace) -> float:
        return float(trace.info["correct"])

    @vf.metric
    async def abstained(self, trace: vf.Trace) -> float:
        return float(trace.info["abstained"])

    @vf.metric
    async def omniscience(self, trace: vf.Trace) -> float:
        # AA-Omniscience per-task index: +1 right, 0 abstain, -1 wrong.
        if trace.info["correct"]:
            return 1.0
        return 0.0 if trace.info["abstained"] else -1.0


class TriviaAbstainConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """`question_id`s to load (empty = the whole train split)."""


class TriviaAbstainTaskset(vf.Taskset[TriviaAbstainTask, TriviaAbstainConfig]):
    def load(self) -> list[TriviaAbstainTask]:
        want = set(self.config.tasks)
        rows = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT,
                            revision=DATASET_REVISION)
        tasks: list[TriviaAbstainTask] = []
        for i, row in enumerate(rows):
            qid = row["question_id"]
            if want and qid not in want:
                continue
            tasks.append(TriviaAbstainTask(
                TriviaQAData(
                    idx=i,
                    name=qid,
                    system_prompt=SYSTEM,
                    prompt=row["question"],
                    answers=list(row["answer"]["normalized_aliases"]),
                    metric=METRIC,
                ),
                self.config.task,
            ))
        return tasks
