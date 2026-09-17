"""affine-trivia-v1: TriviaQA closed-book question answering, `text` dialect.

Wrapper over research-environments' `triviaqa_v1`. The base taskset is an
lm-eval style benchmark (validation split, five-shot Question/Answer blocks).
For the duel corpus we want a chat shape on rows that are NOT the benchmark:

  * the TRAIN split (`rc.wikipedia.nocontext`, 61,888 rows); `validation`
    (7,993) stays out of D as a held-out scorecard;
  * zero-shot: system message + the bare question, one visible answer;
  * a `tasks: list[str]` selector keyed on `question_id` (= TaskData.name).

Grading is the base `correct` reward unchanged: the reply is cut at the first
newline / period / comma, normalized (lower-case, no punctuation, no
articles) and matched against the row's normalized aliases. Hence the
system prompt: the visible reply must start with the answer.
"""

from __future__ import annotations

import verifiers.v1 as vf
from datasets import load_dataset
from triviaqa_v1.taskset import (
    DATASET_NAME,
    DATASET_REVISION,
    TriviaQAData,
    TriviaQATask,
)

DATASET_CONFIG = "rc.wikipedia.nocontext"
SPLIT = "train"
METRIC = "normalized_alias_exact_match"

SYSTEM = (
    "You answer trivia questions from memory. Think about the question "
    "first, then reply with the answer only: the name, term, number or "
    "short phrase asked for, on one line, with no explanation, no full "
    "sentence and no punctuation after it."
)


class TriviaConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """`question_id`s to load (empty = the whole train split)."""


class TriviaTaskset(vf.Taskset[TriviaQATask, TriviaConfig]):
    def load(self) -> list[TriviaQATask]:
        want = set(self.config.tasks)
        rows = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT,
                            revision=DATASET_REVISION)
        tasks: list[TriviaQATask] = []
        for i, row in enumerate(rows):
            qid = row["question_id"]
            if want and qid not in want:
                continue
            tasks.append(TriviaQATask(
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
