"""affine-popqa-abstain-v1: long-tail facts with abstention credit, `text` dialect.

aa-gap-fill-plan §3.1 (Jacob "go", 2026-09-22 15:24 UTC). PopQA (Mallen et
al. 2023; `akariasai/PopQA`, 14,267 questions about Wikidata entities chosen
by LOW popularity — the "confidently wrong on rare facts" failure that
AA-Omniscience penalises). Same prompt and grade as affine_trivia_abstain:

  * `solved` = correct OR "I don't know"; a wrong answer = 0. A teacher
    abstention is an admissible reference; a king that guesses wrong is a
    failed state.
  * `omniscience` = +1 / 0 / -1 (AA's index), `correct`, `abstained` metrics.

Answers: the row's `possible_answers` (object + aliases), normalised with
TriviaQA's rules. Band: `max_pop` keeps only the rarest subjects (Wikipedia
monthly page views of the subject, `s_pop`); the fold's [band_filter]
(teacher abstains-or-answers, king wrong) does the rest. Licence: the source
repository (AlexTMallen/adaptive-retrieval) is MIT; the HF card carries no
licence field — noted for Jacob. Natural Questions is NOT used (CC-BY-SA;
Jacob 15:24 UTC: "PopQA only").
"""

from __future__ import annotations

import json

import verifiers.v1 as vf
from datasets import load_dataset
from triviaqa_v1.taskset import TriviaQAData, normalize_answer

from affine_trivia_abstain_v1.taskset import SYSTEM, TriviaAbstainTask

DATASET_NAME = "akariasai/PopQA"
DATASET_REVISION = "098765c79ea10a2cb19c828324e33281b8336ec0"
SPLIT = "test"
METRIC = "normalized_alias_exact_match"


def task_name(row_id: int | str) -> str:
    return f"popqa-{row_id}"


class PopQAAbstainConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """`popqa-<id>` names to load (empty = the whole split under the band)."""
    max_pop: int = 5000
    """Keep rows whose subject page views (`s_pop`) are at most this — the long tail."""


class PopQAAbstainTaskset(vf.Taskset[TriviaAbstainTask, PopQAAbstainConfig]):
    def load(self) -> list[TriviaAbstainTask]:
        want = set(self.config.tasks)
        rows = load_dataset(DATASET_NAME, split=SPLIT, revision=DATASET_REVISION)
        tasks: list[TriviaAbstainTask] = []
        for i, row in enumerate(rows):
            name = task_name(row["id"])
            if want and name not in want:
                continue
            if not want and int(row.get("s_pop") or 0) > self.config.max_pop:
                continue
            answers = json.loads(row["possible_answers"]) if isinstance(row["possible_answers"], str) \
                else list(row["possible_answers"])
            tasks.append(TriviaAbstainTask(
                TriviaQAData(
                    idx=i,
                    name=name,
                    system_prompt=SYSTEM,
                    prompt=row["question"],
                    answers=sorted({normalize_answer(a) for a in answers if a}),
                    metric=METRIC,
                ),
                self.config.task,
            ))
        return tasks
