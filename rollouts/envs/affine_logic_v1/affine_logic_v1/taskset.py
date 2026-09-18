"""affine-logic-v1: INTELLECT-3-RL logic puzzles, single turn, `text` dialect.

Thin wrapper over research-environments' `i3_logic_v1` (11,647 puzzles:
zebra, ciphers, sudoku, futoshiki, BBH/BBEH families, ...; each graded by a
per-family Python verifier in process — no sandbox, no judge). Three changes
for the duel corpus:

  * a real *system* message — the fold drops any turn whose prefix has none;
  * a `tasks: list[str]` selector so the scheduler addresses rows by name
    through `--env.taskset.tasks`;
  * the grade lands under `correct` (the fold reads `solved` / `correct` /
    `passed_fraction` only; the base reward is named `correct_answer`).

Each puzzle states its own answer layout (a ```python tuple, a ```json
block, "The answer is ..."); the visible reply is the action (`text`).
The difficulty filter defaults to the band a 4B model neither always nor
never solves ([0.1, 0.9] on the shipped `avg@16_qwen3_4b_instruct_2507`
column) — rollouts/catalog.py `_i3_logic_meta` applies the same band, so
the catalog and this taskset agree on the row set.

Task identity: `name = "logic-" + sha256(question)[:12]`.
"""

from __future__ import annotations

import hashlib
import json

import verifiers.v1 as vf
from datasets import load_dataset
from i3_logic_v1.base.data import Data
from i3_logic_v1.task2verifier import verifier_classes
from i3_logic_v1.taskset import Filter, I3LogicData, I3LogicTask, parse_answer

DATASET = "PrimeIntellect/INTELLECT-3-RL"
SUBSET = "logic"
SPLIT = "train"
DIFFICULTY_MIN = 0.1
DIFFICULTY_MAX = 0.9
TASKS_TO_SKIP = ("arc_agi", "arc_agi_2", "buggy_tables")

SYSTEM = (
    "You solve logic and reasoning puzzles. Think the problem through step "
    "by step, then give the final answer in exactly the format the puzzle "
    "asks for. Your visible reply is graded as a whole, so state the answer "
    "once, in that format, and do not add a second candidate answer."
)


def task_name(question: str) -> str:
    return "logic-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]


class LogicTask(I3LogicTask):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        verifier_cls = verifier_classes.get(self.data.task_name)
        if verifier_cls is None:
            raise ValueError(f"Verifier class not found for task: {self.data.task_name}")
        data_obj = Data.from_json_str(self.data.game_data)
        return float(verifier_cls().verify(data_obj, parse_answer(trace.last_reply)))

    async def correct_answer(self, trace: vf.Trace) -> float:
        # Undecorated override: suppresses the base reward so the grade is
        # counted once, under `correct`.
        return await self.correct(trace)


class LogicConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole filtered split)."""
    filter: Filter = Filter(min=DIFFICULTY_MIN, max=DIFFICULTY_MAX)
    tasks_to_skip: list[str] = list(TASKS_TO_SKIP)


class LogicTaskset(vf.Taskset[LogicTask, LogicConfig]):
    def load(self) -> list[LogicTask]:
        want = set(self.config.tasks)
        flt = self.config.filter
        skip = set(self.config.tasks_to_skip)
        rows = load_dataset(DATASET, SUBSET, split=SPLIT)
        tasks: list[LogicTask] = []
        for i, row in enumerate(rows):
            name = task_name(row["question"])
            if want and name not in want:
                continue
            info = json.loads(row["info"])
            if info["task_name"] in skip:
                continue
            if not (flt.min <= (row.get(flt.column) or 0) <= flt.max):
                continue
            game_data = info["game_data_str"] or info["game_data"]
            if not isinstance(game_data, str):
                game_data = json.dumps(game_data)
            tasks.append(LogicTask(
                I3LogicData(
                    idx=i,
                    name=name,
                    system_prompt=SYSTEM,
                    prompt=row["question"],
                    task_name=info["task_name"],
                    game_data=game_data,
                ),
                self.config.task,
            ))
        return tasks
