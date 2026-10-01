"""affine-ifeval-v1: verifiable instruction following, `text` dialect.

Two row families, one taskset, one `correct` reward (1.0 iff every stated
constraint is met, 0.0 otherwise):

  * `allenai/RLVR-IFeval` train (14,973 prompts, ONE constraint each, 25
    constraint types). Its `ground_truth` is `{"func_name": ..., kwargs}`
    for the open-instruct checker (`if_functions.py`, vendored verbatim) —
    the same grader the dataset was built for, so no mapping onto IFEval's
    instruction ids is needed. This is the family D gets by default.
  * `google/IFEval` (541 prompts, several constraints each), graded by
    research-environments' `ifeval_v1` strict checker. OFF by default
    (`include_ifeval = false`): IFEval is a public benchmark, and putting
    its prompts into D burns it as a scorecard (ifbench stays held out).

Changes over the base `ifeval_v1`: a system message (the fold drops
prefixes without one), a `tasks` selector, the RLVR family, and the grade
under `correct` (the base reward is `followed_instructions`).

Task identity: `name = "ifeval-" + sha256(prompt)[:12]` for both families.
"""

from __future__ import annotations

import ast
import hashlib
import json

import nltk
import verifiers.v1 as vf
from datasets import load_dataset
from ifeval_v1.taskset import IFEvalData, IFEvalTask, IFEvalTaskConfig

from affine_ifeval_v1.if_functions import IF_FUNCTIONS_MAP

RLVR_DATASET = "allenai/RLVR-IFeval"
RLVR_SPLIT = "train"
IFEVAL_DATASET = "google/IFEval"
IFEVAL_SPLIT = "train"

SYSTEM = (
    "You are a helpful assistant. Read the request carefully: it contains "
    "explicit formatting or content constraints (length, case, keywords, "
    "sections, JSON, ...). Think about how to satisfy every constraint, "
    "then write the reply itself. Your visible reply is checked "
    "automatically against each constraint, so it must contain only the "
    "requested answer."
)


def task_name(prompt: str) -> str:
    return "ifeval-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def rlvr_prompt(row: dict) -> str:
    """The user turn of the row's `messages` (a list, or its repr as a str)."""
    messages = row["messages"]
    if isinstance(messages, str):
        messages = ast.literal_eval(messages)
    return str(messages[0]["content"])


class RLVRData(vf.TaskData):
    func_name: str
    kwargs: dict
    """Non-null checker arguments from the row's `ground_truth`."""
    constraint_type: str


class RLVRTask(vf.Task[RLVRData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        check = IF_FUNCTIONS_MAP[self.data.func_name]
        try:
            return 1.0 if check(trace.last_reply or "", **self.data.kwargs) else 0.0
        except Exception:
            # A checker that cannot parse the reply is a failed constraint.
            return 0.0


class IFEvalRowTask(IFEvalTask):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        follow = self.follow_list(trace)
        return 1.0 if follow and all(follow) else 0.0

    async def followed_instructions(self, trace: vf.Trace) -> float:
        # Undecorated override: the grade is counted once, under `correct`.
        return await self.correct(trace)


class IFEvalConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = every admitted row)."""
    include_rlvr: bool = True
    include_ifeval: bool = False
    task: IFEvalTaskConfig = IFEvalTaskConfig()


class IFEvalTaskset(vf.Taskset[vf.Task, IFEvalConfig]):
    def load(self) -> list[vf.Task]:
        want = set(self.config.tasks)
        tasks: list[vf.Task] = []
        if self.config.include_rlvr:
            tasks.extend(self._rlvr(want))
        if self.config.include_ifeval:
            tasks.extend(self._ifeval(want))
        return tasks

    def _rlvr(self, want: set[str]) -> list[RLVRTask]:
        rows = load_dataset(RLVR_DATASET, split=RLVR_SPLIT)
        tasks: list[RLVRTask] = []
        for i, row in enumerate(rows):
            prompt = rlvr_prompt(row)
            name = task_name(prompt)
            if want and name not in want:
                continue
            gt = json.loads(row["ground_truth"])
            func_name = gt.pop("func_name")
            if func_name not in IF_FUNCTIONS_MAP:
                continue
            tasks.append(RLVRTask(
                RLVRData(
                    idx=i,
                    name=name,
                    system_prompt=SYSTEM,
                    prompt=prompt,
                    func_name=func_name,
                    kwargs={k: v for k, v in gt.items() if v is not None},
                    constraint_type=str(row.get("constraint_type") or ""),
                ),
                self.config.task,
            ))
        return tasks

    def _ifeval(self, want: set[str]) -> list[IFEvalRowTask]:
        nltk.download("punkt_tab", quiet=True)
        rows = load_dataset(IFEVAL_DATASET, split=IFEVAL_SPLIT)
        tasks: list[IFEvalRowTask] = []
        for i, row in enumerate(rows):
            name = task_name(row["prompt"])
            if want and name not in want:
                continue
            tasks.append(IFEvalRowTask(
                IFEvalData(
                    idx=i,
                    name=name,
                    system_prompt=SYSTEM,
                    prompt=row["prompt"],
                    key=row["key"],
                    instruction_id_list=row["instruction_id_list"],
                    kwargs=row["kwargs"],
                ),
                self.config.task,
            ))
        return tasks
