"""affine-notool-v1: knowledge questions served WITH unrelated tool schemas.

The BFCL "irrelevance" shape: a plain question plus a handful of tools that
cannot answer it. The right reply is prose (the answer), with no tool call.
Reign-12 audit (docs/king-data-spec.md, improvement-loop P1): the king calls
a tool on ~95 % of such prompts (BFCL live_irrelevance 12.3 -> 5.0 %), the
teacher answers in prose ~82 %. D held no state of this shape.

Items: TriviaQA `rc.wikipedia.nocontext` TRAIN questions (61,888; the
validation split stays out of D — it is affine_trivia's held-out card and
BFCL itself is never used). Every question gets 3-6 distractor tools from
`servers.tools.TOOL_NAMES`, chosen by a hash of the question id, so the
menu varies per task. Task names are `notool-<question_id>`
(rollouts/catalog.py `_notool_meta` writes the same uid).

Grades (fold key `solved` first):
  solved       1.0 iff the answer is correct AND the rollout made no tool call
  correct      alias exact match on the first line of the last reply (metric)
  tool_called  1.0 iff at least one tool was called (metric)

Stops: the null harness ends a rollout when a reply carries no tool call
(`agent_completed`). A rollout that keeps calling tools is cut at
`max_turns` (4) under that name, which the fold reads as a clean failure.

Dialect: policies run this source as `tool_call` (null harness with tool
schemas). A prose final reply then folds as a `text` turn through the
`text_final` rule; a king tool call folds as a `tool_call` turn — the
`king_tooluse` material (fold worker, king-data spec §2.2).
"""

from __future__ import annotations

import hashlib
import random

import verifiers.v1 as vf
from datasets import load_dataset
from triviaqa_v1.taskset import (
    ANSWER_STOP_PATTERN,
    DATASET_NAME,
    DATASET_REVISION,
    normalize_answer,
)

from affine_notool_v1.servers.tools import TOOL_NAMES, NoToolToolset, NoToolToolsetConfig

DATASET_CONFIG = "rc.wikipedia.nocontext"
SPLIT = "train"
MIN_TOOLS, MAX_TOOLS = 3, 6

SYSTEM = (
    "You are a helpful assistant with access to the tools listed here. Use a "
    "tool only when the request actually needs it. A question you can "
    "answer from memory is answered directly, in plain text, with no tool "
    "call. For such a question, think it through first, then reply with the "
    "answer only: the name, term, number or short phrase asked for, on one "
    "line, with no explanation, no full sentence and no punctuation after it."
)


def task_name(question_id: str) -> str:
    return f"notool-{question_id}"


def tools_for(question_id: str) -> list[str]:
    """3-6 distractor tools, a deterministic function of the question id."""
    seed = int.from_bytes(hashlib.sha256(question_id.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    k = rng.randint(MIN_TOOLS, MAX_TOOLS)
    return sorted(rng.sample(TOOL_NAMES, k))


class NoToolData(vf.TaskData):
    answers: list[str]
    """Accepted normalized answer aliases."""
    tools: list[str]
    """Distractor tool names offered for this question."""


class NoToolTaskConfig(vf.TaskConfig):
    max_turns: int = 4
    tools: NoToolToolsetConfig = NoToolToolsetConfig()


class NoToolTask(vf.Task[NoToolData, vf.State, NoToolTaskConfig]):
    @classmethod
    def toolsets(cls, config: NoToolTaskConfig) -> list[vf.Toolset]:
        return [NoToolToolset(config.tools)]

    @vf.stop
    async def max_turns(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= self.config.max_turns

    def _answer_correct(self, trace: vf.Trace) -> bool:
        reply = (trace.last_reply or "").strip()
        if not reply:
            return False
        prediction = ANSWER_STOP_PATTERN.split(reply, maxsplit=1)[0]
        return normalize_answer(prediction) in self.data.answers

    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace) -> float:
        return float(self._answer_correct(trace) and not trace.tool_messages)

    @vf.reward(weight=0.0)
    async def correct(self, trace: vf.Trace) -> float:
        return float(self._answer_correct(trace))

    @vf.reward(weight=0.0)
    async def tool_called(self, trace: vf.Trace) -> float:
        return float(bool(trace.tool_messages))


class NoToolConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names (`notool-<question_id>`) to load; empty = the whole train split."""
    task: NoToolTaskConfig = NoToolTaskConfig()


class NoToolTaskset(vf.Taskset[NoToolTask, NoToolConfig]):
    def load(self) -> list[NoToolTask]:
        want = set(self.config.tasks)
        rows = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT,
                            revision=DATASET_REVISION)
        tasks: list[NoToolTask] = []
        for i, row in enumerate(rows):
            qid = row["question_id"]
            name = task_name(qid)
            if want and name not in want:
                continue
            tools = tools_for(qid)
            task_cfg = self.config.task.model_copy(
                update={"tools": self.config.task.tools.model_copy(update={"enabled": tools})})
            tasks.append(NoToolTask(
                NoToolData(
                    idx=i,
                    name=name,
                    system_prompt=SYSTEM,
                    prompt=row["question"],
                    answers=list(row["answer"]["normalized_aliases"]),
                    tools=tools,
                ),
                task_cfg,
            ))
        return tasks
