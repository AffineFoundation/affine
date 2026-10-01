"""affine-wiki-v1: wiki trivia with native tool calls (`tool_call` dialect).

Reuses wiki_search_v1's toolset (search / view sections / read section over
a worker-shared corpus) and question bank, and changes three things the
duel corpus needs:

  * the tool-use instruction is a real *system* message — the teacher's chat
    template renders the tool schemas into that same system block, and the
    fold requires the dialect marker there;
  * the whole question bank (478) instead of the first 20;
  * no judge by default — grading needs a judge-model endpoint the datagen
    box does not have, and a 401 there marks the rollout errored, which the
    duel_turns view then drops. Resolution is telemetry only (Reason v3
    policy), so unjudged rollouts are fine.

Task identity: `name = "wiki-" + sha256(question)[:12]`, computed the same
way in rollouts/catalog.py.
"""

from __future__ import annotations

import hashlib

import verifiers.v1 as vf
from datasets import load_dataset
from pydantic import Field
from wiki_search_v1.servers.wiki import WikiSearchToolset
from wiki_search_v1.taskset import QUESTIONS_DATASET, TriviaTaskData

SYSTEM = (
    "You answer trivia questions with the Wikipedia search tools: "
    "`wiki_search_pages` finds relevant pages, `wiki_view_sections` lists a "
    "page's sections, `wiki_read_section` reads one. Call exactly one tool "
    "per turn, and before each call write one or two sentences saying what "
    "you are looking for and why. When you are confident, reply with a "
    "concise final answer and no tool call."
)


def task_name(question: str) -> str:
    return "wiki-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]


class WikiTaskConfig(vf.TaskConfig):
    judges: vf.Judges = Field(default_factory=list)


class WikiTask(vf.Task[TriviaTaskData, vf.State, WikiTaskConfig]):
    pass


class WikiConfig(vf.TasksetConfig):
    tools: vf.SharedToolsetConfig = vf.SharedToolsetConfig()
    task: WikiTaskConfig = WikiTaskConfig()
    tasks: list[str] = []
    """Task names to load (empty = the whole bank)."""


class WikiTaskset(vf.Taskset[WikiTask, WikiConfig]):
    @classmethod
    def toolsets(cls, config: WikiConfig) -> list[vf.Toolset]:
        return [WikiSearchToolset(config.tools)]

    def load(self) -> list[WikiTask]:
        want = set(self.config.tasks)
        rows = load_dataset(QUESTIONS_DATASET, split="train")
        tasks: list[WikiTask] = []
        for i, row in enumerate(rows):
            name = task_name(row["question"])
            if want and name not in want:
                continue
            tasks.append(WikiTask(
                TriviaTaskData(
                    idx=i,
                    name=name,
                    system_prompt=SYSTEM,
                    prompt=f"Question: {row['question']}",
                    question=row["question"],
                    answer=str(row["answer"]),
                ),
                self.config.task,
            ))
        return tasks
