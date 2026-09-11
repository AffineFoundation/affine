"""affine-wikispeedia-v1: reach a target Wikipedia article by clicking links.

Wrapper over research-environments' `wikispeedia_v1`: (source, target) pairs
on the SNAP Wikispeedia graph (4,604 articles), a stateful `wiki_click_link`
/ `wiki_go_back` toolset per rollout, episode ends when the target is
reached (a clean "done" signal the env enforces). `tool_call` dialect.

Changes for the duel corpus:

  * the game rules move into a real *system* message that names the tools —
    the fold's `tool_call` marker ("tool") must be in the system message,
    and the base taskset ships none (its rules sit in the user prompt);
  * `solved` = the base `reached_target` (a fold key; the base name is not);
  * names are `wikispeedia-<index:05d>` over the seeded pair sample, so
    rollouts/catalog.py (`procedural`, `_wikispeedia_meta`) enumerates the
    pool without loading the graph; `--env.taskset.tasks` selects rows.

Pairs are sampled with the base `sample_pairs` (one seeded stream; the
graph is loaded host-side once per eval, ~100 MB SNAP tarballs cached under
~/.cache/wikispeedia). Distance band defaults to 4-7 hops: on the 3-7
band the teacher solved 12/12 probe tasks (most random pairs are 3 hops
apart), so the floor moves to 4. NOTE: `num_tasks`, `min_dist`, `max_dist`
and `seed` are part of task identity — changing one re-maps every
`wikispeedia-<index>` name to a different pair (traces keep the actual
source/target, so provenance survives; strata do not care).
"""

from __future__ import annotations

import verifiers.v1 as vf
from wikispeedia_v1.graph import WikiGraph, format_article
from wikispeedia_v1.taskset import (
    WikiData,
    WikiTask,
    WikispeediaTaskConfig,
    sample_pairs,
)

DEFAULT_NUM_TASKS = 4000
DEFAULT_SEED = 0

SYSTEM = (
    "You play Wikispeedia: starting from one Wikipedia article, reach a "
    "target article by following links. Each article ends with `Available "
    "links: ...` — those are the only links you may follow. Use the "
    "`wiki_click_link` tool to move to a linked article and `wiki_go_back` "
    "to undo a move. Call exactly one tool per turn, and before each call "
    "write one or two sentences saying which broader concept you are "
    "steering toward and why. The game ends when you reach the target."
)


def task_name(index: int) -> str:
    return f"wikispeedia-{index:05d}"


class WikispeediaTask(WikiTask):
    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace) -> float:
        return float(any("TARGET REACHED" in (m.content or "")
                         for m in trace.tool_messages))

    async def reached_target(self, trace: vf.Trace) -> float:
        # Undecorated override: the grade is counted once, under `solved`.
        return await self.solved(trace)


class WikispeediaConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole seeded pool)."""
    num_tasks: int = DEFAULT_NUM_TASKS
    min_dist: int = 4
    max_dist: int = 7
    seed: int = DEFAULT_SEED
    task: WikispeediaTaskConfig = WikispeediaTaskConfig()


class WikispeediaTaskset(vf.Taskset[WikispeediaTask, WikispeediaConfig]):
    def load(self) -> list[WikispeediaTask]:
        c = self.config
        want = set(c.tasks)
        wiki = WikiGraph.load(include_text=not c.task.tools.links_only)
        pairs = sample_pairs(wiki, c.num_tasks, c.min_dist, c.max_dist, c.seed)
        tasks: list[WikispeediaTask] = []
        for i, (source, target, dist) in enumerate(pairs):
            name = task_name(i)
            if want and name not in want:
                continue
            tasks.append(WikispeediaTask(
                WikiData(
                    idx=i,
                    name=name,
                    system_prompt=SYSTEM,
                    source=source,
                    target=target,
                    shortest_path=dist,
                    prompt=(
                        f"Your mission: {source} >> {target}\n\n"
                        f"Here is the starting article:\n\n"
                        f"{format_article(wiki, source, c.task.tools.links_only)}"
                    ),
                ),
                c.task,
            ))
        return tasks
