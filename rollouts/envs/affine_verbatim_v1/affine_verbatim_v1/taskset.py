"""affine-verbatim-v1: copy a generated text block exactly (single turn, `text` dialect).

Wrapper over prime-envs' `verbatim_copy_v1` (faker-generated words / JSON / CSV /
codes / mixed text, optionally fragmented; the answer inside `<answer>` tags;
`exact_match` reward with Levenshtein similarity as a metric). A copy-fidelity
task for the long-context axis at 1-4k characters. Changes for the duel corpus:

  * per-index seeding: the base draws `num_samples` texts from one stream, so
    task `k` depends on every task before it; here `generate_dataset` is
    called once per requested index with `seed = base_seed + index`, so
    `verbatim-<index:05d>` names a fixed text and `--env.taskset.tasks` loads
    only the requested rows (rollouts/catalog.py `procedural`, `_verbatim_meta`);
  * a real *system* message (the base ships none). The `<answer>` tag is not
    a dialect: the whole visible reply is the action (`text`);
  * the grade lands under `correct` (fold key; the base name is `exact_match`).
"""

from __future__ import annotations

from typing import Literal

import verifiers.v1 as vf
from verbatim_copy_v1.data import generate_dataset
from verbatim_copy_v1.taskset import PROMPT, VerbatimData, VerbatimTask as BaseVerbatimTask

DEFAULT_NUM_SAMPLES = 5000
DEFAULT_SEED = 7
SYSTEM = (
    "You reproduce text exactly. Copy the text between the <text> tags "
    "character for character - no tags, no commentary - inside <answer> and "
    "</answer> tags. Your visible reply is graded as a whole: the content "
    "between the answer tags must match the original exactly."
)


def task_name(index: int) -> str:
    return f"verbatim-{index:05d}"


class VerbatimTask(BaseVerbatimTask):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return await BaseVerbatimTask.exact_match(self, trace)

    async def exact_match(self, trace: vf.Trace) -> float:
        # Undecorated override: the grade is counted once, under `correct`.
        return await self.correct(trace)


class AffineVerbatimConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole pool of `num_samples`)."""
    num_samples: int = DEFAULT_NUM_SAMPLES
    seed: int = DEFAULT_SEED
    content_type: Literal["words", "json", "csv", "codes", "mixed", "all"] = "all"
    target_length: int | None = 1500
    """Characters per text (the base default per type is longer; ~1.5k chars keeps
    the copy under the duel's 1,792-token reference cap)."""
    mean_fragment_length: int | None = None
    task_system_prompt: str = SYSTEM


class VerbatimTaskset(vf.Taskset[VerbatimTask, AffineVerbatimConfig]):
    def load(self) -> list[VerbatimTask]:
        cfg = self.config
        want = set(cfg.tasks)
        tasks: list[VerbatimTask] = []
        for index in range(cfg.num_samples):
            name = task_name(index)
            if want and name not in want:
                continue
            sample = generate_dataset(
                num_samples=1, content_type=cfg.content_type,
                target_length=cfg.target_length,
                mean_fragment_length=cfg.mean_fragment_length,
                seed=cfg.seed + index,
            )[0]
            tasks.append(VerbatimTask(
                VerbatimData(idx=index, name=name, system_prompt=cfg.task_system_prompt,
                             prompt=PROMPT.format(text=sample["text"]), answer=sample["text"]),
                cfg.task,
            ))
        return tasks
