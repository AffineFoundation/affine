"""affine-rgym-v1: reasoning-gym procedural puzzles at a raised curriculum level.

v1 port of the Hub's `primeintellect/reasoning-gym-env` (v0 `ReasoningGymEnv`
over the `reasoning_gym` library: ~100 procedural generators - arithmetic,
algorithmic, logic, games, graphs, geometry, induction, ARC-style grids -
each with its own deterministic `score_answer`). The survey's 6-task teacher
probe solved 6/6 at the library's default settings, so this port generates
every task from the generator's CURRICULUM at a raised level
(`curriculum_level`, default 3 = the fourth rung; a generator with fewer
rungs sits at its top) - the library's own difficulty lever - and keeps
generators without a curriculum at their defaults.

Shape: single turn under the `null` harness, `text` dialect; a real *system*
message asks for the answer inside `<answer>` tags (reasoning-gym's own
DeepSeekZero convention; the scorer gets the tag content).

Task identity: `name = "rgym-<generator>-<index:04d>"`; task `index` of a
generator is `create_dataset`-equivalent with `size=1, seed=base_seed +
index`, so a name is a fixed puzzle (rollouts/catalog.py `rgym` catalog lists
`generators x per_generator` names through the verifiers interpreter).
"""

from __future__ import annotations

import json

import verifiers.v1 as vf
from reasoning_gym.factory import CURRICULA, DATASETS, create_curriculum

DEFAULT_SEED = 1000
DEFAULT_PER_GENERATOR = 60
DEFAULT_LEVEL = 3
# Generators that need external data / heavy deps or whose answers are long
# programs (weak `text` signal) are left out of the default pool.
EXCLUDE = frozenset({"composite", "rush_hour", "sokoban", "mini_sudoku", "sudoku",
                     "futoshiki", "kakurasu", "rubiks_cube", "puzzle24", "emoji_mystery"})
SYSTEM = (
    "You solve puzzles and reasoning problems. Think the problem through "
    "step by step, then give the final answer once, inside <answer> and "
    "</answer> tags, in exactly the format the puzzle asks for. Your visible "
    "reply is graded as a whole."
)


def task_name(generator: str, index: int) -> str:
    return f"rgym-{generator}-{index:04d}"


def default_generators() -> list[str]:
    return sorted(n for n in DATASETS if n not in EXCLUDE)


def extract_answer(text: str) -> str:
    text = text or ""
    if "<answer>" in text:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def make_entry(generator: str, seed: int, level: int) -> tuple[dict, object]:
    """(entry, dataset) for one puzzle of `generator` at curriculum `level`."""
    dataset_cls, config_cls = DATASETS[generator]
    if generator in CURRICULA:
        curriculum = create_curriculum(generator)
        curriculum.set_global_level(level)
        config = curriculum.generate_configuration({"size": 1, "seed": seed})
    else:
        config = config_cls(size=1, seed=seed)
    if hasattr(config, "validate"):
        config.validate()
    dataset = dataset_cls(config=config)
    return dataset[0], dataset


class RGymData(vf.TaskData):
    generator: str
    answer: str
    entry_json: str
    seed: int
    level: int


class RGymTask(vf.Task[RGymData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        entry = json.loads(self.data.entry_json)
        _, dataset = make_entry(self.data.generator, self.data.seed, self.data.level)
        reply = extract_answer(trace.last_reply or "")
        try:
            return float(dataset.score_answer(answer=reply, entry=entry))
        except Exception as exc:  # noqa: BLE001 - a scorer error on a malformed reply is a miss
            trace.info["score_error"] = str(exc)[:300]
            return 0.0


class RGymConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = generators x per_generator)."""
    generators: list[str] = []
    """Generator names (empty = every registered generator minus EXCLUDE)."""
    per_generator: int = DEFAULT_PER_GENERATOR
    seed: int = DEFAULT_SEED
    curriculum_level: int = DEFAULT_LEVEL
    task_system_prompt: str = SYSTEM


class RGymTaskset(vf.Taskset[RGymTask, RGymConfig]):
    def load(self) -> list[RGymTask]:
        cfg = self.config
        want = set(cfg.tasks)
        gens = cfg.generators or default_generators()
        tasks: list[RGymTask] = []
        idx = 0
        for gen in gens:
            for i in range(cfg.per_generator):
                name = task_name(gen, i)
                idx += 1
                if want and name not in want:
                    continue
                seed = cfg.seed + i
                entry, _ = make_entry(gen, seed, cfg.curriculum_level)
                tasks.append(RGymTask(
                    RGymData(idx=idx, name=name, system_prompt=cfg.task_system_prompt,
                             prompt=str(entry["question"]), generator=gen,
                             answer=str(entry.get("answer")), entry_json=json.dumps(entry, default=str),
                             seed=seed, level=cfg.curriculum_level),
                    cfg.task,
                ))
        return tasks
