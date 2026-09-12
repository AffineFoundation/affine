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
# The pool: reasoning-gym 0.1.25 generators whose curriculum yields a valid,
# fast config at DEFAULT_LEVEL (sweep 2026-09-12 on datagen-2: 91 of 96
# registered generators; excluded - caesar_cipher / knight_swap / survo reject
# their level-3 config, jugs / number_sequence take > 6 s per puzzle, and
# composite is a meta-generator). A static tuple, so the catalog and the
# taskset agree without sweeping the library at every load.
DEFAULT_GENERATORS = (
    "ab", "acre", "advanced_geometry", "aiw", "arc_1d", "arc_agi", "base_conversion",
    "basic_arithmetic", "bf", "binary_alternation", "binary_matrix", "bitwise_arithmetic",
    "boxnet", "calendar_arithmetic", "chain_sum", "circuit_logic", "codeio", "coin_flip",
    "color_cube_rotation", "complex_arithmetic", "count_bits", "count_primes", "countdown",
    "course_schedule", "cryptarithm", "decimal_arithmetic", "decimal_chain_sum", "dice",
    "family_relationships", "figlet_font", "fraction_simplification", "game_of_life",
    "game_of_life_halting", "gcd", "graph_color", "group_anagrams", "gsm_symbolic",
    "intermediate_integration", "isomorphic_strings", "knights_knaves", "largest_island",
    "lcm", "leg_counting", "letter_counting", "letter_jumble", "list_functions",
    "mahjong_puzzle", "manipulate_matrix", "maze", "modulo_grid", "n_queens",
    "needle_haystack", "number_filtering", "number_format", "number_sorting",
    "palindrome_generation", "palindrome_partitioning", "path_star", "polynomial_equations",
    "polynomial_multiplication", "pool_matrix", "power_function", "prime_factorization",
    "products", "propositional_logic", "quantum_lock", "ransom_note", "rearc",
    "rectangle_count", "rotate_matrix", "rotten_oranges", "self_reference",
    "sentence_reordering", "shortest_path", "simple_equations", "simple_geometry",
    "simple_integration", "spell_backward", "spiral_matrix", "string_insertion",
    "string_manipulation", "string_splitting", "string_synthesis", "syllogism",
    "time_intervals", "tower_of_hanoi", "tsumego", "word_ladder", "word_sequence_reversal",
    "word_sorting", "zebra_puzzles",
)
SYSTEM = (
    "You solve puzzles and reasoning problems. Think the problem through "
    "step by step, then give the final answer once, inside <answer> and "
    "</answer> tags, in exactly the format the puzzle asks for. Your visible "
    "reply is graded as a whole."
)


def task_name(generator: str, index: int) -> str:
    return f"rgym-{generator}-{index:04d}"


def default_generators() -> list[str]:
    return [g for g in DEFAULT_GENERATORS if g in DATASETS]


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
    """Generator names (empty = DEFAULT_GENERATORS)."""
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
