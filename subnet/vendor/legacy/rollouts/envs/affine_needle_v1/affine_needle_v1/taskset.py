"""affine-needle-v1: find the line that breaks the hidden word-order pattern.

Wrapper over research-environments' `patterned_needle_in_haystack_v1`, a
synthetic long-context task: N lines mostly follow a few "haystack" word
patterns, one or more "needle" lines follow another; the model boxes the
needle segment(s). The base `correct` reward (exact boxed match) is a fold
key already and is reused unchanged; the base system prompt (`hint_level`
"moderate") already carries the `\\boxed{}` mandate the fold's `boxed`
marker looks for.

What changes: identity. The base draws all `num_samples` problems from one
random stream; here each index has its own, `Random(f"{seed}:{index}")`,
name = `needle-<index:05d>`, so rollouts/catalog.py (`procedural`,
`_needle_meta`) enumerates the pool without this package and
`--env.taskset.tasks` generates only the requested rows. `num_lines`
defaults to 200 (the base's 50 is a short prompt; 200 lines ~ 8-10k tokens
makes it a real long-context read) with 2 needles.
"""

from __future__ import annotations

from random import Random
from typing import Literal

import verifiers.v1 as vf
from patterned_needle_in_haystack_v1.problem import generate_problem
from patterned_needle_in_haystack_v1.taskset import SYSTEM_PROMPTS, NeedleData, NeedleTask

DEFAULT_SEED = 7
DEFAULT_NUM_SAMPLES = 5000


def task_name(index: int) -> str:
    return f"needle-{index:05d}"


class NeedleConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole pool of `num_samples`)."""
    num_haystack_patterns: int = 5
    num_needles: int = 2
    min_pattern_length: int = 5
    max_pattern_length: int = 5
    min_patterns_per_line: int = 1
    max_patterns_per_line: int = 1
    pattern_separator: str = " | "
    min_haystack_appearances: int = 2
    num_lines: int = 200
    vocab_size: int = 30
    mode: Literal["spaces", "no_spaces", "alphanumeric"] = "spaces"
    hint_level: Literal["none", "minimal", "moderate", "full"] = "moderate"
    num_samples: int = DEFAULT_NUM_SAMPLES
    seed: int = DEFAULT_SEED


class NeedleTaskset(vf.Taskset[NeedleTask, NeedleConfig]):
    def load(self) -> list[NeedleTask]:
        c = self.config
        want = set(c.tasks)
        system_prompt = SYSTEM_PROMPTS[c.hint_level]
        tasks: list[NeedleTask] = []
        for i in range(c.num_samples):
            name = task_name(i)
            if want and name not in want:
                continue
            sample = generate_problem(
                num_haystack_patterns=c.num_haystack_patterns,
                num_needles=c.num_needles,
                min_pattern_length=c.min_pattern_length,
                max_pattern_length=c.max_pattern_length,
                min_patterns_per_line=c.min_patterns_per_line,
                max_patterns_per_line=c.max_patterns_per_line,
                pattern_separator=c.pattern_separator,
                num_lines=c.num_lines,
                vocab_size=c.vocab_size,
                mode=c.mode,
                rng=Random(f"{c.seed}:{i}"),
                min_haystack_appearances=c.min_haystack_appearances,
            )
            tasks.append(NeedleTask(
                NeedleData(idx=i, name=name, prompt=sample["question"],
                           system_prompt=system_prompt, answer=sample["answer"]),
                c.task,
            ))
        return tasks
