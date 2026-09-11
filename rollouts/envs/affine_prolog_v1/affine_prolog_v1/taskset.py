"""affine-prolog-v1: SWI-Prolog constraint solving in a sandbox (any shell harness).

Wrapper over research-environments' `prolog_v1`: nine procedurally generated
problem kinds (sudoku, graph_coloring, zebra, nqueens, cryptarithm,
scheduling, nonogram, bin_packing, hamiltonian) x three difficulties, solved
by editing `/workspace/solution.pl` in a `swipl:latest` container; the
`solved` reward runs `solve/1` and checks the answer against held-out
generator metadata (already a fold key — unchanged here).

What changes: task identity. The base taskset draws every problem from ONE
random stream, so task `k` depends on all tasks before it and a pool cannot
be addressed by name. Here each index has its own stream,
`Random(f"{seed}:{index}")`, kind = `kinds[index % len(kinds)]`, name =
`prolog-<kind>-<index:04d>`; rollouts/catalog.py (`procedural` catalog,
`_prolog_meta`) enumerates the same names without importing this package,
and `--env.taskset.tasks` loads exactly the requested rows (generation cost
is per requested task). Difficulty defaults to `medium` (the inventory's
starting tier; `hard` / `expert` are built to defeat heuristics — raise only
after the teacher solve rate on medium is known).

The base system prompt (capabilities, libraries) carries no dialect word on
purpose: the shell harness's own system prompt supplies the `bash` / `tool`
marker, as on every sandbox source today.
"""

from __future__ import annotations

import json
from random import Random

import verifiers.v1 as vf
from prolog_v1.generators import GENERATORS
from prolog_v1.taskset import (
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_WORKDIR,
    DIFFICULTIES,
    PROLOG_SYSTEM_PROMPT,
    PrologData,
    PrologTask,
    PrologTaskConfig,
)

KINDS = tuple(sorted(GENERATORS))
DEFAULT_SEED = 42
DEFAULT_NUM_EXAMPLES = 3000


def task_name(kind: str, index: int) -> str:
    return f"prolog-{kind}-{index:04d}"


def kind_of(index: int) -> str:
    return KINDS[index % len(KINDS)]


def build_prompt(description: str, path: str) -> str:
    """The base taskset's user prompt, verbatim (prolog_v1 `_build_prompt`)."""
    return (
        f"{description}\n\n"
        f"The problem facts are already in `{path}` together with a `solve/1` "
        f"stub. Implement `solve/1` in that file so the verification query "
        f"produces the answer. Edit the file and run the query to debug:\n\n"
        f'    swipl -g "solve(X), write_canonical(X), nl, halt" -t "halt(1)" {path}\n\n'
        f"The answer is verified against the original problem definition, so do "
        f"not change the problem facts — solve the instance as given. `solve/1` "
        f"must succeed deterministically with exactly one solution."
    )


class PrologConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole pool of `num_examples`)."""
    difficulty: str = "medium"
    num_examples: int = DEFAULT_NUM_EXAMPLES
    seed: int = DEFAULT_SEED
    docker_image: str = DEFAULT_DOCKER_IMAGE
    workdir: str = DEFAULT_WORKDIR
    task_system_prompt: str = PROLOG_SYSTEM_PROMPT
    task: PrologTaskConfig = PrologTaskConfig()


class PrologTaskset(vf.Taskset[PrologTask, PrologConfig]):
    def load(self) -> list[PrologTask]:
        config = self.config
        if config.difficulty not in DIFFICULTIES:
            raise ValueError(f"Unknown difficulty {config.difficulty!r}. "
                             f"Available: {', '.join(DIFFICULTIES)}")
        want = set(config.tasks)
        generators = {k: GENERATORS[k]() for k in KINDS}
        resources = vf.TaskResources(cpu=2, memory=2, disk=5)
        tasks: list[PrologTask] = []
        for index in range(config.num_examples):
            kind = kind_of(index)
            name = task_name(kind, index)
            if want and name not in want:
                continue
            rng = Random(f"{config.seed}:{index}")
            problem = generators[kind].generate(rng, config.difficulty)
            metadata = json.loads(json.dumps(problem.metadata, default=str))
            tasks.append(PrologTask(
                PrologData(
                    idx=index,
                    name=name,
                    prompt=build_prompt(problem.description,
                                        config.task.solution_file_path),
                    system_prompt=config.task_system_prompt,
                    image=config.docker_image,
                    workdir=config.workdir,
                    resources=resources,
                    kind=kind,
                    starter_file=problem.starter_file,
                    expected_answer=problem.expected_answer,
                    metadata=metadata,
                ),
                config.task,
            ))
        return tasks
