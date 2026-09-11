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
    PrologTask as BasePrologTask,
    PrologTaskConfig,
)
from prolog_v1.verify import verify_answer

KINDS = tuple(sorted(GENERATORS))
DEFAULT_SEED = 42
DEFAULT_NUM_EXAMPLES = 3000
# Probe 2026-09-11 (medium, bash harness): 5 of 12 teacher rollouts hung
# for 20+ minutes inside one `swipl` call — an exhaustive search on
# nqueens / nonogram / bin_packing with no time limit, the container at
# 100 % CPU, the batch slot held until the rollout timeout. The scorer runs
# solve/1 under `timeout 60`; the agent is told to do the same.
SYSTEM_PROMPT = PROLOG_SYSTEM_PROMPT + (
    "\nAlways run swipl under a time limit, e.g. `timeout 60 swipl ...`: a "
    "query that does not finish within a minute must be reformulated (use "
    "library(clpfd) constraints and labeling instead of generate-and-test), "
    "not waited for. The final verification query is run with a 60 s limit."
)


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


# The agent does not reliably follow the time-limit rule (re-probe: `time
# swipl ...` and `swipl -g "repeat, ..."` pegged 4 containers again), and the
# bash harness's own command timeout is a hard-coded 3600 s. So the sandbox
# enforces it: at setup `swipl` becomes a shim that runs the real binary
# under `timeout`. The scorer's `solve/1` query goes through the same shim
# (its own 60 s wrapper is tighter, so nothing changes for grading).
SWIPL_CMD_TIMEOUT_S = 120
SWIPL_SHIM = (
    'set -e; real=$(command -v swipl); '
    '[ -e "$real.real" ] || mv "$real" "$real.real"; '
    "printf '#!/bin/sh\\nexec timeout " + str(SWIPL_CMD_TIMEOUT_S)
    + ' "%s.real" "$@"\\n\' "$real" > "$real"; '
    'chmod +x "$real"; swipl --version'
)


class PrologTask(BasePrologTask):
    async def setup(self, runtime: vf.Runtime) -> None:
        await super().setup(runtime)
        result = await runtime.run(["bash", "-lc", SWIPL_SHIM], {})
        if result.exit_code != 0:
            raise RuntimeError(f"swipl shim failed: {(result.stderr or '')[-300:]}")

    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """The base reward, minus its `trace.has_error` early return.

        On the pods' verifiers (a298bcf) `Trace.ok` is False until the
        rollout's `finally` block, i.e. throughout scoring, so the base
        guard made every rollout 0.0 without ever running `solve/1`
        (probe 2026-09-11: 12/12 teacher rollouts "failed" with an empty
        `info`). Recorded errors are checked directly instead."""
        if trace.errors:
            return 0.0
        exit_code, clean_output = await self._run_solve(runtime)
        trace.info["solve_exit_code"] = exit_code
        trace.info["solve_output"] = clean_output[-4000:]
        if exit_code != 0:
            trace.info["solution_correct"] = False
            return 0.0
        correct = verify_answer(
            kind=self.data.kind,
            output=clean_output,
            metadata=self.data.metadata,
            expected=self.data.expected_answer,
        )
        trace.info["solution_correct"] = correct
        return 1.0 if correct else 0.0


class PrologConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole pool of `num_examples`)."""
    difficulty: str = "medium"
    num_examples: int = DEFAULT_NUM_EXAMPLES
    seed: int = DEFAULT_SEED
    docker_image: str = DEFAULT_DOCKER_IMAGE
    workdir: str = DEFAULT_WORKDIR
    task_system_prompt: str = SYSTEM_PROMPT
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
