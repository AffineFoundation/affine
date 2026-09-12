"""affine-i3code-v1: INTELLECT-3-RL code, single turn, hidden tests run in the rollout runtime.

Wrapper over prime-envs' `i3_code_v1` (8,579 competitive-programming problems;
the model writes one Python solution in a fenced block; `verify.py` runs the
problem's stdin/stdout or functional tests as a uv script IN the rollout's
runtime). The chat-shaped coding family D lacks - every coding turn in D today
is an agent in a sandbox. Changes for the duel corpus:

  * a real *system* message (the base puts its one-line instruction in the
    user turn; the fold drops prefixes without a system message). The
    dialect is `text`: a ```python fence is not a registered dialect, so the
    whole visible reply is the action;
  * `tasks: list[str]` selector by `name = "i3code-" + sha256(question)[:12]`
    (rollouts/catalog.py `_i3_code_meta` computes the same string);
  * the grade lands under `solved` (the base reward is `passed`; kept as the
    undecorated method so it is counted once);
  * the difficulty band [0.1, 0.9] on the shipped `avg@8_qwen3_4b_instruct_2507`
    column is the default (same rule as affine_logic / affine_science).

Safety: the hidden tests execute the model's code in the RUNTIME. Run this
source with the docker runtime (`runner = "verifiers"`), never with
`verifiers_chat` (subprocess = the pod host).
"""

from __future__ import annotations

import hashlib
import json

import verifiers.v1 as vf
from datasets import load_dataset
from i3_code_v1.taskset import (
    DATASET_NAME,
    DATASET_SPLIT,
    DATASET_SUBSET,
    Filter,
    I3CodeConfig,
    I3CodeData,
    I3CodeTask as BaseI3CodeTask,
)

DIFFICULTY_MIN = 0.1
DIFFICULTY_MAX = 0.9
SYSTEM = (
    "You solve competitive-programming tasks in Python. Think the problem "
    "through, then give exactly one complete solution in a single ```python "
    "code block: read from standard input and write to standard output "
    "unless the task names a function to implement. Your visible reply is "
    "graded as a whole by running the hidden tests on the last code block."
)


def task_name(question: str) -> str:
    return "i3code-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]


class I3CodeTask(BaseI3CodeTask):
    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        return await BaseI3CodeTask.passed(self, trace, runtime)

    async def passed(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        # Undecorated override: the grade is counted once, under `solved`.
        return await self.solved(trace, runtime)


class AffineI3CodeConfig(I3CodeConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole filtered split)."""
    filter: Filter = Filter(min=DIFFICULTY_MIN, max=DIFFICULTY_MAX)
    task_system_prompt: str = SYSTEM


class I3CodeTaskset(vf.Taskset[I3CodeTask, AffineI3CodeConfig]):
    def load(self) -> list[I3CodeTask]:
        cfg = self.config
        want = set(cfg.tasks)
        flt = cfg.filter
        rows = load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)
        tasks: list[I3CodeTask] = []
        for i, row in enumerate(rows):
            question = row["question"]
            name = task_name(question)
            if want and name not in want:
                continue
            if flt.column is not None:
                value = row.get(flt.column)
                if value is None or not (flt.min <= float(value) <= flt.max):
                    continue
            info = json.loads(row["info"])
            tests = json.loads(info["tests"])
            inputs, outputs = tests["inputs"], tests["outputs"]
            limit = cfg.max_num_tests if cfg.max_num_tests is not None else len(inputs)
            indices = range(min(limit, len(inputs)))
            tasks.append(I3CodeTask(
                I3CodeData(
                    idx=i,
                    name=name,
                    system_prompt=cfg.task_system_prompt,
                    prompt=question,
                    test_case_inputs=[json.dumps(inputs[j]) for j in indices],
                    test_case_outputs=[json.dumps(outputs[j]) for j in indices],
                    fn_name=tests.get("fn_name"),
                    source=info.get("source"),
                ),
                cfg.task,
            ))
        return tasks
