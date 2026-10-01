"""affine-longcot-v1: LongCoT long-horizon reasoning in a sandbox (any shell harness).

Wrapper over research-environments' `longcot_v1` (~2,000 medium + hard
questions in five domains: logic, cs, chemistry, chess, math; each prompt
embeds the whole task; the agent writes its answer to `/workspace/answer.txt`
and upstream `longcot.verify` grades it as a uv script in the container;
reward `correct` — already a fold key). Two changes for the duel corpus:

  * a `tasks: list[str]` selector by question id, so the scheduler addresses
    rows by name (`--env.taskset.tasks`); rollouts/catalog.py (`longcot`
    catalog) lists the same ids through the verifiers interpreter;
  * a real *system* message (the base ships none; the fold drops prefixes
    without one). It carries no dialect word on purpose: the shell harness's
    own prompt supplies the `bash` / `tool` marker, as on every sandbox
    source.

Task identity: `name = question_id` (stable upstream ids such as
`logic_medium_0042`). The base's domain / difficulty / template filters keep
working; sources.toml pins `--env.taskset.difficulty medium` first (the
inventory's estimate for `hard` is under the headroom band).
"""

from __future__ import annotations

import verifiers.v1 as vf
from longcot_v1.taskset import (
    LongCoTConfig,
    LongCoTData,
    LongCoTTask,
    LongCoTTaskset as BaseLongCoTTaskset,
)

SYSTEM = (
    "You are an expert long-horizon reasoner working in a Linux sandbox. "
    "The task below is self-contained: read it fully, plan, and use the "
    "shell to compute, simulate or check anything you are unsure about "
    "(python3 is available). Work step by step and verify intermediate "
    "results. When you have the final answer, write it - and only it, in "
    "exactly the format the question requests - to /workspace/answer.txt, "
    "then print it as your last message."
)


class AffineLongCoTConfig(LongCoTConfig):
    tasks: list[str] = []
    """Question ids to load (empty = every question in the domain / difficulty filter)."""
    task_system_prompt: str = SYSTEM


class AffineLongCoTTaskset(BaseLongCoTTaskset, vf.Taskset[LongCoTTask, AffineLongCoTConfig]):
    def load(self) -> list[LongCoTTask]:
        want = set(self.config.tasks)
        out: list[LongCoTTask] = []
        for task in super().load():
            data: LongCoTData = task.data
            if want and data.question_id not in want:
                continue
            out.append(LongCoTTask(
                data.model_copy(update={
                    "name": data.question_id,
                    "system_prompt": self.config.task_system_prompt,
                }),
                self.config.task,
            ))
        if want and not out:
            raise ValueError(f"no LongCoT question matched {sorted(want)[:5]}...")
        return out
