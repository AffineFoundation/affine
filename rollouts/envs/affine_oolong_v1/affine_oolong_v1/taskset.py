"""affine-oolong-v1: Oolong synthetic long-context questions, answered from a file in the sandbox.

Wrapper over prime-envs' `oolong_synth_v1`: the long context is uploaded to
`/workspace/context.txt`, the agent scans it from the shell / a REPL and
writes a one-token answer to `/workspace/answer.txt` (or says it last);
scored by the official deterministic Oolong rules (`correct`, partial credit
0.75^|diff| for numeric answers - the fold's rule is score >= 1.0). The
optional LLM judge stays OFF. Changes for the duel corpus:

  * `tasks: list[str]` selector by `name = "oolong-<context_len>-<row index:05d>"`
    (rollouts/catalog.py `_oolong_meta` enumerates the same names from the
    HF rows without loading the contexts);
  * a real *system* message (the base ships none); no dialect word - the
    shell harness supplies it;
  * `context_len` defaults to 16,384 tokens (the base default is 262,144;
    the 16k bucket is the largest a 27B teacher reads in a few shell turns).

Data note: `oolongbench/oolong-synth` has `validation` and `test` splits only;
`validation` (1,300 rows across 13 buckets, ~100 per bucket) is the train-ish
supply here and `test` stays out of D.
"""

from __future__ import annotations

import verifiers.v1 as vf
from datasets import load_dataset
from oolong_synth_v1.taskset import (
    INSTRUCTIONS,
    WORKDIR,
    OolongSynthConfig,
    OolongSynthData,
    OolongSynthTask,
)

DATASET = "oolongbench/oolong-synth"
SYSTEM = (
    "You answer questions about a long document that is stored in the sandbox "
    "at /workspace/context.txt. Do not try to read the whole file at once: "
    "search, count and sample it with shell commands or Python, verify your "
    "finding, then write ONLY the final answer (one token, word, date or "
    "label) to /workspace/answer.txt and print it as your last message."
)


def task_name(context_len: int, index: int) -> str:
    return f"oolong-{context_len}-{index:05d}"


class AffineOolongConfig(OolongSynthConfig):
    tasks: list[str] = []
    """Task names to load (empty = every row of the context_len bucket)."""
    context_len: int = 16384
    task_system_prompt: str = SYSTEM


class OolongTaskset(vf.Taskset[OolongSynthTask, AffineOolongConfig]):
    def load(self) -> list[OolongSynthTask]:
        cfg = self.config
        want = set(cfg.tasks)
        column = "context_window_text_with_labels" if cfg.with_labels else "context_window_text"
        rows = load_dataset(DATASET, split=cfg.split, streaming=True)
        tasks: list[OolongSynthTask] = []
        for i, row in enumerate(rows):
            if row.get("context_len") != cfg.context_len:
                continue
            name = task_name(cfg.context_len, i)
            if want and name not in want:
                continue
            answer_type = row.get("answer_type", "")
            if answer_type not in ("ANSWER_TYPE.NUMERIC", "ANSWER_TYPE.DATE"):
                answer_type = ""
            tasks.append(OolongSynthTask(
                OolongSynthData(
                    idx=i, name=name, system_prompt=cfg.task_system_prompt,
                    prompt=f"{row['question']}\n\n{INSTRUCTIONS}",
                    question=row["question"], answer=row["answer"],
                    context=row[column], answer_type=answer_type, workdir=WORKDIR,
                ),
                cfg.task,
            ))
        return tasks
