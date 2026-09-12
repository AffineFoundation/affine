"""affine-uuidctf-v1: forensic CTF in a sandbox - find, decode, order five shard UUIDs.

Wrapper over research-environments' `uuid_ctf_v1`: each task synthesizes an
incident filesystem (logs, JSON, CSV, base64 / hex / URL-encoded fragments,
decoy UUIDs) under /workspace; the agent must find the five shard UUIDs,
order them by the evidence, and write the derived recovery UUID as JSON to
the answer file (or its last message). Reward `solved` (a fold key) is exact
match on the derived UUID - deterministic, no judge, no external data.
Fills the games / puzzles / security-forensics gap of D (inventory §4).

Changes for the duel corpus:

  * per-index task names `uuidctf-<index:05d>` (the base generates task `i`
    from `Random(seed + i)`, so index alone determines the task) and a
    `tasks` selector, so only the requested rows are generated;
    rollouts/catalog.py (`procedural`, `_uuidctf_meta`) enumerates the names;
  * a real *system* message (the base ships none; the fold drops prefixes
    without one). No dialect word: the shell harness supplies `bash`/`tool`.

Difficulty defaults to `standard` (the base's calibrated default is the
easier `easy`; the teacher is a 27B model - the probe measures headroom).
"""

from __future__ import annotations

import base64

import verifiers.v1 as vf
from uuid_ctf_v1.taskset import (
    DEFAULT_WORKDIR,
    DIFFICULTY_PRESETS,
    UUIDCTFConfig,
    UUIDCTFData,
    UUIDCTFTask,
    _build_task,
    _make_corpus_tar,
)

DEFAULT_NUM_SAMPLES = 3000
SYSTEM = (
    "You are a digital-forensics analyst working in a Linux sandbox. An "
    "incident corpus is extracted under /workspace (a file manifest is at "
    "/workspace/manifest.txt). Use the shell to search, decode and "
    "cross-reference the evidence; verify each candidate before you commit. "
    "Finish by writing the requested JSON object to the answer file named in "
    "the task and printing it as your last message."
)


def task_name(index: int) -> str:
    return f"uuidctf-{index:05d}"


class AffineUUIDCTFConfig(UUIDCTFConfig):
    tasks: list[str] = []
    """Task names to load (empty = the whole pool of `num_samples`)."""
    num_samples: int = DEFAULT_NUM_SAMPLES
    difficulty: str | None = "standard"
    task_system_prompt: str = SYSTEM


class AffineUUIDCTFTaskset(vf.Taskset[UUIDCTFTask, AffineUUIDCTFConfig]):
    def load(self) -> list[UUIDCTFTask]:
        cfg = self.config
        want = set(cfg.tasks)
        preset = DIFFICULTY_PRESETS[cfg.difficulty or "standard"]
        tasks: list[UUIDCTFTask] = []
        for i in range(cfg.num_samples):
            name = task_name(i)
            if want and name not in want:
                continue
            row = _build_task(
                cfg.seed, i,
                num_noise_files=preset.num_noise_files,
                decoys_per_file=preset.decoys_per_file,
                primary_decoy_counts=preset.primary_decoy_counts,
                guidance_level=preset.guidance_level,
            )
            info = row["info"]
            tar = _make_corpus_tar(info["files"])
            tasks.append(UUIDCTFTask(
                UUIDCTFData(
                    idx=i,
                    name=name,
                    system_prompt=cfg.task_system_prompt,
                    prompt=(f"{info['instruction']}\n\n"
                            "If you cannot write files, output the JSON object as your final message."),
                    result_uuid=info["result_uuid"],
                    source_uuids=info["source_uuids"],
                    corpus_tar_b64=base64.b64encode(tar).decode("ascii"),
                    workdir=DEFAULT_WORKDIR,
                ),
                cfg.task,
            ))
        return tasks
