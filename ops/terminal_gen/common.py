"""Shared bits for the terminal task generator (ops/terminal_gen).

The pipeline turns Stack Exchange Q&A posts into Harbor terminal tasks that
`rollouts/envs/affine_terminal_gen_v1` serves to the datagen pods:

    posts.py -> synth.py -> harbor.py -> validate.py -> decontam.py -> package.py

Every stage reads and writes plain JSONL under `<out>/e<epoch>/` so a stage
can be re-run alone. Task ids are `task_e<epoch>_<sha1(site, qid)[:10]>`;
the epoch is the fold epoch the set is generated for (seed of the post
sample), so sets never collide across refreshes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Iterator

GENERATOR_VERSION = "terminal-gen-v1"
TASK_NAME_PREFIX = "terminal_gen"
IMAGE_PREFIX = "affine/terminal-gen"
LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"

REQUIRED_FILES = ("task.toml", "instruction.md", "tests/test.sh",
                  "tests/test_final_state.py", "environment/Dockerfile",
                  "solution/solve.sh")


def out_dir(root: str | os.PathLike, epoch: int) -> Path:
    p = Path(root).expanduser() / f"e{epoch}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def task_id(epoch: int, site: str, qid: str | int) -> str:
    digest = hashlib.sha1(f"{site}#{qid}".encode()).hexdigest()[:10]
    return f"task_e{epoch}_{digest}"


def open_text(path: str | os.PathLike, mode: str = "rt", *, gz: bool | None = None):
    """gzip when the name ends in .gz (or `gz` says so — the atomic writer's
    temp file is `<name>.gz.tmp`, whose suffix is .tmp)."""
    path = Path(path)
    if gz is None:
        gz = path.suffix == ".gz" or path.name.endswith(".gz.tmp")
    if gz:
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return iter(())
    with open_text(path) as fh:
        return iter([json.loads(line) for line in fh if line.strip()])


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open_text(tmp, "wt", gz=path.suffix == ".gz") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
