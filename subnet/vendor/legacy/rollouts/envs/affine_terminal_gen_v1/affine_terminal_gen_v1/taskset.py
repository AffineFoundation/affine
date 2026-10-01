"""affine-terminal-gen-v1: generated terminal tasks, images built on the pod.

The tasks come from `ops/terminal_gen` (Stack Exchange posts -> LLM task
spec -> Harbor task dir -> Docker validation: pristine image scores 0, the
reference solution scores 1 twice -> 16-gram + id decontamination against
Terminal-Bench 2.0 / 4.0 and tmax). Each fold epoch's set ships inside this
package as `data/e<epoch>/tasks.tar.gz` + `manifest.json` +
`decontam_report.md`; the seed is the epoch, so sets never collide and every
refresh is fresh tasks (RT-6: nothing to memorise across crowns).

The tarball holds plain Harbor task dirs (task.toml + instruction.md +
tests/ + environment/{Dockerfile,_fixtures} + solution/), the exact layout
`affine_tmax_v1` reads, so this file is tmax's wrapper over an extracted
directory: `parse_task` per dir, `solved` = HarborTask's own reward, and the
pod builds each image from its Dockerfile under the declared tag
(`local_docker_build = true` in sources.toml). `solution/` is never mounted
into the agent's container (Harbor ignores it); it exists for validation.

Extraction root: `TERMINAL_GEN_ROOT` (env) or `~/.cache/affine/terminal_gen`,
one sub-directory per epoch. `ensure_extracted` is shared with
rollouts/catalog.py (`build_terminal_gen_catalog`).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor.taskset import (
    HarborConfig,
    HarborTask,
    HarborTaskset,
    parse_task,
)

DATA_ROOT = Path(__file__).resolve().parent / "data"
DEFAULT_EPOCH = 63
REQUIRED_FILES = ("task.toml", "instruction.md", "tests/test.sh",
                  "environment/Dockerfile")


def epoch_dir(epoch: int) -> Path:
    return DATA_ROOT / f"e{epoch}"


def cache_root() -> Path:
    root = os.environ.get("TERMINAL_GEN_ROOT")
    if root:
        return Path(root).expanduser()
    return Path("~/.cache/affine/terminal_gen").expanduser()


def manifest(epoch: int) -> dict:
    path = epoch_dir(epoch) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no terminal_gen set for epoch {epoch}: {path} "
                                f"(ops/terminal_gen/run.sh {epoch})")
    return json.loads(path.read_text())


def ensure_extracted(epoch: int) -> Path:
    """Extract data/e<epoch>/tasks.tar.gz once (keyed by its sha256); return the dir."""
    tar_path = epoch_dir(epoch) / "tasks.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(f"no terminal_gen tarball for epoch {epoch}: {tar_path}")
    sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    dest = cache_root() / f"e{epoch}"
    stamp = dest / ".sha256"
    if stamp.exists() and stamp.read_text().strip() == sha and any(dest.glob("task_*")):
        return dest
    tmp = dest.parent / f".tmp-e{epoch}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError(f"unsafe path in {tar_path}: {member.name}")
        tar.extractall(tmp)
    (tmp / ".sha256").write_text(sha)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return dest


def task_dirs(epoch: int) -> list[Path]:
    root = ensure_extracted(epoch)
    return [d for d in sorted(root.iterdir())
            if d.is_dir() and d.name.startswith("task_")
            and all((d / f).is_file() for f in REQUIRED_FILES)]


class TerminalGenConfig(HarborConfig):
    """Harbor config over the extracted epoch dir; the inherited hub selectors
    (`dataset` / `repo` / `registry_*`) are unused."""

    data_epoch: int = DEFAULT_EPOCH
    """Which generated set to serve (`data/e<epoch>/`)."""


class TerminalGenTask(HarborTask):
    NEEDS_CONTAINER = True


class TerminalGenTaskset(HarborTaskset, vf.Taskset[TerminalGenTask, TerminalGenConfig]):
    def load(self) -> list[TerminalGenTask]:
        want = set(self.config.tasks or ())
        dirs = [d for d in task_dirs(self.config.data_epoch) if not want or d.name in want]
        if not dirs:
            raise ValueError(f"no terminal_gen tasks matched (epoch {self.config.data_epoch})")
        tasks: list[TerminalGenTask] = []
        for idx, task_dir in enumerate(dirs):
            data = parse_task(task_dir, idx, self.config)
            if not data.image:
                raise ValueError(f"{task_dir.name}: task.toml [environment].docker_image missing")
            tasks.append(TerminalGenTask(data, self.config.task))
        return tasks
