"""affine-tmax-v1: TMax terminal tasks from a local checkout, images built on the pod.

TMax (research-environments `tmax@2026-07-01`, 14,600 tasks) is the largest
terminal pool on offer. Its task dirs live in the PUBLIC GitHub repo
`PrimeIntellect-ai/prime-tasks` (`datasets/tmax/task_*`), each a Harbor task:
task.toml + instruction.md + tests/ + environment/{Dockerfile, base_install.sh,
post_install.sh, _fixtures}. The upstream `tmax-v1` taskset pins every task to a
prebuilt PRIME image (`prime/primeintellect/tmax:<task>`). Those refs resolve
only inside Prime sandboxes (Prime's compute, which we do not use): they are
not in any docker-pullable registry, with or without a Prime API key
(checked 2026-09-12: `GET /api/v1/images` lists internal `rootfs-cas` paths,
`prime images` has no pull command).

So this wrapper does what terminal_lego does: the pod builds each task's
image from its own Dockerfile (rollouts/runners/verifiers.py
`build_local_images`, `local_docker_build = true` in sources.toml) under the
exact tag the task.toml declares, and the docker runtime finds it locally.
Tasks sharing a base (`[metadata].base_image`) share the apt layer through
Docker's build cache, so only the first build of a base pays the ~5-10 min.

The taskset side is Harbor's parser (`parse_task`) over the local dir, like
terminal_lego_v1; the `solved` reward is HarborTask's own (already a fold
key). `tasks` selects by dir name (`task_000000_c19dda5b`), which is what
rollouts/catalog.py (`tmax` catalog) enumerates from the same checkout.
The checkout root is `TMAX_ROOT` (env) or `~/.cache/affine/prime-tasks`,
a sparse clone of `datasets/tmax` at the registry's pinned commit; the
catalog builder creates it (`ensure_tmax_checkout`) and the taskset only
reads it.
"""

from __future__ import annotations

import os
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor.taskset import (
    HarborConfig,
    HarborTask,
    HarborTaskset,
    parse_task,
)

PRIME_TASKS_REPO = "https://github.com/PrimeIntellect-ai/prime-tasks.git"
# registry.json (research-environments) pins tmax@2026-07-01 to this commit.
PRIME_TASKS_COMMIT = "8b38d35b53271a5f955dfc5dd8197d562cebf46e"
TMAX_SUBDIR = "datasets/tmax"
REQUIRED_FILES = ("task.toml", "instruction.md", "tests/test.sh",
                  "environment/Dockerfile")


def tmax_root() -> Path:
    root = os.environ.get("TMAX_ROOT")
    if root:
        return Path(root).expanduser()
    return Path("~/.cache/affine/prime-tasks").expanduser()


def task_dirs(root: Path) -> list[Path]:
    base = root / TMAX_SUBDIR
    if not base.is_dir():
        raise FileNotFoundError(
            f"tmax checkout missing at {base} (sparse clone of {PRIME_TASKS_REPO} "
            f"@ {PRIME_TASKS_COMMIT}; rollouts/catalog.py ensure_tmax_checkout)")
    return [d for d in sorted(base.iterdir())
            if d.is_dir() and d.name.startswith("task_")
            and all((d / f).is_file() for f in REQUIRED_FILES)]


class TMaxConfig(HarborConfig):
    """Harbor config over the local prime-tasks checkout; the inherited hub
    selectors (`dataset` / `repo` / `registry_*`) are unused."""


class TMaxTask(HarborTask):
    NEEDS_CONTAINER = True


class TMaxTaskset(HarborTaskset, vf.Taskset[TMaxTask, TMaxConfig]):
    def load(self) -> list[TMaxTask]:
        want = set(self.config.tasks or ())
        dirs = [d for d in task_dirs(tmax_root()) if not want or d.name in want]
        if not dirs:
            raise ValueError("no tmax tasks matched")
        tasks: list[TMaxTask] = []
        for idx, task_dir in enumerate(dirs):
            data = parse_task(task_dir, idx, self.config)
            if not data.image:
                raise ValueError(f"{task_dir.name}: task.toml [environment].docker_image missing")
            tasks.append(TMaxTask(data, self.config.task))
        return tasks
