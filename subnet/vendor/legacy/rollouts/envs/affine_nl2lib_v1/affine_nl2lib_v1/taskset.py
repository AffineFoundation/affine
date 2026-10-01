"""affine-nl2lib-v1: implement a Python library from its specification (nl2repo group).

Source: Commit0 (`commit0/commit0`, 54 libraries). Each row points at a
skeleton repository (`commit-0/<name>` at `base_commit`): every public
function and class keeps its signature and docstring, the bodies are removed.
The unit tests are visible; the reference implementation lives in a later
commit of the same repository.

What the sandbox does at setup:
  * clone the skeleton at `base_commit` into /workspace and DROP `.git`
    (the reference solution is reachable from the clone's history — with the
    history gone the agent cannot `git checkout` its way to it);
  * install the row's requirement files and pinned pip packages, then the
    package itself (`pip install -e .`), all best-effort: a skeleton that does
    not install is still a valid task, the agent can fix the packaging;
  * the agent phase then runs with network blocked (`network_block=["*"]`):
    the finished library is on PyPI, and the first probe's teacher simply
    downloaded it and diffed.

Grading (`pytest` pass fraction) is telemetry only, like affine_wiki: it is
capped and never raises — an errored rollout is dropped from the duel corpus,
and the corpus needs the trajectory, not the score.

Task identity: `name = <repo basename>` (e.g. `minitorch`), the same string
rollouts/catalog.py writes as the catalog uid.
"""

from __future__ import annotations

import re
import shlex

import verifiers.v1 as vf
from datasets import load_dataset

DATASET = "commit0/commit0"
SPLIT = "test"
WORKDIR = "/workspace"
GITHUB = "https://github.com"
PYTEST_SUMMARY_RE = re.compile(r"\b(?P<count>\d+)\s+(?P<kind>passed|failed|errors?)\b")

INSTRUCTION = (
    "Implement the `{name}` Python library in {workdir}.\n\n"
    "The repository is a skeleton: every public function and class keeps its "
    "signature and docstring, but the bodies have been removed (they `pass` or "
    "raise NotImplementedError). The original project's documentation is at "
    "{spec}; treat the docstrings and the unit tests under `{test_dir}` as "
    "the authoritative specification.\n\n"
    "Fill in the implementation under `{src_dir}` so that `{test_cmd}` passes "
    "when run from {workdir}. Work incrementally: read the skeleton and the "
    "tests, implement a module, run its tests, fix, move on. Do not modify or "
    "delete the tests. When the tests pass, or you have done what you can, "
    "stop."
)


def task_name(repo: str) -> str:
    return repo.rsplit("/", 1)[-1]


class NL2LibData(vf.TaskData):
    repo: str
    base_commit: str
    src_dir: str
    test_dir: str
    test_cmd: str
    install: str
    packages: list[str]
    pip_packages: list[str]
    pre_install: list[str]


class NL2LibTaskConfig(vf.TaskConfig):
    test_timeout: int = 600
    """Wall-clock budget (seconds) for the guarded pytest run."""
    grade: bool = True
    """Run the tests after the agent stops (telemetry only; never fails the rollout)."""


class NL2LibTask(vf.Task[NL2LibData, vf.State, NL2LibTaskConfig]):
    NEEDS_CONTAINER = True

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        d = self.data
        url = f"{GITHUB}/{d.repo}.git"
        lines = [
            "set -euo pipefail",
            # The runtime shell starts inside WORKDIR; leave it before wiping it.
            "cd /",
            f"rm -rf {shlex.quote(WORKDIR)}",
            f"git clone --quiet {shlex.quote(url)} {shlex.quote(WORKDIR)}",
            f"cd {shlex.quote(WORKDIR)}",
            f"git checkout --quiet {shlex.quote(d.base_commit)}",
            # History holds the reference implementation; drop it.
            "rm -rf .git",
            "git init --quiet && git add -A && git -c user.email=a@b -c user.name=affine commit --quiet -m skeleton || true",
            "python -m pip install --quiet --upgrade pip setuptools wheel >/dev/null 2>&1 || true",
            "python -m pip install --quiet pytest >/dev/null 2>&1 || true",
        ]
        for cmd in d.pre_install:
            lines.append(f"({cmd}) >/dev/null 2>&1 || true")
        for req in d.packages:
            lines.append(f"[ -f {shlex.quote(req)} ] && python -m pip install --quiet -r {shlex.quote(req)} >/dev/null 2>&1 || true")
        if d.pip_packages:
            pkgs = " ".join(shlex.quote(p) for p in d.pip_packages)
            lines.append(f"python -m pip install --quiet {pkgs} >/dev/null 2>&1 || true")
        if d.install:
            lines.append(f"({d.install}) >/dev/null 2>&1 || true")
        result = await runtime.run(["bash", "-lc", "\n".join(lines)], {})
        if result.exit_code != 0:
            output = (result.stdout or "") + (result.stderr or "")
            raise RuntimeError(f"affine-nl2lib setup failed for {d.name}: {output[:1000]}")

    @vf.reward(weight=1.0)
    async def passed_fraction(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        if not self.config.grade:
            return 0.0
        d = self.data
        script = (
            f"cd {shlex.quote(WORKDIR)} && "
            f"timeout {int(self.config.test_timeout)} {d.test_cmd} -q -p no:cacheprovider "
            f"{shlex.quote(d.test_dir)} 2>&1 | tail -n 5"
        )
        try:
            result = await runtime.run(["bash", "-lc", script], {})
            output = (result.stdout or "") + (result.stderr or "")
        except Exception:  # noqa: BLE001 - telemetry must never error the rollout
            return 0.0
        passed = failed = errors = 0
        for line in output.splitlines():
            if " in " not in line:
                continue
            for m in PYTEST_SUMMARY_RE.finditer(line):
                n = int(m.group("count"))
                kind = m.group("kind")
                if kind == "passed":
                    passed = n
                elif kind == "failed":
                    failed = n
                else:
                    errors = n
        total = passed + failed + errors
        trace.record_metrics({
            "nl2lib_passed": float(passed),
            "nl2lib_failed": float(failed),
            "nl2lib_errors": float(errors),
        })
        return passed / total if total else 0.0


class NL2LibConfig(vf.TasksetConfig):
    dataset_name: str = DATASET
    dataset_split: str = SPLIT
    tasks: list[str] = []
    """Task names to load (repo basenames; empty = all 54)."""
    task: NL2LibTaskConfig = NL2LibTaskConfig()


class NL2LibTaskset(vf.Taskset[NL2LibTask, NL2LibConfig]):
    def load(self) -> list[NL2LibTask]:
        want = set(self.config.tasks)
        rows = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        tasks: list[NL2LibTask] = []
        for i, row in enumerate(rows):
            name = task_name(row["repo"])
            if want and name not in want:
                continue
            setup = row.get("setup") or {}
            test = row.get("test") or {}
            python = str(setup.get("python") or "3.11")
            test_dir = str(test.get("test_dir") or "tests/")
            test_cmd = str(test.get("test_cmd") or "pytest")
            src_dir = str(row.get("src_dir") or name)
            tasks.append(NL2LibTask(
                NL2LibData(
                    idx=i,
                    name=name,
                    image=f"python:{python}",
                    workdir=WORKDIR,
                    # Agent phase is framework-only: the reference
                    # implementation is one `pip download <name>` away
                    # (probe 2026-09-07: the teacher fetched wcwidth from
                    # PyPI and diffed against it). Setup runs before the
                    # policy applies, so clone + dependency install still work.
                    network_block=["*"],
                    prompt=INSTRUCTION.format(
                        name=name, workdir=WORKDIR,
                        spec=setup.get("specification") or "the project's public documentation",
                        test_dir=test_dir, test_cmd=test_cmd, src_dir=src_dir),
                    repo=row["repo"],
                    base_commit=row["base_commit"],
                    src_dir=src_dir,
                    test_dir=test_dir,
                    test_cmd=test_cmd,
                    install=str(setup.get("install") or ""),
                    packages=[str(p) for p in (setup.get("packages") or [])],
                    pip_packages=[str(p) for p in (setup.get("pip_packages") or [])],
                    pre_install=[str(p) for p in (setup.get("pre_install") or [])],
                ),
                self.config.task,
            ))
        if not tasks:
            raise ValueError("No Commit0 tasks match the configured filters")
        return tasks


__all__ = ["NL2LibTaskset"]
