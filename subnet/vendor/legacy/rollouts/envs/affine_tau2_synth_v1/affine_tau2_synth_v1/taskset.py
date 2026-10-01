"""affine-tau2-synth-v1: τ²'s orchestrator on the tau2-synth customer-service domains.

Second user-interactive source (env wave 4, 2026-09-18; docs/ask-envs-survey.md
datagen #1). `mikasenghaas/tau2-synth` @ 798589e (MIT) is sierra's τ²-bench
(337326e is an ancestor) plus TEN synthetic domains - library, fitness_gym,
tech_support, auto_repair, online_shopping, travel_agency, vet_clinic
(≈ 10,960 generated tasks: issue combination × persona × variant) and three
dual-control domains where the customer runs tools too - cloud_incident_
response, daily_planner, ev_charging_support (≈ 650). Every policy opens
with "verify the customer's identity before making changes" - the turn-1
ask state - and no domain is on a benchmark card, so the whole pool is usable
(`base` = all tasks; the fork has no split files).

The fork's `tau2` package REPLACES sierra's in the pods' verifiers venv (same
name, superset: added domains, registry entries and a toolkit helper;
telecom / airline / retail code unchanged), so `affine_tau2_v1` keeps
working. Each wrapper points `TAU2_DATA_DIR` at its own bootstrap
(`~/.cache/tau2-synth/data`, fetched from the fork at the pinned revision,
like upstream `tau2_synth`).

Names: `tau2s-<domain>-<τ² task id>` (ids start with `[`, which the eval CLI
would parse as JSON). `solved` = τ²'s reward. Harness: affine_tau2_v1's with
the synth data dir and a generic example-value patch (harness.py).
"""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

SYNTH_DATA_DIR = Path.home() / ".cache" / "tau2-synth" / "data"
os.environ["TAU2_DATA_DIR"] = str(SYNTH_DATA_DIR)   # before any tau2 import

import verifiers.v1 as vf  # noqa: E402
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE  # noqa: E402
from tau2.run import load_tasks  # noqa: E402
from tau2.utils.utils import DATA_DIR  # noqa: E402
from tau2_bench_v1.taskset import Tau2Data  # noqa: E402

from affine_tau2_v1.taskset import AffineTau2Task  # noqa: E402

TAU2_SYNTH_REPOSITORY = "https://github.com/mikasenghaas/tau2-synth.git"
TAU2_SYNTH_REVISION = "798589e02ca91ea61e85557eb672be0a915592eb"
DOMAINS = (
    "library", "fitness_gym", "tech_support", "auto_repair", "online_shopping",
    "travel_agency", "vet_clinic",
    "cloud_incident_response", "daily_planner", "ev_charging_support",
)
NAME_PREFIX = "tau2s-"


def task_name(domain: str, task_id: str) -> str:
    return f"{NAME_PREFIX}{domain}-{task_id}"


def split_name(name: str) -> tuple[str, str]:
    """`tau2s-<domain>-<id>` -> (domain, id)."""
    rest = name[len(NAME_PREFIX):]
    domain, _, task_id = rest.partition("-")
    return domain, task_id


class AffineTau2SynthData(Tau2Data):
    domain: str  # type: ignore[assignment]  - the synth domains are not in tau2_bench_v1's literal


class AffineTau2SynthConfig(vf.TasksetConfig):
    tasks: list[str] = []
    """Task names to load (empty = every task of `domains`)."""
    domains: list[str] = list(DOMAINS)


def bootstrap_data() -> None:
    """Fetch the fork's `data/` once into TAU2_DATA_DIR (file lock + revision marker)."""
    assert str(DATA_DIR) == str(SYNTH_DATA_DIR), (DATA_DIR, SYNTH_DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    marker = DATA_DIR / ".tau2_revision"
    with (DATA_DIR / ".tau2_bootstrap.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (DATA_DIR / "tau2" / "domains").exists() and marker.exists() and marker.read_text() == TAU2_SYNTH_REVISION:
            return
        with tempfile.TemporaryDirectory(prefix="tau2_synth_") as tmp:
            subprocess.run(["git", "init", tmp], check=True, capture_output=True)
            subprocess.run(["git", "-C", tmp, "fetch", "--depth", "1", TAU2_SYNTH_REPOSITORY, TAU2_SYNTH_REVISION],
                           check=True, capture_output=True)
            subprocess.run(["git", "-C", tmp, "checkout", "FETCH_HEAD", "--", "data"], check=True, capture_output=True)
            shutil.copytree(Path(tmp) / "data", DATA_DIR, dirs_exist_ok=True)
            marker.write_text(TAU2_SYNTH_REVISION)


class AffineTau2SynthTaskset(vf.Taskset[AffineTau2Task, AffineTau2SynthConfig]):
    def load(self) -> list[AffineTau2Task]:
        cfg = self.config
        bootstrap_data()
        want = set(cfg.tasks)
        domains = list(cfg.domains)
        if want:
            domains = [d for d in domains if any(split_name(n)[0] == d for n in want)]
        out: list[AffineTau2Task] = []
        for domain in domains:
            if domain not in DOMAINS:
                raise ValueError(f"unknown tau2-synth domain {domain!r}")
            for index, task in enumerate(load_tasks(task_set_name=domain, task_split_name="base")):
                name = task_name(domain, task.id)
                if want and name not in want:
                    continue
                out.append(AffineTau2Task(
                    AffineTau2SynthData(
                        **task.model_dump(exclude={"description"}),
                        idx=index,
                        name=name,
                        description=str(task.description) if task.description else None,
                        prompt=DEFAULT_FIRST_AGENT_MESSAGE.content or "",
                        domain=domain,
                        tau_description=task.description,
                    ),
                    cfg.task,
                ))
        if want and not out:
            raise ValueError(f"no tau2-synth task matched {sorted(want)[:3]}...")
        return out
