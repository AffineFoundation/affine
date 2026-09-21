"""affine-tau2-gen-v1: tau2-gen tasks (Alan's generator) for the three τ²-bench domains.

Admitted as a D source by operator directive 2026-09-21 10:51 UTC ("admit",
held-out policy (b)): train on airline, retail and telecom; the τ² kingboard
cells become "trained environment"; τ³ banking and Gaia2 stay the clean
held-outs. Data event, no wvk.

Source: https://github.com/catoneone/tau2-gen (MIT; our fork unarbos/tau2-gen),
which keeps τ²-bench's environment (policy, tools, user simulator, scorer) and
regenerates the TASKS and their DATABASES: every task carries a fresh database
in `initial_state.initialization_data` (no more `John Smith` / `C1001` in every
telecom task), ten personas, varied instructions, replay-verified reference
trajectories, and a leakage guard (held-out `(intent, composition, persona)`
triples excluded, identifier intersection with the τ² clone must be empty).

What ships in `data/e<epoch>/`: one `<domain>.json.gz` per domain = the
`scripts/export_affine.py` export (every τ² Task field + our idx / name /
prompt / domain / tau_description), the generator's `manifest.json`,
`meta.jsonl.gz` (case / group / persona / n_writes per task) and Alan's
`fidelity_report.md` (task-level shape vs the benchmark + the leakage result).
`bench_task_ids.json` = the τ² `base` ids of the three domains (50 / 114 /
114) for the fold's decontamination list. The seed is the fold epoch the set
is generated for (`data_epoch`), so a refresh (`ops/tau2gen/refresh.sh`)
gives every fold fresh tasks under a new epoch directory; task names carry
the epoch so pools never collide.

Reward bases as exported: airline `[DB, COMMUNICATE]` with a must-mention
`communicate_info` on every task (a refusal cannot pass on silence); retail
`[DB]` only (τ²'s NL_ASSERTION leg would call an LLM judge -- dropped, and
`nl_assertions` cleared); telecom `[ENV_ASSERTION]` (+ `ACTION` on the
escalate-only tasks), τ²'s own basis. No judge anywhere.

Names: `tau2g-e<epoch>-<domain>-<τ² id>`. Harness: `affine-tau2-synth-v1`
(τ² orchestrator, DeepSeek customer, fold-clean stops, generic example values
on identifier parameters; the fork's `set_state` applies `initialization_data`
and `initialization_actions`).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from affine_tau2_synth_v1.taskset import (  # sets TAU2_DATA_DIR before any tau2 import
    AffineTau2SynthConfig,
    AffineTau2SynthData,
    bootstrap_data,
)
import verifiers.v1 as vf  # noqa: E402

from affine_tau2_v1.taskset import AffineTau2Task  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parent / "data"
DOMAINS = ("airline", "retail", "telecom")
DEFAULT_EPOCH = 61
NAME_PREFIX = "tau2g-"


def task_name(epoch: int, domain: str, task_id: str) -> str:
    return f"{NAME_PREFIX}e{epoch}-{domain}-{task_id}"


def split_name(name: str) -> tuple[int, str, str]:
    """`tau2g-e61-airline-<id>` -> (61, "airline", "<id>")."""
    rest = name[len(NAME_PREFIX):]
    epoch, _, rest = rest.partition("-")
    domain, _, task_id = rest.partition("-")
    return int(epoch.lstrip("e")), domain, task_id


def epoch_dir(epoch: int) -> Path:
    return DATA_ROOT / f"e{epoch}"


def load_export(epoch: int, domain: str) -> dict:
    path = epoch_dir(epoch) / f"{domain}.json.gz"
    if not path.exists():
        raise FileNotFoundError(f"no tau2-gen export for epoch {epoch} / {domain}: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def load_meta(epoch: int, domain: str) -> dict[str, dict]:
    path = epoch_dir(epoch) / f"{domain}.meta.jsonl.gz"
    if not path.exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {r["id"]: r for r in rows}


class AffineTau2GenConfig(AffineTau2SynthConfig):
    domains: list[str] = list(DOMAINS)
    data_epoch: int = DEFAULT_EPOCH
    """Which generated set to serve (`data/e<epoch>/`)."""


class AffineTau2GenTaskset(vf.Taskset[AffineTau2Task, AffineTau2GenConfig]):
    def load(self) -> list[AffineTau2Task]:
        cfg = self.config
        bootstrap_data()  # the fork's data dir holds the airline / retail / telecom domain code + db
        want = set(cfg.tasks)
        domains = list(cfg.domains)
        if want:
            wanted = {split_name(n)[1] for n in want}
            domains = [d for d in domains if d in wanted]
        out: list[AffineTau2Task] = []
        for domain in domains:
            if domain not in DOMAINS:
                raise ValueError(f"unknown tau2-gen domain {domain!r}")
            export = load_export(cfg.data_epoch, domain)
            for index, rec in enumerate(export["tasks"]):
                name = task_name(cfg.data_epoch, domain, rec["id"])
                if want and name not in want:
                    continue
                data = {k: v for k, v in rec.items() if k not in ("idx", "name", "prompt", "domain", "tau_description", "description", "system_prompt")}
                out.append(AffineTau2Task(
                    AffineTau2SynthData(
                        **data,
                        idx=index,
                        name=name,
                        description=rec.get("description"),
                        prompt=rec.get("prompt") or "Hi! How can I help you today?",
                        domain=domain,
                        tau_description=rec.get("tau_description"),
                    ),
                    cfg.task,
                ))
        if want and not out:
            raise ValueError(f"no tau2-gen task matched {sorted(want)[:3]}...")
        return out
