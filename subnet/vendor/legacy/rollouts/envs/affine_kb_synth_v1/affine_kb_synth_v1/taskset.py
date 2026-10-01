"""affine-kb-synth-v1: the tau2-synth customer-service domains with the policy
in a knowledge base (docs/env-targets-for-lagging-axes.md §7.1, env wave 5).

Same tasks, data dir, customer simulator and rewards as `affine_tau2_synth_v1`
(mikasenghaas/tau2-synth @ 798589e, ten domains, ~11.6k tasks, none on a
kingboard card). What changes is the AGENT'S prompt: the domain policy is not
in the system prompt; it is chunked into documents behind τ³'s `KB_search`
tool (see kb.py), and the prompt carries τ³'s header ("do not make up
policies; all instructions are in the knowledge base").

Why (tau3-banking read, 2026-09-20): the teacher searches the knowledge base
before its first substantive answer on 97 % of τ³ tasks; the king does so on
34 % and answers policy from memory. D held no state where a customer-service
agent must retrieve policy before acting. τ³ itself has one domain (banking,
97 tasks) and all of it is on the card, so the state is reproduced on the
synth domains instead -- nothing of the banking data is used.

Names: `tau2k-<domain>-<τ² task id>` (a distinct prefix from `tau2s-`, so the
same task can exist in both sources without a name collision).
"""

from __future__ import annotations

import verifiers.v1 as vf
from affine_tau2_synth_v1.taskset import (  # sets TAU2_DATA_DIR before any tau2 import
    DOMAINS,
    AffineTau2SynthConfig,
    AffineTau2SynthData,
    AffineTau2SynthTaskset,
    bootstrap_data,
)
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.run import load_tasks

from affine_tau2_v1.taskset import AffineTau2Task

NAME_PREFIX = "tau2k-"


def task_name(domain: str, task_id: str) -> str:
    return f"{NAME_PREFIX}{domain}-{task_id}"


def split_name(name: str) -> tuple[str, str]:
    rest = name[len(NAME_PREFIX):]
    domain, _, task_id = rest.partition("-")
    return domain, task_id


class AffineKBSynthConfig(AffineTau2SynthConfig):
    pass


class AffineKBSynthTaskset(AffineTau2SynthTaskset, vf.Taskset[AffineTau2Task, AffineKBSynthConfig]):
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
            raise ValueError(f"no tau2-kb task matched {sorted(want)[:3]}...")
        return out
