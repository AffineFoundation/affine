"""Load and validate the declarative registry: sources.toml (what tasks,
what mix) + policies.toml (who plays the seat).

The packaged TOML files are the defaults; ROLLOUTS_SOURCES /
ROLLOUTS_POLICIES point at operator overrides on a pod.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from rollouts.schema import Endpoint, Policy

RUNNERS = ("verifiers", "mini_swe")
CATALOG_KINDS = ("hf", "hf_swebench", "swesmith", "terminal_lego",
                 "terminal_bench_2", "nl2repobench")
SELECT_MODES = ("filter_fn", "tasks")

_PKG_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Source:
    name: str
    group: str
    runner: str
    catalog: str
    policies: tuple[str, ...]
    taskset_id: str = ""
    dataset: str = ""
    split: str = "train"
    uid_field: str = "instance_id"
    row_meta: str = ""
    select: str = "filter_fn"
    pass_dataset: bool = True
    task_id_basename: bool = False
    local_docker_build: bool = False
    swe_eval: bool = False
    swe_namespace: str = "swerebench"   # prebuilt eval image namespace
    share: float = 1.0
    extra_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Registry:
    mix: dict[str, float]
    sources: dict[str, Source]
    policies: dict[str, Policy]

    def target_shares(self) -> dict[str, float]:
        """Per-source target share of kept turns; sums to 1. A source's
        target = its group's mix target x its normalized share within the
        group."""
        by_group: dict[str, float] = {}
        for src in self.sources.values():
            by_group[src.group] = by_group.get(src.group, 0.0) + src.share
        return {
            name: self.mix[src.group] * src.share / by_group[src.group]
            for name, src in self.sources.items()
        }

    def policies_for(self, source: str) -> list[Policy]:
        return [self.policies[p] for p in self.sources[source].policies]


def _load_policies(path: Path) -> dict[str, Policy]:
    raw = tomllib.loads(path.read_text())
    policies: dict[str, Policy] = {}
    for pid, cfg in raw.get("policy", {}).items():
        endpoints = tuple(
            Endpoint(name=e["name"], model=e["model"],
                     base_url=e["base_url"], key_env=e["key_env"],
                     litellm_model=e.get("litellm_model", ""))
            for e in cfg.get("endpoints", ()))
        if not endpoints:
            raise ValueError(f"policy {pid!r} has no endpoints")
        policies[pid] = Policy(
            id=pid, harness=cfg["harness"], endpoints=endpoints,
            sampling=cfg.get("sampling", {}),
            share=float(cfg.get("share", 1.0)))
    if not policies:
        raise ValueError(f"no policies defined in {path}")
    return policies


def _load_sources(path: Path, policies: dict[str, Policy],
                  ) -> tuple[dict[str, float], dict[str, Source]]:
    raw = tomllib.loads(path.read_text())
    mix = {g: float(v) for g, v in raw.get("mix", {}).items()}
    if abs(sum(mix.values()) - 1.0) > 0.01:
        raise ValueError(f"[mix] must sum to 1.0, got {sum(mix.values())}")
    default_policies = tuple(raw.get("defaults", {}).get("policies", ()))

    sources: dict[str, Source] = {}
    for name, cfg in raw.get("source", {}).items():
        pids = tuple(cfg.get("policies", default_policies))
        if not pids:
            raise ValueError(f"source {name!r} has no policies")
        for pid in pids:
            if pid not in policies:
                raise ValueError(f"source {name!r} references unknown "
                                 f"policy {pid!r}")
        src = Source(
            name=name,
            group=cfg["group"],
            runner=cfg["runner"],
            catalog=cfg["catalog"],
            policies=pids,
            taskset_id=cfg.get("taskset_id", ""),
            dataset=cfg.get("dataset", ""),
            split=cfg.get("split", "train"),
            uid_field=cfg.get("uid_field", "instance_id"),
            row_meta=cfg.get("row_meta", ""),
            select=cfg.get("select", "filter_fn"),
            pass_dataset=cfg.get("pass_dataset", True),
            task_id_basename=cfg.get("task_id_basename", False),
            local_docker_build=cfg.get("local_docker_build", False),
            swe_eval=cfg.get("swe_eval", False),
            share=float(cfg.get("share", 1.0)),
            extra_flags=tuple(cfg.get("extra_flags", ())),
        )
        if src.group not in mix:
            raise ValueError(f"source {name!r} group {src.group!r} missing "
                             "from [mix]")
        if src.runner not in RUNNERS:
            raise ValueError(f"source {name!r} runner {src.runner!r} not in "
                             f"{RUNNERS}")
        if src.catalog not in CATALOG_KINDS:
            raise ValueError(f"source {name!r} catalog {src.catalog!r} not "
                             f"in {CATALOG_KINDS}")
        if src.select not in SELECT_MODES:
            raise ValueError(f"source {name!r} select {src.select!r} not in "
                             f"{SELECT_MODES}")
        sources[name] = src
    if not sources:
        raise ValueError(f"no sources defined in {path}")
    return mix, sources


def load_registry(sources_path: Path | None = None,
                  policies_path: Path | None = None) -> Registry:
    sources_path = sources_path or Path(
        os.environ.get("ROLLOUTS_SOURCES", _PKG_DIR / "sources.toml"))
    policies_path = policies_path or Path(
        os.environ.get("ROLLOUTS_POLICIES", _PKG_DIR / "policies.toml"))
    policies = _load_policies(policies_path)
    mix, sources = _load_sources(sources_path, policies)
    return Registry(mix=mix, sources=sources, policies=policies)
