"""Env-driven runtime configuration (paths, budgets, upload cadence).

What to generate lives in sources.toml / policies.toml (declarative,
versioned); how hard to push the box lives here (per-pod knobs). Secrets
(provider API keys, HF_TOKEN) stay in the environment — never in any file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "")


@dataclass(frozen=True)
class RolloutsConfig:
    data_dir: Path            # store + index + catalogs + state + outbox
    verifiers_dir: Path       # verifiers checkout `uv run eval` runs from
    turns_hf_repo: str        # staging dataset the fold consumes
    traces_hf_repo: str       # trace-chunk mirror (system of record backup)
    batch_size: int
    # Single capacity budget: rollout containers across BOTH runners.
    # Replaces the two hand-tuned knobs (LANE_CONCURRENCY vs mini-swe
    # workers); the supervisor runs one batch at a time, so this is the
    # in-batch parallelism cap.
    max_containers: int
    max_turns: int            # verifiers agent max turns
    step_limit: int           # mini-swe agent step limit
    cost_limit: float         # mini-swe per-instance $ cap (litellm)
    rollout_timeout_s: int
    batch_timeout_s: int
    eval_workers: int         # swebench telemetry eval parallelism
    eval_timeout_s: int
    upload_min_turns: int
    upload_interval_s: int
    prune_images: bool
    seed: int
    langs: frozenset[str] | None   # None = all languages

    @property
    def catalog_dir(self) -> Path:
        return self.data_dir / "catalogs"

    @property
    def store_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.jsonl"

    @property
    def pending_turns(self) -> Path:
        return self.data_dir / "pending_turns.jsonl"


def load_config() -> RolloutsConfig:
    raw_langs = os.environ.get("ROLLOUTS_LANGS", "all").strip().lower()
    langs = None if raw_langs in ("all", "*") else frozenset(
        s.strip() for s in raw_langs.split(",") if s.strip())
    return RolloutsConfig(
        data_dir=Path(os.environ.get("ROLLOUTS_DATA_DIR", "/root/rollouts-data")),
        verifiers_dir=Path(os.environ.get(
            "ROLLOUTS_VERIFIERS_DIR", "/root/prime-pilot/verifiers")),
        turns_hf_repo=os.environ.get(
            "ROLLOUTS_TURNS_HF_REPO", "unconst/affine-datagen-turns"),
        traces_hf_repo=os.environ.get(
            "ROLLOUTS_TRACES_HF_REPO", "unconst/affine-rollout-traces"),
        batch_size=_int("ROLLOUTS_BATCH_SIZE", 10),
        max_containers=_int("ROLLOUTS_MAX_CONTAINERS", 24),
        max_turns=_int("ROLLOUTS_MAX_TURNS", 80),
        step_limit=_int("ROLLOUTS_STEP_LIMIT", 100),
        cost_limit=float(os.environ.get("ROLLOUTS_COST_LIMIT", 2.0)),
        rollout_timeout_s=_int("ROLLOUTS_ROLLOUT_TIMEOUT_S", 3600),
        batch_timeout_s=_int("ROLLOUTS_BATCH_TIMEOUT_S", 7200),
        eval_workers=_int("ROLLOUTS_EVAL_WORKERS", 16),
        eval_timeout_s=_int("ROLLOUTS_EVAL_TIMEOUT_S", 3600),
        upload_min_turns=_int("ROLLOUTS_UPLOAD_MIN_TURNS", 400),
        upload_interval_s=_int("ROLLOUTS_UPLOAD_INTERVAL_S", 3600),
        prune_images=_bool("ROLLOUTS_PRUNE_IMAGES", True),
        seed=_int("ROLLOUTS_SEED", 0),
        langs=langs,
    )
