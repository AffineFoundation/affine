"""Env-driven runtime configuration (paths, budgets, publish targets).

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


def _shard(name: str) -> tuple[int, int]:
    """`i/N` -> (i, N); unset = the whole pool (1 pod). Fail-closed on a
    malformed value: two pods silently sharing a shard would double-roll
    the same tasks."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return (0, 1)
    try:
        i_s, n_s = raw.split("/")
        i, n = int(i_s), int(n_s)
    except ValueError:
        raise SystemExit(f"{name}={raw!r}: expected i/N (e.g. 1/3)") from None
    if n < 1 or not 0 <= i < n:
        raise SystemExit(f"{name}={raw!r}: need 0 <= i < N")
    return (i, n)


@dataclass(frozen=True)
class RolloutsConfig:
    data_dir: Path            # store + index + catalogs + state
    verifiers_dir: Path       # verifiers checkout `uv run eval` runs from
    # Canonical trace publish: the public corpus bucket (data.affine.io),
    # `traces/` prefix. The fold derives D from these on the validator box.
    r2_bucket: str
    r2_endpoint: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_prefix: str            # traces/ in production; staging/traces/ for dry runs
    traces_hf_repo: str       # HF cold copy of the chunks (secondary)
    hf_trace_mirror: bool     # keep the HF cold copy running
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
    prune_images: bool
    seed: int
    langs: frozenset[str] | None   # None = all languages
    # Task partition for a fleet of pods: this pod owns the tasks whose
    # blake2b(uid) % N == i. Deterministic in the uid alone (not the pool
    # order), so pods agree on ownership without talking to each other.
    shard: tuple[int, int]         # (i, N); (0, 1) = single pod

    @property
    def catalog_dir(self) -> Path:
        return self.data_dir / "catalogs"

    @property
    def store_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.jsonl"



def load_config() -> RolloutsConfig:
    raw_langs = os.environ.get("ROLLOUTS_LANGS", "all").strip().lower()
    langs = None if raw_langs in ("all", "*") else frozenset(
        s.strip() for s in raw_langs.split(",") if s.strip())
    return RolloutsConfig(
        data_dir=Path(os.environ.get("ROLLOUTS_DATA_DIR", "/root/rollouts-data")),
        verifiers_dir=Path(os.environ.get(
            "ROLLOUTS_VERIFIERS_DIR", "/root/prime-pilot/verifiers")),
        r2_bucket=os.environ.get("ROLLOUTS_R2_BUCKET", "affine-data"),
        r2_endpoint=os.environ.get("ROLLOUTS_R2_ENDPOINT", "").rstrip("/"),
        r2_access_key_id=os.environ.get("ROLLOUTS_R2_ACCESS_KEY_ID", ""),
        r2_secret_access_key=os.environ.get("ROLLOUTS_R2_SECRET_ACCESS_KEY", ""),
        r2_prefix=os.environ.get("ROLLOUTS_R2_PREFIX", "traces/"),
        traces_hf_repo=os.environ.get(
            "ROLLOUTS_TRACES_HF_REPO", "unconst/affine-rollout-traces"),
        hf_trace_mirror=_bool("ROLLOUTS_HF_TRACE_MIRROR", True),
        batch_size=_int("ROLLOUTS_BATCH_SIZE", 10),
        max_containers=_int("ROLLOUTS_MAX_CONTAINERS", 24),
        max_turns=_int("ROLLOUTS_MAX_TURNS", 80),
        step_limit=_int("ROLLOUTS_STEP_LIMIT", 100),
        cost_limit=float(os.environ.get("ROLLOUTS_COST_LIMIT", 2.0)),
        rollout_timeout_s=_int("ROLLOUTS_ROLLOUT_TIMEOUT_S", 3600),
        batch_timeout_s=_int("ROLLOUTS_BATCH_TIMEOUT_S", 7200),
        eval_workers=_int("ROLLOUTS_EVAL_WORKERS", 16),
        eval_timeout_s=_int("ROLLOUTS_EVAL_TIMEOUT_S", 3600),
        prune_images=_bool("ROLLOUTS_PRUNE_IMAGES", True),
        seed=_int("ROLLOUTS_SEED", 0),
        langs=langs,
        shard=_shard("ROLLOUTS_SHARD"),
    )
