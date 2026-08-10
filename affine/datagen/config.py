"""Env-driven configuration for the datagen service.

Secrets (provider API keys, HF_TOKEN) stay in the environment — written to
the machine's .datagen_env by the operator, never read from any config file.
The inference provider pool lives in providers.py (DATAGEN_PROVIDERS).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Placeholder until the AffineFoundation dataset repo is decided.
DEFAULT_HF_REPO = "AffineFoundation/affine-datagen-turns"


@dataclass(frozen=True)
class DatagenConfig:
    temperature: float
    datasets: tuple[tuple[str, str], ...]
    namespace: str
    hf_repo: str
    hf_private: bool
    data_dir: Path
    batch_size: int
    eval_workers: int
    step_limit: int
    cost_limit: float
    seed: int
    agent_timeout_s: int
    eval_timeout_s: int
    upload_min_turns: int
    upload_interval_s: int
    max_batch_retries: int
    prune_images: bool


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "")


def _datasets() -> tuple[tuple[str, str], ...]:
    """Parse DATAGEN_DATASETS: comma-separated "repo[:split]" entries.

    Default split is DATAGEN_SPLIT (`filtered` — the SWE-rebench pool where
    every instance has a prebuilt swerebench/* docker image; the raw `test`
    split is ~69% image-less, and both agent rollout and eval need images).
    """
    raw = (os.environ.get("DATAGEN_DATASETS")
           or os.environ.get("DATAGEN_DATASET", "nebius/SWE-rebench"))
    default_split = os.environ.get("DATAGEN_SPLIT", "filtered")
    out: list[tuple[str, str]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        repo, _, split = entry.partition(":")
        out.append((repo, split or default_split))
    return tuple(out)


def load_config() -> DatagenConfig:
    return DatagenConfig(
        temperature=_float("DATAGEN_TEMPERATURE", 0.7),
        datasets=_datasets(),
        namespace=os.environ.get("DATAGEN_NAMESPACE", "swerebench"),
        hf_repo=os.environ.get("DATAGEN_HF_REPO", DEFAULT_HF_REPO),
        hf_private=_bool("DATAGEN_HF_PRIVATE", True),
        data_dir=Path(os.environ.get("DATAGEN_DATA_DIR", "/root/datagen")),
        batch_size=_int("DATAGEN_BATCH_SIZE", 24),
        # Agent-phase parallelism is per-provider (providers.py concurrency);
        # this only caps swebench eval workers. Eval overlaps the next
        # batch's generation, so leave the rollout containers some headroom.
        eval_workers=_int("DATAGEN_EVAL_WORKERS", 16),
        # Measured live at 50: ~57% of finished trajectories died as
        # LimitsExceeded at the cap (~103 msgs) — pure waste.
        step_limit=_int("DATAGEN_STEP_LIMIT", 100),
        # Inert while litellm cannot price the model (cost_tracking is set to
        # ignore_errors); a working price map makes it a per-instance $ cap.
        cost_limit=_float("DATAGEN_COST_LIMIT", 2.0),
        seed=_int("DATAGEN_SEED", 0),
        # Per provider-group cap. A saturated lane blocked a whole wave for
        # 2h live; requeue-elsewhere beats waiting out a crawling provider.
        agent_timeout_s=_int("DATAGEN_AGENT_TIMEOUT_S", 5400),
        eval_timeout_s=_int("DATAGEN_EVAL_TIMEOUT_S", 3600),
        upload_min_turns=_int("DATAGEN_UPLOAD_MIN_TURNS", 400),
        upload_interval_s=_int("DATAGEN_UPLOAD_INTERVAL_S", 3600),
        max_batch_retries=_int("DATAGEN_MAX_BATCH_RETRIES", 2),
        prune_images=_bool("DATAGEN_PRUNE_IMAGES", True),
    )
