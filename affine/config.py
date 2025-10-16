from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


def _float(env_key: str, default: float) -> float:
    try:
        return float(os.getenv(env_key, default))
    except ValueError:
        return default


def _int(env_key: str, default: int) -> int:
    try:
        return int(os.getenv(env_key, default))
    except ValueError:
        return default


def _bool(env_key: str, default: bool) -> bool:
    raw = os.getenv(env_key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", ""}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration sourced from environment variables."""

    netuid: int = _int("AFFINE_NETUID", 0)
    wilson_confidence: float = _float("AFFINE_WILSON_CONFIDENCE", 0.95)
    ratio_decay_seconds: float = _float("AFFINE_RATIO_DECAY_SECONDS", 6 * 3600)
    block_size: int = _int("AFFINE_VALIDATOR_BLOCK", 16)
    chutes_timeout: float = _float("AFFINE_CHUTES_TIMEOUT", 120.0)
    chutes_api_key: str = os.getenv("CHUTES_API_KEY", "")
    chutes_url: str = os.getenv("AFFINE_CHUTES_URL", "http://localhost:8000")
    validator_hotkey: str = os.getenv("AFFINE_VALIDATOR_HOTKEY", "")
    epoch_anchor: str = os.getenv("AFFINE_EPOCH_ANCHOR", "0" * 16)
    epsilon: float = _float("AFFINE_EPSILON", 0.01)
    max_trials: int = _int("AFFINE_MAX_TRIALS", 200)
    bucket_endpoint: str = os.getenv("AFFINE_BUCKET_ENDPOINT", os.getenv("R2_ENDPOINT", ""))
    bucket_name: str = os.getenv("AFFINE_BUCKET_NAME", os.getenv("R2_BUCKET_ID", ""))
    bucket_prefix: str = os.getenv("AFFINE_BUCKET_PREFIX", os.getenv("R2_FOLDER", "affine"))
    bucket_access_key: str = os.getenv("AFFINE_BUCKET_ACCESS_KEY", os.getenv("R2_WRITE_ACCESS_KEY_ID", ""))
    bucket_secret_key: str = os.getenv("AFFINE_BUCKET_SECRET_KEY", os.getenv("R2_WRITE_SECRET_ACCESS_KEY", ""))
    bucket_region: str = os.getenv("AFFINE_BUCKET_REGION", os.getenv("R2_REGION", "auto"))
    bucket_create: bool = _bool("AFFINE_BUCKET_CREATE", True)
    bucket_public_base: str = os.getenv("AFFINE_BUCKET_PUBLIC_BASE", os.getenv("R2_PUBLIC_BASE", ""))
    cache_dir: str = os.getenv("AFFINE_CACHE_DIR", str(Path.home() / ".cache" / "affine" / "blocks"))

    @property
    def ratio_decay_half_life(self) -> timedelta:
        return timedelta(seconds=self.ratio_decay_seconds)

    @property
    def ratio_to_beat(self) -> float:
        return 0.5 + self.epsilon

    @property
    def cache_path(self) -> Path:
        path = Path(self.cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def bucket_write_enabled(self) -> bool:
        return bool(self.bucket_name and self.bucket_access_key and self.bucket_secret_key)

    @property
    def bucket_configured(self) -> bool:
        return bool(self.bucket_name)


settings = Settings()
