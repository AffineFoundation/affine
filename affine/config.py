from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta


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

    @property
    def ratio_decay_half_life(self) -> timedelta:
        return timedelta(seconds=self.ratio_decay_seconds)

    @property
    def ratio_to_beat(self) -> float:
        return 0.5 + self.epsilon


settings = Settings()
