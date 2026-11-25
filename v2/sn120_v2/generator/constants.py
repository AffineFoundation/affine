"""Immutable Generator protocol and scoring constants.

Generator is mechanism 1 of SN120. These values describe the protocol only;
this package never submits weights, writes to a chain, or changes another pool.
"""

from __future__ import annotations

from dataclasses import dataclass

PUBLIC_NAME = "Generator"
PUBLIC_URL = "https://affine.io/generator"
SCHEMA_VERSION = "affine-generator-v1"
NETUID = 120
MECH_ID = 1
EPOCH_SECONDS = 4 * 60 * 60
MAX_SUBMISSION_BYTES = 16 * 1024
MIN_TASKS_PER_SUBMISSION = 1
MAX_TASKS_PER_SUBMISSION = 32

POOL_NAME = "sn120-math-prior"
POOL_VERSION = "v1"
POOL_MANIFEST_SHA256 = "45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115"
POOL_MANIFEST_URL = (
    "https://github.com/AffineFoundation/affine/releases/download/"
    "sn120-math-prior-v1/MANIFEST.json"
)
POOL_ID = f"{POOL_NAME}-{POOL_VERSION}:{POOL_MANIFEST_SHA256}"
POOL_ROWS = 811_609
POOL_COLUMNS = (
    "id",
    "problem",
    "answer",
    "answer_type",
    "source",
    "source_id",
    "license",
    "subsource",
    "topic",
    "difficulty_meta",
    "also_in",
    "in_sn120_pool",
)
POOL_MANIFEST_ENV = "AFFINE_GENERATOR_POOL_MANIFEST_URL"
POOL_CACHE_ENV = "AFFINE_GENERATOR_POOL_CACHE"
MAX_POOL_MANIFEST_BYTES = 1024 * 1024

BETA_PRIOR = (1, 1)  # Beta(1,1), the uniform prior over the task pass rate p
MIDDLE_BAND = (0.25, 0.75)
Q_ACCEPT = 0.8
Q_REJECT = 0.2
FINAL_Q_ACCEPT = 0.5
MIN_SAMPLES = 8
SAMPLE_INCREMENT = 8
MAX_SAMPLES = 32
SAMPLE_COUNTS = tuple(range(MIN_SAMPLES, MAX_SAMPLES + 1, SAMPLE_INCREMENT))


@dataclass(frozen=True, slots=True)
class SamplingSettings:
    """The inference settings used for every Generator sample."""

    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 65_536
    thinking: bool = True


@dataclass(frozen=True, slots=True)
class KingSettings:
    """The king is frozen at the epoch deadline; no FP8 substitute is allowed."""

    frozen_at: str = "deadline"
    dtype: str = "BF16"
    kv_cache_dtype: str = "BF16"
    fp8: bool = False


SAMPLING = SamplingSettings()
KING = KingSettings()

ACCEPT = "accept"
REJECT = "reject"
CONTINUE = "continue"

PAYOUT_SCHEMA_VERSION = "affine-generator-payout-v1"
PAYOUT_WINDOW_SECONDS = 24 * 60 * 60
PAYOUT_MAX_AGE_SECONDS = 6 * 60 * 60
BURN_UID = 0
