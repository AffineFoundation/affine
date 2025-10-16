from __future__ import annotations

import struct
from typing import Optional

import numpy as np

from .hashing import hash_bytes


def challenge_seed(env_id: str, spec_version: int, challenge_id: str) -> int:
    """Derive a deterministic 64-bit seed for the given challenge."""
    challenge_clean = challenge_id.lower().removeprefix("0x")
    data = f"{env_id}:{spec_version}:{challenge_clean}".encode()
    digest = hash_bytes(data)
    return int.from_bytes(digest[:8], "little", signed=False)


def make_rng(env_id: str, spec_version: int, challenge_id: str) -> np.random.Generator:
    """Return a numpy PCG64 generator for the challenge."""
    seed = challenge_seed(env_id, spec_version, challenge_id)
    return np.random.Generator(np.random.PCG64(seed))


def derive_challenge_id(
    validator_hotkey: str,
    env_id: str,
    counter: int,
    epoch_anchor: str,
    spec_hash: str,
    size: int = 16,
) -> str:
    """Generate a reproducible challenge identifier (hex string)."""
    payload = f"{validator_hotkey}:{env_id}:{counter}:{epoch_anchor}:{spec_hash}".encode()
    digest = hash_bytes(payload)
    size = max(8, min(size, len(digest)))
    return digest[:size].hex()


def reseed_sequence(seed: Optional[int], env_id: str, spec_version: int, count: int) -> list[int]:
    """Utility for deterministic sequences of seeds (used for testing)."""
    if seed is None:
        base = np.random.SeedSequence().entropy
    else:
        base = seed
    seeds: list[int] = []
    for ctr in range(count):
        cid = f"{base:016x}{ctr:016x}"
        seeds.append(challenge_seed(env_id, spec_version, cid))
    return seeds


__all__ = ["challenge_seed", "derive_challenge_id", "make_rng", "reseed_sequence"]
