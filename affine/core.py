from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

JsonPrimitive = Union[str, int, float, bool, None]
JsonValue = Union[JsonPrimitive, Sequence["JsonValue"], Mapping[str, "JsonValue"]]  # type: ignore

HASH_DIGEST_SIZE = 32


def canonical_bytes(payload: Any) -> bytes:
    """Encode a payload into canonical JSON bytes."""

    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def hash_bytes(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=HASH_DIGEST_SIZE).digest()


def hash_hex(data: bytes) -> str:
    return f"b2:{hash_bytes(data).hex()}"


def merkle_root(leaves: Sequence[bytes]) -> str:
    if not leaves:
        return hash_hex(hash_bytes(b""))
    layer = [hash_bytes(b"\x00" + leaf) for leaf in leaves]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [
            hash_bytes(b"\x01" + layer[i] + layer[i + 1])
            for i in range(0, len(layer), 2)
        ]
    return hash_hex(layer[0])


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _freeze(v) for k, v in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(v) for v in value)
    return value


def _ensure_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"Expected integer-compatible value, got {type(value)!r}")


def canonical_timestamp(dt: datetime | None = None) -> int:
    target = dt or datetime.now(timezone.utc)
    return int(target.timestamp() * 1000)


@dataclass(frozen=True)
class Challenge:
    env_id: str
    challenge_id: str
    info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "challenge_id": self.challenge_id,
            "info": _freeze(self.info),
        }


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": bool(self.ok), "reason": self.reason}


@dataclass
class Sample:
    version: int
    env_id: str
    env_spec_version: int
    challenge_id: str
    validator: str
    miner_id: str
    role: str
    request_id: Optional[str]
    prompt: str
    response: str
    info: Mapping[str, Any]
    ok: bool
    reason: str
    timing_ms: int
    bytes: int
    sample_hash: Optional[str] = None

    def canonical_dict(self) -> Dict[str, Any]:
        return {
            "version": _ensure_int(self.version),
            "env_id": self.env_id,
            "env_spec_version": _ensure_int(self.env_spec_version),
            "challenge_id": self.challenge_id,
            "validator": self.validator,
            "miner_id": self.miner_id,
            "role": self.role,
            "request_id": self.request_id,
            "prompt": self.prompt,
            "response": self.response,
            "info": _freeze(self.info),
            "ok": bool(self.ok),
            "reason": self.reason,
            "timing_ms": _ensure_int(self.timing_ms),
            "bytes": _ensure_int(self.bytes),
        }

    def compute_hash(self) -> str:
        digest = hash_hex(canonical_bytes(self.canonical_dict()))
        self.sample_hash = digest
        return digest

    def to_dict(self) -> Dict[str, Any]:
        payload = self.canonical_dict()
        payload["sample_hash"] = self.sample_hash or self.compute_hash()
        return payload


@dataclass
class BlockHeader:
    version: int
    prev_hash: str
    block_index: int
    timestamp_ms: int
    validator: str
    env_spec_versions: Mapping[str, int]
    sample_count: int
    merkle_root: str
    signature: str = ""

    def canonical_dict(self, include_signature: bool = True) -> Dict[str, Any]:
        payload = {
            "version": _ensure_int(self.version),
            "prev_hash": self.prev_hash,
            "block_index": _ensure_int(self.block_index),
            "timestamp": _ensure_int(self.timestamp_ms),
            "validator": self.validator,
            "env_spec_versions": _freeze(self.env_spec_versions),
            "sample_count": _ensure_int(self.sample_count),
            "merkle_root": self.merkle_root,
        }
        if include_signature:
            payload["signature"] = self.signature
        return payload

    def signing_bytes(self) -> bytes:
        return canonical_bytes(self.canonical_dict(include_signature=False))


@dataclass
class Block:
    header: BlockHeader
    samples: Sequence[Sample | str]

    def sample_hashes(self) -> Tuple[str, ...]:
        hashes = []
        for sample in self.samples:
            if isinstance(sample, Sample):
                hashes.append(sample.sample_hash or sample.compute_hash())
            else:
                hashes.append(sample)
        return tuple(hashes)

    def canonical_dict(self) -> Dict[str, Any]:
        return {
            "version": self.header.version,
            "header": self.header.canonical_dict(),
            "samples": list(self.sample_hashes()),
            "embedded": [s.to_dict() for s in self.samples if isinstance(s, Sample)],
        }


@dataclass(frozen=True)
class DuelResult:
    env_id: str
    outcome: str
    wins: int
    losses: int
    ties: int
    trials: int
    ci: Tuple[float, float]


def challenge_seed(env_id: str, spec_version: int, challenge_id: str) -> int:
    challenge_clean = challenge_id.lower().removeprefix("0x")
    payload = f"{env_id}:{spec_version}:{challenge_clean}".encode()
    return int.from_bytes(hash_bytes(payload)[:8], "little")


def make_rng(env_id: str, spec_version: int, challenge_id: str) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(challenge_seed(env_id, spec_version, challenge_id))
    )


def derive_challenge_id(
    validator_hotkey: str,
    env_id: str,
    counter: int,
    epoch_anchor: str,
    spec_hash: str,
    *,
    size: int = 16,
) -> str:
    payload = (
        f"{validator_hotkey}:{env_id}:{counter}:{epoch_anchor}:{spec_hash}".encode()
    )
    digest = hash_bytes(payload)
    size = max(8, min(size, len(digest)))
    return digest[:size].hex()


__all__ = [
    "Block",
    "BlockHeader",
    "Challenge",
    "DuelResult",
    "JsonPrimitive",
    "JsonValue",
    "Sample",
    "Verdict",
    "canonical_bytes",
    "canonical_timestamp",
    "challenge_seed",
    "derive_challenge_id",
    "hash_bytes",
    "hash_hex",
    "make_rng",
    "merkle_root",
]
