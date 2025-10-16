from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from .hashing import canonical_bytes, hash_bytes

JsonPrimitive = Union[str, int, float, bool, None]
JsonValue = Union[JsonPrimitive, Sequence["JsonValue"], Mapping[str, "JsonValue"]]  # type: ignore


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
    """Return unix epoch milliseconds for ``dt`` (default now in UTC)."""
    target = dt or datetime.now(timezone.utc)
    return int(target.timestamp() * 1000)


@dataclass(frozen=True)
class Challenge:
    """Canonical metadata for a deterministic environment challenge."""

    env_id: str
    challenge_id: str
    info: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "challenge_id": self.challenge_id,
            "info": _freeze(self.info),
        }


@dataclass(frozen=True)
class Verdict:
    """Result of verifying a miner response."""

    ok: bool
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": bool(self.ok), "reason": self.reason}


@dataclass
class Sample:
    """Ground-truth evidence for a duel trial."""

    version: int
    env_id: str
    env_spec_version: int
    challenge_id: str
    validator: str
    miner_id: str
    role: str  # "contender" | "champion"
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
        payload: Dict[str, Any] = {
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
        return payload

    def compute_hash(self) -> str:
        blob = canonical_bytes(self.canonical_dict())
        digest = hash_bytes(blob)
        encoded = digest.hex()
        self.sample_hash = encoded
        return encoded

    def to_dict(self) -> Dict[str, Any]:
        payload = self.canonical_dict()
        payload["sample_hash"] = self.sample_hash or self.compute_hash()
        return payload


@dataclass
class BlockHeader:
    """Signed header anchoring a batch of samples."""

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
        payload: Dict[str, Any] = {
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
    """Hash-chained batch of samples."""

    header: BlockHeader
    samples: Sequence[Sample | str]

    def sample_hashes(self) -> Tuple[str, ...]:
        hashes: list[str] = []
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
            "embedded": [
                s.to_dict() for s in self.samples if isinstance(s, Sample)
            ],
        }

    def payload_bytes(self) -> bytes:
        return canonical_bytes(
            {
                "header": self.header.canonical_dict(),
                "samples": list(self.sample_hashes()),
            }
        )


@dataclass(frozen=True)
class DuelResult:
    """Summary of a single-environment duel."""

    env_id: str
    outcome: str  # "contender" | "champion" | "inconclusive"
    wins: int
    losses: int
    ties: int
    trials: int
    ci: Tuple[float, float]


__all__ = [
    "Block",
    "BlockHeader",
    "Challenge",
    "DuelResult",
    "Sample",
    "Verdict",
    "canonical_timestamp",
]
