from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

from nacl import signing
from nacl.exceptions import BadSignatureError

from ..core.hashing import canonical_bytes, hash_hex, merkle_root
from ..core.types import Block, BlockHeader, Sample, canonical_timestamp


def _leaf_bytes(sample: Sample) -> bytes:
    return canonical_bytes(sample.canonical_dict())


def build_block(
    samples: Sequence[Sample],
    *,
    validator: str,
    block_index: int,
    prev_hash: str,
    env_spec_versions: Mapping[str, int],
    signing_key: signing.SigningKey,
    version: int = 1,
) -> Tuple[Block, str]:
    if not samples:
        raise ValueError("Cannot build block without samples.")
    for sample in samples:
        sample.compute_hash()
    root = merkle_root([_leaf_bytes(sample) for sample in samples])
    header = BlockHeader(
        version=version,
        prev_hash=prev_hash,
        block_index=block_index,
        timestamp_ms=canonical_timestamp(),
        validator=validator,
        env_spec_versions=dict(env_spec_versions),
        sample_count=len(samples),
        merkle_root=root,
    )
    signature = signing_key.sign(header.signing_bytes()).signature
    header.signature = f"ed25519:{signature.hex()}"
    block = Block(header=header, samples=list(samples))
    digest = block_hash(block)
    return block, digest


def block_hash(block: Block) -> str:
    payload = {
        "header": block.header.canonical_dict(),
        "samples": list(block.sample_hashes()),
    }
    return hash_hex(canonical_bytes(payload))


def serialize_block(block: Block) -> str:
    payload = block.canonical_dict()
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_block(data: Mapping[str, object]) -> Block:
    header_data = data["header"]  # type: ignore[index]
    header = BlockHeader(
        version=int(header_data["version"]),
        prev_hash=str(header_data["prev_hash"]),
        block_index=int(header_data["block_index"]),
        timestamp_ms=int(header_data["timestamp"]),
        validator=str(header_data["validator"]),
        env_spec_versions={k: int(v) for k, v in header_data["env_spec_versions"].items()},
        sample_count=int(header_data["sample_count"]),
        merkle_root=str(header_data["merkle_root"]),
        signature=str(header_data.get("signature", "")),
    )
    samples: list[Sample] = []
    embedded = data.get("embedded", [])  # type: ignore[assignment]
    for sample_data in embedded:
        sample = Sample(
            version=int(sample_data["version"]),
            env_id=str(sample_data["env_id"]),
            env_spec_version=int(sample_data["env_spec_version"]),
            challenge_id=str(sample_data["challenge_id"]),
            validator=str(sample_data["validator"]),
            miner_id=str(sample_data["miner_id"]),
            role=str(sample_data["role"]),
            request_id=sample_data.get("request_id"),
            prompt=str(sample_data["prompt"]),
            response=sample_data["response"],
            info=sample_data.get("info", {}),
            ok=bool(sample_data["ok"]),
            reason=str(sample_data.get("reason", "")),
            timing_ms=int(sample_data.get("timing_ms", 0)),
            bytes=int(sample_data.get("bytes", 0)),
            sample_hash=sample_data.get("sample_hash"),
        )
        samples.append(sample)
    return Block(header=header, samples=samples)


def verify_block(block: Block, public_key: signing.VerifyKey) -> bool:
    signature_prefixed = block.header.signature
    if not signature_prefixed.startswith("ed25519:"):
        return False
    signature = bytes.fromhex(signature_prefixed.split(":", 1)[1])
    try:
        public_key.verify(block.header.signing_bytes(), signature)
    except BadSignatureError:
        return False
    expected_root = merkle_root([_leaf_bytes(sample) for sample in block.samples if isinstance(sample, Sample)])
    return expected_root == block.header.merkle_root


__all__ = ["block_hash", "build_block", "load_block", "serialize_block", "verify_block"]
