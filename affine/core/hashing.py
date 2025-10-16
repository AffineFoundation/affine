from __future__ import annotations

import hashlib
import json
from typing import Iterable, Sequence

try:
    from blake3 import blake3  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    blake3 = None


def canonical_bytes(payload) -> bytes:
    """Return canonical UTF-8 JSON suitable for hashing."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def hash_bytes(data: bytes) -> bytes:
    """Hash bytes using BLAKE3 when available, otherwise SHA-256."""
    if blake3 is not None:
        return blake3(data).digest()
    return hashlib.sha256(data).digest()


def hash_hex(data: bytes, prefix: str = "b3") -> str:
    """Return prefixed hexadecimal digest."""
    digest = hash_bytes(data)
    tag = "b3" if blake3 is not None else "sha256"
    return f"{tag}:{digest.hex()}"


def merkle_leaf(data: bytes) -> bytes:
    return hash_bytes(b"\x00" + data)


def merkle_parent(left: bytes, right: bytes) -> bytes:
    return hash_bytes(b"\x01" + left + right)


def merkle_root(leaves: Sequence[bytes]) -> str:
    """Compute binary Merkle root, duplicating odd leaf at each layer."""
    if not leaves:
        return hash_hex(hash_bytes(b""))
    layer = [merkle_leaf(leaf) for leaf in leaves]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [merkle_parent(layer[i], layer[i + 1]) for i in range(0, len(layer), 2)]
    return hash_hex(layer[0])


__all__ = ["canonical_bytes", "hash_bytes", "hash_hex", "merkle_root"]
