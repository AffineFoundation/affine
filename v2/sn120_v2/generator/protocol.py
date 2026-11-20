"""Generator submission schema, canonical identity and Ed25519 signatures.

The signed body is one UTF-8 string, not a JSON representation::

    affine-generator-v1|120|1|<epoch>|<hotkey>|<pool>|sha256(newline.join(task_ids))

``sha256(...)`` denotes the lowercase hexadecimal SHA-256 of that newline
join. ``pool`` is ``sn120-math-prior-v1:<raw manifest SHA-256>``. Task IDs
are the release's original 20-hex IDs, unchanged. No ``mp:`` IDs are minted.

Registration on SN120 and a valid v2 activation are checked at the upload
boundary. This module performs no chain reads or writes.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..canonical import canonical_json, is_sha256, sha256_hex
from ..errors import ConfigError, ContractError
from ..links import activation_message, public_key, valid_activation
from .constants import (
    MAX_SUBMISSION_BYTES,
    MAX_TASKS_PER_SUBMISSION,
    MECH_ID,
    MIN_TASKS_PER_SUBMISSION,
    NETUID,
    POOL_ID,
    SCHEMA_VERSION,
)
from .epoch import epoch_id

_FIELDS = ("version", "netuid", "mech_id", "epoch", "hotkey", "pool", "task_ids", "signature")
_TASK_ID_LENGTH = 20
_HEX = frozenset("0123456789abcdef")


def _reject(reason: str) -> None:
    raise ContractError(f"Generator submission: {reason}")


def valid_task_id(value: Any) -> bool:
    """Whether this is an unmodified ID from the published prior pool."""
    return (
        isinstance(value, str)
        and len(value) == _TASK_ID_LENGTH and _TASK_ID_LENGTH == 20
        and _HEX.issuperset(value)
    )


def task_ids_digest(task_ids: Sequence[str]) -> str:
    """SHA-256 of ``newline.join(task_ids)`` in their submitted order.

    There is no trailing newline. The list is never sorted or deduplicated.
    """
    if not isinstance(task_ids, (list, tuple)):
        _reject("task_ids must be a list")
    for task_id in task_ids:
        if not isinstance(task_id, str):
            _reject("every task ID must be a string")
    return sha256_hex("\n".join(task_ids).encode("utf-8"))


def signing_message(
    epoch: str,
    hotkey: str,
    task_ids: Sequence[str],
    pool: str = POOL_ID,
) -> bytes:
    """The exact bytes a miner signs, using the fixed schema/netuid/mechanism."""
    return (
        f"{SCHEMA_VERSION}|{NETUID}|{MECH_ID}|{epoch}|{hotkey}|{pool}|"
        f"{task_ids_digest(task_ids)}"
    ).encode("utf-8")


def canonical_submission_hash(submission: "Submission | Mapping[str, Any]") -> str:
    """SHA-256 of the whole submission's canonical JSON, including signature."""
    document = submission.to_dict() if isinstance(submission, Submission) else submission
    return sha256_hex(canonical_json(document))


@dataclass(frozen=True, slots=True)
class Submission:
    """One signed Generator task selection. Construction does no I/O."""

    epoch: str
    hotkey: str
    task_ids: tuple[str, ...]
    signature: str
    pool: str = POOL_ID
    version: str = SCHEMA_VERSION
    netuid: int = NETUID
    mech_id: int = MECH_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "netuid": self.netuid,
            "mech_id": self.mech_id,
            "epoch": self.epoch,
            "hotkey": self.hotkey,
            "pool": self.pool,
            "task_ids": list(self.task_ids),
            "signature": self.signature,
        }

    def canonical_json(self) -> bytes:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return canonical_submission_hash(self)

    def signing_message(self) -> bytes:
        return signing_message(self.epoch, self.hotkey, self.task_ids, self.pool)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Submission":
        if not isinstance(data, Mapping):
            _reject("expected a JSON object")
        if set(data) != set(_FIELDS):
            _reject(f"required fields are exactly {', '.join(_FIELDS)}")
        if data["version"] != SCHEMA_VERSION or not isinstance(data["version"], str):
            _reject(f"version must be {SCHEMA_VERSION}")
        if type(data["netuid"]) is not int or data["netuid"] != NETUID:
            _reject(f"netuid must be integer {NETUID}, not a boolean or string")
        if type(data["mech_id"]) is not int or data["mech_id"] != MECH_ID:
            _reject(f"mech_id must be integer {MECH_ID}, not a boolean or string")
        if not isinstance(data["epoch"], str):
            _reject("epoch must be a string")
        if not isinstance(data["hotkey"], str):
            _reject("hotkey must be an SS58 string")
        if data["pool"] != POOL_ID or not isinstance(data["pool"], str):
            _reject(f"pool must be the exact published prior pin {POOL_ID}")
        if not isinstance(data["task_ids"], list):
            _reject("task_ids must be a JSON list")
        task_ids = data["task_ids"]
        if not MIN_TASKS_PER_SUBMISSION <= len(task_ids) <= MAX_TASKS_PER_SUBMISSION:
            _reject("task_ids must contain 1..32 IDs")
        if any(not valid_task_id(task_id) for task_id in task_ids):
            _reject("task IDs must be distinct, original lowercase 20-hex published IDs")
        if len(set(task_ids)) != len(task_ids):
            _reject("task IDs must be distinct")
        if not isinstance(data["signature"], str) or len(data["signature"]) != 128 or not _HEX.issuperset(data["signature"]):
            _reject("signature must be a 64-byte Ed25519 signature as 128 lowercase hex characters")
        return cls(
            version=data["version"],
            netuid=data["netuid"],
            mech_id=data["mech_id"],
            epoch=data["epoch"],
            hotkey=data["hotkey"],
            pool=data["pool"],
            task_ids=tuple(task_ids),
            signature=data["signature"],
        )

    @classmethod
    def from_json(cls, raw: bytes | bytearray | str) -> "Submission":
        """Strictly decode one size-bounded UTF-8 JSON submission."""
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        if not isinstance(raw, (bytes, bytearray)):
            _reject("upload must be bytes or text")
        if not 0 < len(raw) <= MAX_SUBMISSION_BYTES:
            _reject(f"upload must contain 1..{MAX_SUBMISSION_BYTES} bytes")
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            _reject(f"invalid UTF-8 JSON: {exc}")
        submission = cls.from_dict(data)
        try:
            epoch_id(submission.epoch)
        except ValueError as exc:
            _reject(str(exc))
        try:
            public_key(submission.hotkey)
        except ContractError as exc:
            _reject(str(exc))
        return submission

    def verify_signature(self) -> bool:
        """Verify this signed selection against the Ed25519 key in ``hotkey``."""
        return verify_submission(self)


def sign_submission(
    epoch: str,
    hotkey: str,
    task_ids: Sequence[str],
    sign: Callable[[bytes], bytes | bytearray],
    *,
    pool: str = POOL_ID,
    require_activation: bool = True,
) -> Submission:
    """Sign with a wallet-like callable; never read or print a private key.

    ``sign`` receives the exact message and returns the 64-byte signature.
    A Bittensor hotkey's ``sign`` method can be supplied directly.
    """
    if require_activation:
        try:
            signed = bytes(sign(activation_message(hotkey, NETUID)))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"Generator hotkey cannot sign its activation: {exc}") from exc
        text = f"affine2|activate|{hotkey}|{base64.urlsafe_b64encode(signed).decode().rstrip('=')}"
        if not valid_activation(text, hotkey, NETUID):
            raise ContractError("Generator hotkey must have a valid Ed25519 activation")
    message = signing_message(epoch, hotkey, task_ids, pool)
    try:
        signature = bytes(sign(message))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Generator signing failed: {exc}") from exc
    if len(signature) != 64:
        raise ContractError("Generator signer must return a 64-byte Ed25519 signature")
    unsigned = Submission.from_dict(
        {
            "version": SCHEMA_VERSION,
            "netuid": NETUID,
            "mech_id": MECH_ID,
            "epoch": epoch,
            "hotkey": hotkey,
            "pool": pool,
            "task_ids": list(task_ids),
            "signature": "00" * 64,
        }
    )
    return Submission.from_json(canonical_json(unsigned.to_dict() | {"signature": signature.hex()}))


def verify_submission(
    submission: Submission | Mapping[str, Any],
    *,
    registered: bool | None = None,
    activation: str | None = None,
    known_task_ids: Callable[[str], bool] | None = None,
) -> bool:
    """Verify the signature and optionally registration, activation and IDs.

    Registration and activation cannot be inferred from a signature. Supplying
    ``activation`` enforces the existing v2 activation check; leaving it unset
    leaves that lookup to the caller. A submission with an invalid signature is
    always false. Unknown or malformed pool IDs are rejected, never minted.
    """
    try:
        parsed = submission.to_dict() if isinstance(submission, Submission) else submission
        if not isinstance(parsed, Submission):
            parsed = Submission.from_dict(parsed)
        epoch_id(parsed.epoch)
        signature = bytes.fromhex(parsed.signature)
        key = public_key(parsed.hotkey)
    except (ContractError, TypeError, ValueError):
        return False
    if registered is False or (activation is not None and not valid_activation(activation, parsed.hotkey, NETUID)):
        return False
    if known_task_ids is not None and any(not known_task_ids(task_id) for task_id in parsed.task_ids):
        return False
    try:
        from nacl.exceptions import CryptoError
        from nacl.signing import VerifyKey
    except ImportError as exc:
        raise ConfigError("Generator signatures need PyNaCl; install v2's 'links' extra") from exc
    try:
        VerifyKey(key).verify(parsed.signing_message(), signature)
    except (ValueError, CryptoError):
        return False
    return True
