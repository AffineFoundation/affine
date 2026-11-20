"""Generator's once-ever task claims, trailing payout and weight primitives.

These are pure functions. They neither read a chain nor submit a weight
transaction. A caller supplies verified uploads, accepted evidence, the current
UID mapping and an honest freshness timestamp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..canonical import is_sha256
from ..errors import ContractError
from ..payout import check_weights
from .constants import (
    BURN_UID,
    PAYOUT_MAX_AGE_SECONDS,
    PAYOUT_SCHEMA_VERSION,
    PAYOUT_WINDOW_SECONDS,
)
from .protocol import valid_task_id


@dataclass(frozen=True, slots=True)
class Upload:
    """Store-authenticated evidence, not uploader-supplied timestamps or hashes."""

    task_id: str
    uid: int
    hotkey: str
    uploaded_at: float
    file_sha256: str
    valid: bool = True


@dataclass(frozen=True, slots=True)
class Claim:
    """The one permanent owner of an accepted task ID."""

    task_id: str
    uid: int
    hotkey: str
    uploaded_at: float
    file_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "uid": self.uid,
            "hotkey": self.hotkey,
            "uploaded_at": self.uploaded_at,
            "file_sha256": self.file_sha256,
        }


def _timestamp(value: Any, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ContractError(f"{where} must be a finite Unix timestamp")
    return float(value)


def _upload(value: Upload | Mapping[str, Any]) -> Upload:
    if isinstance(value, Upload):
        row = value
    elif isinstance(value, Mapping):
        if set(value) != {"task_id", "uid", "hotkey", "uploaded_at", "file_sha256", "valid"}:
            raise ContractError("invalid Generator upload evidence fields")
        row = Upload(**value)
    else:
        raise ContractError("Generator upload evidence must be an Upload or a complete mapping")
    if not valid_task_id(row.task_id) or type(row.uid) is not int or row.uid < 0:
        raise ContractError("invalid Generator task ID or UID in upload evidence")
    if not isinstance(row.hotkey, str) or not row.hotkey:
        raise ContractError("Generator upload evidence needs an earning hotkey")
    _timestamp(row.uploaded_at, "Generator uploaded_at")
    if not is_sha256(row.file_sha256):
        raise ContractError("Generator file SHA must be lowercase SHA-256 hex")
    if type(row.valid) is not bool:
        raise ContractError("Generator upload validity must be a boolean")
    return row


def winner_key(upload: Upload) -> tuple[float, str]:
    """Earliest authenticated upload; exact timestamp tie -> lower file SHA."""
    return upload.uploaded_at, upload.file_sha256


def first_accepted_claims(
    uploads: Iterable[Upload | Mapping[str, Any]],
    ledger: Mapping[str, Claim | Mapping[str, Any]] | None = None,
) -> dict[str, Claim]:
    """Update an append-only, once-ever task ledger deterministically.

    Every row must already be a valid upload with a passing acceptance. The
    store, not this helper, authenticates its timestamp, file hash, hotkey and
    validity. ``ledger`` contains all earlier claims, including those outside
    the trailing 24-hour payout window; a task can never be paid again.
    """
    claims = dict(ledger or {})
    candidates: dict[str, Upload] = {}
    for item in uploads:
        row = _upload(item)
        if row.task_id in claims:
            continue
        if row.valid and (row.task_id not in candidates or winner_key(row) < winner_key(candidates[row.task_id])):
            candidates[row.task_id] = row
    for task_id, row in candidates.items():
        claims[task_id] = Claim(task_id, row.uid, row.hotkey, row.uploaded_at, row.file_sha256)
    return claims


def trailing_claims(claims: Iterable[Claim], now: float, *, window_s: int = PAYOUT_WINDOW_SECONDS) -> tuple[Claim, ...]:
    """Claims in the trailing 24h, with one deterministic order and unique IDs."""
    now = _timestamp(now, "now")
    if type(window_s) is not int or window_s <= 0:
        raise ContractError("payout window must be a positive integer")
    rows = tuple(claims)
    if len({row.task_id for row in rows}) != len(rows):
        raise ContractError("the once-ever ledger cannot contain the same task twice")
    return tuple(
        sorted(
            (row for row in rows if now - window_s < row.uploaded_at <= now),
            key=lambda row: (row.uploaded_at, row.file_sha256, row.task_id),
        )
    )


def payout_document(
    claims: Iterable[Claim],
    *,
    now: float,
    updated_at: float,
) -> dict[str, Any]:
    """The small, data-only public payout document before UID normalization."""
    now = _timestamp(now, "now")
    updated_at = _timestamp(updated_at, "payout updated_at")
    recent = trailing_claims(claims, now)
    return {
        "version": PAYOUT_SCHEMA_VERSION,
        "updated_at": updated_at,
        "window_seconds": PAYOUT_WINDOW_SECONDS,
        "max_age_seconds": PAYOUT_MAX_AGE_SECONDS,
        "claims": [claim.to_dict() for claim in recent],
    }


def payout_weights(
    claims: Iterable[Claim],
    registered: Mapping[int, str],
    *,
    now: float,
    updated_at: float,
    burn_uid: int = BURN_UID,
) -> dict[int, float]:
    """Equal weight per accepted unique task in the trailing 24 hours.

    The once-ever history must be supplied, not just recent claims. Multiple
    accepted tasks for one UID add their shares. A changed or unregistered earning
    hotkey redirects only that claim's equal share to UID 0; other UIDs are never
    topped up. If no task is payable, or the payout was last updated more than six
    hours ago, all weight goes to UID 0. A stale payout never leaves an all-zero
    table or retains historical winners.
    """
    now = _timestamp(now, "now")
    updated_at = _timestamp(updated_at, "payout updated_at")
    if type(burn_uid) is not int or burn_uid != BURN_UID:
        raise ContractError("Generator burn UID is fixed at 0")
    if updated_at > now or now - updated_at > PAYOUT_MAX_AGE_SECONDS:
        return {BURN_UID: 1.0}
    recent = trailing_claims(claims, now)
    if not recent:
        return {BURN_UID: 1.0}
    shares: dict[int, float] = {BURN_UID: 0.0}
    for claim in recent:
        uid = claim.uid if registered.get(claim.uid) == claim.hotkey and claim.uid != burn_uid else burn_uid
        shares[uid] = shares.get(uid, 0.0) + 1.0 / len(recent)
    total = math.fsum(shares.values())
    if total <= 0:
        return {BURN_UID: 1.0}
    normalized = {uid: share / total for uid, share in shares.items() if share > 0 or uid == BURN_UID and len(shares) == 1}
    residual = 1.0 - math.fsum(normalized.values())
    if residual and BURN_UID in normalized:
        normalized[BURN_UID] = normalized[BURN_UID] + residual
    return check_weights(normalized)


# Names that make their pure scope explicit to other launch components.
select_first_uploads = first_accepted_claims
weights = payout_weights
