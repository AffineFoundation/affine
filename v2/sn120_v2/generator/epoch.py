"""Generator's four-hour UTC clock and disjoint object-store layout.

An epoch starts at midnight UTC plus a whole number of four-hour periods. Its
ID names its deadline (and scoring start), not its opening. Thus
``20261120T0000Z`` covers 2026-11-19 20:00:00 inclusive through 2026-11-20
00:00:00 exclusive. At the deadline the king freezes in BF16 and scoring
starts; a timestamp exactly at the boundary belongs to the new epoch.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from ..errors import ContractError
from .constants import EPOCH_SECONDS

UTC = dt.timezone.utc
_EPOCH = re.compile(r"([0-9]{4})([0-9]{2})([0-9]{2})T([0-9]{2})([0-9]{2})Z")
_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _time(value: dt.datetime | int | float) -> dt.datetime:
    if isinstance(value, bool):
        raise ValueError("a boolean is not a timestamp")
    if isinstance(value, (int, float)):
        value = dt.datetime.fromtimestamp(value, UTC)
    if not isinstance(value, dt.datetime):
        raise TypeError("expected a datetime or Unix timestamp")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("a datetime must have an explicit timezone")
    return value.astimezone(UTC)


def epoch_id(value: dt.datetime | int | float | str) -> str:
    """Canonical deadline ID for a timestamp, or validated canonical epoch ID."""
    if isinstance(value, str):
        match = _EPOCH.fullmatch(value)
        if match is None:
            raise ValueError("epoch must have the form YYYYMMDDTHHMMZ")
        year, month, day, hour, minute = map(int, match.groups())
        try:
            stamp = dt.datetime(year, month, day, hour, minute, tzinfo=UTC)
        except ValueError as exc:
            raise ValueError(f"invalid epoch deadline: {exc}") from exc
        if minute or hour % 4 or stamp.timestamp() == 0:
            raise ValueError("epoch deadline must be a nonzero four-hour UTC boundary")
        return stamp.strftime("%Y%m%dT%H%MZ")
    timestamp = _time(value).timestamp()
    deadline = (int(timestamp // EPOCH_SECONDS) + 1) * EPOCH_SECONDS
    return dt.datetime.fromtimestamp(deadline, UTC).strftime("%Y%m%dT%H%MZ")


def epoch_deadline(epoch: str) -> dt.datetime:
    """The deadline/scoring start of a canonical epoch, as aware UTC."""
    canonical = epoch_id(epoch)
    return dt.datetime.strptime(canonical, "%Y%m%dT%H%MZ").replace(tzinfo=UTC)


def epoch_start(epoch: str) -> dt.datetime:
    return epoch_deadline(epoch) - dt.timedelta(seconds=EPOCH_SECONDS)


def epoch_bounds(epoch: str) -> tuple[dt.datetime, dt.datetime]:
    """Half-open ``[start, deadline)`` interval for task submissions."""
    deadline = epoch_deadline(epoch)
    return deadline - dt.timedelta(seconds=EPOCH_SECONDS), deadline


def epoch_contains(epoch: str, timestamp: dt.datetime | int | float) -> bool:
    start, deadline = epoch_bounds(epoch)
    return start <= _time(timestamp) < deadline


def _part(value: Any) -> str:
    if not isinstance(value, str):
        raise ContractError("Generator object key components must be strings")
    text = value
    if not _PART.fullmatch(text) or text in (".", ".."):
        raise ContractError(f"unsafe Generator object key component {text!r}")
    return text


def pool_pointer() -> str:
    """Small manifest pointer, never the 811,609-row reprocessed dataset."""
    return "public/generator/pool"


def link_object(hotkey: str) -> str:
    return f"public/generator/links/{_part(hotkey)}.json"


def submission_object(epoch: str, hotkey: str) -> str:
    return f"private/generator/sub/{epoch_id(epoch)}/{_part(hotkey)}.json"


def epoch_prefix(epoch: str) -> str:
    return f"public/generator/epochs/{epoch_id(epoch)}"


def epoch_manifest(epoch: str) -> str:
    return f"{epoch_prefix(epoch)}/manifest.json"


def epoch_results(epoch: str) -> str:
    return f"{epoch_prefix(epoch)}/results.json"


def accepted_tasks(epoch: str) -> str:
    return f"{epoch_prefix(epoch)}/accepted.jsonl"


def epoch_rollout_prefix(epoch: str) -> str:
    return f"{epoch_prefix(epoch)}/rollouts"


def rollout_object(epoch: str, name: str) -> str:
    name = _part(name)
    if not name.endswith((".json", ".jsonl", ".jsonl.gz")):
        raise ContractError("a rollout object must be JSON or JSONL, optionally gzip-compressed")
    return f"{epoch_rollout_prefix(epoch)}/{name}"


def payout_object() -> str:
    return "public/generator/payout.json"


def state_object() -> str:
    return "state.json"


def layout() -> dict[str, str]:
    """The fixed top-level contract, convenient for independent publishers."""
    return {
        "pool_pointer": pool_pointer(),
        "links": "public/generator/links/<hotkey>.json",
        "submissions": "private/generator/sub/<epoch>/<hotkey>.json",
        "epoch_manifest": "public/generator/epochs/<epoch>/manifest.json",
        "epoch_results": "public/generator/epochs/<epoch>/results.json",
        "accepted": "public/generator/epochs/<epoch>/accepted.jsonl",
        "rollouts": "public/generator/epochs/<epoch>/rollouts/...",
        "payout": payout_object(),
        "state": state_object(),
    }
