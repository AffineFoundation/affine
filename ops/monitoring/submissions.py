"""Read-only, allowlisted submission telemetry; never instantiate validator state."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import re

try:
    from . import common
except ImportError:
    import common


STATE = "affine/state/state.json"
REGISTRATIONS = "affine/state/registrations.json"
SNAPSHOT = "affine/state/public/snapshot.json"
LIMIT = 12
_STATES = ("activated", "queued", "rejected", "crowned", "unknown")
_STATE_DETAIL = {
    "activated": "Upload admission opened; not a duel queue entry.",
    "queued": "Historically admitted; may already be evaluated or in flight.",
    "rejected": "Registration rejected; not a waiting duel.",
    "crowned": "Promoted to public model storage.",
    "unknown": "Missing or unrecognized registration state.",
}


def _load(path):
    try:
        value = common.read_json(path)
    except (OSError, ValueError, UnicodeError):
        return {}, f"{path}: unavailable or invalid JSON"
    if not isinstance(value, dict):
        return {}, f"{path}: expected an object"
    return value, None


def _timestamp(value):
    if not isinstance(value, str) or len(value) > 40:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            return None
        return stamp.astimezone(timezone.utc).isoformat()
    except (ValueError, OverflowError):
        return None


def _challenge(value):
    return value if isinstance(value, str) and re.fullmatch(r"chal-[0-9]{1,12}", value) else "Unknown"


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _sum(rows, key):
    values = [row[key] for row in rows]
    return sum(values) if all(value is not None for value in values) else None


def _display(value):
    return "Unknown" if value is None else value


def _source(path, data, field):
    stamp = _timestamp(data.get(field))
    return f"{path} — source time: {stamp or 'Unknown'}; age: {common.duration(common.age(stamp))}"


def collect():
    """Return queue and registration panels using only local JSON reads."""
    state, state_error = _load(STATE)
    registrations, registration_error = _load(REGISTRATIONS)
    snapshot, snapshot_error = _load(SNAPSHOT)

    raw_queue = state.get("queue")
    queue_ok = isinstance(raw_queue, list) and all(isinstance(row, dict) for row in raw_queue)
    queue_errors = [state_error] if state_error else []
    if not queue_ok and not queue_errors:
        queue_errors.append(f"{STATE}: invalid or missing queue")
    queue = [{
        "challenge_id": _challenge(row.get("challenge_id")),
        "queued_at": _timestamp(row.get("queued_at")),
        "retry_count": _count(row.get("retry_count", 0)),
        "infra_retry_count": _count(row.get("infra_retry_count", 0)),
    } for row in raw_queue] if queue_ok else []
    waits = [common.age(row["queued_at"]) for row in queue]
    oldest = max(waits) if waits and all(wait is not None for wait in waits) else None
    retries = _sum(queue, "retry_count") if queue_ok else None
    infra_retries = _sum(queue, "infra_retry_count") if queue_ok else None
    retried = (sum(row["retry_count"] > 0 or row["infra_retry_count"] > 0 for row in queue)
               if retries is not None and infra_retries is not None else None)
    if queue_ok and (retries is None or infra_retries is None or any(wait is None for wait in waits)):
        queue_errors.append("Some queue timestamps or retry counters are invalid; affected aggregates are Unknown.")
    queue_panel = common.panel(
        "Duel queue", "Persisted waiting entries, separate from historical admission status",
        status="warn" if queue_errors else "ok",
        metrics=[
            common.metric("Waiting", len(queue) if queue_ok else "Unknown", "state.json queue only; excludes in-flight duels"),
            common.metric("Entries with retries", _display(retried)),
            common.metric("Retries", _display(retries), "Sum of retry_count for currently waiting entries"),
            common.metric("Infra retries", _display(infra_retries), "Separate infrastructure counter; not miner retry budget"),
            common.metric("Oldest wait", "None" if queue_ok and not queue else common.duration(oldest), "Since queued_at; resets on requeue/defer"),
        ],
        sections=[common.table(
            f"Queue head (first {LIMIT}, persisted dispatch order)",
            ["Challenge", "Queued at (UTC)", "Wait", "Retries", "Infra retries"],
            [[row["challenge_id"], _display(row["queued_at"]), common.duration(common.age(row["queued_at"])),
              _display(row["retry_count"]), _display(row["infra_retry_count"])] for row in queue[:LIMIT]],
        )],
        bars=[dict(label="Waiting entries with retries", value=retried, max=max(1, len(queue)),
                   detail="Either retry counter is nonzero; denominator is the waiting queue", tone="warn" if retried else "")]
        if retried is not None else [],
        notes=queue_errors + [
            "Only state.json queue represents waiting duels. Registration state 'queued' records past admission, not current queue membership.",
            "Queue order is the persisted canonical dispatch order, not a timestamp sort. queued_at resets on retries/defer, so wait is not total submission age.",
            "Counts exclude in-flight work and lifetime stats.queued. Retry totals cover waiting entries only, not lifetime failures.",
            "Collection reads independent file snapshots; it does not reconcile, dequeue, retry, or mutate production state.",
        ],
        sources=[_source(STATE, state, "flushed_at")],
    )

    raw_records = registrations.get("records")
    records_ok = isinstance(raw_records, dict) and all(isinstance(row, dict) for row in raw_records.values())
    registration_errors = [registration_error] if registration_error else []
    if not records_ok and not registration_errors:
        registration_errors.append(f"{REGISTRATIONS}: invalid or missing records")
    records = []
    if records_ok:
        for row in raw_records.values():
            value = row.get("state")
            records.append({
                "state": value if isinstance(value, str) and value in _STATES else "unknown",
                "challenge_id": _challenge(row.get("challenge_id")),
                "created_at": _timestamp(row.get("created_at")),
                "updated_at": _timestamp(row.get("updated_at")),
            })
    counts = Counter(row["state"] for row in records)
    if counts["unknown"]:
        registration_errors.append("Unrecognized registration states are grouped as unknown.")
    records.sort(key=lambda row: row["updated_at"] or row["created_at"] or "", reverse=True)
    r2 = snapshot.get("submission_r2")
    enabled = r2.get("enabled") if isinstance(r2, dict) else None
    if type(enabled) is not bool:
        enabled = None
    if snapshot_error:
        registration_errors.append(snapshot_error)
    elif enabled is None:
        registration_errors.append(f"{SNAPSHOT}: R2 enabled flag missing or invalid")
    registration_panel = common.panel(
        "Private submissions", "Registration lifecycle history and public R2 admission flag",
        status="warn" if registration_errors else "ok",
        metrics=[
            common.metric("Registrations", len(records) if records_ok else "Unknown", "Stored lifecycle records, not waiting duels"),
            common.metric("R2 enabled", "Unknown" if enabled is None else ("Yes" if enabled else "No"), "Public snapshot flag; not a connectivity or hygiene health check"),
            common.metric("Historically queued", counts["queued"] if records_ok else "Unknown", "Admission history; never used to calculate the duel queue"),
        ],
        sections=[
            common.table("Registration state counts", ["State", "Count", "Meaning"],
                         [[name, counts[name], _STATE_DETAIL[name]] for name in _STATES
                          if name != "unknown" or counts[name]] if records_ok else []),
            common.table(f"Recent registrations (up to {LIMIT}, latest update first)",
                         ["Challenge", "State", "Created at (UTC)", "Updated at (UTC)"],
                         [[row["challenge_id"], row["state"], _display(row["created_at"]), _display(row["updated_at"])]
                          for row in records[:LIMIT]]),
        ],
        bars=[dict(label=name, value=counts[name], max=max(1, len(records)),
                   detail=_STATE_DETAIL[name], tone="warn" if name in ("rejected", "unknown") else "")
              for name in _STATES if name != "unknown" or counts[name]] if records_ok else [],
        notes=registration_errors + [
            "Registration 'queued' is historical: a completed or in-flight challenge can retain it. Consult the Duel queue panel for current waiting work.",
            "Activate: verify the Ed25519 signature, mint prefix-scoped temporary upload access, and seal delivery to the miner hotkey.",
            "Ready: revoke upload access and remove mailbox blobs before verifying the signed manifest, object inventory/sizes, repository hygiene and pinned architecture; only accepted uploads enqueue.",
            "Eval fetch verifies every file SHA-256 before model loading. Only crowned models are copied to public storage.",
            "Miner-caused manifest/signature/hygiene failures burn the submission slot; infrastructure transport faults defer processing for retry. This panel describes the flow, not a fresh verification.",
            "Allowlisted output only: challenge IDs, known states, timestamps, counts and a public boolean. No credentials, token IDs, mailbox keys, private repository URLs, identities or raw failure details are emitted.",
        ],
        sources=[_source(REGISTRATIONS, registrations, "flushed_at"), _source(SNAPSHOT, snapshot, "generated_at")],
    )
    return {"queue": queue_panel, "registrations": registration_panel}


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2, allow_nan=False))
