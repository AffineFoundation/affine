"""Fail-closed validation of the one reviewed dedup journal before old code runs.

This is not a second controller. It validates complete historical params, checks
charged duplicate/null-update accounting against round_closed, and rejects work
that only new code may finish. Journal records are never rewritten.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from .errors import JournalCorrupt, V2Error
from .journal import JournalRecord
from .params import NEW_PARAMS_ONLY_FIELDS, RELEASE_FIELDS, REVIEWED_NEW_PARAMS_HASHES, Params
from .state import LEASE_FAILED, ROUND_CLOSED, ROUND_OPENED, ROUND_SETTLED, State


def _bad(message: str) -> None:
    raise JournalCorrupt(f"dedup rollback compatibility: {message}")


def historical_params(data: dict) -> Params:
    """Validate the complete old/reviewed-new pin and return old behavior."""
    if not isinstance(data, dict):
        _bad("params_set does not contain an object")
    if set(data) & (set(RELEASE_FIELDS) - {"z_bar"} | NEW_PARAMS_ONLY_FIELDS):
        params_hash = digest_params(data)
        if params_hash not in REVIEWED_NEW_PARAMS_HASHES:
            _bad("historical params differ from the reviewed dedup pin")
        recipe = data.get("recipe")
        from .params import REVIEWED_NEW_CODE_DIGESTS
        if not isinstance(recipe, dict) or recipe.get("code_digest") != REVIEWED_NEW_CODE_DIGESTS[params_hash]:
            _bad("historical params do not pin the reviewed combined worker recipe")
    return Params.from_dict(data)


def digest_params(data: dict) -> str:
    from .canonical import digest
    return digest(data)


def require_closed_dedup_history(records: Iterable[JournalRecord], state: State) -> None:
    """Validate new-history accounting and refuse new-code-only open work.

    Call after read-only replay, before constructing a Journal writer. Old code
    never finishes an open dedup round: stripping historical dedup fields would
    change the ticket Params digest. Finish even a decided round with new-code
    ``run`` before rollback. Closed rounds need no ticket reconstruction.
    """
    for row in records:
        if row.kind == "params_set":
            historical_params(row.data.get("params"))
    if state.round is not None and state.round.params_hash in REVIEWED_NEW_PARAMS_HASHES:
        stage = "decided" if state.round.decision is not None else "undecided"
        raise V2Error(
            f"rollback blocked: {state.round.round_id} is {stage} under dedup params; "
            "finish it with reviewed new-code run before rollback"
        )
    opened = {}
    settlements = {}
    failures = {}
    closes = {}
    for row in records:
        if row.kind == ROUND_OPENED:
            record = row.data["round"]
            if record.get("params_hash") in REVIEWED_NEW_PARAMS_HASHES:
                if record["round_id"] in opened:
                    _bad(f"{record['round_id']} opens twice")
                opened[record["round_id"]] = record
        elif row.kind == ROUND_SETTLED:
            round_id = row.data.get("round_id")
            if round_id in opened:
                if round_id in settlements:
                    _bad(f"{round_id} settles twice")
                decision = row.data.get("decision", {})
                _validate_settlement(opened[round_id], decision, row.data.get("results"))
                settlements[round_id] = decision
        elif row.kind == LEASE_FAILED:
            round_id = row.data.get("round_id")
            if round_id in opened:
                if row.data.get("reason", "").startswith("InfraError:"):
                    failures.setdefault(round_id, {})[row.data.get("lease_id")] = row.data.get("reason")
        elif row.kind == ROUND_CLOSED:
            round_id = row.data.get("round_id")
            if round_id in opened:
                record = opened[round_id]
                # All-duplicate rounds carry their complete decision in
                # round_opened and never issue a lease or separate settlement.
                if round_id not in settlements and isinstance(record.get("decision"), dict):
                    _validate_settlement(record, record["decision"], None)
                    settlements[round_id] = record["decision"]
                if round_id in closes:
                    _bad(f"{round_id} closes without exactly one valid decision")
                if round_id not in settlements:
                    failed = failures.get(round_id, {})
                    lease_ids = {lease["lease_id"] for lease in record["leases"]}
                    if not (record.get("decision") is None and not record.get("duplicate_decisions")
                            and lease_ids and lease_ids == set(failed)
                            and row.data.get("outcome") == "no_decision"
                            and row.data.get("reason") in failed.values()
                            and all(not row.data.get(name) for name in ("served", "front", "attempted", "held", "released"))):
                        _bad(f"{round_id} closes without exactly one valid decision")
                closes[round_id] = row.data
    for round_id, record in opened.items():
        if round_id not in closes:
            # A decided-but-not-closed round must also finish with new code.
            continue
        entries = settlements.get(round_id, {}).get("entries", [])
        outcome = closes[round_id].get("outcome")
        expected_served = sorted(
            (entry["uid"], entry["registration_block"], entry["hotkey"])
            for entry in entries if entry.get("status") in ("winner", "lost", "duplicate", "null_update")
        ) if outcome != "abandoned" else []
        actual = sorted((int(uid), int(block), hotkey) for uid, block, hotkey in closes[round_id]["served"])
        if actual != expected_served:
            _bad(f"{round_id} charged-seat accounting differs: expected {expected_served}, got {actual}")
        positions = [entry["queue_pos"] for entry in entries]
        if len(positions) != len(set(positions)):
            _bad(f"{round_id} has duplicate decision queue positions")


def _validate_settlement(record: dict, decision: dict, results: object) -> None:
    entries = decision.get("entries")
    if not isinstance(entries, list):
        _bad(f"{record['round_id']} has no decision entry list")
    stage_a = record.get("duplicate_decisions") or []
    by_pos = {entry.get("queue_pos"): entry for entry in entries if isinstance(entry, dict)}
    for duplicate in stage_a:
        settled = by_pos.get(duplicate.get("queue_pos"))
        if settled != duplicate:
            _bad(f"{record['round_id']} changed or omitted an open-time duplicate decision")
    if not results and stage_a:
        allowed = {entry["queue_pos"] for entry in stage_a}
        for lease in record.get("leases", []):
            allowed.update(entry["queue_pos"] for entry in lease["entries"])
        if set(by_pos) != allowed or any(by_pos[pos]["status"] != "no_decision" for pos in set(by_pos) - {e["queue_pos"] for e in stage_a}):
            _bad(f"{record['round_id']} no-result close does not preserve duplicate charges and genuine survivors")
    charged = [entry for entry in entries if entry.get("status") in ("winner", "lost", "duplicate", "null_update")]
    counts = Counter((entry.get("uid"), entry.get("registration_block")) for entry in charged)
    if any(count != 1 for count in counts.values()):
        _bad(f"{record['round_id']} charges one seat more than once")
