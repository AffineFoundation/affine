"""Read-only validator and weight telemetry; no production objects or RPCs."""
from __future__ import annotations

import json
import math
import time
import tomllib
from datetime import datetime, timezone

try:
    from . import common
except ImportError:
    import common

STATE = "affine/state/state.json"
SNAPSHOT = "affine/state/public/snapshot.json"
CONTRACT = "affine/affine.toml"
_PHASES = frozenset(("boot", "tick", "process_challenge", "duel")) | frozenset(
    f"{action}_{role}_machine"
    for action in ("provisioning", "bootstrapping") for role in ("eval", "bench", "chat")
)


def _load(path):
    try:
        if path == CONTRACT:
            value = tomllib.loads((common.ROOT / path).read_text())
        else:
            value = common.read_json(path)
    except (OSError, ValueError, UnicodeError):
        return {}, f"{path}: unavailable or invalid"
    return (value, None) if isinstance(value, dict) else ({}, f"{path}: expected an object")


def _dict(value):
    return value if isinstance(value, dict) else {}


def _number(value):
    if type(value) not in (int, float):
        return None
    try:
        return value if math.isfinite(value) and 0 <= value <= 10**15 else None
    except OverflowError:
        return None


def _count(value):
    return value if type(value) is int and 0 <= value <= 10**15 else None


def _positive(value):
    value = _number(value)
    return value if value is not None and value >= 0.001 else None


def _display(value):
    return "Unknown" if value is None else value


def _rows(value):
    return value if isinstance(value, list) and all(isinstance(row, dict) for row in value) else None


def _timestamp(value, now):
    if not isinstance(value, str) or len(value) > 40:
        return None, None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            return None, None
        age = now - stamp.timestamp()
        if age < -60:
            return None, None
        return stamp.astimezone(timezone.utc).isoformat(), max(0, age)
    except (ValueError, OverflowError, OSError):
        return None, None


def collect():
    """Project only bounded numbers, normalized timestamps and known phase names."""
    state, state_error = _load(STATE)
    snapshot, snapshot_error = _load(SNAPSHOT)
    contract, contract_error = _load(CONTRACT)
    errors = [e for e in (state_error, snapshot_error, contract_error) if e]
    settings = _dict(contract.get("validator"))
    subnet = _dict(contract.get("subnet"))
    now = time.time()
    flushed_at, state_age = _timestamp(state.get("flushed_at"), now)
    generated_at, snapshot_age = _timestamp(snapshot.get("generated_at"), now)
    poll = _positive(settings.get("poll_interval_s"))
    warn = _positive(settings.get("tick_warn_after_s"))
    publish = _positive(settings.get("dashboard_flush_min_interval_s"))
    stale_after = max(warn, 3 * poll, 3 * publish) if all(v is not None for v in (poll, warn, publish)) else None
    freshness = ("Unknown" if stale_after is None or state_age is None or snapshot_age is None
                 else "Stale" if max(state_age, snapshot_age) > stale_after else "Fresh")
    phase = _dict(snapshot.get("phase"))
    phase_name = phase.get("name")
    phase_name = phase_name if isinstance(phase_name, str) and phase_name in _PHASES else "Unknown"
    _, phase_age = _timestamp(phase.get("since"), now)
    queue = _rows(state.get("queue"))
    benches = _rows(state.get("bench_jobs"))
    inflight = state.get("in_flight")
    inflight_count = (0 if inflight is None else 1 if isinstance(inflight, dict) else None) if "in_flight" in state else None
    stats = _dict(state.get("stats"))
    stat_counts = {k: _count(stats.get(k)) for k in ("queued", "accepted", "rejected", "failed")}
    validator_notes = errors + [
        "Heartbeat is a file-publication proxy: state.flushed_at and snapshot.generated_at, not the in-memory TickWatchdog beat or the 600s log heartbeat. Fresh files do not prove chain or GPU health.",
        "Stale threshold is max(tick_warn_after_s, 3 × poll_interval_s, 3 × dashboard_flush_min_interval_s); this is monitoring policy, not a production watchdog change.",
        "Phase comes from the public snapshot. Its age is time since set_phase, not time since last progress; a long duel is not itself stale.",
        "Waiting and in-flight counts come from persisted state. Lifetime queued counts admissions, not waiting work. Independent reads can straddle a dispatch or crown.",
        "Allowlisted output only; identities, model references, private URLs, credentials, phase extras and raw errors are never emitted.",
    ]
    if freshness != "Fresh":
        validator_notes.append("File heartbeat is stale or unavailable; invalid/future timestamps are Unknown, not healthy.")
    incomplete = phase_name == "Unknown" or phase_age is None or queue is None or benches is None or inflight_count is None or any(v is None for v in stat_counts.values())
    if incomplete:
        validator_notes.append("Some phase or count fields are unavailable or invalid; affected values are Unknown.")
    sources = [f"{STATE} — flushed_at: {flushed_at or 'Unknown'}",
               f"{SNAPSHOT} — generated_at: {generated_at or 'Unknown'}", CONTRACT]
    validator_panel = common.panel(
        "Validator", "Read-only control-plane snapshot · publication freshness, not a process probe",
        status="warn" if errors or incomplete or freshness != "Fresh" else "ok",
        metrics=[
            common.metric("Heartbeat proxy", freshness, f"Stale after {common.duration(stale_after)}"),
            common.metric("State age", common.duration(state_age)),
            common.metric("Snapshot age", common.duration(snapshot_age)),
            common.metric("Phase", phase_name, f"Set {common.duration(phase_age)} ago"),
            common.metric("Waiting", len(queue) if queue is not None else "Unknown", "Excludes in-flight work"),
            common.metric("In flight", _display(inflight_count), "Persisted unresolved entry, not independently probed"),
            common.metric("Bench jobs", len(benches) if benches is not None else "Unknown", "Persisted pending/running jobs"),
        ],
        sections=[common.table("Lifetime admission / duel counters", ["Counter", "Count"],
                               [[k, _display(v)] for k, v in stat_counts.items()])],
        notes=validator_notes, sources=sources,
    )

    success_at, weights_age = _timestamp(state.get("last_weights_at"), now)
    interval = _positive(settings.get("weight_interval_s"))
    intervals = weights_age / interval if weights_age is not None and interval is not None else None
    cadence = ("Unknown" if intervals is None else "Past interval" if intervals > 1 else "Within interval")
    depth = _count(subnet.get("king_chain_size"))
    reign = _dict(snapshot.get("reign"))
    members = _rows(reign.get("members"))
    flags_ok = members is not None and all(type(m.get("earning")) is bool and type(m.get("inaccessible")) is bool for m in members)
    candidates = sum(m["earning"] for m in members) if flags_ok else None
    inaccessible = sum(m["inaccessible"] for m in members) if flags_ok else None
    weights_notes = errors + [
        "Latest success is validator-recorded, not independently chain-confirmed by this monitor. chain.set_rolling_weights waits for inclusion and checks dispatch failure; only its True result causes State.record_weights_set to stamp last_weights_at.",
        "Past interval is a cadence warning, not proof of failed weights: ticks, accessibility probes and chain rate limits can delay a successful write. Failed attempts do not advance the timestamp; crowns/reverts can force an early attempt.",
        "Rolling design: current king then newest prior distinct hotkeys, capped at king_chain_size. Empty-hotkey genesis takes no seat. Proven inaccessible models lose their seat and deeper accessible kings backfill; unknown accessibility probes retain prior status.",
        "The chain setter then skips unregistered members and deduplicates UIDs; remaining UIDs receive equal 1/N weights. Registration filtering does not backfill beyond the selected window. No registered recipient means the burn UID receives all weight; a stale metagraph refuses the write entirely.",
        "Public earning flags describe intended window candidates before registration filtering, not confirmed recipients or paid emissions. Snapshot UIDs and rounded weight_bps are not treated as chain truth; the monitor does no RPC or accessibility probe.",
        "Accessibility exclusions are in-memory validator state, not persisted in state.json. Counts therefore use the public snapshot and may lag; no payout list is reconstructed from private identities.",
    ]
    weights_incomplete = cadence == "Unknown" or depth is None or not flags_ok
    if weights_incomplete:
        weights_notes.append("Weight timestamp, interval or public window data is incomplete; missing values are Unknown, never zero successes.")
    if depth is not None and (_count(reign.get("size")) != depth or (candidates is not None and candidates > depth)):
        weights_incomplete = True
        weights_notes.append("Public payout window and local TOML disagree; they may represent different configuration snapshots.")
    weights_panel = common.panel(
        "Weights", "Recorded successful weight-setting cadence · rolling equal-share design",
        status="warn" if errors or weights_incomplete or cadence != "Within interval" or freshness != "Fresh" else "ok",
        metrics=[
            common.metric("Last recorded success", success_at or "Unknown", "Not independently chain-confirmed"),
            common.metric("Success age", common.duration(weights_age)),
            common.metric("Configured interval", common.duration(interval)),
            common.metric("Cadence", cadence, f"{intervals:.2f} × configured interval" if intervals is not None else "Unknown age / interval"),
            common.metric("Window limit", _display(depth), "Maximum distinct candidate hotkeys"),
            common.metric("Reported candidates", _display(candidates), "Snapshot earning=true; before registration filtering"),
            common.metric("Reported inaccessible", _display(inaccessible), "Across published lineage, not only the payout window"),
        ],
        notes=weights_notes + (["Source freshness is stale or unknown; intended payout counts may be outdated."] if freshness != "Fresh" else []),
        sources=sources + ["affine/affine/validator.py::_maybe_set_weights",
                           "affine/affine/state.py::king_lineage_members / record_weights_set",
                           "affine/affine/chain.py::set_rolling_weights"],
    )
    return {"validator": validator_panel, "weights": weights_panel}


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2, allow_nan=False))
