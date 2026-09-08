"""Read-only teacher swarm and router panels; never publish backend addresses."""
from __future__ import annotations

import http.client
import math
import re
import time
from datetime import datetime, timezone

from . import common

__all__ = ["collect"]

_STATE = "ops/teacher-swarm/state/state.json"
_HEALTH = "http://localhost:9100/health"
_METRICS = "http://localhost:9100/metrics"
_STALE_SECONDS = 300
_WINDOWS = ("10s", "60s", "300s")
_RATES = (("Requests/s", "req_per_s"), ("Samples/s", "samples_per_s"),
          ("Echoes/s", "echoes_per_s"))


def _number(value):
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return value
    return None


def _count(value):
    value = _number(value)
    return int(value) if value is not None and value == int(value) else None


def _display(value):
    return "Unknown" if value is None else value


def _name(value, model=False):
    pattern = r"[A-Za-z0-9_-][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_-][A-Za-z0-9_.-]*)?" if model else r"[A-Za-z][A-Za-z0-9_.-]*"
    if isinstance(value, str) and len(value) <= 160 and re.fullmatch(pattern, value):
        return value
    return "Unknown"


def _load(source, remote=False):
    try:
        value = common.get_json(source, timeout=3) if remote else common.read_json(source)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError, http.client.HTTPException):
        return None


def _backends(data):
    value = data.get("backends") if data is not None else None
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    return None


def _sum(backends, key):
    if backends is None:
        return None
    values = [_count(b.get(key)) for b in backends]
    return sum(values) if all(v is not None for v in values) else None


def _freshness(state):
    value = state.get("updated") if state else None
    try:
        if isinstance(value, str):
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                return "Unknown", None
            timestamp = stamp.astimezone(timezone.utc).timestamp()
        else:
            timestamp = _number(value)
        if timestamp is None or not math.isfinite(timestamp):
            return "Unknown", None
        seconds = time.time() - timestamp
    except (ValueError, TypeError, OverflowError, OSError):
        return "Unknown", None
    if seconds < -60:
        return "Clock skew", None
    seconds = max(0, seconds)
    return ("Stale" if seconds > _STALE_SECONDS else "Fresh"), seconds


def collect():
    """Return ``teacher`` and ``router`` panel payloads without mutating services."""
    health = _load(_HEALTH, remote=True)
    telemetry = _load(_METRICS, remote=True)
    state = _load(_STATE)
    live = _backends(telemetry)
    recorded = _backends(state)
    freshness, state_age = _freshness(state)
    notes = []
    status = "ok"

    total = _count(health.get("backends")) if health is not None else None
    healthy = _count(health.get("healthy")) if health is not None else None
    ready = health.get("ok") if health is not None else None
    live_healthy = (sum(b.get("healthy") is True for b in live)
                    if live is not None and all(type(b.get("healthy")) is bool for b in live)
                    else None)
    if total is None and live is not None:
        total = len(live)
    if healthy is None:
        healthy = live_healthy

    if health is None:
        status = "warn"
        notes.append("Router health probe unavailable or invalid; no cached health is assumed.")
    elif type(ready) is not bool or _count(health.get("backends")) is None or _count(health.get("healthy")) is None:
        status = "warn"
        notes.append("Router health response is incomplete.")
    if live is None:
        status = "warn"
        notes.append("Router metrics unavailable or invalid; occupancy and counters are unknown.")
    elif any(type(b.get("healthy")) is not bool or any(_count(b.get(k)) is None for k in ("in_flight", "ok", "err")) for b in live):
        status = "warn"
        notes.append("Some replica metrics are incomplete.")
    if total is not None and healthy is not None and healthy > total:
        status = "warn"
        notes.append("Router health counts are inconsistent.")
        healthy = live_healthy
    if live is not None and (total != len(live) or (live_healthy is not None and healthy != live_healthy)):
        status = "warn"
        notes.append("Health and metrics snapshots disagree; they are sampled separately.")
    if ready is False or healthy == 0 or live_healthy == 0 or (live is not None and not live) or (health is None and live is None):
        status = "error"
        notes.append("No healthy router service is confirmed.")
    elif (healthy is not None and total is not None and healthy < total) or (live is not None and any(b.get("healthy") is False for b in live)):
        status = "warn"
        notes.append("One or more replica circuits are open.")

    rates = telemetry.get("rates") if telemetry else None
    rates = rates if isinstance(rates, dict) else {}
    rate_values = {key: [_number(rates.get(w, {}).get(key))
                         if isinstance(rates.get(w), dict) else None for w in _WINDOWS]
                   for _, key in _RATES}
    if any(v is None for values in rate_values.values() for v in values):
        if status == "ok":
            status = "warn"
        notes.append("Some rolling rate windows are unavailable; missing values are not zero.")
    inflight = _sum(live, "in_flight")
    request_rate = rate_values["req_per_s"][0]
    traffic = ("In flight" if inflight else "Idle" if request_rate == 0 and inflight == 0
               else "Recent completions" if request_rate is not None and request_rate > 0 else "Unknown")
    notes.extend([
        "Zero traffic is idle, not a failure. Rates count router-recorded completions, not arrivals or tokens.",
        "Healthy means the router circuit is eligible (closed), not an independent GPU health probe.",
        "Success/error counters are cumulative since router start for currently listed backends; re-added backends reset their counters. Removed backends are not included.",
        "Success is the router's ok counter (including upstream non-5xx responses); errors are failed attempts, including retries, not unique failed requests.",
    ])

    active = live if live is not None else recorded
    pod_groups = {}
    if active is not None:
        for backend in active:
            pod = _name(backend.get("pod"))
            pod_groups.setdefault(pod, []).append(backend)
    active_count = len(pod_groups) if active is not None and "Unknown" not in pod_groups else None
    active_detail = ("Unique pods in current router backends; excludes historical pod records."
                     if live is not None else "Recorded advertised pods only; live membership unavailable.")
    pod_rows = []
    for pod, replicas in sorted(pod_groups.items()):
        eligible = (sum(b.get("healthy") is True for b in replicas)
                    if live is not None and all(type(b.get("healthy")) is bool for b in replicas) else None)
        types = ", ".join(sorted({_name(b.get("type")) for b in replicas}))
        pod_rows.append([pod, types, len(replicas), _display(eligible),
                         _display(_sum(replicas, "in_flight") if live is not None else None)])

    backend_rows, bars = [], []
    peak = max([_count(b.get("in_flight")) or 0 for b in live or []] + [1])
    for ordinal, backend in enumerate(live or [], 1):
        label = f"Replica {ordinal}"
        pod = _name(backend.get("pod"))
        circuit = ("Closed / eligible" if backend.get("healthy") is True else
                   "Open / benched" if backend.get("healthy") is False else "Unknown")
        occupancy = _count(backend.get("in_flight"))
        backend_rows.append([label, pod, _name(backend.get("type")), circuit,
                             _display(occupancy), _display(_count(backend.get("ok"))),
                             _display(_count(backend.get("err")))])
        if occupancy is not None:
            bars.append(dict(label=f"{label} · {pod}", value=occupancy, max=peak,
                             detail=f"{circuit}; scale is observed peak, not capacity",
                             tone="green" if backend.get("healthy") is True else "warn"))

    rate_rows = [[w] + [_display(rate_values[key][i]) for _, key in _RATES]
                 for i, w in enumerate(_WINDOWS)]
    charts = [dict(title="Rolling completion rates (/s)", labels=list(_WINDOWS),
                   series=[dict(name=label, values=rate_values[key]) for label, key in _RATES])]
    router = common.panel(
        "Teacher router", "Live loopback health, traffic and replica circuits", status=status,
        metrics=[common.metric("Healthy replicas", _display(healthy), f"Of {_display(total)} routed replicas"),
                 common.metric("Active pods", _display(active_count), active_detail),
                 common.metric("In flight", _display(inflight)), common.metric("Traffic", traffic),
                 common.metric("Successes since router start", _display(_sum(live, "ok")), "Current backend lifetimes; includes retries."),
                 common.metric("Errors since router start", _display(_sum(live, "err")), "Cumulative failed attempts, not current health.")],
        sections=[common.table("Rolling completion rates", ["Window"] + [label for label, _ in _RATES], rate_rows),
                  common.table("Backend counters since router start",
                               ["Replica", "Pod", "Type", "Circuit health", "In flight", "Success", "Error"], backend_rows)],
        bars=bars, charts=charts, notes=notes, sources=[_HEALTH, _METRICS])

    teacher_notes = []
    teacher_status = status
    if freshness != "Fresh":
        if teacher_status == "ok":
            teacher_status = "warn"
        teacher_notes.append("Recorded swarm state is stale, undated or clock-skewed; model and cost are not confirmed live.")
    if state is None:
        teacher_notes.append("Swarm state is unavailable or invalid.")
    model = _name(state.get("model"), model=True) if state else "Unknown"
    cost = _number(state.get("spend_usd_hr")) if state else None
    if model == "Unknown" or cost is None or recorded is None:
        if teacher_status == "ok":
            teacher_status = "warn"
        teacher_notes.append("Recorded model, cost or advertised membership is incomplete.")
    if live is not None and recorded is not None:
        def membership(backends):
            return {(str(b.get("url", "")).rstrip("/"), _name(b.get("pod"))) for b in backends}
        if membership(live) != membership(recorded):
            if teacher_status == "ok":
                teacher_status = "warn"
            teacher_notes.append("Recorded and routed membership differ; active pods use the router snapshot.")
    teacher_notes.extend([
        active_detail,
        "Recorded spend is the state's hourly estimate, not a live billing quote; it may include non-serving pods.",
        "State freshness threshold: 300s. Router health is sampled now; stale state alone does not prove service failure.",
        "Replica health reflects circuit eligibility. Historical error totals and idle traffic do not imply a current failure.",
    ])
    if status != "ok":
        teacher_notes.append("Router health or telemetry is degraded; see the router panel for probe and circuit details.")
    health_bars = []
    if total is not None and healthy is not None and total > 0 and healthy <= total:
        health_bars.append(dict(label="Healthy routed replicas", value=healthy, max=total,
                                detail="Router circuit eligibility", tone="green" if healthy == total else "warn"))
    teacher = common.panel(
        "Teacher swarm", "Live serving membership with recorded model and cost", status=teacher_status,
        metrics=[common.metric("Teacher model", model, "Recorded in swarm state, not independently probed."),
                 common.metric("Healthy replicas", _display(healthy), f"Of {_display(total)} routed replicas"),
                 common.metric("Active pods", _display(active_count), active_detail),
                 common.metric("Recorded cost", f"${cost:,.2f}/h" if cost is not None else "Unknown", "Recorded spend_usd_hr; not live billing."),
                 common.metric("State freshness", freshness, f"Age: {common.duration(state_age)}", "green" if freshness == "Fresh" else "warn"),
                 common.metric("Recorded advertised replicas", len(recorded) if recorded is not None else "Unknown")],
        sections=[common.table("Active routed pods" if live is not None else "Recorded advertised pods",
                               ["Pod", "Type", "Replicas", "Healthy", "In flight"], pod_rows)],
        bars=health_bars, notes=teacher_notes, sources=[_STATE, _HEALTH, _METRICS])
    return {"teacher": teacher, "router": router}
