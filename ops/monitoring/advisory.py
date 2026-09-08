"""Read-only benchmark and post-crown audit projections for the monitoring wall."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from .common import metric, panel, read_json, table

_PUBLIC = "affine/state/public/"
_LIMIT = 12


def _load(name, kind):
    try:
        value = read_json(_PUBLIC + name)
    except (OSError, ValueError, TypeError):
        return kind(), f"{name}: unavailable or invalid JSON."
    if not isinstance(value, kind):
        return kind(), f"{name}: unexpected data shape."
    return value, None


def _dict(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _number(value):
    if type(value) not in (int, float):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def _fraction(value):
    value = _number(value)
    return value if value is not None and 0 <= value <= 1 else None


def _reign(row):
    value = row.get("reign_number")
    if type(value) is int and value >= 0:
        return value
    label = row.get("label")
    if isinstance(label, str):
        match = re.fullmatch(r"reign-(\d{1,9})", label)
        if match:
            return int(match[1])
    return None


def _label(reign):
    return f"Reign {reign}" if reign is not None else "Unknown reign"


def _text(value, limit=800):
    """Plain text only; the UI owns HTML escaping, not this collector."""
    if not isinstance(value, str):
        return "Unknown"
    value = re.sub(r"(?:[a-z][a-z0-9+.-]*://|models/registrations/)[^\s<>\"']+",
                   "[reference withheld]", value, flags=re.I)
    return " ".join(value.split())[:limit]


def _time(value):
    if not isinstance(value, str):
        return "Unknown"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return "Unknown"


def _stamp(value):
    value = _time(value)
    if value == "Unknown":
        return float("-inf")
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def _benchmarks(data, snapshot, errors):
    notes = [
        "Benchmarks are advisory only: never scored, never part of the duel or crown decision.",
        "Small-n caveat: single 25-task runs are noisy, not reliable per-model evidence. "
        "Unknown n stays unknown; no sample count is inferred from a score.",
        "RT-7 remains open: the historical live-board score/benchmark relationship inverted. "
        "A crown measures teacher-anchored distillation, not coding isomorphism or demonstrated programmability.",
        "Public results retain latest successful scores; failed attempts may be absent. "
        "Chart gaps mean unavailable results, not zero performance.",
    ] + errors
    king = _reign(_dict(snapshot.get("king")))
    models = _rows(data.get("models"))
    records = {}
    for model in models:
        reign = _reign(model)
        if reign is None:
            continue
        for suite, result in _dict(model.get("suites")).items():
            if not isinstance(suite, str) or not isinstance(result, dict):
                continue
            key = (reign, suite)
            if key not in records or _stamp(result.get("finished_at")) > _stamp(records[key].get("finished_at")):
                records[key] = result
    reigns = sorted({r for r, _ in records} | ({king} if king is not None else set()))[-_LIMIT:]
    declared_suites = data.get("suites")
    declared_suites = declared_suites if isinstance(declared_suites, list) else []
    suites = sorted({s for _, s in records} | {s for s in declared_suites if isinstance(s, str)})
    rows, series, bars = [], [], []
    missing = False
    for suite in suites:
        values = []
        for reign in reigns:
            result = records.get((reign, suite), {})
            score = _fraction(result.get("score")) if result.get("ok") is True else None
            values.append(round(score * 100, 3) if score is not None else None)
            n = result.get("n_sims")
            n = n if type(n) is int and n > 0 else "Unknown"
            rows.append([_label(reign), _text(suite, 100),
                         f"{score:.1%}" if score is not None else "Unavailable", n,
                         _time(result.get("finished_at"))])
            if reign == king or (king is None and reign == reigns[-1]):
                missing |= score is None
                if score is not None:
                    bars.append(dict(label=_text(suite, 100), value=round(score * 100, 3), max=100,
                                     detail=f"{_label(reign)}; n={n}; advisory only", tone=""))
        series.append(dict(name=_text(suite, 100), values=values))
    jobs = snapshot.get("bench_jobs")
    if not isinstance(jobs, list):
        jobs = data.get("active")
        notes.append("Snapshot bench_jobs unavailable; using public benchmark active jobs as fallback.")
        missing = True
    active = []
    for job in _rows(jobs):
        state = job.get("state")
        if state in ("DONE", "COMPLETED", "FAILED", "CANCELLED"):
            continue
        state = state if state in ("QUEUED", "RUNNING", "PENDING", "STARTING") else "Unknown"
        active.append([_label(_reign(job)), _text(job.get("suite"), 100), state,
                       _time(job.get("queued_at")), _time(job.get("started_at"))])
    if not records:
        notes.append("No reign benchmark results are available.")
    if king is None:
        notes.append("Current king reign is unknown; latest result is not assumed to be the current king.")
    elif not any(r == king for r, _ in records):
        notes.append(f"{_label(king)} benchmark is pending or unavailable.")
        missing = True
    return panel(
        "Benchmarks", "Latest reign performance · advisory, never scored",
        status="warn" if errors or missing or not records or king is None else "ok",
        metrics=[metric("Current king", _label(king)),
                 metric("Latest benchmark", _label(max((r for r, _ in records), default=None))),
                 metric("Active jobs", len(active)),
                 metric("Published", _time(data.get("generated_at")))],
        sections=[table("Latest reign performance (%)", ["Reign", "Suite", "Score", "n", "Finished"], rows),
                  table("Active benchmark jobs", ["Reign", "Suite", "State", "Queued", "Started"], active)],
        charts=[dict(title="Advisory benchmark performance (%)", labels=[_label(r) for r in reigns],
                     series=series)] if series and reigns else [],
        bars=bars, notes=notes,
        sources=[_PUBLIC + "benchmarks.json", _PUBLIC + "snapshot.json (bench_jobs, king)"],
    )


def _audit_state(entry):
    seed = _reign(entry) == 0 and entry.get("challenge_id") == "seed"
    inputs = _dict(entry.get("inputs"))
    required = ["prompt.md", "evidence.json"] + ([] if seed else ["duel_record.json.gz"])
    missing = [name for name in required if not isinstance(inputs.get(name), str) or not inputs[name].strip()]
    confidence = _fraction(entry.get("confidence"))
    status = entry.get("status")
    status = status if status in ("ok", "error", "pending", "running") else "unknown"
    exploit = entry.get("exploit")
    verdict = "Exploit" if exploit is True else "No exploit reported" if exploit is False else "Unknown"
    issues = []
    if missing:
        issues.append("Missing artifact: " + ", ".join(missing))
    if status != "ok":
        issues.append(f"Audit status: {status}")
    if confidence is None:
        issues.append("Confidence unavailable")
    elif confidence < 0.5:
        issues.append("Low reported confidence (<50%; monitoring warning only)")
    if type(exploit) is not bool:
        issues.append("Verdict unavailable")
    if entry.get("dry_run") is True:
        issues.append("Dry-run audit; enforcement not live")
    coverage = "Missing artifact" if missing else "Inputs declared present"
    if seed and not inputs.get("duel_record.json.gz"):
        coverage += "; seed duel artifact absent (expected)"
    return status, verdict, confidence, coverage, issues, seed


def _audits(data, snapshot, errors):
    notes = [
        "Audits are post-crown policy, not consensus scoring. An ok status is not proof of complete evidence.",
        "Coverage uses published input declarations, not a fresh download/hash verification. "
        "The historical seed has no competitive duel: its missing duel artifact is expected.",
        "Confidence is the auditor's reported confidence, not a calibrated probability of safety.",
    ] + errors
    latest = {}
    for entry in _rows(data):
        reign = _reign(entry)
        if reign is not None and (reign not in latest or _stamp(entry.get("audited_at")) > _stamp(latest[reign].get("audited_at"))):
            latest[reign] = entry
    king_row = _dict(snapshot.get("king"))
    king = _reign(king_row)
    current = latest.get(king)
    if current and king_row.get("revision") and current.get("revision") != king_row["revision"]:
        current = None
    pending = king is not None and king != 0 and (
        current is None or current.get("status") in ("pending", "running") or current.get("dry_run") is True
    )
    if pending:
        notes.append(f"{_label(king)}: current king audit pending; no completed matching live audit.")
    elif king is None:
        notes.append("Current king unknown; current audit coverage cannot be established.")
    if not latest:
        notes.append("No published audit records are available.")
    issues_by_reign = {}
    rows, bars = [], []
    covered = competitive = 0
    has_exploit = False
    for reign in sorted(latest, reverse=True):
        entry = latest[reign]
        status, verdict, confidence, coverage, issues, seed = _audit_state(entry)
        has_exploit |= entry.get("exploit") is True
        if not seed:
            competitive += 1
            covered += int(status == "ok" and not coverage.startswith("Missing") and entry.get("dry_run") is not True)
        if issues:
            issues_by_reign[reign] = issues
        if len(rows) < _LIMIT:
            rows.append([_label(reign), status, verdict,
                         f"{confidence:.0%}" if confidence is not None else "Unknown",
                         coverage, _time(entry.get("audited_at"))])
            if confidence is not None:
                bars.append(dict(label=_label(reign), value=round(confidence * 100, 3), max=100,
                                 detail=coverage, tone="warn" if issues else ""))
            summary = _text(entry.get("summary"))
            notes.append(f"{_label(reign)}: {summary}")
    for reign, issues in list(issues_by_reign.items())[:_LIMIT]:
        notes.append(f"{_label(reign)} warning: {'; '.join(issues)}.")
    if pending:
        rows.insert(0, [_label(king), "pending", "Unknown", "Unknown", "Current king audit pending", "Unknown"])
    latest_reign = max(latest, default=None)
    latest_entry = latest.get(latest_reign, {})
    confidence = _fraction(latest_entry.get("confidence"))
    warn = bool(errors or issues_by_reign or pending or king is None or not latest or has_exploit)
    return panel(
        "Audits", "Latest post-crown coverage, status and reported confidence",
        status="warn" if warn else "ok",
        metrics=[metric("Current king", _label(king), "Audit pending" if pending else "", "warn" if pending else ""),
                 metric("Latest audit", _label(latest_reign), _time(latest_entry.get("audited_at"))),
                 metric("Latest confidence", f"{confidence:.0%}" if confidence is not None else "Unknown"),
                 metric("Evidence coverage", f"{covered}/{competitive}", "Competitive audits; latest per reign; seed excluded"),
                 metric("Audit warnings", len(issues_by_reign), "Reigns with incomplete evidence or review")],
        sections=[table("Latest audit per reign", ["Reign", "Status", "Verdict", "Confidence", "Coverage", "Audited"], rows)],
        bars=bars, notes=notes,
        sources=[_PUBLIC + "audits.json", _PUBLIC + "snapshot.json (king)"],
    )


def collect():
    """Return common.panel payloads without model refs, wallet IDs or raw jobs."""
    benchmarks, bench_error = _load("benchmarks.json", dict)
    audits, audit_error = _load("audits.json", list)
    snapshot, snapshot_error = _load("snapshot.json", dict)
    return {
        "benchmarks": _benchmarks(benchmarks, snapshot, [e for e in (bench_error, snapshot_error) if e]),
        "audits": _audits(audits, snapshot, [e for e in (audit_error, snapshot_error) if e]),
    }
