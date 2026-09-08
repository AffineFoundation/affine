"""Read-only scoring panels from the public snapshot and bounded duel history."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import re
import tomllib

try:
    from . import common
except ImportError:
    import common

__all__ = ["collect"]

SNAPSHOT = "affine/state/public/snapshot.json"
CONTRACT = "affine/affine.toml"
HISTORY_URL = "http://localhost:8787/api/v1/history?limit=40"
KINDS = ("bash", "tool_call", "boxed")
PARAMS = (
    "weight_version_key", "score_mode", "n_turns", "k_sigma", "min_margin",
    "tau", "band_c", "band_floor", "forfeit_turn_score", "min_thought_chars",
    "causality_gate", "causality_tau", "causality_gamma", "n_teacher_samples",
    "n_miner_samples", "allowed_action_kinds", "max_thought_tokens",
    "max_action_tokens", "max_tokens_by_kind", "temperature", "top_p",
)
STAGES = {"dispatching", "load_challenger", "load_king", "load_teacher",
          "scoring", "sampling", "loading", "finalizing", "complete"}
MODES = {"min_rg", "min_rga", "reason"}
UNKNOWN = "Unknown"


def _obj(value):
    return value if isinstance(value, dict) else {}


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return value if math.isfinite(value) else None
        except OverflowError:
            pass
    return None


def _fmt(value, percent=False):
    n = _number(value)
    if n is None:
        return UNKNOWN
    if percent:
        return f"{n:.1%}"
    return str(n) if isinstance(n, int) else f"{n:.6g}"


def _stamp(value):
    try:
        if isinstance(value, str):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return None
        elif _number(value) is not None:
            dt = datetime.fromtimestamp(value, timezone.utc)
        else:
            return None
        return dt.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _cid(row):
    value = row.get("challenge_id")
    return value if isinstance(value, str) and re.fullmatch(r"chal-\d{1,12}", value) else UNKNOWN


def _param(key, value):
    if value is None:
        return "null"
    if key == "score_mode":
        return value if isinstance(value, str) and value in MODES else UNKNOWN
    if key == "allowed_action_kinds":
        if isinstance(value, list) and all(isinstance(k, str) and k in KINDS for k in value):
            return ", ".join(sorted(value)) or "(empty)"
        return UNKNOWN
    if key == "max_tokens_by_kind":
        if not isinstance(value, dict):
            return UNKNOWN
        clean = {}
        for kind, limits in value.items():
            if kind not in KINDS or not isinstance(limits, dict):
                return UNKNOWN
            if any(k not in {"max_thought_tokens", "max_action_tokens"}
                   or _number(v) is None for k, v in limits.items()):
                return UNKNOWN
            clean[kind] = limits
        return json.dumps(clean, sort_keys=True)
    if isinstance(value, bool):
        return str(value).lower()
    return _fmt(value)


def _bar(label, value, maximum=1, detail="", tone=""):
    value, maximum = _number(value), _number(maximum)
    if value is None or maximum is None or maximum <= 0 or not 0 <= value <= maximum:
        return []
    return [dict(label=label, value=value, max=maximum, detail=detail, tone=tone)]


def _load_snapshot():
    try:
        data = common.read_json(SNAPSHOT)
        if not isinstance(data, dict):
            raise ValueError
        return data, []
    except Exception:
        return {}, ["Public snapshot unavailable; live progress is unknown."]


def _load_history():
    try:
        data = common.get_json(HISTORY_URL, timeout=8)
        items = data.get("items") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError
        rows = [r for r in items[:40] if isinstance(r, dict) and r.get("event") == "verdict"]
        rows.sort(key=lambda r: _stamp(r.get("at")) or "", reverse=True)
        return rows, []
    except Exception:
        return [], ["Recent history unavailable; completed-duel telemetry is unknown."]


def _load_contract():
    try:
        with (common.ROOT / CONTRACT).open("rb") as stream:
            data = tomllib.load(stream)
        values = dict(_obj(data.get("duel")))
        values["weight_version_key"] = _obj(data.get("subnet")).get("weight_version_key")
        values["allowed_action_kinds"] = _obj(data.get("dataset")).get("allowed_action_kinds")
        return values, []
    except Exception:
        return {}, ["Live TOML unavailable; parameter comparison is incomplete."]


def _duel(snapshot, notes):
    stamp = _stamp(snapshot.get("generated_at"))
    age = common.age(stamp)
    notes = list(notes)
    stale = age is None or age > 120
    if age is None:
        notes.append("Snapshot freshness is unknown.")
    elif stale:
        notes.append("Snapshot is older than 120s; displayed progress may be stale.")
    active = _obj(snapshot.get("current_eval"))
    progress = _obj(active.get("progress"))
    stage = active.get("stage", progress.get("phase"))
    stage = stage if isinstance(stage, str) and stage in STAGES else UNKNOWN
    side = progress.get("miner", progress.get("side"))
    side = side if isinstance(side, str) and side in {"king", "challenger", "teacher"} else UNKNOWN
    done, total = _number(progress.get("done")), _number(progress.get("total"))
    if not active:
        stage = "Idle" if snapshot and snapshot.get("current_eval", False) is None else UNKNOWN
    elapsed = common.age(_stamp(active.get("started_at")))
    bars = _bar(f"{side} turns", done, total, "Last reported side only; not combined duel completion.") if active else []
    return common.panel(
        "Duel progress", "Public snapshot · live progress", status="warn" if stale or notes else "ok",
        metrics=[common.metric("Challenge", _cid(active) if active else "None"),
                 common.metric("Stage", stage), common.metric("Side", side),
                 common.metric("Turns", f"{_fmt(done)} / {_fmt(total)}"),
                 common.metric("Elapsed", common.duration(elapsed)),
                 common.metric("Snapshot age", common.duration(age), stamp or UNKNOWN,
                               "warn" if stale else "")],
        bars=bars, notes=notes + ["Progress reports only the most recently updated side; no ETA or other-side progress is inferred."],
        sources=[SNAPSHOT])


def _historical_note(latest):
    if not latest:
        return "No completed verdict in the latest 40 history events."
    return (f"Latest completed verdict: {_cid(latest)} at {_stamp(latest.get('at')) or UNKNOWN}. "
            "This is historical telemetry, not a service heartbeat; its age does not imply a dead service.")


def _scores(latest, live, history_notes, contract_notes):
    stamp = _obj(latest.get("duel_params"))
    comparisons = []
    different = []
    for key in PARAMS:
        old = _param(key, stamp[key]) if key in stamp else "Not stamped"
        new = _param(key, live[key]) if key in live else "Not configured"
        state = "Unknown" if key not in stamp or key not in live or UNKNOWN in (old, new) else ("Match" if old == new else "Different (historical)")
        if state == "Different (historical)":
            different.append(key)
        comparisons.append([key, old, new, state])
    rows, metrics, bars = [], [], []
    fields = [("R mean", "mean_r_leg", False), ("G mean", "mean_g_leg", False),
              ("G binding", "g_bind_frac", True), ("B mean", "mean_b", False),
              ("B pass", "b_gate_pass_rate", True), ("Forfeits", "n_forfeits", False),
              ("Forfeit rate", "forfeit_rate", True), ("Turns", "n_turns", False),
              ("Thought mean chars", "mean_len_z", False), ("Thought median chars", "median_len_z", False),
              ("Action mean chars", "mean_len_y", False)]
    sides = [_obj(latest.get(side)) for side in ("king", "challenger", "teacher")]
    for label, key, percent in fields:
        rows.append([label] + [_fmt(s.get(key), percent) for s in sides])
    for name, side, score_key in zip(("King", "Challenger"), sides, ("score_king", "score")):
        value = latest.get(score_key)
        if value is None:
            value = side.get("reason", side.get("S"))
        metrics.append(common.metric(f"{name} recorded score", _fmt(value), "Stored score under that verdict's regime."))
        bars += _bar(f"{name} B pass", side.get("b_gate_pass_rate"), detail="Historical recorded pass rate")
        bars += _bar(f"{name} forfeits", side.get("forfeit_rate"), detail="Historical recorded forfeit rate")
    metrics += [common.metric("Paired turns", _fmt(latest.get("n_paired_turns"))),
                common.metric("Forfeit turns (paired)", _fmt(latest.get("n_forfeit_turns"))),
                common.metric("Live weight version", _fmt(live.get("weight_version_key")))]
    notes = [_historical_note(latest),
             "R/G/B and lengths are published summaries, not recomputed scores. Unknown fields are not zero.",
             "Missing stamps are not filled from live TOML; TOML describes configured values, not proof of the running pod's configuration.",
             "B is a licensing gate, not a ranked score. No verdict or crown decision is recomputed."]
    if different:
        notes.append("Stamped parameters differ from current TOML: " + ", ".join(different) + ". Historical differences are not a liveness failure.")
    return common.panel(
        "Scoring telemetry", "Latest completed duel · stamped versus configured", status="warn" if history_notes or contract_notes else "ok",
        metrics=metrics, bars=bars,
        sections=[common.table("Published side metrics", ["Metric", "King", "Challenger", "Teacher"], rows),
                  common.table("Latest stamped parameters vs live TOML", ["Parameter", "Verdict stamp", "Live TOML", "Comparison"], comparisons)],
        notes=notes + history_notes + contract_notes, sources=[HISTORY_URL, CONTRACT])


def _dialects(latest, history_notes):
    stamp = _obj(latest.get("duel_params"))
    k = _number(stamp.get("n_teacher_samples"))
    teacher = _obj(_obj(latest.get("teacher")).get("by_dialect"))
    sides = {name: _obj(_obj(latest.get(name)).get("by_dialect")) for name in ("king", "challenger")}
    kinds = [kind for kind in KINDS if kind in teacher or any(kind in s for s in sides.values())]
    rows, refs, bars = [], [], []
    for kind in kinds:
        for name, side in sides.items():
            d = _obj(side.get(kind))
            rows.append([kind, name, _fmt(d.get("n_turns")), _fmt(d.get("n_valid")),
                         _fmt(d.get("parse_rate"), True), _fmt(d.get("b_gate_pass_rate"), True),
                         _fmt(d.get("mean_b")), _fmt(d.get("median_len_z"))])
            bars += _bar(f"{kind} {name} parse", d.get("parse_rate"), detail="Parseable turns / side's dialect turns")
            bars += _bar(f"{kind} {name} B pass", d.get("b_gate_pass_rate"), detail="Published B pass rate; not parse rate")
        d = _obj(teacher.get(kind))
        refs.append([kind, _fmt(d.get("n_turns")), _fmt(d.get("zero_ref_turns")), _fmt(d.get("mean_refs")), _fmt(k)])
        bars += _bar(f"{kind} refs / turn", d.get("mean_refs"), k, "Teacher yield; maximum from this verdict's stamped k")
    notes = [_historical_note(latest),
             "Parse and B rates have different denominators. Teacher zero-ref turns are unscorable and are excluded from side telemetry; they are not miner forfeits.",
             "Reference yield uses stamped k only; missing historical fields stay unknown."]
    if not kinds:
        notes.append("No per-dialect telemetry published for this verdict.")
    return common.panel(
        "Action dialects", "Latest completed duel · parse, B license and reference yield", status="warn" if history_notes else "ok",
        bars=bars, sections=[common.table("Side telemetry", ["Dialect", "Side", "Turns", "Parsed", "Parse rate", "B pass", "B mean", "Thought median chars"], rows),
                              common.table("Teacher reference yield", ["Dialect", "Drawn turns", "Zero-ref turns", "Mean refs / turn", "Stamped k"], refs)],
        notes=notes + history_notes, sources=[HISTORY_URL])


def _history(rows, history_notes):
    chronological = list(reversed(rows))
    labels, regimes, assignments, regime_rows = [], {}, [], []
    for row in chronological:
        stamp = _obj(row.get("duel_params"))
        signature = tuple(_param(key, stamp[key]) if key in stamp else "Not stamped" for key in PARAMS)
        if signature not in regimes:
            name = f"Regime {len(regimes) + 1}"
            regimes[signature] = name
            regime_rows.append([name] + [_param(key, stamp[key]) if key in stamp else "Not stamped" for key in
                                        ("score_mode", "tau", "forfeit_turn_score", "allowed_action_kinds", "min_margin")])
        assignments.append(regimes[signature])
        labels.append(_cid(row))
    series = [{"name": name + " margin", "values": [
        _number(row.get("margin")) if assignment == name else None
        for row, assignment in zip(chronological, assignments)]} for name in regimes.values()]
    table_rows = []
    for row, regime in reversed(list(zip(chronological, assignments))):
        won = row.get("challenger_wins")
        outcome = "Challenger won" if won is True else "Challenger did not win" if won is False else UNKNOWN
        table_rows.append([_cid(row), _stamp(row.get("at")) or UNKNOWN, regime, _fmt(row.get("margin")),
                           _fmt(row.get("se")), _fmt(row.get("z")), _fmt(row.get("n_paired_turns")), outcome])
    return common.panel(
        "Duel history", "Completed verdicts within the latest 40 events · oldest to newest chart", status="warn" if history_notes else "ok",
        metrics=[common.metric("Completed verdicts", len(rows), "Failures and non-verdict events excluded; no extra pages fetched.")],
        charts=[dict(title="Recorded challenger − king margin (historical regimes)", labels=labels, series=series)] if series else [],
        sections=[common.table("Historical parameter regimes", ["Regime", "Mode", "Tau", "Forfeit floor", "Dialects", "Margin floor"], regime_rows),
                  common.table("Recent completed duels", ["Challenge", "Completed UTC", "Regime", "Margin", "SE", "z", "Paired turns", "Recorded outcome"], table_rows)],
        notes=[_historical_note(rows[0] if rows else {}),
               "Historical regime caveat: margins across scoring forks, teachers, corpus changes and incumbents are not directly comparable. Each point retains its original verdict; no replay or normalization is performed.",
               "Series split on available parameter stamps only; identical or missing stamps do not establish identical teachers, corpora or regimes. Missing margins remain gaps.",
               "Positive margin alone does not imply a crown: statistical, margin and licensing requirements also apply. Outcomes shown are recorded, not inferred."] + history_notes,
        sources=[HISTORY_URL])


def collect() -> dict:
    """Return four common.panel dictionaries without artifacts or private identifiers."""
    snapshot, snapshot_notes = _load_snapshot()
    rows, history_notes = _load_history()
    live, contract_notes = _load_contract()
    latest = rows[0] if rows else {}
    return {
        "duel": _duel(snapshot, snapshot_notes),
        "scores": _scores(latest, live, history_notes, contract_notes),
        "dialects": _dialects(latest, history_notes),
        "history": _history(rows, history_notes),
    }
