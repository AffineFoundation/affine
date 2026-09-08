"""Lightweight corpus metadata panels; never materialize turns or deferred records."""
from __future__ import annotations

import hashlib
import json

from .common import ROOT, age, duration, get_json, metric, panel, read_json, table

DATASET_URL = "http://localhost:8787/api/v1/dataset"
MANIFEST_PATH = "affine/state/corpus_cache/current_manifest.json"
FOLD_PATH = "ops/corpus_build/state.json"
_DAILY_GRACE = 48 * 3600
_SYNC_GRACE = 30 * 60


def _object(value):
    return value if isinstance(value, dict) else {}


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _display(value):
    return "Unknown" if value is None or value == "" else str(value)


def _sum_counts(records, key):
    values = [_count(r.get(key)) for r in records]
    return sum(values) if values and None not in values else None


def _age(value):
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    try:
        return age(value)
    except (ValueError, OverflowError, OSError):
        return None


def _alignment(left_sha, left_epoch, right_sha, right_epoch):
    if not left_sha or not right_sha or left_epoch is None or right_epoch is None:
        return "Unknown"
    return "Aligned" if left_sha == right_sha and left_epoch == right_epoch else "Mismatch"


def _mix(counts, total, title, unit):
    if not isinstance(counts, dict) or not counts or not total:
        return [], []
    if any(_count(n) is None for n in counts.values()) or sum(counts.values()) != total:
        return [], []
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    rows = [[str(label), str(n), f"{100 * n / total:.2f}%"] for label, n in ordered]
    bars = [dict(label=str(label), value=100 * n / total, max=100,
                 detail=f"{n:,} / {total:,} {unit} ({100 * n / total:.2f}%)", tone="")
            for label, n in ordered]
    return [table(title, ["Source / group", unit.title(), "Share"], rows)], bars


def _strata_counts(state):
    groups = state.get("group_strata")
    if state.get("mix_seeded") is not True or not isinstance(groups, dict) or not groups:
        return None
    seen, counts = set(), {}
    for group, keys in groups.items():
        if not isinstance(keys, list) or any(not isinstance(k, str) or not k for k in keys):
            return None
        unique = set(keys)
        if len(unique) != len(keys) or seen.intersection(unique):
            return None
        seen.update(unique)
        counts[group] = len(unique)
    return counts if seen else None


def _queue_metric(state, key, label):
    if key not in state:
        return metric(label, "Unknown", "Field absent from fold state", "warn")
    value = state[key]
    if value is None:
        return metric(label, "None")
    if not isinstance(value, dict) or not value:
        return metric(label, "Unknown", "Invalid fold record", "warn")
    turns = value.get("n_turns") if key == "pending" else value.get("n_added")
    return metric(label, f"Epoch {_display(value.get('epoch'))}",
                  f"{_display(turns)} new turns; recorded work awaiting completion", "warn")


def collect():
    """Return independent corpus and daily-fold panels from three cheap reads."""
    api, manifest, state = {}, {}, {}
    corpus_notes, fold_notes = [], []
    local_sha = None
    try:
        api = get_json(DATASET_URL, timeout=4)
        if not isinstance(api, dict) or api.get("error") or _count(api.get("n_turns")) is None:
            raise ValueError("invalid dataset summary")
    except (OSError, ValueError, TypeError) as exc:
        api = {}
        corpus_notes.append(f"Cached dataset API unavailable ({type(exc).__name__}); local metadata only.")
    try:
        raw = (ROOT / MANIFEST_PATH).read_bytes()
        manifest = json.loads(raw)
        if not isinstance(manifest, dict) or _count(manifest.get("corpus_epoch")) is None:
            raise ValueError("invalid manifest")
        local_sha = hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError) as exc:
        manifest = {}
        corpus_notes.append(f"Local manifest unavailable ({type(exc).__name__}).")
    try:
        state = read_json(FOLD_PATH)
        if not isinstance(state, dict) or not state:
            raise ValueError("invalid fold state")
    except (OSError, ValueError, TypeError) as exc:
        state = {}
        fold_notes.append(f"Fold state unavailable ({type(exc).__name__}).")

    history = [r for r in state.get("history", []) if isinstance(r, dict)] if isinstance(state.get("history"), list) else []
    last = history[-1] if history else {}
    api_epoch, local_epoch = api.get("corpus_epoch"), manifest.get("corpus_epoch")
    api_sha = api.get("manifest_sha256")
    api_alignment = _alignment(api_sha, api_epoch, local_sha, local_epoch)
    fold_alignment = _alignment(last.get("manifest_sha256"), last.get("epoch"),
                                api_sha or local_sha, api_epoch if api else local_epoch)
    metadata = api or manifest
    epoch = metadata.get("corpus_epoch")
    shards = manifest.get("shards")
    active = [s for s in shards if isinstance(s, dict) and s.get("active") is True] if isinstance(shards, list) else []
    turns = _count(api.get("n_turns")) if api else _count(_object(manifest.get("index")).get("n_turns"))
    if turns is None and not api:
        turns = _sum_counts(active, "n_turns")
    trajectories = _count(api.get("n_trajectories")) if api else _sum_counts(active, "n_trajectories")
    chunks = _count(api.get("n_chunks")) if api else (len(active) if isinstance(shards, list) else None)
    created_at = manifest.get("created_at") if not api or api_alignment == "Aligned" else None
    publish_age, sync_age = _age(created_at), _age(api.get("synced_at"))
    corpus_warn = (not api or not manifest or api_alignment != "Aligned" or api.get("stale") is True
                   or sync_age is None or sync_age > _SYNC_GRACE
                   or publish_age is None or publish_age > _DAILY_GRACE or fold_alignment == "Mismatch")
    corpus_notes.extend([
        "Source mix counts corpus turns, NOT duel slice shares. A duel samples strata, not turns uniformly.",
        "Daily refresh: publication age is a data event, not a daemon heartbeat. A 48h publication-age warning allows one daily cycle of grace.",
        "Cache sync age is separate from publication age; >30m warns about cached telemetry freshness, not fold service health.",
        "Alignment compares cached metadata and exact local manifest bytes, not a new fetch of the remote corpus pointer.",
    ])
    source_counts = _object(api.get("mix")).get("source")
    sections, bars = _mix(source_counts, turns, "Source mix — turn shares, NOT duel slice shares", "turns")
    if not sections:
        corpus_notes.append("Exact source turn shares unavailable or counts do not sum to the API turn total; no dataset scan attempted.")
    sections.append(table("Manifest alignment", ["Snapshot", "Epoch", "Manifest SHA256"], [
        ["Cached dataset API", _display(api_epoch), _display(api_sha)],
        ["Local current_manifest", _display(local_epoch), _display(local_sha)],
        ["Last fold publication", _display(last.get("epoch")), _display(last.get("manifest_sha256"))],
    ]))
    corpus_panel = panel(
        "Corpus D", "Cached dataset metadata · source TURN shares, not duel slice shares",
        status="warn" if corpus_warn else "ok",
        metrics=[
            metric("Epoch", _display(epoch)), metric("Turns", _display(turns)),
            metric("Trajectories", _display(trajectories), "API distinct trajectories" if api else "Sum of active manifest trajectory counts"),
            metric("Chunks", _display(chunks), "Active corpus view/trajectory chunks, not folded trace chunks"),
            metric("Schema", _display(metadata.get("schema_version"))),
            metric("View", _display(metadata.get("view_spec"))),
            metric("Published age", duration(publish_age), _display(created_at), "warn" if publish_age is not None and publish_age > _DAILY_GRACE else ""),
            metric("Cache sync age", duration(sync_age), _display(api.get("synced_at")), "warn" if sync_age is None or sync_age > _SYNC_GRACE else ""),
            metric("Cache stale flag", "Unknown" if not isinstance(api.get("stale"), bool) else str(api["stale"]).lower(), tone="warn" if api.get("stale") is True else ""),
            metric("API / local manifest", api_alignment, tone="warn" if api_alignment != "Aligned" else ""),
            metric("Fold / corpus manifest", fold_alignment, tone="warn" if fold_alignment != "Aligned" else ""),
        ], sections=sections, bars=bars, notes=corpus_notes,
        sources=[DATASET_URL + " (cached summary only)", MANIFEST_PATH, FOLD_PATH],
    )

    fold_age = _age(last.get("at"))
    pending = _queue_metric(state, "pending", "Pending publish")
    unannounced = _queue_metric(state, "unannounced", "Unannounced publish")
    folded = state.get("folded_chunks")
    folded_count = len(folded) if isinstance(folded, list) else None
    fold_warn = (not state or not last or fold_alignment != "Aligned" or pending["tone"] == "warn"
                 or unannounced["tone"] == "warn" or fold_age is None or fold_age > _DAILY_GRACE)
    fold_notes.extend([
        "Daily refresh job, not a continuously running daemon. Last publish is historical; an old event alone does not mean the service is dead. Cron/service health is handled elsewhere.",
        "Last publish turns are newly added turns, not the total corpus size. Folded chunks count processed trace chunks; some records may still be deferred.",
        "Pending is a resumable publication, not the ingestion backlog. Deferred records and large dataset files are never opened or scanned here; deferred backlog is unknown.",
    ])
    counts = _strata_counts(state)
    group_turns = state.get("group_counts")
    coverage = (isinstance(group_turns, dict) and bool(group_turns)
                and all(_count(n) is not None for n in group_turns.values())
                and turns is not None and sum(group_turns.values()) == turns)
    exact = (counts is not None and coverage and set(counts) == set(group_turns)
             and fold_alignment == "Aligned" and (not api or api_alignment == "Aligned")
             and "pending" in state and state["pending"] is None)
    fold_sections, fold_bars = [], []
    if exact:
        fold_sections, fold_bars = _mix(counts, sum(counts.values()),
                                       "Group strata — expected duel slice mix", "strata")
        fold_notes.append("Exact stored, disjoint group_strata pool shares for the aligned manifest; expected slice shares under one-turn-per-stratum sampling, NOT an exact realized duel slice.")
    else:
        fold_notes.append("Exact slice mix unavailable: requires seeded, disjoint group_strata, complete group turn counts, aligned manifests and no pending publish. No turn-share proxy is substituted.")
    recent = history[-20:]
    fold_sections.append(table("Recent publications (new turns)", ["Epoch", "Published at", "New turns", "Trace chunks folded"], [
        [_display(r.get("epoch")), _display(r.get("at")), _display(r.get("n_turns")), _display(r.get("n_chunks"))]
        for r in reversed(recent)
    ]))
    chart_rows = [r for r in recent if _count(r.get("n_turns")) is not None]
    charts = ([dict(title="New turns per publication (not corpus totals)",
                    labels=[f"Epoch {_display(r.get('epoch'))}" for r in chart_rows],
                    series=[dict(name="New turns", values=[r["n_turns"] for r in chart_rows])])]
              if chart_rows else [])
    fold_panel = panel(
        "Corpus fold", "Daily refresh · publication state and stratum-based expected slice mix",
        status="warn" if fold_warn else "ok",
        metrics=[
            metric("Last published epoch", _display(last.get("epoch"))),
            metric("Last publish age", duration(fold_age), _display(last.get("at")), "warn" if fold_age is not None and fold_age > _DAILY_GRACE else ""),
            metric("Last publish new turns", _display(last.get("n_turns"))),
            metric("Last publish folded chunks", _display(last.get("n_chunks")), "Trace chunks processed in this publication"),
            metric("Folded trace chunks", _display(folded_count), "Cumulative processed chunks; not a deferred backlog count"),
            pending, unannounced,
            metric("Fold / corpus manifest", fold_alignment, tone="warn" if fold_alignment != "Aligned" else ""),
            metric("Slice mix evidence", "Exact stratum pool" if exact else "Unavailable"),
        ], sections=fold_sections, bars=fold_bars, charts=charts, notes=fold_notes,
        sources=[FOLD_PATH, MANIFEST_PATH, DATASET_URL + " (alignment only)"],
    )
    return {"corpus": corpus_panel, "corpus-fold": fold_panel}
