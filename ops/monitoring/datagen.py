"""Read-only datagen and trace panels for the 120-second slow collector.

Inventory, worker probes and the public manifest are independent observations.
Only allowlisted worker summaries leave this module; no raw inventory or logs do.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import re
import subprocess
import time
import urllib.error
import urllib.request

from . import common, remote_workers

__all__ = ["collect"]
MANIFEST_URL = "https://data.affine.io/traces/manifest.json"
_MAX_MANIFEST_BYTES = 6 * 1024 * 1024
_SOURCE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,79}\Z")


def _failure(title, source, code):
    return common.panel(
        title, "Read-only • refreshed every 120 seconds", status="error",
        notes=[f"Collection unavailable: {code}. Other sources collect independently."],
        sources=[source],
    )


def _inventory():
    result = subprocess.run(
        ["lium", "ps", "--format", "json"], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=12, check=True,
    )
    if len(result.stdout) > 2 * 1024 * 1024:
        raise ValueError("inventory_too_large")
    pods = json.loads(result.stdout)
    if not isinstance(pods, list):
        raise ValueError("invalid_inventory")
    return pods


def _worker_panel():
    try:
        pods = _inventory()
    except subprocess.TimeoutExpired:
        return _failure("Datagen workers", "lium ps --format json", "inventory_timeout")
    except (OSError, subprocess.CalledProcessError):
        return _failure("Datagen workers", "lium ps --format json", "inventory_unavailable")
    except (ValueError, TypeError):
        return _failure("Datagen workers", "lium ps --format json", "invalid_inventory")
    workers = remote_workers.collect_workers(pods)
    running = sum((w["rollout_processes"] or 0) > 0 for w in workers)
    reachable = sum(w["reachable"] for w in workers)
    unknown = sum(w["rollout_processes"] is None for w in workers)
    rows = []
    for w in workers:
        state = w.get("error") or ("running" if w["rollout_processes"] else "idle")
        rows.append([
            w["name"], state, w["rollout_processes"], w["bootstrap_processes"],
            w["last_cycle_at"], w["last_cycle_source"], w["last_cycle_policy"],
            w["log_updated_at"],
        ])
    maximum = max([w["rollout_processes"] or 0 for w in workers] + [1])
    notes = [
        "SSH is read-only, at most 8 concurrent connections, with a shared 14-second deadline.",
        "Process counts are rollout supervisors and bootstrap shells, not individual tasks. Zero means idle; null means unknown.",
        "Last cycle is historical worker-local wall time (timezone unspecified); log update is UTC. Neither proves successful rollouts.",
        "Cycle search reads at most four 64 KiB log tails per worker; missing cycles can be outside those tails.",
    ]
    if not workers:
        notes.append("No affine-datagen pods were found in the inventory.")
    return common.panel(
        "Datagen workers", "Read-only inventory + bounded SSH • every 120 seconds",
        status="ok" if workers and running == len(workers) and not any(w.get("error") for w in workers) else "warn",
        metrics=[
            common.metric("Datagen pods", len(workers)),
            common.metric("Reachable", reachable, f"of {len(workers)} pods"),
            common.metric("Running workers", running, f"{unknown} unknown", "green" if running else "warn"),
            common.metric("Idle workers", sum(w["rollout_processes"] == 0 for w in workers)),
        ],
        sections=[common.table(
            "Worker observations",
            ["Pod", "Status", "Rollout processes", "Bootstrap processes", "Last cycle (local)", "Source", "Policy", "Log updated (UTC)"],
            rows,
        )],
        bars=[dict(label=w["name"], value=w["rollout_processes"] or 0, max=maximum,
                   detail="Unknown" if w["rollout_processes"] is None else f'{w["rollout_processes"]} rollout supervisors',
                   tone="green" if w["rollout_processes"] else "warn") for w in workers],
        notes=notes,
        sources=["lium ps --format json", "Read-only SSH: /proc process counts and bounded cycle-log metadata"],
    )


def _fetch_manifest():
    request = urllib.request.Request(MANIFEST_URL, headers={
        "User-Agent": "Mozilla/5.0 (Affine read-only monitoring)",
        "Accept": "application/json", "Accept-Encoding": "identity",
    })
    deadline = time.monotonic() + 12
    with urllib.request.urlopen(request, timeout=8) as response:
        body = bytearray()
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError
            chunk = response.read1(min(65536, _MAX_MANIFEST_BYTES + 1 - len(body)))
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > _MAX_MANIFEST_BYTES:
                raise ValueError("manifest_too_large")
    return json.loads(body)


def _timestamp(value):
    if not isinstance(value, str) or len(value) > 40:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None


def _number(value):
    return value if type(value) is int and 0 <= value <= 10**18 else None


def _summary(manifest, now=None):
    if not isinstance(manifest, dict) or not isinstance(manifest.get("chunks"), list):
        raise ValueError("invalid_manifest")
    now = now or datetime.now(timezone.utc)
    groups = {}
    totals = dict(chunks=0, rollouts=0, bytes=0, recent=0, latest=None, bad=0)
    for chunk in manifest["chunks"]:
        if not isinstance(chunk, dict):
            totals["bad"] += 1
            continue
        sources = chunk.get("sources")
        valid_sources = isinstance(sources, list) and sources and all(
            isinstance(s, str) and _SOURCE.fullmatch(s) for s in sources
        )
        if not valid_sources:
            label = "Unknown source"
        elif len(set(sources)) != 1:
            label = "Mixed sources (unattributed)"
        else:
            label = sources[0]
        group = groups.setdefault(label, dict(chunks=0, rollouts=0, bytes=0, recent=0, latest=None, bad=0))
        count, size = _number(chunk.get("n_rollouts")), _number(chunk.get("bytes"))
        created = _timestamp(chunk.get("created_at"))
        bad = count is None or size is None or created is None or not valid_sources
        recent = count if count is not None and created is not None and 0 <= (now - created).total_seconds() <= 86400 else 0
        for target in (totals, group):
            target["chunks"] += 1
            target["rollouts"] += count or 0
            target["bytes"] += size or 0
            target["recent"] += recent
            target["bad"] += bool(bad)
            if created is not None and (target["latest"] is None or created > target["latest"]):
                target["latest"] = created
    return totals, groups


def _display_count(value, partial):
    return f"≥{value}" if partial else value


def _trace_panel():
    try:
        manifest = _fetch_manifest()
        totals, groups = _summary(manifest)
    except urllib.error.HTTPError as exc:
        return _failure("Trace corpus", MANIFEST_URL, f"http_{exc.code}")
    except (TimeoutError, urllib.error.URLError, OSError):
        return _failure("Trace corpus", MANIFEST_URL, "manifest_unavailable")
    except (ValueError, TypeError):
        return _failure("Trace corpus", MANIFEST_URL, "invalid_or_oversize_manifest")
    latest = totals["latest"]
    stale = latest is None or (datetime.now(timezone.utc) - latest).total_seconds() > 86400
    published = _timestamp(manifest.get("published_at"))
    mismatched = (manifest.get("n_chunks") != totals["chunks"] or
                  manifest.get("n_rollouts") != totals["rollouts"])
    partial = totals["bad"] > 0
    notes = [
        "24h counts use chunk creation time, not rollout execution or upload time; future-dated chunks are excluded.",
        "Bytes are compressed chunk bytes. Only the public manifest is fetched (6 MiB limit); no trace payloads are downloaded.",
        "Multi-source chunks are counted once under Mixed sources (unattributed); the manifest cannot split their rollouts by source.",
        "Latest creation and publication timestamps are UTC; this collector stores no historical manifest copies.",
    ]
    if partial:
        notes.append(f'{totals["bad"]} malformed chunk records/fields; affected counts are lower bounds. Other sources remain available.')
    if mismatched:
        notes.append("Manifest header totals disagree with usable chunk records; displayed totals are derived from chunk records.")
    if stale:
        notes.append("No chunk with a creation timestamp in the last 24 hours was observed.")
    rows = []
    for label, group in sorted(groups.items()):
        rows.append([label, group["chunks"],
                     _display_count(group["rollouts"], group["bad"]),
                     _display_count(group["bytes"], group["bad"]),
                     group["latest"].isoformat() if group["latest"] else None,
                     _display_count(group["recent"], group["bad"]),
                     "partial" if group["bad"] else "ok"])
    maximum = max([group["recent"] for group in groups.values()] + [1])
    return common.panel(
        "Trace corpus", "Public trace manifest • every 120 seconds",
        status="warn" if partial or mismatched or stale else "ok",
        metrics=[
            common.metric("Total chunks", _display_count(totals["chunks"], partial)),
            common.metric("Total rollouts", _display_count(totals["rollouts"], partial)),
            common.metric("Total bytes", _display_count(totals["bytes"], partial), "Compressed trace chunks"),
            common.metric("Rollouts / 24h", _display_count(totals["recent"], partial), "By chunk creation time"),
            common.metric("Latest creation", latest.isoformat() if latest else "Unknown"),
            common.metric("Manifest published", published.isoformat() if published else "Unknown"),
        ],
        sections=[common.table("Trace production by source",
                               ["Source", "Chunks", "Rollouts", "Bytes", "Latest creation (UTC)", "Rollouts / 24h", "Status"], rows)],
        bars=[dict(label=label, value=group["recent"], max=maximum,
                   detail=f'{group["recent"]} rollouts / 24h' + (" (partial)" if group["bad"] else ""),
                   tone="warn" if group["bad"] else "accent")
              for label, group in sorted(groups.items(), key=lambda pair: (-pair[1]["recent"], pair[0]))],
        notes=notes, sources=[MANIFEST_URL],
    )


def collect():
    """Return ``datagen`` and ``traces`` panels without mutating production."""
    collectors = {"datagen": (_worker_panel, "Datagen workers", "lium ps --format json"),
                  "traces": (_trace_panel, "Trace corpus", MANIFEST_URL)}
    panels = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {key: executor.submit(spec[0]) for key, spec in collectors.items()}
        for key, future in futures.items():
            try:
                panels[key] = future.result()
            except Exception:
                panels[key] = _failure(collectors[key][1], collectors[key][2], "collector_failed")
    return panels
