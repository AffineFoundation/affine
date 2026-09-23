#!/usr/bin/env python
"""Per-task outcome table from the env-backfill traces, for the fold's band filter.

The env backfill (`rollouts.backfill`, coverage queue) rolls every king on
every env — 24+ graded rollouts per (env, reign) — and publishes them under
`traces-backfill/` on data.affine.io, OUTSIDE the fold's traces manifest by
design (those rollouts never enter D). Their pass / fail per task is exactly
the king-side evidence `[band_filter]` wants (fold worker, requests.md
2026-09-22 21:15): "the king seat did NOT solve it" can only be judged where
the seat has tried the task. This job reduces the backfill traces to one row
per (source, task, seat model) and publishes it next to the corpus:

    https://data.affine.io/backfill/task_outcomes.jsonl      one JSON per line
    https://data.affine.io/backfill/task_outcomes.meta.json  manifest sha, counts

Row: {"source", "sid", "uid", "seat": "king"|"teacher"|"frontier",
      "king_digest": <12 hex or "">, "model": <policy.model>, "n": graded
      rollouts, "n_solved", "n_errored", "policies": [...], "last_at"}.
Grading = affine.corpus.view.rollout_outcome (the fold's own rule); errored /
unscored rollouts are listed in n_errored and never in n. Only drops can
come out of this table on the fold side (no backfill turn enters D).

Incremental: every chunk is reduced once and cached under
affine/state/backfill_outcomes/chunks/<sha256>.json; a run re-reads the
manifest, reduces new chunks, re-aggregates and republishes only when the
manifest sha changed. Credentials as ops/corpus_build.py:
DATA_R2_ACCESS_KEY_ID / DATA_R2_SECRET_ACCESS_KEY (+ DATA_R2_ENDPOINT).

    .venv/bin/python ops/coverage/backfill_outcomes.py [--no-publish] [--force]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "affine")]
from affine.corpus.view import rollout_outcome  # noqa: E402

BASE = os.environ.get("AFFINE_DATA_BASE", "https://data.affine.io")
MANIFEST_URL = f"{BASE}/traces-backfill/manifest.json"
STATE = REPO / "affine" / "state" / "backfill_outcomes"
CHUNKS = STATE / "chunks"
OUT_JSONL = STATE / "task_outcomes.jsonl"
OUT_META = STATE / "task_outcomes.meta.json"
R2_KEY = "backfill/task_outcomes.jsonl"
R2_META_KEY = "backfill/task_outcomes.meta.json"
UA = {"User-Agent": "affine-backfill-outcomes/0.1"}
DIGEST_RE = re.compile(r"^(?:backfill_)?([0-9a-f]{12})_")


def env_value(key: str) -> str:
    if os.environ.get(key):
        return os.environ[key]
    for p in (REPO / ".env", Path.home() / ".affine-validator.env"):
        try:
            for line in p.read_text().splitlines():
                line = line.strip().removeprefix("export ").strip()
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


def seat_of(policy: dict) -> tuple[str, str]:
    """(seat, king_digest) from the backfill policy stamp.
    Kings: id backfill_<d12>_<harness>, model king/king-<d12>; teacher:
    backfill_teacher_*; frontier: frontier_<slug>_*."""
    pid = str(policy.get("id") or policy.get("policy_id") or "")
    model = str(policy.get("model") or "")
    if pid.startswith("frontier_"):
        return "frontier", ""
    if pid.startswith("backfill_teacher_") or pid.startswith("teacher_"):
        return "teacher", ""
    m = re.search(r"king-([0-9a-f]{12})", model)
    if m:
        return "king", m.group(1)
    m = DIGEST_RE.match(pid.removeprefix("backfill_"))
    if m:
        return "king", m.group(1)
    if pid.startswith("king_"):
        return "king", ""
    return "other", ""


def reduce_chunk(raw: bytes) -> list[dict]:
    rows = []
    with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            task = e.get("task") or {}
            policy = e.get("policy") or {}
            seat, digest = seat_of(policy)
            trace = e.get("trace") or e
            rows.append({
                "source": e.get("source"), "sid": task.get("sid"), "uid": task.get("uid"),
                "seat": seat, "king_digest": digest, "model": policy.get("model") or "",
                "policy": policy.get("id") or policy.get("policy_id") or "",
                "outcome": rollout_outcome(trace), "at": e.get("stored_at") or e.get("created_at") or "",
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-publish", action="store_true", help="write the local files only")
    ap.add_argument("--force", action="store_true", help="re-aggregate and publish even if the manifest sha is unchanged")
    args = ap.parse_args()
    CHUNKS.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=120, headers=UA, follow_redirects=True)
    raw = client.get(MANIFEST_URL)
    raw.raise_for_status()
    manifest_sha = hashlib.sha256(raw.content).hexdigest()
    manifest = raw.json()
    prev = json.loads(OUT_META.read_text()) if OUT_META.exists() else {}
    if prev.get("traces_backfill_manifest_sha256") == manifest_sha and not args.force:
        print(f"unchanged: manifest {manifest_sha[:12]} already reduced ({prev.get('n_tasks')} tasks)")
        return 0
    new = 0
    for c in manifest["chunks"]:
        cache = CHUNKS / f"{c['sha256']}.json"
        if cache.exists():
            continue
        r = client.get(f"{BASE}/{c['key']}")
        if r.status_code != 200:
            print(f"skip {c['key']}: HTTP {r.status_code}")
            continue
        if hashlib.sha256(r.content).hexdigest() != c["sha256"]:
            print(f"skip {c['key']}: sha mismatch")
            continue
        cache.write_text(json.dumps(reduce_chunk(r.content)))
        new += 1
    agg: dict[tuple, dict] = {}
    n_rollouts = 0
    listed = {c["sha256"] for c in manifest["chunks"]}
    for cache in CHUNKS.glob("*.json"):
        if cache.stem not in listed:
            continue                      # a chunk the manifest no longer lists
        for r in json.loads(cache.read_text()):
            n_rollouts += 1
            if not r.get("source") or not r.get("sid"):
                continue
            key = (r["source"], r["sid"], r["seat"], r["king_digest"], r["model"])
            a = agg.setdefault(key, {"source": r["source"], "sid": r["sid"], "uid": r.get("uid"),
                                     "seat": r["seat"], "king_digest": r["king_digest"], "model": r["model"],
                                     "n": 0, "n_solved": 0, "n_errored": 0, "policies": set(), "last_at": ""})
            if r["outcome"] == "solved":
                a["n"] += 1; a["n_solved"] += 1
            elif r["outcome"] == "failed":
                a["n"] += 1
            else:
                a["n_errored"] += 1
            if r.get("policy"):
                a["policies"].add(r["policy"])
            a["last_at"] = max(a["last_at"], r.get("at") or "")
    rows = sorted(agg.values(), key=lambda a: (a["source"], a["sid"], a["seat"], a["king_digest"], a["model"]))
    body = "".join(json.dumps({**a, "policies": sorted(a["policies"])}, sort_keys=True) + "\n" for a in rows).encode()
    by_seat = defaultdict(int)
    for a in rows:
        by_seat[a["seat"]] += 1
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "traces_backfill_manifest_sha256": manifest_sha,
        "traces_backfill_manifest_published_at": manifest.get("published_at"),
        "n_chunks": len(listed), "n_rollouts": n_rollouts, "n_tasks": len(rows),
        "rows_by_seat": dict(by_seat), "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
        "fields": ["source", "sid", "uid", "seat", "king_digest", "model", "n", "n_solved", "n_errored", "policies", "last_at"],
        "grading": "affine.corpus.view.rollout_outcome; n = solved + failed, errored/unscored in n_errored",
    }
    STATE.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_bytes(body)
    OUT_META.write_text(json.dumps(meta, indent=1))
    print(f"reduced {new} new chunk(s); {n_rollouts} rollouts -> {len(rows)} (source, task, seat, model) rows {dict(by_seat)}")
    if args.no_publish:
        return 0
    ak, sk = env_value("DATA_R2_ACCESS_KEY_ID"), env_value("DATA_R2_SECRET_ACCESS_KEY")
    endpoint = env_value("DATA_R2_ENDPOINT")
    if not (ak and sk and endpoint):
        print("DATA_R2_* missing; local files written, not published")
        return 2
    import boto3
    s3 = boto3.client("s3", endpoint_url=endpoint, region_name="auto", aws_access_key_id=ak, aws_secret_access_key=sk)
    bucket = env_value("DATA_R2_BUCKET") or "affine-data"
    s3.put_object(Bucket=bucket, Key=R2_KEY, Body=body, ContentType="application/x-ndjson", CacheControl="no-cache")
    s3.put_object(Bucket=bucket, Key=R2_META_KEY, Body=OUT_META.read_bytes(), ContentType="application/json", CacheControl="no-cache")
    print(f"published {BASE}/{R2_KEY} ({len(body)} B, {len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
