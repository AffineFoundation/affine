#!/usr/bin/env python
"""Publish a benchmark-suite run: copy the run directory to R2
(`affine-data` bucket, `research/benchsuite/<run_id>/`, public at
https://data.affine.io/research/benchsuite/<run_id>/) and write the compact
scorecard JSON the kingboard + report read (`affine/state/benchsuite/
<run_id>.json`). Never touches `traces/` or any prefix the fold reads.

  python publish.py --run-dir ~/benchsuite/runs/<run_id> [--no-r2] [--state-dir affine/state/benchsuite]

R2 credentials: DATA_R2_ACCESS_KEY_ID / DATA_R2_SECRET_ACCESS_KEY /
DATA_R2_ENDPOINT (the corpus fold's affine-data-only key) or the repo .env.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import tomllib
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
FORBIDDEN_PREFIXES = ("traces/", "views/", "corpus/", "turns/")


def log(msg: str) -> None:
    print(f"[publish] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def env_from_files() -> None:
    """Load DATA_R2_* from ~/.affine-validator.env / repo .env when unset (never overrides)."""
    for path in (Path.home() / ".affine-validator.env", REPO / ".env"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            if k.startswith(("DATA_R2_", "R2_")) and k not in os.environ:
                os.environ[k] = v.strip().strip("'\"")


def r2_client():
    env_from_files()
    endpoint = os.environ.get("DATA_R2_ENDPOINT") or os.environ.get("R2_ENDPOINT")
    key = os.environ.get("DATA_R2_ACCESS_KEY_ID")
    secret = os.environ.get("DATA_R2_SECRET_ACCESS_KEY")
    if not (endpoint and key and secret):
        raise SystemExit("DATA_R2_ENDPOINT / DATA_R2_ACCESS_KEY_ID / DATA_R2_SECRET_ACCESS_KEY not set")
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=key,
                        aws_secret_access_key=secret, region_name="auto",
                        config=BotoConfig(retries={"max_attempts": 8, "mode": "adaptive"}))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def scorecard(run_dir: Path) -> dict:
    """Compact per-run scorecard from manifest.json + every cell's summary.json
    (rollout rows dropped — those stay in R2)."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    # Concurrent runners (chat sets vs sandbox sets) keep their own manifest*.json.
    for extra in sorted(run_dir.glob("manifest-*.json")):
        m2 = json.loads(extra.read_text())
        manifest.setdefault("cells", {}).update(m2.get("cells") or {})
        manifest["prime_wallet_log"] = (manifest.get("prime_wallet_log") or []) + (m2.get("prime_wallet_log") or [])
        manifest["prime_spent_usd"] = round(float(manifest.get("prime_spent_usd") or 0)
                                            + float(m2.get("prime_spent_usd") or 0), 2)
    cells = {}
    for summ_path in sorted(run_dir.glob("*/*/summary.json")):
        s = json.loads(summ_path.read_text())
        s = {k: v for k, v in s.items() if k != "rollouts"}
        cells.setdefault(s["env"], {}).setdefault(f"t{s['temperature']:g}", {})[s["model"]] = s
    by_id = {e["id"]: e for e in SUITE["envs"]}
    rows = []
    for env_id, temps in cells.items():
        e = by_id.get(env_id, {})
        for tkey, models in temps.items():
            k, t = models.get("king"), models.get("teacher")
            rows.append({
                "env": env_id, "group": e.get("group"), "temperature": float(tkey[1:]),
                "note": e.get("note"),
                "n": (k or t or {}).get("n"),
                "king": None if not k else {"score": k["score"], "ci95": k["ci95"], "n": k["n"],
                                            "n_errored": k["n_errored"], "n_timeout": k.get("n_timeout"), "n_context_overflow": k.get("n_context_overflow"), "finished_only": k.get("finished_only"), "completion_tokens": k["completion_tokens"],
                                            "prompt_tokens": k["prompt_tokens"], "wall_seconds": k.get("wall_seconds"),
                                            "finish_length_frac": k.get("finish_length_frac")},
                "teacher": None if not t else {"score": t["score"], "ci95": t["ci95"], "n": t["n"],
                                               "n_errored": t["n_errored"], "n_timeout": t.get("n_timeout"), "n_context_overflow": t.get("n_context_overflow"), "finished_only": t.get("finished_only"), "completion_tokens": t["completion_tokens"],
                                               "prompt_tokens": t["prompt_tokens"], "wall_seconds": t.get("wall_seconds"),
                                               "finish_length_frac": t.get("finish_length_frac"),
                                               "reused_from": t.get("reused_from")},
                "delta": None if not (k and t) else round(k["score"] - t["score"], 4),
                "prime_eval_url": {m: s.get("prime_eval_url") for m, s in models.items() if s.get("prime_eval_url")} or None,
                "cost": (manifest.get("cells", {}).get(f"{env_id}__{tkey}") or {}),
            })
    unfinished = sorted(str(d.relative_to(run_dir)) for d in run_dir.glob("*/*")
                        if d.is_dir() and (d / "cmd.txt").exists() and not (d / "summary.json").exists())
    return {
        "run_id": manifest.get("run_id"),
        "status": "partial" if unfinished else "complete",
        "unfinished_cells": unfinished,
        "king": manifest.get("king"), "teacher": manifest.get("teacher"),
        "where": manifest.get("where"), "code": manifest.get("code"),
        "created_at": manifest.get("created_at"),
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prime_spent_usd": manifest.get("prime_spent_usd"),
        "lock": manifest.get("lock"),
        "duel": (manifest.get("king") or {}).get("duel"),
        "prime_evals": manifest.get("prime_evals"),
        "prime_evals_account": manifest.get("prime_evals_account"),
        "mode": manifest.get("mode"),
        "sandbox_trigger": manifest.get("sandbox_trigger"),
        "pod": manifest.get("pod"),
        "r2_prefix": f"{SUITE['suite']['r2_prefix']}{manifest.get('run_id')}/",
        "rows": rows,
        "skipped": manifest.get("skipped", []),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--state-dir", default=str(REPO / SUITE["suite"]["state_dir"]))
    ap.add_argument("--no-r2", action="store_true")
    ap.add_argument("--only-state", action="store_true", help="write the scorecard JSON only")
    a = ap.parse_args()
    run_dir = Path(a.run_dir).expanduser().resolve()
    run_id = run_dir.name
    card = scorecard(run_dir)
    state_dir = Path(a.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{run_id}.json").write_text(json.dumps(card, indent=1))
    log(f"scorecard -> {state_dir / (run_id + '.json')} ({len(card['rows'])} rows)")
    if card.get("status") == "complete" and (card.get("king") or {}).get("digest"):
        marker = HERE / "state" / f"inflight-{card['king']['digest'][:12]}"
        if marker.exists():
            marker.unlink()
            log(f"removed {marker.name} (watch mode resumes for this king)")
    if a.no_r2 or a.only_state:
        return 0

    prefix = f"{SUITE['suite']['r2_prefix']}{run_id}/"
    assert not prefix.startswith(FORBIDDEN_PREFIXES), prefix
    bucket = SUITE["suite"]["r2_bucket"]
    s3 = r2_client()
    files = []
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(run_dir).as_posix()
        if rel.endswith("traces.jsonl"):
            gz = p.with_suffix(".jsonl.gz")
            if not gz.exists() or gz.stat().st_mtime < p.stat().st_mtime:
                with p.open("rb") as fi, gzip.open(gz, "wb", compresslevel=6) as fo:
                    shutil.copyfileobj(fi, fo)
            p, rel = gz, rel + ".gz"
        elif rel.endswith("traces.jsonl.gz") and p.with_suffix("").exists():
            continue      # uploaded via its .jsonl sibling above
        files.append((p, rel))
    index = []
    for p, rel in files:
        key = prefix + rel
        s3.upload_file(str(p), bucket, key)
        index.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    index_doc = {"run_id": run_id, "bucket": bucket, "prefix": prefix,
                 "public_base": f"https://data.affine.io/{prefix}",
                 "files": index, "published_at": card["published_at"]}
    s3.put_object(Bucket=bucket, Key=prefix + "index.json",
                  Body=json.dumps(index_doc, indent=1).encode(), ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=prefix + "scorecard.json",
                  Body=json.dumps(card, indent=1).encode(), ContentType="application/json")
    log(f"uploaded {len(files)} files ({sum(f['bytes'] for f in index)/1e6:.1f} MB) -> "
        f"https://data.affine.io/{prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
