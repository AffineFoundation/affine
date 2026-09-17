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

FAILED_CELL_ERROR_SHARE = 0.90   # >= this share errored -> the cell is a failed run, score null
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
FORBIDDEN_PREFIXES = ("traces/", "views/", "corpus/", "turns/")
PARTIAL = False



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

    def side(x: dict | None, teacher: bool = False) -> dict | None:
        """Card cell for one model side. A cell whose rollouts (nearly) all
        errored is a run failure — infrastructure, not the model — and must
        never publish as a score (2026-09-17: Genesis SWE-bench showed 0.0 on
        affine.io after docker never started on the pod: 500/500 errored, 0
        tokens). `score` is null and `status` = "failed"; the numbers stay for
        the record so the gap check can retry the cell."""
        if not x:
            return None
        n = int(x.get("n") or 0)
        n_err = int(x.get("n_errored") or 0)
        failed = n == 0 or n_err >= n or (n and n_err / n >= FAILED_CELL_ERROR_SHARE)
        out = {"score": None if failed else x["score"], "ci95": None if failed else x["ci95"], "n": x["n"],
               "n_errored": x["n_errored"], "n_timeout": x.get("n_timeout"), "n_context_overflow": x.get("n_context_overflow"),
               "finished_only": None if failed else x.get("finished_only"), "completion_tokens": x["completion_tokens"],
               "prompt_tokens": x["prompt_tokens"], "wall_seconds": x.get("wall_seconds"),
               "finish_length_frac": x.get("finish_length_frac"),
               "by_class": x.get("by_class") or None,
               "status": "failed" if failed else "ok"}
        if failed:
            out["raw_score"] = x["score"]
            out["failure"] = f"{n_err}/{n} rollouts errored (run failure, not a model score)"
        if teacher:
            out["reused_from"] = x.get("reused_from")
        return out

    rows = []
    for env_id, temps in cells.items():
        e = by_id.get(env_id, {})
        for tkey, models in temps.items():
            k, t = models.get("king"), models.get("teacher")
            rows.append({
                "env": env_id, "group": e.get("group"), "temperature": float(tkey[1:]),
                "note": e.get("note"),
                "show_classes": e.get("show_classes"),
                "graded": e.get("graded", "deterministic"),   # "llm_judge" = advisory, never in the score
                "judge": e.get("judge"),
                "n": (k or t or {}).get("n"),
                "king": side(k),
                "teacher": side(t, teacher=True),
                "delta": None if not (k and t and side(k)["score"] is not None and side(t)["score"] is not None)
                else round(k["score"] - t["score"], 4),
                "prime_eval_url": {m: s.get("prime_eval_url") for m, s in models.items() if s.get("prime_eval_url")} or None,
                "cost": (manifest.get("cells", {}).get(f"{env_id}__{tkey}") or {}),
            })
    unfinished = sorted(str(d.relative_to(run_dir)) for d in run_dir.glob("*/*")
                        if d.is_dir() and (d / "cmd.txt").exists() and not (d / "summary.json").exists())
    return {
        "run_id": manifest.get("run_id"),
        "status": "partial" if (unfinished or PARTIAL) else "complete",
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
    ap.add_argument("--partial", action="store_true", help="mark the card partial (more cells will follow, e.g. the sandbox phase)")
    ap.add_argument("--only-cells", default="", help="comma list of <model>/<env>__t<T>: upload just those "
                    "cell dirs (+ manifests), merge into the existing R2 index (a cell added to a published run)")
    a = ap.parse_args()
    global PARTIAL
    PARTIAL = a.partial
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
    only = [c.strip("/") for c in a.only_cells.split(",") if c.strip()]
    files = []
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(run_dir).as_posix()
        if only and not (any(rel.startswith(c + "/") for c in only) or "/" not in rel):
            continue      # partial publish: the named cells + the run's top-level manifests
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
    if only:
        try:
            old = json.loads(s3.get_object(Bucket=bucket, Key=prefix + "index.json")["Body"].read())
            uploaded = {rel for _, rel in files}
            index = [f for f in old.get("files", []) if f["path"] not in uploaded]
        except s3.exceptions.NoSuchKey:
            index = []
    for p, rel in files:
        key = prefix + rel
        s3.upload_file(str(p), bucket, key)
        index.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    index.sort(key=lambda f: f["path"])
    index_doc = {"run_id": run_id, "bucket": bucket, "prefix": prefix,
                 "public_base": f"https://data.affine.io/{prefix}",
                 "files": index, "published_at": card["published_at"]}
    s3.put_object(Bucket=bucket, Key=prefix + "index.json",
                  Body=json.dumps(index_doc, indent=1).encode(), ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=prefix + "scorecard.json",
                  Body=json.dumps(card, indent=1).encode(), ContentType="application/json")
    log(f"uploaded {len(files)} files ({sum(p.stat().st_size for p, _ in files)/1e6:.1f} MB; index {len(index)} files) -> "
        f"https://data.affine.io/{prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
