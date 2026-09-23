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
import fnmatch
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import tomllib

# >= this share of rollouts errored (infrastructure) -> the cell is a failed run: score null,
# status "failed", nothing quotable. 0.90 let King 14's SWE-bench cell through with 406/500
# docker pulls failed ("Unable to find image", Docker Hub cap) and a 70.4 finished-only over
# 54 tasks on the board (2026-09-17 18:18 UTC); half the tasks missing is not a score.
FAILED_CELL_ERROR_SHARE = 0.50
DATASET_SIZE = {"swebench-verified": 500, "terminal-bench-2": 89}


def job_running(run_dir, model: str, env_id: str):
    """A launcher / Harbor job for this cell is alive (the watcher's SWE and TB2 jobs run outside the pass)."""
    import subprocess
    base, _, tag = env_id.partition("@")
    d12 = run_dir.name.split("-")[1][:12] if "-" in run_dir.name else ""
    out = subprocess.run(["pgrep", "-fa", f"harbor_cell.py (run|resume) --env {base} .*--model (king-)?{d12}"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        tagged = "--budget-tag" in line
        if (tag and f"--budget-tag {tag}" in line) or (not tag and not tagged):
            return {"n_expected": DATASET_SIZE.get(base)}
    return None
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
FORBIDDEN_PREFIXES = ("traces/", "views/", "corpus/", "turns/")
PARTIAL = False
ETA = False
STATE_DIR_FOR_ETA = ""
ETA_CAP_S = 150 * 60   # fast topology: no cell should outlive ~2.5 h (SWE/TB2 on Daytona, shards in parallel)


def reference_walls(state_dir: Path) -> dict:
    """env -> king wall_seconds from the newest complete king card (for the ETA)."""
    best, walls = None, {}
    for p in state_dir.glob("*.json"):
        try:
            c = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if c.get("status") == "complete" and (c.get("king") or {}).get("reign") is not None:
            if best is None or (c.get("created_at") or "") > (best.get("created_at") or ""):
                best = c
    for r in (best or {}).get("rows", []):
        w = ((r.get("king") or {}).get("wall_seconds"))
        if w:
            walls[f"{r['env']}__t{r['temperature']:g}"] = float(w)
    return walls



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
    # A cell that was started (cmd.txt) and has no result at the FINAL publish is a run
    # failure — it must show as "run failed" on the card, never blank (Jacob 2026-09-20;
    # reign 20's card went out with 5 of 26 king cells and nothing said why). While the pass
    # is still running (--partial) these stay "unfinished" with an ETA.
    unfinished_now = []
    for d in sorted(run_dir.glob("*/*")):
        if d.is_dir() and (d / "cmd.txt").exists() and not (d / "summary.json").exists():
            unfinished_now.append(d)
            if not PARTIAL:
                model = d.parent.name
                env_id, _, t = d.name.rpartition("__t")
                try:
                    float(t)
                except ValueError:
                    continue   # a moved-aside cell dir (__t0.contended-HHMM, .dead-...) is not a live cell
                why = (d / "cmd.txt").read_text(errors="replace").strip().splitlines()
                # a Harbor job still running under this cell (the watcher's SWE / TB2 jobs share the run dir) is not
                # a failure: publish it as RUNNING with its trial count (Jacob 2026-09-23: "failed" placeholders were
                # shown for reign 21's @4h250 while it was 400/500 in)
                running = job_running(run_dir, model, env_id)
                n_done = len(list((d / "harbor").glob("*/result.json"))) if (d / "harbor").is_dir() else 0
                entry = {"env": env_id, "temperature": float(t), "model": model, "n": n_done, "n_errored": 0, "score": None, "ci95": None,
                         "completion_tokens": 0, "prompt_tokens": 0}
                if running:
                    entry.update(running=True, n_expected=running.get("n_expected"),
                                 failure_note=f"running: {n_done} of {running.get('n_expected') or '?'} trials done")
                else:
                    entry.update(run_failed=True,
                                 failure_note=("no pod (stock)" if why and why[0].startswith("no pod") else "the pass ended without a result for this cell"))
                cells.setdefault(env_id, {}).setdefault(f"t{float(t):g}", {}).setdefault(model, entry)
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
        if x.get("running"):
            return {"score": None, "ci95": None, "n": x.get("n", 0), "n_errored": 0, "status": "running", "running": True,
                    "n_expected": x.get("n_expected"), "failure": x.get("failure_note"), "finished_only": None, "by_class": None,
                    "completion_tokens": 0, "prompt_tokens": 0, "wall_seconds": None, "finish_length_frac": None}
        if x.get("run_failed"):
            return {"score": None, "ci95": None, "n": 0, "n_errored": 0, "status": "failed", "run_failed": True,
                    "failure": f"run failed: {x.get('failure_note')}", "finished_only": None, "by_class": None,
                    "completion_tokens": 0, "prompt_tokens": 0, "wall_seconds": None, "finish_length_frac": None}
        n = int(x.get("n") or 0)
        n_err = int(x.get("n_errored") or 0)
        # the error share is over trials that ran against a LIVE model: a reaped serving box or a
        # Daytona outage (n_infra_env) is our failure to retry, not the model's (Jacob 2026-09-22)
        n_live = int(x.get("n_live") if x.get("n_live") is not None else n)
        failed = n_live == 0 or n_err >= n_live or (n_live and n_err / n_live >= FAILED_CELL_ERROR_SHARE)
        n_env = int(x.get("n_infra_env") or 0)
        # a job our infrastructure cut down (half or more of its trials never saw the model) is not a
        # result yet either: it shows as failed WITH the reason, and the watcher resumes it
        infra_cut = bool(n) and n_env / n >= 0.5
        # an interrupted Harbor job (fewer live trials than the task set) is PARTIAL: the number is shown
        # with its n, but the cell is not final — the watcher resumes the job (2026-09-22: reign 15's
        # @4h250 published 0.389 on 216 of 500 live trials after a Daytona capacity stop)
        n_exp = x.get("n_expected")
        # <= 2 % of the task set never reaching a live model is noise-level (reign 15: 5 of 500 after three resumes) --
        # publish as final with the count in the note instead of holding the cell "partial" forever
        partial = bool(n_exp) and (int(n_exp) - n_live) > max(2, 0.02 * int(n_exp)) and not (failed or infra_cut)   # <= 2 trials or <= 2 % missing = final
        out = {"score": None if (failed or infra_cut) else x["score"], "ci95": None if (failed or infra_cut) else x["ci95"], "n": x["n"],
               "n_errored": x["n_errored"], "n_timeout": x.get("n_timeout"), "n_context_overflow": x.get("n_context_overflow"),
               "n_infra_env": x.get("n_infra_env"), "n_live": x.get("n_live"), "served_gpu": x.get("served_gpu"),
               "finished_only": None if (failed or infra_cut) else x.get("finished_only"), "completion_tokens": x["completion_tokens"],
               "prompt_tokens": x["prompt_tokens"], "wall_seconds": x.get("wall_seconds"),
               "finish_length_frac": x.get("finish_length_frac"),
               "by_class": x.get("by_class") or None,
               "status": "failed" if (failed or infra_cut) else ("partial" if partial else "ok")}
        if n_exp and not partial and n_live < int(n_exp) and not (failed or infra_cut):
            out["n_expected"] = int(n_exp)
            out["note"] = f"{int(n_exp) - n_live} of {n_exp} trials never ran against a live model (infrastructure); scored on {n_live}"
        if partial:
            out["n_expected"] = int(n_exp)
            out["failure"] = f"partial: {n_live} of {n_exp} trials ran against a live model; the job is being resumed"
        if infra_cut and not failed:
            out["failure"] = f"infrastructure: {n_env} of {n} trials never ran against a live model (serving box / Daytona); resuming"
        # cloud-sandbox / harness-change provenance (harbor_cell.py cells): the kingboard
        # flags a cell whose harness differs from the card's default for that env
        for key in ("sandbox", "harness", "harness_change", "harness_note", "budget", "served_by"):
            if x.get(key) is not None:
                out[key] = x[key]
        if failed:
            out["raw_score"] = x["score"]
            out["failure"] = f"{n_err}/{n} rollouts errored (run failure, not a model score)"
        if teacher:
            out["reused_from"] = x.get("reused_from")
        return out

    rows = []
    for env_id, temps in cells.items():
        # "<env>@<budget tag>" = the same env at a non-default budget (a separate column)
        base_env, _, budget_tag = env_id.partition("@")
        e = by_id.get(base_env, {})
        for tkey, models in temps.items():
            k, t = models.get("king"), models.get("teacher")
            if e.get("unverified"):
                # suite.toml env.unverified: the grader is suspect (2026-09-20 Gaia2) — the number is
                # withheld from the board (score null, status "unverified"), the raw value kept for the record
                rows.append({"env": env_id, "base_env": base_env, "budget_tag": budget_tag or None, "group": e.get("group"),
                             "temperature": float(tkey[1:]), "note": e.get("note"), "show_classes": None, "graded": e.get("graded", "deterministic"),
                             "judge": e.get("judge"), "n": (k or t or {}).get("n"), "unverified": e["unverified"],
                             "king": {**side(models.get("king")), **{"score": None, "ci95": None, "finished_only": None, "status": "unverified", "unverified": e["unverified"], "raw_score": models["king"].get("score")}} if models.get("king") else None,
                             "teacher": {**side(models.get("teacher"), teacher=True), **{"score": None, "ci95": None, "finished_only": None, "status": "unverified", "unverified": e["unverified"], "raw_score": models["teacher"].get("score")}} if models.get("teacher") else None,
                             "delta": None, "prime_eval_url": None, "cost": (manifest.get("cells", {}).get(f"{env_id}__{tkey}") or {})})
                continue
            rows.append({
                "env": env_id, "base_env": base_env, "budget_tag": budget_tag or None,
                "group": e.get("group"), "temperature": float(tkey[1:]),
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
    unfinished = sorted(str(d.relative_to(run_dir)) for d in unfinished_now) if PARTIAL else []
    failed_cells = [] if PARTIAL else sorted(str(d.relative_to(run_dir)) for d in unfinished_now)
    done_cells = sorted(str(d.relative_to(run_dir)) for d in run_dir.glob("*/*") if d.is_dir() and (d / "summary.json").exists())
    partial = bool(PARTIAL)
    return {
        "run_id": manifest.get("run_id"),
        "status": "partial" if partial else "complete",
        "running": partial,   # cells publish as they finish; the card fills in until the final publish
        "progress": {"done": len(done_cells), "total": len(done_cells) + len(unfinished), "remaining": unfinished,
                     "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     **({"eta": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + max(
                         [min(reference_walls(Path(STATE_DIR_FOR_ETA)).get(c.split("/")[-1], 2700.0), ETA_CAP_S) for c in unfinished] or [0.0])))}
                        if (ETA and unfinished) else {})},
        "unfinished_cells": unfinished,
        "failed_cells": failed_cells,      # started, no result at the final publish -> "run failed" rows
        "missing_roles": manifest.get("missing_roles") or None,
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
    ap.add_argument("--eta", action="store_true", help="estimate the card ETA from the remaining cells' reference wall times (fast pass)")
    ap.add_argument("--only-cells", default="", help="comma list of <model>/<env>__t<T>: upload just those "
                    "cell dirs (+ manifests), merge into the existing R2 index (a cell added to a published run)")
    a = ap.parse_args()
    global PARTIAL, ETA, STATE_DIR_FOR_ETA
    PARTIAL = a.partial
    ETA = a.eta
    STATE_DIR_FOR_ETA = a.state_dir
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
        if only and not (any(fnmatch.fnmatch(rel, c + "/*") for c in only) or "/" not in rel):
            continue      # partial publish: the named cells (globs ok, e.g. king/*) + the run's top-level manifests
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
