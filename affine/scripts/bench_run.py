"""Run an advisory bench suite on the bench pod for a model that is NOT a
submission (teacher, reference base, ...) and record it exactly like the
validator does: bench_history.jsonl row + state/benches artifact + index line.

The validator's own bench queue only ever holds kings (policy = accepted), so
reference points such as the frozen teacher never got a row. This talks to the
same bench pod (through the validator's local tunnel) with the same request
shape; job ids carry a name instead of the shared counter so they cannot
collide with validator-issued ids. The dashboard picks the row up from
bench_history on its next flush.

  source ~/.affine-validator.env
  python affine/scripts/bench_run.py --repo Qwen/Qwen3.8-27B --label teacher \
      --suite swe_rebench_lite_300 --suite swe_rebench_lite
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine.config import load_config  # noqa: E402

STATE = REPO / "affine/state"
POLL_S = 30


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hf_main_sha(repo: str) -> str:
    r = httpx.get(f"https://huggingface.co/api/models/{repo}/revision/main", timeout=30,
                  headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {})
    r.raise_for_status()
    return r.json()["sha"]


def run_suite(base: str, headers: dict, cfg, repo: str, revision: str,
              suite: str, label: str) -> dict:
    job_id = f"bench-{label}-{suite}"
    job = {"job_id": job_id, "repo": repo, "revision": revision, "hotkey": "",
           "suite": suite, "label": label, "state": "QUEUED", "queued_at": now_iso()}
    while True:
        r = httpx.post(f"{base}/bench", json={
            "repo": repo, "revision": revision, "suite": suite,
            "num_trials": cfg.bench.num_trials, "max_concurrency": cfg.bench.max_concurrency,
            "user_llm": cfg.bench.user_llm}, timeout=30, headers=headers)
        if r.status_code == 409:
            print(f"{job_id}: bench pod busy, waiting"); time.sleep(POLL_S); continue
        r.raise_for_status()
        remote_id = r.json()["job_id"]
        break
    job["state"] = "RUNNING"
    job["started_at"] = now_iso()
    print(f"{job_id}: dispatched as {remote_id}")
    t0 = time.time()
    while True:
        time.sleep(POLL_S)
        r = httpx.get(f"{base}/bench/{remote_id}", timeout=30, headers=headers)
        r.raise_for_status()
        remote = r.json()
        state = remote.get("state", "")
        if state in ("queued", "LOADING_MODEL", "RUNNING"):
            prog = remote.get("progress") or remote.get("phase") or ""
            print(f"  {state} {prog} ({time.time() - t0:.0f}s)", flush=True)
            continue
        if state == "aborted":
            raise SystemExit(f"{job_id}: aborted on the pod")
        result = remote.get("result", {"ok": False, "error": f"state={state}"})
        break
    job["state"] = "DONE" if result.get("ok") else "FAILED"
    job["finished_at"] = now_iso()
    job["result"] = result
    # Artifact first (same order as the validator: fetch-now beats fetch-later).
    art = httpx.get(f"{base}/bench/{remote_id}/artifact", timeout=120, headers=headers)
    if art.status_code == 200:
        benches = STATE / "benches"
        benches.mkdir(exist_ok=True)
        name = f"{job_id}.json.gz"
        (benches / name).write_bytes(art.content)
        with open(benches / "index.jsonl", "a") as f:
            f.write(json.dumps({"key": f"benches/{name}", "bytes": len(art.content),
                                "at": now_iso(), "job_id": job_id, "repo": repo,
                                "revision": revision, "hotkey": "", "suite": suite,
                                "label": label, "ok": bool(result.get("ok")),
                                "score": result.get("score")}) + "\n")
        n_inst = len(json.loads(gzip.decompress(art.content)).get("instances", {}))
        print(f"{job_id}: artifact saved ({len(art.content)} bytes, {n_inst} instances)")
    with open(STATE / "bench_history.jsonl", "a") as f:
        f.write(json.dumps(job, default=str) + "\n")
    print(f"{job_id}: {job['state']} {json.dumps(result, default=str)[:300]}")
    return job


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default="", help="default: HF main sha")
    ap.add_argument("--label", required=True)
    ap.add_argument("--suite", action="append", required=True)
    args = ap.parse_args()
    cfg = load_config()
    token = cfg.secrets.eval_token or os.environ.get("AFFINE_EVAL_TOKEN", "")
    if not token:
        raise SystemExit("AFFINE_EVAL_TOKEN missing (source ~/.affine-validator.env)")
    base = f"http://127.0.0.1:{cfg.bench_machine.port}"
    headers = {"X-Affine-Token": token}
    h = httpx.get(f"{base}/health", timeout=15, headers=headers)
    h.raise_for_status()
    print("bench pod health:", json.dumps(h.json())[:200])
    revision = args.revision or hf_main_sha(args.repo)
    print(f"{args.repo}@{revision[:12]} label={args.label} suites={args.suite}")
    for suite in args.suite:
        run_suite(base, headers, cfg, args.repo, revision, suite, args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
