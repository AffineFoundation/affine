#!/usr/bin/env python
"""Benchmark-suite watcher (pm2 `affine-benchsuite`): run the suite on every
crown and re-run the current king weekly.

Loop every --interval seconds:
  1. read affine/state/state.json -> king (digest, reign, hotkey).
  2. if no published run exists for this digest, or the newest one is older
     than --weekly-days, start one pass via `run_pass.sh` (rent a pod, serve
     king + teacher, run every env, publish to R2 + affine/state/benchsuite,
     release the pod). One pass at a time; a failed pass is retried after
     --retry-hours.
  3. state in ops/benchsuite/state/watch.json.

`--once` does one tick; `--dry-run` reports what it would start. The pass
itself is `run_pass.sh <digest> <reign>`; see that script for the
provider (Prime pod by default) and the budget guard.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
STATE_DIR = HERE / "state"
WATCH_JSON = STATE_DIR / "watch.json"
CARDS_DIR = REPO / SUITE["suite"]["state_dir"]
VALIDATOR_STATE = REPO / "affine" / "state" / "state.json"


def log(msg: str) -> None:
    print(f"[benchsuite-watch] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}",
          flush=True)


def load_watch() -> dict:
    if WATCH_JSON.exists():
        return json.loads(WATCH_JSON.read_text())
    return {"passes": []}


def save_watch(w: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = WATCH_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(w, indent=1))
    tmp.replace(WATCH_JSON)


def current_king() -> dict | None:
    try:
        st = json.loads(VALIDATOR_STATE.read_text())
    except (OSError, ValueError):
        return None
    k = st.get("king") or {}
    if not k.get("revision"):
        return None
    return {"digest": k["revision"], "reign": k.get("reign_number"), "hotkey": k.get("hotkey"),
            "repo": k.get("repo"), "crowned_at": k.get("crowned_at")}


def latest_card_for(digest: str) -> dict | None:
    best = None
    for p in CARDS_DIR.glob("*.json"):
        try:
            c = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if (c.get("king") or {}).get("digest") == digest:
            if best is None or (c.get("created_at") or "") > (best.get("created_at") or ""):
                best = c
    return best


def iso_to_ts(s: str | None) -> float:
    if not s:
        return 0.0
    return time.mktime(time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))


def tick(a: argparse.Namespace) -> None:
    w = load_watch()
    king = current_king()
    if king is None:
        log("no king in state.json; nothing to do")
        return
    running = [p for p in w["passes"] if p.get("state") == "running"]
    if running:
        p = running[0]
        if a.dry_run or p.get("pid") is None:
            log(f"pass {p['run_id']} marked running")
            return
        rc = subprocess.run(["kill", "-0", str(p["pid"])], capture_output=True).returncode
        if rc == 0:
            log(f"pass {p['run_id']} still running (pid {p['pid']})")
            return
        done_marker = Path(p["log"]).with_suffix(".exit")
        code = int(done_marker.read_text().strip()) if done_marker.exists() else -1
        p.update(state="done" if code == 0 else "failed", exit_code=code,
                 finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        save_watch(w)
        log(f"pass {p['run_id']} finished exit={code}")
    card = latest_card_for(king["digest"])
    why = None
    if card is None:
        why = f"no run for reign {king['reign']} ({king['digest'][:12]})"
    elif time.time() - iso_to_ts(card.get("created_at")) > a.weekly_days * 86400:
        why = f"last run for reign {king['reign']} is older than {a.weekly_days} d"
    if why is None:
        return
    last_fail = [p for p in w["passes"] if p.get("state") == "failed" and p.get("digest") == king["digest"]]
    if last_fail and time.time() - iso_to_ts(last_fail[-1].get("finished_at")) < a.retry_hours * 3600:
        log(f"last pass for this king failed < {a.retry_hours} h ago; waiting")
        return
    run_id = time.strftime("%Y%m%dT%H%MZ", time.gmtime()) + f"-{king['digest'][:12]}"
    log(f"start pass {run_id}: {why}")
    if a.dry_run:
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logp = STATE_DIR / f"pass-{run_id}.log"
    cmd = ["bash", str(HERE / "run_pass.sh"), king["digest"], str(king["reign"]), run_id]
    with logp.open("a") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE),
                                start_new_session=True)
    w["passes"].append({"run_id": run_id, "digest": king["digest"], "reign": king["reign"],
                        "state": "running", "pid": proc.pid, "log": str(logp), "why": why,
                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    save_watch(w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--weekly-days", type=float, default=7.0)
    ap.add_argument("--retry-hours", type=float, default=6.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    while True:
        try:
            tick(a)
        except Exception as e:  # keep the service alive; the next tick retries
            log(f"tick error: {e!r}")
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
