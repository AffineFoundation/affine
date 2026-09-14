#!/usr/bin/env python
"""Benchmark-suite watcher (pm2 `affine-benchsuite`): run the suite on every
crown, re-run the current king weekly, and (when [challenger].enabled) bench
the day's top-k near-miss challengers chat-only — all on Prime's stack
([modes].default = prime).

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
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from challengers import near_misses
from weights_fingerprint import same_weights

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
STATE_DIR = HERE / "state"
FINGERPRINT_TIMEOUT_S = 120     # the identity check must not delay the pass
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


def latest_card() -> dict | None:
    best = None
    for p in CARDS_DIR.glob("*.json"):
        try:
            c = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if c.get("status") in ("complete", "partial") and (c.get("king") or {}).get("digest"):
            if best is None or (c.get("created_at") or "") > (best.get("created_at") or ""):
                best = c
    return best


def skip_if_identical_weights(king: dict, run_id: str) -> bool:
    """Guard: if the new king's TENSOR set equals the last benchmarked king's,
    do not spend a pass — publish a stub card that points at the previous run
    (the kingboard shows the previous numbers under the new reign with a note)."""
    prev = latest_card()
    if prev is None or prev["king"]["digest"] == king["digest"]:
        return False
    # Cheap by construction (manifest hashes, then sampled range reads — no
    # shard is pulled) and hard-capped: a slow or broken check must never hold
    # the pass. 2026-09-14 the old full-shard fingerprint took 90 min.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(same_weights, king["digest"], prev["king"]["digest"])
    try:
        same, how = fut.result(timeout=FINGERPRINT_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        log(f"weight-identity check did not finish in {FINGERPRINT_TIMEOUT_S} s; benching anyway")
        pool.shutdown(wait=False, cancel_futures=True)
        return False
    except Exception as e:  # manifest / range trouble: fall through and bench
        log(f"weight-identity check failed ({e!r}); benching anyway")
        return False
    finally:
        pool.shutdown(wait=False)
    if not same:
        log(f"weights differ from reign {prev['king'].get('reign')} ({how}); benching")
        return False
    log(f"identical weights: {how}")
    stub = dict(prev)
    stub.update({
        "run_id": run_id, "king": king, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "skipped_identical_weights",
        "identical_to": {"run_id": prev["run_id"], "digest": prev["king"]["digest"],
                         "reign": prev["king"].get("reign"), "how": how},
        "mode": "skipped", "prime_spent_usd": 0.0,
    })
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    (CARDS_DIR / f"{run_id}.json").write_text(json.dumps(stub, indent=1))
    log(f"reign {king.get('reign')} ({king['digest'][:12]}) has the same weights as "
        f"reign {prev['king'].get('reign')} ({prev['king']['digest'][:12]}); pass skipped, stub card written")
    return True


def start_pass(w: dict, ref: str, label: str, run_id: str, mode: str, why: str,
               extra_env: dict | None = None, **fields) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logp = STATE_DIR / f"pass-{run_id}.log"
    env = dict(os.environ)
    env.update(extra_env or {})
    cmd = ["bash", str(HERE / "run_pass.sh"), ref, label, run_id, mode]
    with logp.open("a") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE),
                                start_new_session=True, env=env)
    w["passes"].append({"run_id": run_id, "mode": mode, "state": "running", "pid": proc.pid,
                        "log": str(logp), "why": why,
                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields})
    save_watch(w)
    log(f"start pass {run_id} mode={mode}: {why}")


def start_challenger_if_due(w: dict, a: argparse.Namespace) -> bool:
    """Once per UTC day, queue the top-k losing challengers (margin > 0) of the
    last window and start them one at a time (chat sets only, Prime pod)."""
    cfg = SUITE.get("challenger") or {}
    if not cfg.get("enabled"):
        return False
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if w.get("challenger_day") != today:
        picks = near_misses(float(cfg.get("window_hours", 24)), int(cfg.get("per_day", 3)))
        done = {p.get("revision") for p in w["passes"] if p.get("mode") == "challenger"}
        king = current_king() or {}
        # skip revisions already benched and the sitting king (it has its own pass)
        w["challenger_queue"] = [p for p in picks if p["revision"] not in done
                                 and p["revision"] != king.get("digest")]
        w["challenger_day"] = today
        save_watch(w)
        log(f"challenger picks for {today}: {[p['challenge_id'] for p in w['challenger_queue']]}")
    if not w.get("challenger_queue"):
        return False
    pick = w["challenger_queue"].pop(0)
    run_id = time.strftime("%Y%m%dT%H%MZ", time.gmtime()) + f"-{pick['challenge_id']}"
    king = current_king() or {}
    start_pass(w, pick["repo"], pick["challenge_id"], run_id, "challenger",
               f"near-miss loser margin {pick['margin']:+.5f} (z {pick.get('z')})",
               extra_env={"CHALLENGER_REVISION": pick["revision"],
                          "CHALLENGER_MARGIN": str(pick["margin"]), "CHALLENGER_Z": str(pick.get("z") or ""),
                          "CHALLENGER_VS_REIGN": str(king.get("reign") or ""),
                          "CHALLENGER_VS_KING_DIGEST": str(king.get("digest") or ""),
                          "CHALLENGER_JUDGED_AT": str(pick.get("at") or ""), "CHALLENGER_HOTKEY": str(pick.get("hotkey") or "")},
               revision=pick["revision"], challenge_id=pick["challenge_id"])
    return True


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
    # An operator-driven pass for this king (e.g. the reference full pass on a
    # Prime pod) is marked by state/inflight-<digest12>; the watcher stands down
    # for that king until the marker is removed (publish.py removes it).
    if (STATE_DIR / f"inflight-{king['digest'][:12]}").exists():
        log(f"pass for {king['digest'][:12]} in flight elsewhere (inflight marker); standing down")
        return
    if start_challenger_if_due(w, a):
        return
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
    if a.dry_run:
        log(f"would start pass {run_id}: {why}")
        return
    if card is None and skip_if_identical_weights(king, run_id):
        return
    start_pass(w, king["digest"], str(king["reign"]), run_id,
               os.environ.get("BENCHSUITE_MODE") or SUITE["modes"]["default"], why,
               digest=king["digest"], reign=king["reign"])


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
