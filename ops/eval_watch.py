#!/usr/bin/env python3
"""Live validator/eval watchdog (2026-08-14, operator directive).

Polls the public snapshot + pm2 every POLL_S and prints an `ALERT <kind> ...`
line when something needs a human/agent fix. Silent-ish otherwise (heartbeat
lines start with `ok` so log watchers can key on `^ALERT`).

Watched invariants:
  - validator pm2 process online
  - snapshot generated_at fresh (dash alive)
  - current duel wall time <= DUEL_BUDGET_S (evals should average <1h)
  - duel scoring progress advances between polls
  - a verdict landed within VERDICT_GAP_S
  - evalsrv log DuelFault count is not growing (checked every 5th poll)
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone

SNAP_URL = "https://affine.io/api/v1/snapshot"
HIST_URL = "https://affine.io/api/v1/history"
EVAL_SSH = ["ssh", "-o", "ConnectTimeout=10", "-o",
            "StrictHostKeyChecking=accept-new",
            "-p", "40298", "root@69.63.236.165"]

POLL_S = 120
DUEL_BUDGET_S = 70 * 60      # alert if one duel runs past this
PROGRESS_STALL_S = 20 * 60   # alert if scoring counter frozen this long
VERDICT_GAP_S = 95 * 60      # alert if no verdict landed for this long
SNAP_STALE_S = 10 * 60


def now() -> float:
    return time.time()


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def parse_iso(s: str) -> float | None:
    try:
        return datetime.fromisoformat(s).timestamp()
    except (TypeError, ValueError):
        return None


def curl_json(url: str) -> dict | list | None:
    try:
        raw = subprocess.check_output(
            ["curl", "-s", "--max-time", "20", url], text=True)
        return json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def pm2_online() -> bool:
    try:
        raw = subprocess.check_output(
            ["pm2", "jlist"], text=True, stderr=subprocess.DEVNULL)
        for p in json.loads(raw):
            if p.get("name") == "affine-validator":
                return p.get("pm2_env", {}).get("status") == "online"
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        pass
    return False


def eval_fault_count() -> int | None:
    try:
        raw = subprocess.check_output(
            EVAL_SSH + ["grep -ac 'DuelFault:' /root/logs/evalsrv.log"],
            text=True, timeout=30)
        return int(raw.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def main() -> None:
    last_progress: tuple[str, int] | None = None   # (challenge_id, done)
    progress_since = now()
    last_fault_count: int | None = None
    poll = 0

    while True:
        poll += 1
        alerts: list[str] = []

        if not pm2_online():
            alerts.append("ALERT validator-down: pm2 affine-validator not online")

        snap = curl_json(SNAP_URL)
        if not isinstance(snap, dict):
            alerts.append("ALERT snapshot-unreachable: affine.io snapshot fetch failed")
            snap = {}

        gen = parse_iso(snap.get("generated_at") or "")
        if gen and now() - gen > SNAP_STALE_S:
            alerts.append(f"ALERT snapshot-stale: generated {int(now()-gen)}s ago")

        phase = snap.get("phase") or {}
        ce = snap.get("current_eval") or {}
        cid = ce.get("challenge_id")
        if phase.get("name") == "duel" and cid:
            since = parse_iso(phase.get("since") or "")
            if since and now() - since > DUEL_BUDGET_S:
                alerts.append(
                    f"ALERT duel-slow: {cid} running {int((now()-since)/60)}m "
                    f"(budget {DUEL_BUDGET_S//60}m)")
            done = ((ce.get("progress") or {}).get("done"))
            if isinstance(done, int):
                key = (cid, done)
                if key != last_progress:
                    last_progress = key
                    progress_since = now()
                elif now() - progress_since > PROGRESS_STALL_S:
                    alerts.append(
                        f"ALERT duel-stalled: {cid} scoring frozen at "
                        f"{done} for {int((now()-progress_since)/60)}m")

        hist = curl_json(HIST_URL)
        items = hist if isinstance(hist, list) else []
        if items:
            newest = parse_iso(
                items[0].get("finished_at") or items[0].get("decided_at")
                or items[0].get("at") or "")
            if newest and now() - newest > VERDICT_GAP_S:
                alerts.append(
                    f"ALERT no-verdicts: last verdict "
                    f"{int((now()-newest)/60)}m ago")

        if poll % 5 == 1:
            fc = eval_fault_count()
            if fc is not None:
                if last_fault_count is not None and fc > last_fault_count:
                    alerts.append(
                        f"ALERT duel-faults: +{fc - last_fault_count} new "
                        f"DuelFault lines on eval pod")
                last_fault_count = fc

        if alerts:
            for a in alerts:
                print(f"{ts()} {a}", flush=True)
        else:
            prog = ce.get("progress") or {}
            print(f"{ts()} ok phase={phase.get('name')} cid={cid} "
                  f"done={prog.get('done')}/{prog.get('total')} "
                  f"queue={len(snap.get('queue') or [])}", flush=True)

        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
