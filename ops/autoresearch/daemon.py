#!/usr/bin/env python
"""Auto-research daemon (pm2 `affine-autoresearch`) — docs/auto-research-loop.md §1.

Stage 1 shipped here: **divergence tables per king**. On every tick the daemon
syncs the trace cache, and for the sitting king and the previous kings (newest
first) that have no `affine/state/king_divergence/<digest12>.jsonl` yet — or
whose failed-rollout pool grew by --regrow since the last table — it runs the
first-divergence probe (3 teacher references at every king turn, Engy) and the
analysis that writes the side-table the fold's `king_divergence` group reads.

Rails (§2): daily $ ledger with a hard cap, STOP file (finish the running stage,
then idle), DRY_RUN file (probe into a scratch dir, publish nothing), one JSON
line per stage run in log/<stage>.jsonl, Discord line per published table
(private Arbos channel only).

  daemon.py --once            one tick
  daemon.py --interval 600    the pm2 loop
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "ops" / "king-review"))
import krlib  # noqa: E402

STATE_DIR = REPO / "affine" / "state" / "autoresearch"
LOG_DIR = STATE_DIR / "log"
DIV_DIR = REPO / "affine" / "state" / "king_divergence"
TRACES = REPO / "affine" / "state" / "king_review" / "traces"
VALIDATOR_STATE = REPO / "affine" / "state" / "state.json"
PY = REPO / ".venv" / "bin" / "python"
PROBE = HERE / "divergence_probe.py"
ANALYZE = HERE / "divergence_analyze.py"
DISCORD_CHANNEL = "1510910974498967613"
DISCORD_TOKEN_ENV = "DISCORD_BOT_TOKEN_ARBOS_BITTENSOR"
DAILY_USD_CAP = float(os.environ.get("AUTORESEARCH_DAILY_USD", "60"))


def log(msg: str) -> None:
    print(f"[autoresearch] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def stage_log(stage: str, row: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, **row}
    with open(LOG_DIR / f"{stage}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def load_state() -> dict:
    p = STATE_DIR / "state.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except ValueError:
            pass
    return {"tables": {}, "budget": {}}


def save_state(s: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_DIR / "state.json.tmp"
    tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(STATE_DIR / "state.json")
    os.chmod(STATE_DIR / "state.json", 0o600)


def budget_left(s: dict) -> float:
    day = time.strftime("%Y-%m-%d", time.gmtime())
    spent = s.setdefault("budget", {}).get(day, 0.0)
    return DAILY_USD_CAP - spent


def budget_add(s: dict, usd: float) -> None:
    day = time.strftime("%Y-%m-%d", time.gmtime())
    s.setdefault("budget", {})[day] = s["budget"].get(day, 0.0) + usd


def env_file_value(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for path in (Path.home() / ".affine-validator.env", REPO / ".env"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def post_discord(text: str, dry_run: bool) -> None:
    if dry_run:
        log(f"DRY-RUN discord: {text}")
        return
    token = env_file_value(DISCORD_TOKEN_ENV)
    if not token:
        log("no discord token")
        return
    try:
        requests.post(f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL}/messages",
                      headers={"Authorization": f"Bot {token}"}, json={"content": text[:1900]}, timeout=20)
    except requests.RequestException as e:
        log(f"discord failed: {e!r}")


def kings() -> list[tuple[int, str]]:
    """(reign, digest12) for the sitting king and its predecessors, newest first."""
    s = json.loads(VALIDATOR_STATE.read_text())
    k = s["king"]
    out = [(int(k["reign_number"]), k["revision"][:12])]
    for p in k.get("previous") or []:
        out.append((int(p["reign_number"]), p["revision"][:12]))
    return out


def sync_traces() -> list[dict]:
    store = krlib.TraceStore(TRACES)
    store.sync(log=log)
    return store.index(log=log)


def failed_pool(rows: list[dict], digest12: str) -> int:
    teacher_solved = {r["sid"] for r in rows if not r.get("king") and r.get("outcome") == "solved"}
    return sum(1 for r in rows if r.get("king") == f"king-{digest12}" and r.get("outcome") == "failed"
               and r["sid"] in teacher_solved)


def run_divergence(digest12: str, *, cap: int, max_turns: int, workers: int, usd_cap: float,
                   dry_run: bool) -> dict:
    env = dict(os.environ, DIV_KING=digest12)
    if dry_run:
        env["DIV_OUT"] = str(STATE_DIR / "dryrun" / "king_divergence")
    t0 = time.time()
    p = subprocess.run([str(PY), str(PROBE), "--failed-only", "--cap", str(cap), "--max-turns", str(max_turns),
                        "--workers", str(workers), "--budget-usd", f"{usd_cap:.2f}"],
                       cwd=str(REPO), env=env, capture_output=True, text=True, timeout=6 * 3600)
    cost = 0.0
    for line in p.stdout.splitlines()[::-1]:
        if "cost $" in line:
            try:
                cost = float(line.split("cost $")[1].split()[0])
            except ValueError:
                pass
            break
    a = subprocess.run([str(PY), str(ANALYZE)], cwd=str(REPO), env=env, capture_output=True, text=True, timeout=1800)
    summary = a.stdout[-4000:]
    n_rows = 0
    table = (Path(env.get("DIV_OUT", str(DIV_DIR))) / f"{digest12}.jsonl")
    if table.exists():
        n_rows = sum(1 for _ in open(table))
    return {"digest": digest12, "cost_usd": cost, "seconds": round(time.time() - t0), "rows": n_rows,
            "probe_rc": p.returncode, "analyze_rc": a.returncode, "summary": summary,
            "probe_tail": p.stdout[-1500:] + p.stderr[-800:]}


def tick(a: argparse.Namespace) -> None:
    s = load_state()
    if (STATE_DIR / "STOP").exists():
        log("STOP present; idle")
        return
    dry_run = (STATE_DIR / "DRY_RUN").exists() or a.dry_run
    rows = sync_traces()
    todo = []
    for reign, dg in kings()[: a.kings]:
        pool = failed_pool(rows, dg)
        prev = s["tables"].get(dg) or {}
        table = DIV_DIR / f"{dg}.jsonl"
        if not table.exists() and pool >= a.min_pool:
            todo.append((reign, dg, pool, "new"))
        elif table.exists() and pool - prev.get("pool", 0) >= a.regrow:
            todo.append((reign, dg, pool, "regrow"))
    if not todo:
        log("nothing to do")
        return
    for reign, dg, pool, why in todo:
        left = budget_left(s)
        if left <= 1.0:
            log(f"budget exhausted for today (${DAILY_USD_CAP}); skipping {dg}")
            stage_log("divergence", {"digest": dg, "skipped": "budget"})
            break
        log(f"divergence for reign {reign} {dg}: pool {pool} ({why}), budget left ${left:.2f}")
        res = run_divergence(dg, cap=a.cap, max_turns=a.max_turns, workers=a.workers,
                             usd_cap=min(a.usd_per_king, left), dry_run=dry_run)
        budget_add(s, res["cost_usd"])
        s["tables"][dg] = {"reign": reign, "pool": pool, "rows": res["rows"], "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "cost_usd": res["cost_usd"], "why": why, "dry_run": dry_run}
        save_state(s)
        stage_log("divergence", {k: v for k, v in res.items() if k not in ("summary", "probe_tail")} | {"reign": reign, "pool": pool, "why": why, "dry_run": dry_run})
        log(f"{dg}: {res['rows']} side-table rows, ${res['cost_usd']:.2f}, {res['seconds']} s")
        if res["rows"]:
            post_discord(f"[autoresearch] king_divergence table for reign {reign} `{dg}`: {res['rows']} first-divergence states "
                         f"(failed pool {pool}, ${res['cost_usd']:.2f}); fold admits at its next run.", dry_run)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--kings", type=int, default=5, help="sitting king + N-1 predecessors")
    ap.add_argument("--min-pool", type=int, default=100, help="failed rollouts on teacher-solved tasks before a first table")
    ap.add_argument("--regrow", type=int, default=300, help="new failed rollouts before a table is rebuilt")
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--usd-per-king", type=float, default=15.0)
    a = ap.parse_args()
    while True:
        try:
            tick(a)
        except Exception as e:  # noqa: BLE001
            log(f"tick failed: {e!r}")
            stage_log("daemon", {"error": repr(e)})
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
