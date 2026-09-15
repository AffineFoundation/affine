"""Benchmark backfill queue: run ops/benchsuite passes for models whose
affine.io/#kings benchmark cells are empty (operator directive 2026-09-15:
every king, the genesis and the teacher must have a score on every held-out
benchmark).

The queue calls `ops/benchsuite/pass.sh` — the same entry point the
benchsuite worker uses by hand — and never forks its code. Up to
MAX_PARALLEL passes run at once, each on its own Lium 1x H200 pod
(rent -> serve -> chat cells -> sandbox cells -> publish -> release, all
inside run_pass.sh). Cards land in affine/state/benchsuite/ and the
kingboard matrix picks them up on its next pass.

    python bench_queue.py add --ref <sha256|hf://repo@rev> --label <reign|genesis> [--chat-envs humaneval]
    python bench_queue.py status
    python bench_queue.py tick          # launch what fits, record exits (pm2 runs this in a loop)
    python bench_queue.py loop [--interval 300]

State: state/bench_queue.json (one entry per pass). Exit codes come from
ops/benchsuite/state/pass-<run_id>.exit; a rent / wait failure (exit 2, 3 =
no Lium stock or the pod never served) is retried after RETRY_S, other
failures stop the entry and are reported.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE_DIR = Path(os.environ.get("COVERAGE_STATE_DIR", HERE / "state"))
QUEUE_PATH = STATE_DIR / "bench_queue.json"
BENCH_STATE = REPO / "ops" / "benchsuite" / "state"
PASS_SH = REPO / "ops" / "benchsuite" / "pass.sh"
MAX_PARALLEL = int(os.environ.get("COVERAGE_BENCH_PARALLEL", "3"))
RETRY_S = 15 * 60            # no Lium stock: try again in 15 min
RETRY_EXITS = {2, 3}         # run_pass.sh: 2 = rent failed, 3 = pod never became ready
MAX_ATTEMPTS = 12
LOG_PREFIX = "[bench_queue]"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {now_iso()} {msg}", flush=True)


def load_queue() -> list[dict]:
    if not QUEUE_PATH.exists():
        return []
    return json.loads(QUEUE_PATH.read_text())


def save_queue(q: list[dict]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(q, indent=1))
    tmp.replace(QUEUE_PATH)


def digest12_of_ref(ref: str) -> str:
    if ref.startswith("hf://"):
        return ref.rsplit("@", 1)[-1][:12]
    return ref[:12]


def run_id_for(ref: str, label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    return f"{stamp}-{digest12_of_ref(ref)}" if label != "genesis" else f"{stamp}-genesis-{digest12_of_ref(ref)}"


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def exit_code(run_id: str) -> int | None:
    p = BENCH_STATE / f"pass-{run_id}.exit"
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except ValueError:
        return -1


def pod_cost(run_id: str) -> dict:
    """Pod name + $/h from the pass log ("Lium pod <name> (<usd>/h) serving")
    plus the wall hours between the first and the last log line."""
    p = BENCH_STATE / f"pass-{run_id}.log"
    out: dict = {}
    try:
        text = p.read_text(errors="replace")
    except OSError:
        return out
    m = re.search(r"Lium pod (\S+) \(([\d.]+)/h\)", text)
    if m:
        out["pod"] = m.group(1)
        out["usd_per_hour"] = float(m.group(2))
    stamps = re.findall(r"^\[\w+\] (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)", text, re.M)
    if len(stamps) >= 2:
        t0 = datetime.fromisoformat(stamps[0].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(stamps[-1].replace("Z", "+00:00"))
        out["wall_h"] = round((t1 - t0).total_seconds() / 3600, 2)
        if "usd_per_hour" in out:
            out["pod_usd"] = round(out["wall_h"] * out["usd_per_hour"], 2)
    return out


def launch(entry: dict) -> None:
    run_id = run_id_for(entry["ref"], entry["label"])
    entry["run_id"] = run_id
    log_path = BENCH_STATE / f"pass-{run_id}.log"
    env = dict(os.environ)
    env["BENCHSUITE_FORCE_SANDBOX"] = "1" if entry.get("sandbox", True) else "0"
    if entry.get("chat_envs"):
        env["BENCHSUITE_CHAT_ENVS"] = entry["chat_envs"]
    BENCH_STATE.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as fh:
        # setsid: the pass must outlive this process (pm2 restarts kill the tree)
        proc = subprocess.Popen(
            ["setsid", "bash", str(PASS_SH), entry["ref"], entry["label"], run_id, entry.get("mode", "lium")],
            stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
            cwd=str(REPO), start_new_session=True)
    entry.update(status="running", pid=proc.pid, started_at=now_iso(),
                 attempts=int(entry.get("attempts") or 0) + 1)
    log(f"launched {run_id} ({entry['label']}, {entry['ref'][:40]}…) pid {proc.pid} attempt {entry['attempts']}")


def tick(q: list[dict]) -> None:
    # 1. record exits
    for e in q:
        if e.get("status") != "running":
            continue
        code = exit_code(e["run_id"])
        if code is None:
            if not pid_alive(e.get("pid")):
                pidf = BENCH_STATE / f"pass-{e['run_id']}.pid"
                # run_pass.sh re-execs under its own pid; trust the .pid file first
                try:
                    if pid_alive(int(pidf.read_text().strip())):
                        continue
                except (OSError, ValueError):
                    pass
                e.update(status="lost", ended_at=now_iso(),
                         note="driver process gone without an exit file (pod may still run: attach mode)")
                log(f"{e['run_id']}: driver gone, no exit file")
            continue
        e.update(exit=code, ended_at=now_iso(), **pod_cost(e["run_id"]))
        if code == 0:
            e["status"] = "done"
            log(f"{e['run_id']} done (${e.get('pod_usd', '?')} pod, {e.get('wall_h', '?')} h)")
        elif code in RETRY_EXITS and e.get("attempts", 0) < MAX_ATTEMPTS:
            e.update(status="pending", not_before=time.time() + RETRY_S)
            log(f"{e['run_id']} exit {code} (no stock / pod not ready): retry in {RETRY_S // 60} min")
        else:
            e["status"] = "failed"
            log(f"{e['run_id']} FAILED exit {code}")
    # 2. launch what fits, in queue order. A stock backoff is global (no pod
    # for anyone), so while the head of the queue waits, nothing behind it
    # jumps ahead — the operator's order (genesis, then kings 10 -> 1) holds.
    running = sum(1 for e in q if e.get("status") == "running")
    for e in q:
        if running >= MAX_PARALLEL:
            break
        if e.get("status") != "pending":
            continue
        if time.time() < float(e.get("not_before") or 0):
            break
        try:
            launch(e)
            running += 1
        except OSError as ex:
            e.update(status="failed", note=f"launch failed: {ex}")
            log(f"launch failed for {e['label']}: {ex}")


def cmd_add(args: argparse.Namespace) -> int:
    q = load_queue()
    q.append({"ref": args.ref, "label": args.label, "mode": args.mode, "sandbox": not args.no_sandbox,
              "chat_envs": args.chat_envs, "status": "pending", "added_at": now_iso(), "attempts": 0,
              "priority": args.priority, "note": args.note})
    q.sort(key=lambda e: (e.get("priority", 100), e.get("added_at", "")))
    save_queue(q)
    print(f"queued {args.label} {args.ref}")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    q = load_queue()
    total = 0.0
    for e in q:
        cost = e.get("pod_usd")
        total += float(cost or 0)
        print(f"{e.get('status', '?'):8} p{e.get('priority', 100):<3} {e['label']:8} {e['ref'][:44]:44} "
              f"{e.get('run_id', '') or '':32} exit={e.get('exit', '')} ${cost if cost is not None else '-'}")
    print(f"total pod spend recorded: ${total:.2f} (Prime miniF2F sandbox fees are not included)")
    return 0


def cmd_tick(_: argparse.Namespace) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / "bench_queue.lock").open("w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        q = load_queue()
        tick(q)
        save_queue(q)
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    while True:
        cmd_tick(args)
        time.sleep(args.interval)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--ref", required=True, help="sha256 digest (public models.affine.io copy) or hf://repo@revision")
    a.add_argument("--label", required=True, help="reign number, or 'genesis'")
    a.add_argument("--mode", default="lium")
    a.add_argument("--no-sandbox", action="store_true", help="chat sets only")
    a.add_argument("--chat-envs", default="", help="comma list to restrict the chat cells (BENCHSUITE_CHAT_ENVS)")
    a.add_argument("--priority", type=int, default=100, help="lower runs first")
    a.add_argument("--note", default="")
    a.set_defaults(fn=cmd_add)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("tick").set_defaults(fn=cmd_tick)
    lp = sub.add_parser("loop")
    lp.add_argument("--interval", type=int, default=300)
    lp.set_defaults(fn=cmd_loop)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
