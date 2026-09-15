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
sys.path.insert(0, str(REPO / "ops" / "teacher-swarm"))
sys.path.insert(0, str(HERE))

import lium_api  # noqa: E402  (Lium listing / removal, same module kingpod.py uses)
from coverage import post_discord  # noqa: E402
STATE_DIR = Path(os.environ.get("COVERAGE_STATE_DIR", HERE / "state"))
QUEUE_PATH = STATE_DIR / "bench_queue.json"
META_PATH = STATE_DIR / "bench_queue_meta.json"     # scale state: active flag, pods seen, posts
KINGPOD = REPO / "ops" / "benchsuite" / "kingpod.py"
# env-backfill pods the datagen worker rents (see internal/coverage/env-backfill-spec.md):
# never released by this queue, summed into the spend lines
LEDGER_PATH = STATE_DIR / "backfill_pods.json"
BENCH_POD_PREFIX = "bench-king-"        # kingpod.py POD_PREFIX: every pod the passes rent
PRIME_MINIF2F_USD = 25.0                # Prime sandbox fee per pass with the sandbox sets (benchsuite.md §9.3)
BENCH_STATE = REPO / "ops" / "benchsuite" / "state"
PASS_SH = REPO / "ops" / "benchsuite" / "pass.sh"
# Operator 2026-09-15 18:16 UTC: "scale as needed, scale down when caught up".
# Each pass rents its own pod; the Docker Hub pull-cap login (200 pulls/h for the
# one account) is the practical ceiling for concurrent SWE-bench passes.
MAX_PARALLEL = int(os.environ.get("COVERAGE_BENCH_PARALLEL", "5"))
# run_pass.sh exit codes worth a retry: 2 = rent failed (no stock), 3 = pod never
# became ready, 10 = suite.lock.json did not match the pod (the benchsuite worker
# is re-pinning the lock after a suite change; nothing wrong with the model)
RETRY_BACKOFF_S = {2: 15 * 60, 3: 15 * 60, 10: 30 * 60}
RETRY_EXITS = set(RETRY_BACKOFF_S)
SUITE_TOML = REPO / "ops" / "benchsuite" / "suite.toml"
SUITE_LOCK = REPO / "ops" / "benchsuite" / "suite.lock.json"
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


def lock_in_sync() -> tuple[bool, str]:
    """suite.toml and suite.lock.json name the same envs. When they do not,
    every pass would rent a pod, install, and refuse at the lock check (exit
    10, ~25 min and ~$2 wasted), so the queue waits for the benchsuite worker
    to re-pin instead of launching."""
    try:
        import tomllib
        suite = {e["id"] for e in tomllib.load(SUITE_TOML.open("rb")).get("envs", [])}
        lock = set(json.loads(SUITE_LOCK.read_text()).get("envs", {}))
    except (OSError, ValueError, KeyError) as ex:
        return False, f"cannot read suite/lock: {ex}"
    if suite == lock:
        return True, ""
    return False, f"suite.toml vs lock: +{sorted(suite - lock)} -{sorted(lock - suite)}"


def load_meta() -> dict:
    try:
        return json.loads(META_PATH.read_text())
    except (OSError, ValueError):
        return {"active": False, "pods_seen": [], "released": []}


def save_meta(m: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = META_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=1))
    tmp.replace(META_PATH)


def pod_of(run_id: str) -> str | None:
    p = BENCH_STATE / f"pass-{run_id}.log"
    try:
        m = re.search(r"Lium pod (\S+) \(", p.read_text(errors="replace"))
    except OSError:
        return None
    return m.group(1) if m else None


def spend(q: list[dict]) -> dict:
    """Pod $ recorded on finished passes + a running estimate for live ones +
    the Prime miniF2F fee per pass that ran the sandbox sets."""
    pod_usd = 0.0
    prime = 0.0
    for e in q:
        if e.get("pod_usd") is not None:
            pod_usd += float(e["pod_usd"])
        elif e.get("status") == "running" and e.get("started_at"):
            c = pod_cost(e["run_id"])
            if c.get("usd_per_hour"):
                t0 = datetime.fromisoformat(e["started_at"]).timestamp()
                pod_usd += c["usd_per_hour"] * (time.time() - t0) / 3600
        if e.get("status") in ("done", "running") and e.get("sandbox", True):
            prime += PRIME_MINIF2F_USD
    return {"pod_usd": round(pod_usd, 2), "prime_usd_est": round(prime, 2),
            "total_usd": round(pod_usd + prime, 2)}


def ledger_pods() -> list[dict]:
    try:
        return json.loads(LEDGER_PATH.read_text())
    except (OSError, ValueError):
        return []


def ledger_spend() -> float:
    total = 0.0
    for e in ledger_pods():
        try:
            t0 = datetime.fromisoformat(e["rented_at"]).timestamp()
            t1 = datetime.fromisoformat(e["released_at"]).timestamp() if e.get("released_at") else time.time()
            total += float(e.get("usd_per_hour") or 0) * max(0.0, t1 - t0) / 3600
        except (KeyError, ValueError, TypeError):
            continue
    return round(total, 2)


def listed_backfill_pods(pods_seen: list[str]) -> list[str]:
    """Pods this queue rented that Lium still lists (the truth for billing);
    env-backfill pods from the datagen worker's ledger are never ours."""
    sess = lium_api.session()
    listing = {lium_api.pod_name(p) for p in (lium_api.pods(sess) or [])}
    theirs = {e.get("pod") for e in ledger_pods()}
    return sorted(n for n in pods_seen if n in listing and n not in theirs)


def scale_down(q: list[dict], meta: dict) -> None:
    """Queue empty: release every pod the passes rented, verify against the
    Lium listing, post one line with the spend, idle. Runs on every tick
    until the listing shows none of our pods (a release can lag)."""
    try:
        alive = listed_backfill_pods(meta.get("pods_seen") or [])
    except Exception as ex:
        log(f"scale-down: Lium listing failed ({ex}); retrying next tick")
        return
    for name in alive:
        r = subprocess.run([sys.executable, str(KINGPOD), "release", name], capture_output=True, text=True,
                           cwd=str(KINGPOD.parent), timeout=300)
        log(f"scale-down: release {name} -> exit {r.returncode} {(r.stdout or r.stderr).strip()[-200:]}")
        meta.setdefault("released", []).append({"pod": name, "at": now_iso(), "exit": r.returncode})
    if alive:
        time.sleep(30)
        try:
            alive = listed_backfill_pods(meta.get("pods_seen") or [])
        except Exception as ex:
            log(f"scale-down: Lium re-listing failed ({ex})")
            return
    if alive:
        log(f"scale-down: still listed after release: {alive}; retrying next tick")
        return
    if meta.get("active"):
        sp = spend(q)
        n_done = sum(1 for e in q if e.get("status") == "done")
        n_failed = sum(1 for e in q if e.get("status") in ("failed", "lost"))
        text = (f"kings coverage — scale-DOWN {now_iso()[:16]}Z: benchmark backfill queue empty "
                f"({n_done} passes done, {n_failed} failed/lost); Lium listing shows 0 backfill pods "
                f"(released this sweep: {len(meta.get('released') or [])} total). Spend: pods ${sp['pod_usd']} "
                f"+ Prime miniF2F ≈ ${sp['prime_usd_est']} = ≈ ${sp['total_usd']}; env-backfill pods (datagen ledger) "
                f"${ledger_spend()}. Driver idles; nightly check keeps running.")
        try:
            post_discord(text)
        except Exception as ex:
            log(f"discord post failed: {ex}")
        log(text)
        meta["active"] = False
        meta["scaled_down_at"] = now_iso()


def tick(q: list[dict]) -> None:
    meta = load_meta()
    # 0. remember every pod a running pass rented (for the scale-down sweep)
    for e in q:
        if e.get("status") == "running" and e.get("run_id"):
            name = pod_of(e["run_id"])
            if name:
                e["pod"] = name
                if name not in meta.setdefault("pods_seen", []):
                    meta["pods_seen"].append(name)
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
            back = RETRY_BACKOFF_S[code]
            e.update(status="pending", not_before=time.time() + back)
            log(f"{e['run_id']} exit {code} ({'lock mismatch' if code == 10 else 'no stock / pod not ready'}): "
                f"retry in {back // 60} min")
        else:
            e["status"] = "failed"
            log(f"{e['run_id']} FAILED exit {code}")
    # 2. launch what fits, in queue order. A stock backoff is global (no pod
    # for anyone), so while the head of the queue waits, nothing behind it
    # jumps ahead — the operator's order (genesis, then kings 10 -> 1) holds.
    running = sum(1 for e in q if e.get("status") == "running")
    launched = 0
    in_sync, why = lock_in_sync()
    if not in_sync:
        if time.time() - float(meta.get("lock_warned_at") or 0) > 1800:
            log(f"launches held: {why} (waiting for the benchsuite worker to re-pin suite.lock.json)")
            meta["lock_warned_at"] = time.time()
    for e in q:
        if not in_sync or running >= MAX_PARALLEL:
            break
        if e.get("status") != "pending":
            continue
        if time.time() < float(e.get("not_before") or 0):
            break
        try:
            launch(e)
            running += 1
            launched += 1
        except OSError as ex:
            e.update(status="failed", note=f"launch failed: {ex}")
            log(f"launch failed for {e['label']}: {ex}")
    # 3. scale state: one Discord line at scale-up, one at scale-down
    pending = sum(1 for e in q if e.get("status") == "pending")
    if launched and not meta.get("active"):
        meta["active"] = True
        meta["scaled_up_at"] = now_iso()
        text = (f"kings coverage — scale-UP {now_iso()[:16]}Z: benchmark backfill queue started, "
                f"{running} pass(es) running, {pending} pending, cap {MAX_PARALLEL} pods "
                f"(1×H200 first, then B200 / RTX PRO 6000 / 2×H200 / 2×B200 / 2×H100 as stock allows). "
                f"Spend so far ≈ ${spend(q)['total_usd']}.")
        try:
            post_discord(text)
        except Exception as ex:
            log(f"discord post failed: {ex}")
        log(text)
    if running == 0 and pending == 0 and (meta.get("active") or meta.get("pods_seen")):
        scale_down(q, meta)
    save_meta(meta)


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
    for e in q:
        cost = e.get("pod_usd")
        print(f"{e.get('status', '?'):8} p{e.get('priority', 100):<3} {e['label']:8} {e['ref'][:44]:44} "
              f"{e.get('run_id', '') or '':32} exit={e.get('exit', '')} ${cost if cost is not None else '-'}")
    sp = spend(q)
    meta = load_meta()
    print(f"spend: pods ${sp['pod_usd']} (running passes estimated) + Prime miniF2F ≈ ${sp['prime_usd_est']} "
          f"= ≈ ${sp['total_usd']}; scale state: {'ACTIVE' if meta.get('active') else 'idle'}, "
          f"pods seen {len(meta.get('pods_seen') or [])}, cap {MAX_PARALLEL}")
    return 0


def cmd_scale_down(_: argparse.Namespace) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / "bench_queue.lock").open("w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        q = load_queue()
        meta = load_meta()
        meta["active"] = True     # force the post
        scale_down(q, meta)
        save_meta(meta)
        save_queue(q)
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
    sub.add_parser("scale-down", help="release every pod this queue rented that Lium still lists (manual sweep)").set_defaults(fn=cmd_scale_down)
    sub.add_parser("tick").set_defaults(fn=cmd_tick)
    lp = sub.add_parser("loop")
    lp.add_argument("--interval", type=int, default=300)
    lp.set_defaults(fn=cmd_loop)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
