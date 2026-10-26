#!/usr/bin/env python3
"""Track M crown watcher: bench every new king on fastbench, sparingly.

Watches the local mirror of the Track M status log for CROWN lines. For each
new king it: copies the LoRA adapter (eval box -> here -> fastbench box),
merges it into the cached base on the fastbench box (CPU only), waits until
the service is idle, submits exactly ONE bench (n=25, seed=0 — fixed so all
kings share a panel), polls to completion, and appends a BENCH line to the
status log on the eval box.

Budget rules enforced here:
  * one bench in flight at a time; /health busy+queued checked before submit
  * n=25 always; only crown events trigger jobs
  * never aborts or touches jobs it did not create
  * if several crowns arrive while busy, only the NEWEST is benched; older
    ones are recorded as superseded (kings are dethroned already anyway)

State (which crowns were benched/skipped, in-flight job ids) is persisted to
state.json next to this script, so restarts never re-bench or double-submit.
A missing state file marks every crown already in the log as pre_integration
— a fresh start can never mass-bench the backlog.

Tunnel recovery: on repeated API failures the SSH tunnel (local 9310 ->
fastbench 9100) is relaunched from ops/state.json of the fastbench repo.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
STATE_PATH = HERE / "state.json"
TMP_DIR = HERE / "tmp"
MIRROR_LOG = Path("/home/const/subnet120/research/logs/trackM_status.log")
FASTBENCH_ENV = Path.home() / ".fastbench.env"
FB_OPS_STATE = Path("/home/const/subnet120/fastbench/ops/state.json")
EVAL_SSH = "root@REDACTED_IP -p 40301"
STATUS_LOG_REMOTE = "/dshare/koth/status.log"
KINGS_DIR_REMOTE = "/dshare/koth/kings"

POLL_S = 30            # mirror-log scan interval
JOB_POLL_S = 45        # bench job poll interval
BENCH_N = 25
BENCH_SEED = 0
JOB_TIMEOUT_S = 3 * 3600
MERGED_KEEP = 2        # merged model dirs kept on the fastbench box

CROWN_RE = re.compile(r"CROWN reign=(\d+)->(\d+): (sub_\S+) dethrones")
SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=20"]

log = logging.getLogger("crownwatch")


def _env_from_file(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


_fb_env = _env_from_file(FASTBENCH_ENV)
BASE_URL = _fb_env.get("FASTBENCH_URL", "http://127.0.0.1:9310")
HEADERS = {"X-Fastbench-Token": _fb_env.get("FASTBENCH_TOKEN", "")}


def _ssh_args(target: str) -> list[str]:
    return ["ssh", *SSH_OPTS, *shlex.split(target)]


def ssh_run(target: str, command: str, timeout: int = 120,
            input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(_ssh_args(target) + [command], capture_output=True,
                          text=True, timeout=timeout, input=input_text)


def fb_ssh_target() -> str:
    return json.loads(FB_OPS_STATE.read_text())["ssh"]


# -- state -----------------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    # First run: everything already in the log predates the watcher.
    state = {"crowns": {}}
    for reign_to, sub in scan_crowns():
        state["crowns"][sub] = {"reign": reign_to,
                                "status": "pre_integration"}
    save_state(state)
    log.info("fresh state: %d historical crowns marked pre_integration",
             len(state["crowns"]))
    return state


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.rename(STATE_PATH)


def scan_crowns() -> list[tuple[int, str]]:
    """All (new_reign, sub_id) crown events in log order."""
    out: list[tuple[int, str]] = []
    try:
        text = MIRROR_LOG.read_text(errors="replace")
    except OSError:
        return out
    for m in CROWN_RE.finditer(text):
        out.append((int(m.group(2)), m.group(3)))
    return out


# -- fastbench API with tunnel recovery -------------------------------------

_api_fail_streak = 0


def restore_tunnel() -> None:
    """Relaunch the local SSH tunnel (9310 -> fastbench box 9100)."""
    try:
        target = fb_ssh_target()
    except Exception:
        log.error("cannot read fastbench ops state; no tunnel restore")
        return
    subprocess.run(["pkill", "-f", r"ssh.*-L 9310:127\.0\.0\.1:9100"],
                   capture_output=True)
    time.sleep(2)
    cmd = ["ssh", "-N", "-f", *SSH_OPTS,
           "-o", "ServerAliveInterval=30",
           "-o", "ExitOnForwardFailure=yes",
           "-L", "9310:127.0.0.1:9100", *shlex.split(target)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    log.warning("tunnel restore rc=%d %s", p.returncode,
                (p.stderr or "").strip()[-200:])
    time.sleep(5)


def api(method: str, path: str, body: dict | None = None) -> dict | None:
    """One fastbench call; None on failure. Restores the tunnel after 4
    consecutive failures, with exponential-ish backoff handled by callers'
    poll loops."""
    global _api_fail_streak
    try:
        r = httpx.request(method, BASE_URL + path, json=body,
                          headers=HEADERS, timeout=30)
        r.raise_for_status()
        _api_fail_streak = 0
        return r.json()
    except Exception as e:
        _api_fail_streak += 1
        log.warning("api %s %s failed (streak %d): %s",
                    method, path, _api_fail_streak, e)
        if _api_fail_streak % 4 == 0:
            restore_tunnel()
        return None


def wait_idle() -> None:
    """Block until the service is reachable, not busy, and nothing queued."""
    while True:
        h = api("GET", "/health")
        if h and h.get("ok") and not h.get("busy") and not h.get("queued"):
            return
        if h:
            log.info("service busy (job=%s queued=%s); waiting",
                     h.get("current_job"), h.get("queued"))
        time.sleep(POLL_S)


# -- pipeline stages ---------------------------------------------------------

def deliver(sub: str, reign: int) -> bool:
    """eval box king dir -> local tmp -> fastbench box /root/kings/<sub>."""
    local = TMP_DIR / sub
    local.mkdir(parents=True, exist_ok=True)
    src = f"{KINGS_DIR_REMOTE}/king_r{reign}/"
    parts = shlex.split(EVAL_SSH)
    rsh = " ".join(["ssh", *SSH_OPTS, *parts[1:]])
    p = subprocess.run(["rsync", "-a", "-e", rsh,
                        f"{parts[0]}:{src}", f"{local}/"],
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0 or not (local / "adapter_config.json").exists():
        log.error("adapter pull failed: %s", (p.stderr or "")[-300:])
        return False
    fb = fb_ssh_target()
    fparts = shlex.split(fb)
    rsh = " ".join(["ssh", *SSH_OPTS, *fparts[1:]])
    p = subprocess.run(["rsync", "-a", "-e", rsh, str(local),
                        f"{fparts[0]}:/root/kings/"],
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        log.error("adapter push failed: %s", (p.stderr or "")[-300:])
        return False
    return True


def merge_on_box(sub: str) -> bool:
    """Merge adapter into the cached base on the fastbench box (CPU only).
    Also prunes old merged dirs, keeping the newest MERGED_KEEP."""
    fb = fb_ssh_target()
    prune = ("ls -1dt /root/models/*/ 2>/dev/null | "
             f"tail -n +{MERGED_KEEP + 1} | xargs -r rm -rf")
    ssh_run(fb, prune, timeout=300)
    cmd = ("source /root/venv/bin/activate && "
           "HF_HOME=/root/hf "
           f"python /root/merge_lora.py --adapter /root/kings/{sub} "
           f"--out /root/models/{sub}")
    p = ssh_run(fb, cmd, timeout=5400)
    tail = ((p.stdout or "") + (p.stderr or ""))[-500:]
    if p.returncode != 0 or "[merge] OK" not in (p.stdout or ""):
        log.error("merge failed rc=%d: %s", p.returncode, tail)
        return False
    log.info("merge done: %s", tail.strip().splitlines()[-1])
    return True


def poll_job(job_id: str) -> dict | None:
    """Poll one bench job to a terminal state. Never aborts anything."""
    t0 = time.time()
    while time.time() - t0 < JOB_TIMEOUT_S:
        j = api("GET", f"/bench/{job_id}")
        if j:
            state = j.get("state")
            if state in ("completed", "failed", "aborted"):
                return j
            prog = j.get("progress") or {}
            log.info("job %s: %s/%s %s", job_id, state, j.get("phase"),
                     prog.get("done"))
        time.sleep(JOB_POLL_S)
    log.error("job %s not terminal after %ds; leaving it alone",
              job_id, JOB_TIMEOUT_S)
    return None


def append_status(line: str) -> bool:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    full = f"{ts} [bench] {line}"
    p = ssh_run(EVAL_SSH, f"cat >> {STATUS_LOG_REMOTE}",
                input_text=full + "\n")
    if p.returncode != 0:
        log.error("status append failed: %s", (p.stderr or "")[-200:])
        return False
    log.info("status appended: %s", full)
    return True


def bench_crown(state: dict, sub: str, rec: dict) -> None:
    """Run the full pipeline for one crown; resumes from a persisted stage."""
    reign = rec["reign"]
    tag = f"fastbench_king_r{reign}_{sub}"
    model_path = f"/root/models/{sub}"

    if not rec.get("job_id"):
        rec["status"] = "in_progress"
        save_state(state)
        if not deliver(sub, reign):
            rec["status"] = "failed"
            rec["error"] = "adapter delivery failed"
            save_state(state)
            return
        # The king dir may have been overwritten by a newer crown while we
        # copied; if so, this crown is superseded and the newest wins later.
        newer = [s for _, s in scan_crowns()]
        if newer and newer[-1] != sub:
            rec["status"] = "superseded"
            rec["superseded_by"] = newer[-1]
            save_state(state)
            log.info("crown %s superseded by %s during delivery",
                     sub, newer[-1])
            return
        if not merge_on_box(sub):
            rec["status"] = "failed"
            rec["error"] = "merge failed"
            save_state(state)
            return
        wait_idle()
        resp = api("POST", "/bench", {"model": model_path,
                                      "n": BENCH_N, "seed": BENCH_SEED})
        if not resp or "job_id" not in resp:
            rec["status"] = "failed"
            rec["error"] = f"submit failed: {resp}"
            save_state(state)
            return
        rec["job_id"] = resp["job_id"]
        save_state(state)
        log.info("submitted %s for crown %s", resp["job_id"], sub)

    job = poll_job(rec["job_id"])
    if job is None:
        rec["status"] = "failed"
        rec["error"] = "poll timeout"
        save_state(state)
        return
    if job["state"] != "completed":
        rec["status"] = "failed"
        rec["error"] = f"job {job['state']}: {job.get('error')}"
        save_state(state)
        append_status(f"BENCH tag={tag} FAILED seed={BENCH_SEED} "
                      f"job={rec['job_id']} error={job.get('error')}")
        return
    res = job.get("result") or {}
    rec["status"] = "benched"
    rec["score"] = res.get("score")
    rec["n_resolved"] = res.get("n_resolved")
    rec["wall_s"] = res.get("wall_time_s")
    save_state(state)
    append_status(
        f"BENCH tag={tag} score={res.get('score')} "
        f"resolved={res.get('n_resolved')}/{res.get('n')} "
        f"wall={res.get('wall_time_s')}s seed={BENCH_SEED} "
        f"job={rec['job_id']}")


# -- main loop --------------------------------------------------------------

def tick(state: dict) -> None:
    known = state["crowns"]
    # Restart recovery: a job we already submitted must be polled to its end
    # (and its BENCH line written) before any new crown is considered.
    for sub, rec in list(known.items()):
        if rec.get("status") == "in_progress" and rec.get("job_id"):
            log.info("resuming poll of in-flight job %s (crown %s)",
                     rec["job_id"], sub)
            bench_crown(state, sub, rec)
    crowns = scan_crowns()
    pending = [(r, s) for r, s in crowns
               if known.get(s, {}).get("status")
               in (None, "in_progress")]
    if not pending:
        return
    # Newest crown wins; older unbenched crowns are already dethroned.
    for reign, sub in pending[:-1]:
        rec = known.setdefault(sub, {"reign": reign})
        rec["status"] = "superseded"
        rec["superseded_by"] = pending[-1][1]
        log.info("crown %s superseded by %s", sub, pending[-1][1])
    save_state(state)
    reign, sub = pending[-1]
    rec = known.setdefault(sub, {"reign": reign})
    log.info("processing crown reign=%d sub=%s", reign, sub)
    bench_crown(state, sub, rec)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    log.info("crown watcher up: log=%s url=%s n=%d seed=%d",
             MIRROR_LOG, BASE_URL, BENCH_N, BENCH_SEED)
    while True:
        try:
            tick(state)
        except Exception:
            log.exception("tick crashed; continuing")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
