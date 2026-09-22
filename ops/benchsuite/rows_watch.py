#!/usr/bin/env python
"""Overnight row filler (Jacob 2026-09-20 16:16 UTC: "when we come back tomorrow all of those
will be full"). Every 30 min: read the kingboard's own matrix (what Jacob sees), find every
benchmark column a target row is missing, launch the job that fills it, retry a job whose cell
is still missing/failed (max 2 retries), and page the private Discord channel when the
projected finish slips past the deadline.

  rows_watch.py --targets "20,teacher,19,18,17,16,15" --deadline 2026-09-21T07:00Z

Cell -> job map (each job is one of the existing launchers, into the row's main run id
unless noted):
  <env>@8k / @16k old-cap variants   cap_backfill.sh at the OLD cap (BENCHSUITE_SETTINGS_JSON
                                     max_tokens) into a separate run id <ts>-<digest12>-oldcap
                                     (a pull into the main run would overwrite the current-cap
                                     cells); recap.py renames the cells to @8k / @16k
  swebench-verified                  swe_resume.sh when a harbor job dir exists, else swe_rerun.sh
  swebench-verified@4h250            swe_rerun.sh with BUDGET_TAG=4h250 (4 h / 250 steps)
  minif2f                            cap_backfill.sh CAP_ENVS=minif2f CAP_RUNTIME=prime
  chat / long-context cells          cap_backfill.sh CAP_ENVS=<missing>
  terminal-bench-2                   fast_pass.sh FAST_ONLY_ROLES=agentic FAST_GROUPS=tb2
  tau2-* / tau3-banking              fast_pass.sh FAST_ONLY_ROLES=agentic FAST_GROUPS=agentic
                                     FAST_AGENTIC_ENVS=<missing>
  gaia2-ambiguity                    skipped (unverified grader, 2026-09-20)
Daytona: at most MAX_SWE_JOBS SWE jobs at once (memory quota 1000 GB / 4 GB per sandbox).
State: state/rows_watch.json; status line for reports: state/rows_watch_status.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE = HERE / "state"
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
CARDS = REPO / "affine" / "state" / "benchsuite"
BENCH_HOME = Path(os.environ.get("BENCH_HOME", str(Path.home() / "benchsuite")))
MATRIX_URL = os.environ.get("KINGBOARD_MATRIX_URL", "http://127.0.0.1:8790/api/matrix.json")
PY = os.environ.get("BENCHSUITE_PYTHON", str(REPO / ".venv" / "bin" / "python"))
TEACHER_FROM = SUITE["modes"]["teacher_from"]
SKIP = {"gaia2-ambiguity"}
OLDCAP = {"mmlu-pro@8k": ("mmlu-pro", 8192), "math500@16k": ("math500", 16384),
          "gpqa-diamond@16k": ("gpqa-diamond", 16384), "livecodebench@16k": ("livecodebench", 16384)}
AGENTIC_POD = set(SUITE["fast"]["agentic_envs_on_pod"])
MAX_SWE_JOBS = int(os.environ.get("ROWS_MAX_SWE_JOBS", "2"))
MAX_RETRIES = 2          # model / harness failures
MAX_INFRA_EXITS = 12     # infrastructure exits (no stock, reaped box, Daytona) are not attempts, but not forever either
INFRA_ENV_SHARE = 0.20   # a job whose results are >= 20 % infra_env trials was cut down by our infrastructure


def infra_exit(info: dict, job: dict, code: int) -> str:
    """Why this launcher exit was OUR infrastructure (empty string = the model / harness result stands)."""
    if code in (2, 3):
        return {2: "no stock / no job dir", 3: "pod never served"}[code]
    if job["kind"] in ("swe", "swe4h"):
        cell = "swebench-verified@4h250__t0" if job["kind"] == "swe4h" else "swebench-verified__t0"
        summ = BENCH_HOME / "runs" / info["run_id"] / "king" / cell / "summary.json"
        try:
            x = json.loads(summ.read_text())
        except (OSError, ValueError):
            return "no summary written"
        n, env_n = int(x.get("n") or 0), int(x.get("n_infra_env") or 0)
        if n == 0:
            return "empty job"
        if env_n / n >= INFRA_ENV_SHARE:
            return f"{env_n}/{n} trials never ran against a live model (reaped box / Daytona)"
        if n < 500 and x.get("exit_code", 0) != 0:
            return f"harbor exited {x.get('exit_code')} at {n}/500"
    return ""
# typical wall time per job kind (minutes), for the ETA projection
DUR = {"oldcap8k": 150, "oldcap16k": 150, "swe": 150, "swe4h": 330, "minif2f": 90, "cells": 120, "tb2": 240, "agentic": 120}


def log(msg: str) -> None:
    print(f"[rows-watch] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def load_state() -> dict:
    p = STATE / "rows_watch.json"
    return json.loads(p.read_text()) if p.exists() else {"jobs": {}, "paged": {}}


def save_state(s: dict) -> None:
    (STATE / "rows_watch.json").write_text(json.dumps(s, indent=1))


def king_state() -> dict:
    return json.loads((REPO / "affine" / "state" / "state.json").read_text())["king"]


def target_info(t: str) -> dict:
    """label -> {row_label, ref, digest12, label, run_id, side}."""
    if t == "teacher":
        return {"row": "Teacher", "ref": "hf://Qwen/Qwen3.8-27B@main", "d12": "hf-main", "label": "teacher", "run_id": TEACHER_FROM, "side": "teacher"}
    if t == "genesis":
        return {"row": "Genesis", "ref": "hf://Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0", "d12": "hf-995ad96ea", "label": "genesis",
                "run_id": "20260915T1415Z-genesis", "side": "king"}
    k = king_state()
    n = int(t)
    rev = k["revision"] if k["reign_number"] == n else next(p["revision"] for p in k["previous"] if p["reign_number"] == n)
    # the row's main card = the newest complete-or-partial card for this digest that has a fast/lium mode
    cards = []
    for p in CARDS.glob(f"*{rev[:12]}*.json"):
        try:
            c = json.loads(p.read_text())
        except ValueError:
            continue
        if (c.get("king") or {}).get("reign") == n and c.get("mode") in ("fast", "lium", "prime", "full", None) and not p.name.endswith("-agentic.json"):
            cards.append((len([r for r in c.get("rows", []) if (r.get("king") or {}).get("score") is not None]), c.get("run_id")))
    run_id = max(cards)[1] if cards else None
    return {"row": f"King {n}", "ref": rev, "d12": rev[:12], "label": str(n), "run_id": run_id, "side": "king"}


def matrix() -> dict:
    r = httpx.get(MATRIX_URL, timeout=60)
    r.raise_for_status()
    return r.json()


def env_gaps(mx: dict, row_label: str) -> list[str]:
    """env-table columns (kings.affine.io) under 24 rollouts or absent for a row.

    affine_wiki grades nothing and is 0 for every model. affine_tau2 is not a gap either: its harness
    fails before the first model call on every row (reign 13: 300 infra rows, 0 graded) and the board
    renders it "errored" -- an env-side fix, not something more rollouts can fill (kingboard worker,
    2026-09-22). Reign 13's affine_tau2_synth 0.0% is real (50 graded, 0 solved), not an ingest gap.
    """
    cols = [c.get("key") if isinstance(c, dict) else c for c in mx["columns"]]
    envs = [c[4:] for c in cols if c.startswith("env:") and c not in ("env:affine_wiki", "env:affine_tau2")]
    row = next((r for r in mx["rows"] if r.get("label") == row_label), None)
    if row is None:
        return envs
    cells = row.get("cells") or {}
    out = []
    for e in envs:
        c = cells.get(f"env:{e}")
        n = (c or {}).get("n") or 0
        if n < 24:
            out.append(f"{e}:{n}")
    return out


def missing_cells(mx: dict, row_label: str) -> tuple[list[str], list[str]]:
    """(missing bench envs, envs the board shows as failed) for a row, by the board's own columns."""
    cols = [c.get("key") if isinstance(c, dict) else c for c in mx["columns"]]
    bench = [c[6:] for c in cols if c.startswith("bench:")]
    row = next((r for r in mx["rows"] if r.get("label") == row_label), None)
    if row is None:
        return bench, []
    cells = row.get("cells") or {}
    missing, failed = [], []
    for e in bench:
        if e in SKIP:
            continue
        c = cells.get(f"bench:{e}")
        if c is None:
            missing.append(e)
        elif (c.get("status") or "").startswith("fail") or (c.get("score") is None and not c.get("unverified")):
            failed.append(e)
    return missing, failed


def jobs_for(info: dict, envs: list[str]) -> list[dict]:
    """Group missing envs into launchable jobs."""
    jobs = []
    envs = [e for e in envs if e not in SKIP]
    old8 = [OLDCAP[e][0] for e in envs if e in OLDCAP and OLDCAP[e][1] == 8192]
    old16 = [OLDCAP[e][0] for e in envs if e in OLDCAP and OLDCAP[e][1] == 16384]
    if old8:
        jobs.append({"kind": "oldcap8k", "envs": old8})
    if old16:
        jobs.append({"kind": "oldcap16k", "envs": old16})
    if "swebench-verified" in envs:
        jobs.append({"kind": "swe", "envs": ["swebench-verified"]})
    if "swebench-verified@4h250" in envs:
        jobs.append({"kind": "swe4h", "envs": ["swebench-verified@4h250"]})
    if "minif2f" in envs:
        jobs.append({"kind": "minif2f", "envs": ["minif2f"]})
    if "terminal-bench-2" in envs:
        jobs.append({"kind": "tb2", "envs": ["terminal-bench-2"]})
    ag = [e for e in envs if e in AGENTIC_POD]
    if ag:
        jobs.append({"kind": "agentic", "envs": ag})
    rest = [e for e in envs if e not in OLDCAP and e not in ("swebench-verified", "swebench-verified@4h250", "minif2f", "terminal-bench-2") and e not in AGENTIC_POD]
    if rest:
        jobs.append({"kind": "cells", "envs": rest})
    return jobs


def pass_alive(run_id: str) -> bool:
    """The row's own pass or a cap backfill into this run id is still running."""
    for p in [STATE / f"pass-{run_id}.pid", *STATE.glob(f"pass-capfill-{run_id}-*.pid")]:
        try:
            pid = int(p.read_text().strip())
        except (OSError, ValueError):
            continue
        if subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode == 0:
            return True
    return False


def harbor_busy(d12: str, env: str) -> bool:
    """A Harbor job for this model+env is already running (the row's own pass, an earlier launch, or a
    chain script). `<env>@<tag>` cells run as `--env <env> --budget-tag <tag>`; the plain cell must NOT
    match a tagged one (2026-09-22: the watcher launched a second reign-19 @4h250 resume next to the
    running one because swe4h never checked)."""
    if "@" in env:
        base, tag = env.split("@", 1)
        pat = f"harbor_cell.py (run|resume) --env {base} --budget-tag {tag} .*--model king-{d12}"
    else:
        pat = f"harbor_cell.py (run|resume) --env {env} --model king-{d12}"
    out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.strip()
    return bool(out)


def swe_jobs_running() -> int:
    """SWE Daytona jobs in flight or about to be (a swe_rerun/swe_resume still renting its pod counts)."""
    def n(pat: str) -> int:
        # distinct command lines, not pids: a bash script forks a copy of itself (same cmdline) for every
        # `$(...)` substitution, and a script polling for stock did that non-stop -> two scripts counted as four
        out = subprocess.run(["pgrep", "-fa", pat], capture_output=True, text=True).stdout
        return len({line.split(" ", 1)[1] for line in out.splitlines() if " " in line})
    harbor = n("harbor_cell.py run --env swebench-verified") + n("harbor_cell.py resume --env swebench-verified")
    scripts = n("^bash .*swe_rerun.sh ") + n("^bash .*swe_resume.sh ")
    return max(harbor, scripts)


def launch(info: dict, job: dict) -> subprocess.Popen:
    env = dict(os.environ)
    run_dir = BENCH_HOME / "runs" / info["run_id"]
    label, ref = info["label"], info["ref"]
    logf = STATE / f"rows-{info['d12']}-{job['kind']}-{time.strftime('%H%M', time.gmtime())}.log"
    fh = open(logf, "a")
    kind = job["kind"]
    if kind in ("oldcap8k", "oldcap16k"):
        cap = 8192 if kind == "oldcap8k" else 16384
        run_id = f"{time.strftime('%Y%m%dT%H%MZ', time.gmtime())}-{info['d12']}-oldcap{cap // 1024}k"
        env.update(CAP_ENVS=",".join(job["envs"]), CAP_NEW_RUN="1", BENCHSUITE_SETTINGS_JSON=json.dumps({"max_tokens": cap}),
                   CAP_RUN_NOTE=f"old completion cap {cap} (the default before 2026-09-19) for the board's @{cap // 1024}k columns")
        cmd = ["bash", str(HERE / "cap_backfill.sh"), ref, label, run_id, info["side"]]
    elif kind == "swe":
        harbor = run_dir / "king" / "swebench-verified__t0" / "harbor"
        script = "swe_resume.sh" if harbor.is_dir() else "swe_rerun.sh"
        cmd = ["bash", str(HERE / script), ref, label, info["run_id"]]
    elif kind == "swe4h":
        # an interrupted job is RESUMED (finished trials stay, infra-errored ones re-run); `run` would
        # skip on the stale summary.json and exit 0 — that was reign 15's and 16's "3 attempts"
        cell = run_dir / "king" / "swebench-verified@4h250__t0"
        harbor = cell / "harbor"
        if harbor.is_dir() and (harbor / "config.json").exists():
            cmd = ["bash", str(HERE / "swe_resume.sh"), ref, label, info["run_id"], "4h250"]
        else:
            if cell.exists():   # a job that died before harbor wrote its config cannot be resumed; `run` would skip on its summary
                dead = cell.with_name(cell.name + f".dead-{time.strftime('%d%H%M', time.gmtime())}")
                cell.rename(dead); log(f"{info['row']} swe4h: unresumable cell moved to {dead.name}")
            env.update(BUDGET_TAG="4h250")
            cmd = ["bash", str(HERE / "swe_rerun.sh"), ref, label, info["run_id"]]
    elif kind == "minif2f":
        env.update(CAP_ENVS="minif2f", CAP_RUNTIME="prime")
        cmd = ["bash", str(HERE / "cap_backfill.sh"), ref, label, info["run_id"], info["side"]]
    elif kind == "cells":
        env.update(CAP_ENVS=",".join(job["envs"]))
        cmd = ["bash", str(HERE / "cap_backfill.sh"), ref, label, info["run_id"], info["side"]]
    elif kind == "tb2":
        env.update(FAST_ONLY_ROLES="agentic", FAST_GROUPS="tb2")
        cmd = ["bash", str(HERE / "pass.sh"), ref, label, info["run_id"], "fast"]
    elif kind == "agentic":
        env.update(FAST_ONLY_ROLES="agentic", FAST_GROUPS="agentic", FAST_AGENTIC_ENVS=",".join(job["envs"]))
        cmd = ["bash", str(HERE / "pass.sh"), ref, label, info["run_id"], "fast"]
    else:
        raise ValueError(kind)
    log(f"launch {info['row']} {kind} {job['envs']}: {' '.join(cmd)} -> {logf.name}")
    return subprocess.Popen(cmd, cwd=str(HERE), env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)


def discord(msg: str) -> None:
    tok = os.environ.get("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR")
    ch = os.environ.get("ROWS_DISCORD_CHANNEL", "1510910974498967613")
    if not tok:
        log("no Discord token; would have posted: " + msg[:200])
        return
    try:
        httpx.post(f"https://discord.com/api/v10/channels/{ch}/messages", headers={"Authorization": f"Bot {tok}"},
                   json={"content": msg[:1900]}, timeout=30)
    except httpx.HTTPError as e:
        log(f"discord post failed: {e!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="20,teacher,19,18,17,16,15")
    ap.add_argument("--deadline", default="2026-09-21T07:00Z")
    ap.add_argument("--interval", type=int, default=1800)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    deadline = datetime.strptime(a.deadline, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    procs: dict[str, subprocess.Popen] = {}
    while True:
        st = load_state()
        try:
            mx = matrix()
        except Exception as e:
            log(f"matrix unavailable ({e!r}); retry next cycle")
            time.sleep(300)
            continue
        lines = []
        worst_eta = None
        swe_blocked = False   # a higher-priority row is waiting for a Daytona slot: lower rows do not take it
        for t in [x.strip() for x in a.targets.split(",") if x.strip()]:
            try:
                info = target_info(t)
            except Exception as e:
                log(f"{t}: cannot resolve ({e!r})")
                continue
            if not info.get("run_id"):
                log(f"{t}: no card yet (a full pass is the watcher's job); skipping")
                continue
            missing, failed = missing_cells(mx, info["row"])
            todo = missing + failed
            running_here = []
            main_alive = pass_alive(info["run_id"])
            for job in jobs_for(info, todo):
                key = f"{info['d12']}:{job['kind']}"
                # the row's own pass is still producing cells: only the variants it never makes are ours now
                if main_alive and job["kind"] not in ("oldcap8k", "oldcap16k", "swe4h"):
                    running_here.append(job["kind"] + "(pass)"); continue
                if job["kind"] in ("swe", "swe4h", "tb2") and harbor_busy(info["d12"], job["envs"][0]):
                    running_here.append(job["kind"] + "(harbor)"); continue
                rec = st["jobs"].setdefault(key, {"attempts": 0, "state": "idle"})
                p = procs.get(key)
                if p is None and rec.get("state") == "running" and rec.get("pid"):
                    # watcher restarted: the launcher it started earlier may still be running
                    if subprocess.run(["kill", "-0", str(rec["pid"])], capture_output=True).returncode == 0:
                        running_here.append(job["kind"] + "(prev)"); continue
                    rec["state"] = "ended"
                if p is not None and p.poll() is None:
                    running_here.append(job["kind"])
                    continue
                if p is not None:                      # ended: the cell decides whether it worked
                    rec["state"] = "ended"; rec["exit"] = p.returncode; procs.pop(key, None)
                    why = infra_exit(info, job, p.returncode)
                    if why:
                        # our infrastructure, not the model: no stock, a pod that never served, a reaped serving
                        # box or Daytona outage mid-job -> the attempt does not count (Jacob 2026-09-22 02:55)
                        rec["attempts"] = max(0, rec["attempts"] - 1); rec["infra_exits"] = rec.get("infra_exits", 0) + 1
                        log(f"{info['row']} {job['kind']}: exit {p.returncode} classified infra ({why}); not counted "
                            f"(infra exits so far {rec['infra_exits']})")
                        if rec["infra_exits"] >= MAX_INFRA_EXITS:
                            rec["state"] = "gave_up"
                            discord(f"[rows-watch] {info['row']} {job['kind']}: {rec['infra_exits']} infrastructure exits in a row ({why}); stopping — needs a human")
                            continue
                if rec.get("state") == "gave_up":
                    continue
                if rec["attempts"] > MAX_RETRIES:
                    rec["state"] = "gave_up"
                    log(f"{info['row']} {job['kind']}: gave up after {rec['attempts']} attempts — {job['envs']}")
                    discord(f"[rows-watch] {info['row']} {job['kind']} {job['envs']}: gave up after {rec['attempts']} attempts (cells stay 'run failed')")
                    continue
                lp = STATE / f"swe-local-{info['d12']}.pid"
                if job["kind"] == "swe4h" and lp.exists() and subprocess.run(["kill", "-0", lp.read_text().strip()], capture_output=True).returncode == 0:
                    running_here.append("swe4h(+docker lane)")   # swe_prime.sh runs it too; both lanes race, first to publish wins
                if job["kind"] in ("swe", "swe4h") and (swe_blocked or swe_jobs_running() >= MAX_SWE_JOBS):
                    running_here.append(job["kind"] + "(waiting: Daytona)"); swe_blocked = True
                    continue
                if job["kind"] == "minif2f" and int(subprocess.run(["pgrep", "-fc", "CAP_ENVS=minif2f|eval minif2f"], capture_output=True, text=True).stdout.strip() or 0) > 0:
                    # Prime sandboxes: one MiniF2F job at a time (three at once hit "Total CPU limit exceeded" 429s today)
                    running_here.append("minif2f(waiting: Prime sandboxes)"); continue
                rec["attempts"] += 1; rec["state"] = "running"; rec["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); rec["envs"] = job["envs"]
                procs[key] = launch(info, job); rec["pid"] = procs[key].pid
                running_here.append(job["kind"])
            eta = None
            if todo:
                mins = max([DUR.get(j["kind"], 120) for j in jobs_for(info, todo)] or [0]) + 25
                eta = datetime.now(timezone.utc).timestamp() + mins * 60
                worst_eta = max(worst_eta or 0, eta)
            eg = env_gaps(mx, info["row"])
            lines.append(f"{info['row']}: missing {len(missing)} {missing} failed {failed} running {running_here}"
                         + (f" eta {datetime.fromtimestamp(eta, timezone.utc).strftime('%H:%M')}Z" if eta else " FULL")
                         + (f" | env rows <24: {eg}" if eg else " | env rows FULL"))
        save_state(st)
        status = f"{time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())}\n" + "\n".join(lines)
        (STATE / "rows_watch_status.txt").write_text(status + "\n")
        log("status:\n" + status)
        if worst_eta and worst_eta > deadline.timestamp():
            hour = time.strftime("%Y%m%d%H", time.gmtime())
            if st["paged"].get("slip") != hour:
                st["paged"]["slip"] = hour; save_state(st)
                discord(f"[rows-watch] projected finish {datetime.fromtimestamp(worst_eta, timezone.utc).strftime('%H:%M')}Z is past the "
                        f"{deadline.strftime('%H:%M')}Z deadline:\n{status}")
        # 3-hourly status to Discord (the agent cannot self-report; the channel gets the same lines)
        hh = time.gmtime().tm_hour
        if hh % 3 == 0 and st["paged"].get("status_hour") != f"{time.strftime('%Y%m%d')}{hh}":
            st["paged"]["status_hour"] = f"{time.strftime('%Y%m%d')}{hh}"; save_state(st)
            discord(f"[rows-watch] {status}")
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
