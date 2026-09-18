#!/usr/bin/env python
"""Run one sandbox-heavy benchmark cell through Harbor on a cloud sandbox
backend (Daytona) while the GPU box only serves the model.

Why: our verifiers docker runtime runs every task container ON the GPU pod,
so a SWE-bench Verified pass is 48 tasks in flight for 6+ h on one H200 and
the pass is stock-bound on Lium. With Harbor + Daytona (org quota 500 vCPU /
1000 GB, region us) the containers run on Daytona, the pod only answers
model calls, and 100–200 tasks run at once.

What it runs: `harbor run -d <dataset@version> -a <agent> -e daytona -m
openai/<served model>` with the model connection passed as agent env
(OPENAI_API_BASE / OPENAI_API_KEY / MSWEA_API_KEY). Agents:
  mini-swe-agent  SWE-bench Verified / Pro (Harbor installs upstream
                  mini-swe-agent v2 inside the sandbox; our verifiers cells
                  use verifiers' own mini_swe_agent harness — a HARNESS
                  CHANGE, flagged on the card as harness_change = true)
  terminus-2      Terminal-Bench 2 (the leaderboard's reference agent; our
                  verifiers cell uses verifiers' terminus_2 harness)

Output: the cell dir gets Harbor's job under harbor/ (config, per-trial
result.json, agent trajectories, verifier output) plus a summary.json in the
benchsuite schema (n, score, ci95, finished_only, error classes, per-rollout
rows) so publish.py / report.py / the kingboard read it unchanged. The
summary records sandbox = "daytona", harness = "harbor:<agent>", the budget
(agent timeout, step limit, max_tokens, temperature) and harness_change.

Budget tag: a cell run at a non-default budget is written as
<env>@<tag>__t<T> (e.g. swebench-verified@4h250__t0) — a separate column on
the cards, never overwriting the 1-h cell.

  harbor_cell.py run --env swebench-verified --model-url http://h:p/v1 --model king-xxx \
      --model-key-env BENCH_API_KEY --out <run_dir>/king --concurrency 100 \
      [--budget-tag 4h250 --agent-timeout-s 14400 --step-limit 250] [--n-tasks 2]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
HARBOR_BIN = Path(os.environ.get("HARBOR_BIN", str(Path.home() / "benchsuite" / "harborenv" / "bin" / "harbor")))


def log(msg: str) -> None:
    print(f"[harbor-cell] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def env_by_id(env_id: str) -> dict:
    for e in SUITE["envs"]:
        if e["id"] == env_id:
            return e
    raise SystemExit(f"unknown env {env_id}")


def mini_swe_config(step_limit: int, temperature: float, top_p: float | None) -> str:
    kw = {"temperature": temperature}
    if top_p is not None:
        kw["top_p"] = top_p
    cfg = {"agent": {"step_limit": step_limit}, "model": {"model_kwargs": kw}}
    return json.dumps(cfg)  # YAML-compatible JSON


def build_cmd(env: dict, a: argparse.Namespace, job_dir: Path, cfg_path: Path | None) -> list[str]:
    hb = env["harbor"]
    model_key = os.environ.get(a.model_key_env, "")
    cmd = [str(HARBOR_BIN), "run", "-d", hb["dataset"], "-a", hb["agent"], "-e", "daytona",
           "-m", f"openai/{a.model}", "-n", str(a.concurrency), "-y", "-q",
           "-o", str(job_dir.parent), "--job-name", job_dir.name,
           "--ae", f"OPENAI_API_BASE={a.model_url}", "--ae", f"OPENAI_BASE_URL={a.model_url}",
           "--ae", f"OPENAI_API_KEY={model_key}", "--ae", f"MSWEA_API_KEY={model_key}",
           "--agent-timeout-multiplier", f"{a.agent_timeout_s / hb['task_agent_timeout_s']:.4f}",
           "--max-retries", str(a.max_retries)]
    if hb["agent"] == "mini-swe-agent":
        cmd += ["--ak", f"max_tokens={int(env['max_tokens'])}"]
        if cfg_path is not None:
            cmd += ["--ak", f"config_file={cfg_path}"]
    elif hb["agent"] == "terminus-2":
        # terminus-2 runs in the harbor process (host side) and talks to the model
        # through LiteLLM: endpoint + sampling as agent kwargs, key from the host env
        # (OPENAI_API_KEY, exported by cmd_run). No per-call max_tokens knob: the
        # agent's own default applies — a forced difference vs our docker cell (8k).
        cmd += ["--ak", f"api_base={a.model_url}", "--ak", f"temperature={a.temperature:g}"]
        if a.step_limit:
            cmd += ["--ak", f"max_turns={int(a.step_limit)}"]
    if a.n_tasks and a.n_tasks > 0:
        cmd += ["-l", str(a.n_tasks)]
    for x in hb.get("extra_args") or []:
        cmd.append(str(x))
    return cmd


def agent_log_has(trial_dir: Path, needle: str) -> bool:
    for name in ("agent/mini-swe-agent.txt", "agent/terminus-2.txt", "trial.log"):
        p = trial_dir / name
        if p.exists():
            try:
                if needle in p.read_text(errors="replace")[-20000:]:
                    return True
            except OSError:
                pass
    return False


def summarize(job_dir: Path, env: dict, a: argparse.Namespace, wall: float, exit_code: int) -> dict:
    rows = []
    for rp in sorted(job_dir.glob("*/result.json")):
        try:
            r = json.loads(rp.read_text())
        except ValueError:
            continue
        if "trial_name" not in r:
            continue
        rewards = ((r.get("verifier_result") or {}).get("rewards") or {})
        score = rewards.get("reward")
        exc = r.get("exception_info") or {}
        etype = exc.get("exception_type") or ""
        emsg = (exc.get("exception_message") or "")[:160]
        ag = r.get("agent_execution") or {}
        t0, t1 = ag.get("started_at"), ag.get("finished_at")
        secs = None
        if t0 and t1:
            from datetime import datetime
            secs = (datetime.fromisoformat(t1.replace("Z", "+00:00")) - datetime.fromisoformat(t0.replace("Z", "+00:00"))).total_seconds()
        ares = r.get("agent_result") or {}
        # error classes as in run_suite.summarize_traces: the model's budget
        # exhaustion is a 0, infrastructure is excluded from n and retried
        if "Timeout" in etype and "Agent" in etype:
            err_class = "timeout"
        elif etype == "NonZeroAgentExitCodeError" and agent_log_has(rp.parent, "ContextWindowExceeded"):
            # mini-swe-agent exits 1 when the conversation outgrows the model's context:
            # the MODEL's failure (our verifiers cells call it context_overflow, score 0),
            # not infrastructure (reign 13 @4h: 110 of 500)
            err_class = "context_overflow"
        elif etype:
            err_class = "infra"
        else:
            err_class = None
        if score is None and err_class in ("timeout", "context_overflow"):
            score = 0.0
        rows.append({
            "task_key": r.get("task_name"), "trial": r.get("trial_name"),
            "score": None if score is None else float(score),
            "error_class": err_class, "errors": [etype] if etype else [], "error_messages": [emsg] if emsg else [],
            "agent_seconds": None if secs is None else round(secs, 1),
            "prompt_tokens": int(ares.get("n_input_tokens") or 0), "completion_tokens": int(ares.get("n_output_tokens") or 0),
            "cost_usd": ares.get("cost_usd"),
        })
    scored = [r["score"] for r in rows if r["score"] is not None]
    k = int(sum(1 for s in scored if s >= 1.0))
    lo, hi = wilson(k, len(scored))
    fin = [r["score"] for r in rows if r["score"] is not None and r["error_class"] is None]
    fk = int(sum(1 for s in fin if s >= 1.0))
    flo, fhi = wilson(fk, len(fin))
    hb = env["harbor"]
    return {
        "n": len(rows), "n_scored": len(scored),
        "n_errored": sum(1 for r in rows if r["error_class"] == "infra"),
        "n_timeout": sum(1 for r in rows if r["error_class"] == "timeout"),
        "n_context_overflow": sum(1 for r in rows if r["error_class"] == "context_overflow"),
        "score": round(k / len(scored), 4) if scored else 0.0, "ci95": [round(lo, 4), round(hi, 4)],
        "finished_only": {"n": len(fin), "score": round(fk / len(fin), 4) if fin else 0.0, "ci95": [round(flo, 4), round(fhi, 4)]},
        "binary": True,
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "reasoning_tokens": 0,
        "finish_length_frac": None,   # harbor does not expose per-call finish reasons
        "env": a.env if not a.budget_tag else f"{a.env}@{a.budget_tag}", "base_env": a.env,
        "taskset": env["taskset"], "model": a.model_label, "temperature": a.temperature,
        "rollouts_per_task": 1,
        "harness": f"harbor:{hb['agent']}", "harness_change": True,
        "harness_note": ("upstream mini-swe-agent v2 installed in the sandbox by Harbor (our 1-h cells: verifiers' own "
                         "mini_swe_agent harness)" if hb["agent"] == "mini-swe-agent" else
                         "Harbor's terminus-2 agent (our docker cells: verifiers' terminus_2 harness)"),
        "sandbox": "daytona", "runtime": "harbor/daytona",
        "budget": {"tag": a.budget_tag or "default", "agent_timeout_s": a.agent_timeout_s,
                   "step_limit": a.step_limit, "max_tokens": int(env["max_tokens"]),
                   "temperature": a.temperature, "top_p": a.top_p, "concurrency": a.concurrency},
        "max_tokens": int(env["max_tokens"]), "reward": env["reward"],
        "wall_seconds": round(wall, 1), "exit_code": exit_code,
        "task_subset": {"n": int(a.n_tasks)} if a.n_tasks and a.n_tasks > 0 else {"n": "all"},
        "harbor": {"dataset": hb["dataset"], "agent": hb["agent"], "job_dir": str(job_dir),
                   "harbor_version": harbor_version()},
        "rollouts": rows,
    }


def harbor_version() -> str:
    try:
        return subprocess.run([str(HARBOR_BIN), "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return "?"


def cmd_run(a: argparse.Namespace) -> int:
    env = env_by_id(a.env)
    if "harbor" not in env:
        raise SystemExit(f"{a.env} has no [envs].harbor block in suite.toml")
    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit("DAYTONA_API_KEY not set")
    cell = f"{a.env}@{a.budget_tag}" if a.budget_tag else a.env
    d = Path(a.out).expanduser() / f"{cell}__t{a.temperature:g}"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "summary.json").exists() and not a.force:
        log(f"skip {d}: summary.json exists")
        return 0
    job_dir = d / "harbor"
    cfg_path = None
    if env["harbor"]["agent"] == "mini-swe-agent":
        cfg_path = d / "mini_swe_config.yaml"
        cfg_path.write_text(mini_swe_config(int(a.step_limit or 0) or 10**9, a.temperature, a.top_p))
    os.environ["OPENAI_API_KEY"] = os.environ.get(a.model_key_env, "")   # host-side agents (terminus-2)
    cmd = build_cmd(env, a, job_dir, cfg_path)
    (d / "cmd.txt").write_text(" ".join(c if "API_KEY" not in c else c.split("=")[0] + "=***" for c in cmd) + "\n")
    log(f"start {a.model_label}/{cell}: {(d / 'cmd.txt').read_text().strip()}")
    t0 = time.time()
    with (d / "harbor.log").open("a") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=os.environ.copy())
    wall = time.time() - t0
    if not (job_dir / "result.json").exists() and not list(job_dir.glob("*/result.json")):
        log(f"FAIL {cell}: exit={p.returncode}, no results (see {d / 'harbor.log'})")
        return 1
    summ = summarize(job_dir, env, a, wall, p.returncode)
    (d / "summary.json").write_text(json.dumps(summ, indent=1))
    log(f"done {a.model_label}/{cell}: n={summ['n']} score={summ['score']} ci={summ['ci95']} "
        f"finished-only={summ['finished_only']['score']} (n={summ['finished_only']['n']}) "
        f"timeouts={summ['n_timeout']} infra={summ['n_errored']} wall={wall / 60:.1f}min exit={p.returncode}")
    return 0 if p.returncode == 0 else 1


def cmd_resume(a: argparse.Namespace) -> int:
    """Resume an interrupted Harbor job in place (`harbor job resume`): finished
    trials stay, unfinished ones rerun. Harbor masks secrets in the saved
    config (MSWEA_API_KEY -> 'sk-e****'), so the masked entries are rewritten to
    ${VAR} references and the real values exported. Never kill a running harbor
    with SIGKILL / by closing its tmux pane: its sandboxes stay up on Daytona
    (2026-09-17: 107 orphans, deleted by hand); SIGINT lets it clean up."""
    env = env_by_id(a.env)
    cell = f"{a.env}@{a.budget_tag}" if a.budget_tag else a.env
    d = Path(a.out).expanduser() / f"{cell}__t{a.temperature:g}"
    job_dir = d / "harbor"
    cfg_path = job_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    key = os.environ.get(a.model_key_env, "")
    os.environ["OPENAI_API_KEY"] = key
    os.environ["MSWEA_API_KEY"] = key
    n_conc = int(a.concurrency) if a.concurrency else int(cfg.get("n_concurrent_trials") or 0)

    def patch(o) -> bool:
        # harbor refuses to resume unless job config, job lock and every trial's
        # config/lock agree, so the same rewrite goes into all of them
        changed = False
        if isinstance(o, dict):
            for k, v in list(o.items()):
                if k in ("OPENAI_API_KEY", "MSWEA_API_KEY") and isinstance(v, str) and v != "${" + k + "}":
                    o[k] = "${" + k + "}"; changed = True
                elif k == "n_concurrent_trials" and n_conc and o[k] != n_conc:
                    o[k] = n_conc; changed = True
                else:
                    changed |= patch(v)
        elif isinstance(o, list):
            for v in o:
                changed |= patch(v)
        return changed

    n_patched = 0
    for fp in [cfg_path, job_dir / "lock.json", *job_dir.glob("*/config.json"), *job_dir.glob("*/lock.json")]:
        try:
            c = json.loads(fp.read_text())
        except (OSError, ValueError):
            continue
        if patch(c):
            fp.write_text(json.dumps(c, indent=2)); n_patched += 1
    cfg = json.loads(cfg_path.read_text())
    log(f"resume: {n_patched} config/lock files rewritten (secret refs, n_concurrent_trials={cfg.get('n_concurrent_trials')})")
    cmd = [str(HARBOR_BIN), "job", "resume", "-p", str(job_dir)]
    log(f"resume {a.model_label}/{cell}: {' '.join(cmd)} (n_concurrent_trials={cfg.get('n_concurrent_trials')})")
    t0 = time.time()
    with (d / "harbor.log").open("a") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=os.environ.copy())
    prev = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    wall = float(prev.get("wall_seconds") or 0) + time.time() - t0
    summ = summarize(job_dir, env, a, wall, p.returncode)
    summ["resumed"] = int(prev.get("resumed") or 0) + 1
    (d / "summary.json").write_text(json.dumps(summ, indent=1))
    log(f"done {a.model_label}/{cell}: n={summ['n']} score={summ['score']} ci={summ['ci95']} "
        f"finished-only={summ['finished_only']['score']} (n={summ['finished_only']['n']}) timeouts={summ['n_timeout']} infra={summ['n_errored']} exit={p.returncode}")
    return 0 if p.returncode == 0 else 1


def cmd_resummarize(a: argparse.Namespace) -> int:
    env = env_by_id(a.env)
    cell = f"{a.env}@{a.budget_tag}" if a.budget_tag else a.env
    d = Path(a.out).expanduser() / f"{cell}__t{a.temperature:g}"
    prev = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    summ = summarize(d / "harbor", env, a, float(prev.get("wall_seconds") or 0), int(prev.get("exit_code") or 0))
    (d / "summary.json").write_text(json.dumps(summ, indent=1))
    log(f"{cell}: n={summ['n']} score={summ['score']} ci={summ['ci95']} timeouts={summ['n_timeout']} infra={summ['n_errored']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run", cmd_run), ("resume", cmd_resume), ("resummarize", cmd_resummarize)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--env", required=True)
        s.add_argument("--out", required=True, help="<run_dir>/<king|teacher>")
        s.add_argument("--model", default="", help="served model name (openai/<model> for harbor)")
        s.add_argument("--model-label", default="king")
        s.add_argument("--model-url", default="")
        s.add_argument("--model-key-env", default="BENCH_API_KEY")
        s.add_argument("--concurrency", type=int, default=None, help="in-flight trials (default: suite.toml sandbox_daytona.concurrency; resume: keep the job's)")
        s.add_argument("--temperature", type=float, default=float(SUITE["sampling"]["primary_temperature"]))
        s.add_argument("--top-p", type=float, default=None)
        s.add_argument("--budget-tag", default="", help="e.g. 4h250: the cell becomes <env>@<tag>")
        s.add_argument("--agent-timeout-s", type=int, default=3600)
        s.add_argument("--step-limit", type=int, default=0, help="mini-swe-agent step_limit / terminus max_episodes (0 = env default)")
        s.add_argument("--n-tasks", type=int, default=0)
        s.add_argument("--max-retries", type=int, default=1)
        s.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.concurrency is None and a.cmd != "resume":
        a.concurrency = int(SUITE.get("sandbox_daytona", {}).get("concurrency", 100))
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
