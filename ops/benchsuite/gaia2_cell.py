#!/usr/bin/env python
"""Gaia2 `ambiguity` split (Meta ARE) as a benchsuite cell — the "did it ask?"
benchmark: impossible / contradictory / under-specified phone-assistant tasks
whose pass condition is asking the user instead of acting.

Runs Meta's own runner (`are-benchmark run`, package
meta-agents-research-environments, pinned in suite.toml) against our model:
  agent  -> ARE's default agent, provider "local" = any OpenAI-compatible
            endpoint, reached through auth_proxy.py so the pod's bearer and the
            judge's key do not collide (ARE reads ONE OPENAI_API_KEY)
  user   -> scripted in the scenario (no user simulator, $0)
  judge  -> ARE's GraphPerEventJudge: hard per-event checks + an LLM soft
            checker for free-text arguments (messages, emails). We pin the LLM
            to openai/gpt-4.1 on Prime Inference, T as ARE sets it -> the cell
            is graded = "llm_judge" (advisory), like tau3-banking.
  data   -> HF meta-agents-research-environments/gaia2, config `ambiguity`,
            split `validation` (200 scenarios), CC-BY-4.0 "benchmarking only"
            -> eval-only, never in D. The scenarios' file systems come from
            gaia2_filesystem; ARE lazy-loads them through the HF API per file,
            which trips the 1000-req/5-min limit at scale, so we mirror that
            dataset once (GAIA2_FS_LOCAL) and point ARE at the mirror.

Output: <out>/gaia2-ambiguity__t0/{are/, summary.json} in the benchsuite schema.
One run per scenario (Meta's leaderboard uses 3; a card row is one pass).

  gaia2_cell.py run --model-url http://pod:port/v1 --model king-xxx --model-key-env BENCH_API_KEY \
      --judge-key-env PRIME_API_KEY --out <run_dir>/king [--n-tasks 2] [--concurrency 16]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
ARE_BIN = Path(os.environ.get("ARE_BIN", str(Path.home() / "benchsuite" / "areenv" / "bin" / "are-benchmark")))
ENV_ID = "gaia2-ambiguity"
HF_DATASET = "meta-agents-research-environments/gaia2"
HF_CONFIG = "ambiguity"
HF_SPLIT = "validation"
JUDGE_MODEL = "openai/openai/gpt-4.1"   # LiteLLM strips one "openai/"; Prime wants "openai/gpt-4.1"
JUDGE_ENDPOINT = "https://api.pinference.ai/api/v1"


def log(msg: str) -> None:
    print(f"[gaia2-cell] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def env_cfg() -> dict:
    for e in SUITE["envs"]:
        if e["id"] == ENV_ID:
            return e
    raise SystemExit(f"{ENV_ID} missing from suite.toml")


def summarize(out_dir: Path, a: argparse.Namespace, wall: float, exit_code: int) -> dict:
    rows = []
    stats = {}
    if (out_dir / "benchmark_stats.json").exists():
        try:
            stats = json.loads((out_dir / "benchmark_stats.json").read_text())
        except ValueError:
            stats = {}
    for line in (out_dir / "output.jsonl").read_text().splitlines() if (out_dir / "output.jsonl").exists() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        md = r.get("metadata") or {}
        status = md.get("status")
        exc = bool(md.get("has_exception"))
        score = r.get("score")
        calls = None
        tname = Path(r.get("trace_id") or "").name
        tp = out_dir / tname if tname else None
        prompt_toks = compl_toks = 0
        if tp is not None and tp.is_file():
            try:
                t = json.loads(tp.read_text())
                u = list((t.get("per_agent_llm_usage_stats") or {}).values())
                if u:
                    calls = int(u[0].get("total_llm_calls") or 0)
                    prompt_toks = int(sum(u[0].get("prompt_tokens") or []))
                    compl_toks = int(sum(u[0].get("completion_tokens") or []))
            except ValueError:
                pass
        rows.append({
            "task_key": r.get("task_id"), "score": None if exc else (float(score) if score is not None else 0.0),
            "error_class": "infra" if exc else None, "errors": ["exception"] if exc else [],
            "error_messages": [(md.get("rationale") or "")[:160]] if exc else [],
            "status": status, "rationale": (md.get("rationale") or "")[:300],
            "n_calls": calls, "prompt_tokens": prompt_toks, "completion_tokens": compl_toks,
            "trace": tp.name if (tp is not None and tp.is_file()) else None,
        })
    scored = [r["score"] for r in rows if r["score"] is not None]
    k = int(sum(1 for s in scored if s >= 1.0))
    lo, hi = wilson(k, len(scored))
    return {
        "n": len(rows), "n_scored": len(scored), "n_errored": sum(1 for r in rows if r["error_class"] == "infra"),
        "n_timeout": 0, "n_context_overflow": 0,
        "score": round(k / len(scored), 4) if scored else 0.0, "ci95": [round(lo, 4), round(hi, 4)],
        "finished_only": {"n": len(scored), "score": round(k / len(scored), 4) if scored else 0.0, "ci95": [round(lo, 4), round(hi, 4)]},
        "binary": True, "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows), "reasoning_tokens": 0,
        "finish_length_frac": None,
        "env": ENV_ID, "taskset": "are:gaia2", "model": a.model_label, "temperature": a.temperature,
        "rollouts_per_task": 1, "harness": "are:default_agent", "runtime": "are/in-process",
        "graded": "llm_judge",
        "judge": {"model": JUDGE_MODEL, "via": "Prime Inference (api.pinference.ai)", "role": "ARE GraphPerEventJudge soft checker (free-text args); hard checks are rule-based"},
        "max_tokens": None, "reward": "gaia2_success",
        "wall_seconds": round(wall, 1), "exit_code": exit_code,
        "task_subset": {"n": int(a.n_tasks)} if a.n_tasks else {"n": "all"},
        "dataset": {"hf": HF_DATASET, "config": HF_CONFIG, "split": HF_SPLIT, "revision": a.hf_revision or None,
                    "license": "CC-BY-4.0, benchmarking purposes only (eval-only, never in D)"},
        "are": {"stats": (stats.get("statistics") or {}).get("global"), "version": are_version()},
        "rollouts": rows,
    }


def are_version() -> str:
    try:
        out = subprocess.run([str(ARE_BIN.parent / "python"), "-c",
                              "import importlib.metadata as m; print(m.version('meta-agents-research-environments'))"],
                             capture_output=True, text=True, timeout=60)
        return out.stdout.strip() or "?"
    except Exception:
        return "?"


def cmd_run(a: argparse.Namespace) -> int:
    env = env_cfg()
    d = Path(a.out).expanduser() / f"{ENV_ID}__t{a.temperature:g}"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "summary.json").exists() and not a.force:
        log(f"skip {d}: summary.json exists")
        return 0
    judge_key = os.environ.get(a.judge_key_env, "")
    model_key = os.environ.get(a.model_key_env, "")
    if not judge_key or not model_key:
        raise SystemExit(f"need {a.judge_key_env} and {a.model_key_env} in the environment")
    port = free_port()
    proxy = subprocess.Popen([sys.executable, str(HERE / "auth_proxy.py"), "--port", str(port),
                              "--upstream", a.model_url, "--key-env", a.model_key_env],
                             stdout=(d / "proxy.log").open("a"), stderr=subprocess.STDOUT, env=os.environ.copy())
    time.sleep(1)
    out_dir = d / "are"
    cmd = [str(ARE_BIN), "run", "-a", "default", "--num_runs", "1",
           "--hf-dataset", HF_DATASET, "--hf-config", HF_CONFIG, "--hf-split", HF_SPLIT,
           "--max_concurrent_scenarios", str(a.concurrency),
           "--provider", "local", "--endpoint", f"http://127.0.0.1:{port}/v1", "--model", f"openai/{a.model}",
           "--judge_provider", "local", "--judge_endpoint", JUDGE_ENDPOINT, "--judge_model", JUDGE_MODEL,
           "--output_dir", str(out_dir), "--trace_dump_format", "lite",
           "--scenario_timeout", str(int(env.get("rollout_timeout") or 1800)), "--log-level", "WARNING"]
    if a.n_tasks:
        cmd += ["-l", str(a.n_tasks)]
    (d / "cmd.txt").write_text(" ".join(cmd) + "\n")
    run_env = os.environ.copy()
    run_env["OPENAI_API_KEY"] = judge_key      # the agent's key is injected by the proxy
    run_env.setdefault("DEMO_FS_PATH", str(Path.home() / "benchsuite" / "gaia2_fs" / "demo_filesystem"))   # local mirror (install_are.sh)
    log(f"start {a.model_label}/{ENV_ID}: {' '.join(cmd)}")
    t0 = time.time()
    try:
        with (d / "are.log").open("a") as fh:
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=run_env, cwd=str(d))
    finally:
        proxy.terminate()
    wall = time.time() - t0
    if not (out_dir / "output.jsonl").exists():
        log(f"FAIL {ENV_ID}: exit={p.returncode}, no output.jsonl (see {d / 'are.log'})")
        return 1
    summ = summarize(out_dir, a, wall, p.returncode)
    (d / "summary.json").write_text(json.dumps(summ, indent=1))
    log(f"done {a.model_label}/{ENV_ID}: n={summ['n']} score={summ['score']} ci={summ['ci95']} infra={summ['n_errored']} wall={wall / 60:.1f}min")
    return 0 if p.returncode == 0 else 1


def cmd_resummarize(a: argparse.Namespace) -> int:
    d = Path(a.out).expanduser() / f"{ENV_ID}__t{a.temperature:g}"
    prev = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    summ = summarize(d / "are", a, float(prev.get("wall_seconds") or 0), int(prev.get("exit_code") or 0))
    (d / "summary.json").write_text(json.dumps(summ, indent=1))
    log(f"{a.model_label}/{ENV_ID}: n={summ['n']} score={summ['score']} ci={summ['ci95']} infra={summ['n_errored']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rs = sub.add_parser("resummarize")
    rs.add_argument("--out", required=True); rs.add_argument("--model-label", default="king"); rs.add_argument("--temperature", type=float, default=0.0)
    rs.add_argument("--n-tasks", type=int, default=0); rs.add_argument("--hf-revision", default="")
    rs.set_defaults(fn=cmd_resummarize)
    s = sub.add_parser("run")
    s.set_defaults(fn=cmd_run)
    s.add_argument("--out", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--model-label", default="king")
    s.add_argument("--model-url", required=True)
    s.add_argument("--model-key-env", default="BENCH_API_KEY")
    s.add_argument("--judge-key-env", default="PRIME_API_KEY")
    s.add_argument("--concurrency", type=int, default=16)
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--n-tasks", type=int, default=0)
    s.add_argument("--hf-revision", default="")
    s.add_argument("--force", action="store_true")
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
