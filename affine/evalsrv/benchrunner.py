"""Advisory tau2 benchmarks against a served miner model.

Runs the tau2-bench CLI with the agent pointed at the local vLLM
OpenAI-compatible endpoint (litellm `openai/<repo>` + OPENAI_BASE_URL) and
the user simulator at either the warm teacher or any configured litellm
model string. Scores are ADVISORY ONLY — published on the dashboard for the
benchmark-isomorphism claim, never consumed by S* or the duel rule.

Suites map to tau2 domains: tau2_airline → airline, etc.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import statistics as st
import subprocess
import threading
import time
from pathlib import Path

from . import swerunner

log = logging.getLogger("evalsrv.bench")

SUITE_DOMAINS = {
    "tau2_airline": "airline",
    "tau2_retail": "retail",
    "tau2_telecom": "telecom",
}

BENCH_DATA_DIR = Path(os.environ.get("AFFINE_BENCH_DIR", "/root/bench"))


def run_suite(suite: str, model_repo: str, model_port: int,
              user_llm: str, teacher_repo: str, teacher_port: int,
              num_trials: int, max_concurrency: int,
              timeout_s: int = 14400,
              abort_event: threading.Event | None = None) -> dict:
    """Dispatch a named advisory suite (tau2_* or swe_rebench_lite)."""
    if suite == swerunner.SUITE_NAME or suite == "swe_rebench_lite":
        return swerunner.run_swe_lite(
            model_repo, model_port,
            workers=max_concurrency, timeout_s=timeout_s,
            abort_event=abort_event)
    return _run_tau2(
        suite, model_repo, model_port, user_llm, teacher_repo, teacher_port,
        num_trials, max_concurrency, timeout_s=timeout_s,
        abort_event=abort_event)


def _find_rewards(obj) -> list[float]:
    """Collect simulation rewards from a tau2 results JSON, defensively:
    any numeric `reward` field found in the tree."""
    out: list[float] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "reward" and isinstance(v, (int, float)):
                out.append(float(v))
            else:
                out.extend(_find_rewards(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_find_rewards(v))
    return out


def _run_tau2(suite: str, model_repo: str, model_port: int,
              user_llm: str, teacher_repo: str, teacher_port: int,
              num_trials: int, max_concurrency: int,
              timeout_s: int = 14400,
              abort_event: threading.Event | None = None) -> dict:
    """Run one tau2 domain against the served model. Returns a result dict:
    {ok, suite, score, n_sims, wall_time_s, raw_path?, error?}.

    On a shared duel pod, abort_event lets a duel reclaim GPUs; on the
    dedicated bench machine this is unused."""
    domain = SUITE_DOMAINS.get(suite)
    if domain is None:
        return {"ok": False, "suite": suite, "error": f"unknown suite {suite!r}"}

    BENCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Keep package domain data; write sims under BENCH_DATA_DIR/simulations.
    sim_dir = BENCH_DATA_DIR / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{suite}-{int(time.time())}"
    env = dict(
        os.environ,
        OPENAI_API_KEY="local",
        # Agent always hits the miner. Do NOT set a global OPENAI_BASE_URL when
        # the user simulator needs a different host — litellm would send both
        # there. Prefer per-call api_base via the openai/ host:port form below.
        TAU2_DATA_DIR=str(BENCH_DATA_DIR),
    )
    agent_base = f"http://localhost:{model_port}/v1"
    agent_model = f"openai/{model_repo}"
    env["OPENAI_BASE_URL"] = agent_base
    if user_llm == "teacher":
        user_model = f"openai/{teacher_repo}"
        teacher_base = f"http://localhost:{teacher_port}/v1"
        # Optional CLI flags (sierra tau2 ≥ recent): separate bases for agent/user.
        # Detect once; if absent, both sides share OPENAI_BASE_URL (miner) —
        # still better than silently pretending TAU2_USER_BASE_URL works.
        help_txt = ""
        try:
            help_txt = subprocess.check_output(
                ["tau2", "run", "--help"], text=True, stderr=subprocess.STDOUT,
                timeout=30)
        except Exception:
            pass
        extra: list[str] = []
        if "--agent-base-url" in help_txt:
            extra += ["--agent-base-url", agent_base]
        if "--user-base-url" in help_txt:
            extra += ["--user-base-url", teacher_base]
        elif "--user-llm-args" in help_txt:
            # last-resort hook some builds expose
            extra += ["--user-llm-args", f"api_base={teacher_base}"]
        if "--user-base-url" not in help_txt and "--user-llm-args" not in help_txt:
            log.warning(
                "tau2 CLI has no user-base-url flag; user simulator will hit "
                "the miner endpoint (OPENAI_BASE_URL). Install a newer tau2 "
                "or point user_llm at an external model string.")
    else:
        user_model = user_llm
        extra = []

    cmd = [
        "tau2", "run",
        "--domain", domain,
        "--agent-llm", agent_model,
        "--user-llm", user_model,
        "--num-trials", str(num_trials),
        "--max-concurrency", str(max_concurrency),
        "--save-to", run_name,
        *extra,
    ]
    t0 = time.time()
    log.info("bench %s: %s", suite, " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)

    def _kill():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            log.warning("could not kill bench subprocess", exc_info=True)

    while True:
        if proc.poll() is not None:
            break
        if abort_event is not None and abort_event.is_set():
            _kill()
            return {"ok": False, "suite": suite, "aborted": True,
                    "error": "aborted to free GPUs for a duel"}
        if time.time() - t0 > timeout_s:
            _kill()
            return {"ok": False, "suite": suite, "error": f"timeout after {timeout_s}s"}
        time.sleep(2)
    out = proc.stdout.read() if proc.stdout else ""
    wall = time.time() - t0
    if proc.returncode != 0:
        return {"ok": False, "suite": suite, "wall_time_s": round(wall, 1),
                "error": out[-2000:]}

    # Upstream writes data/simulations/<save_to>/results.json (or under
    # TAU2_DATA_DIR/simulations/<save_to>/). Prefer the exact path.
    exact = sim_dir / run_name / "results.json"
    candidates: list[Path] = [exact] if exact.exists() else []
    if not candidates:
        candidates = sorted(BENCH_DATA_DIR.rglob(f"*{run_name}*/results.json"),
                            key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(BENCH_DATA_DIR.rglob(f"*{run_name}*.json"),
                            key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        # Fall back to the newest simulations JSON.
        candidates = sorted(BENCH_DATA_DIR.rglob("*.json"),
                            key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        # Last resort: parse an average-reward line out of stdout.
        m = re.search(r"(?:avg|average)[ _]reward[:= ]+([0-9.]+)",
                      out, re.IGNORECASE)
        if m:
            return {"ok": True, "suite": suite, "score": float(m.group(1)),
                    "n_sims": None, "wall_time_s": round(wall, 1)}
        return {"ok": False, "suite": suite, "wall_time_s": round(wall, 1),
                "error": "no results file found and no score in stdout"}

    raw_path = candidates[0]
    try:
        rewards = _find_rewards(json.loads(raw_path.read_text()))
    except json.JSONDecodeError as e:
        return {"ok": False, "suite": suite, "error": f"bad results json: {e}",
                "raw_path": str(raw_path)}
    if not rewards:
        return {"ok": False, "suite": suite, "error": "no rewards in results",
                "raw_path": str(raw_path)}
    return {
        "ok": True, "suite": suite,
        "score": round(st.mean(rewards), 4),
        "n_sims": len(rewards),
        "wall_time_s": round(wall, 1),
        "raw_path": str(raw_path),
    }
