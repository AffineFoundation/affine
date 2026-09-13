#!/usr/bin/env python
"""Run coached and plain teacher continuations per state on a pod, through the
coach proxy (research/hints/coached/coach_proxy.py).

Same machinery as ops/recoverable/run_states.py (verifiers eval on the task's
own taskset / image / reward, the `recoverable_resume` plugin for the agent
harnesses, the stock harness for same-task proxies), with the model endpoint
routed through the proxy:

    http://127.0.0.1:<port>/u/<unit_stem>/c<k>/<arm>/v1   arm = coached | plain

so the proxy knows which continuation it is coaching. Arms per state: `coached`
x N always; `plain` x N unless the state carries a `plain_control` block
(side-table continuations of the same teacher, same T) and --reuse-plain is
set. Results: results/<unit_stem>.<arm>.c<k>.json + results.jsonl; traces
under traces/. Idempotent like run_states.py.

  PYTHONPATH=/root/affine:/root/rollouts python run_coached.py \
      --states RUN/states/states.jsonl --out RUN/out --proxy-port 8765 \
      --workers 24 --continuations 3 --reuse-plain
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from rollouts.config import load_config
from rollouts.registry import load_registry
from rollouts.runners.verifiers import build_local_images, eval_cmd
from rollouts.schema import Endpoint

HERE = Path(__file__).resolve().parent
RECOV = Path(os.environ.get("RECOVERABLE_SRC", str(HERE.parents[2] / "ops" / "recoverable")))
sys.path.insert(0, str(RECOV))
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
import run_states as RS  # noqa: E402  (ops/recoverable/run_states.py helpers)

log = logging.getLogger("coached.run")
_lock = threading.Lock()
ARMS = ("coached", "plain")


def endpoint_for(port: int, unit: str, k: int, arm: str, key_env: str) -> Endpoint:
    return Endpoint(name=f"coach-{arm}", model=RS.TEACHER.model,
                    base_url=f"http://127.0.0.1:{port}/u/{C.unit_stem(unit)}/c{k}/{arm}/v1",
                    key_env=key_env)


def build_cmd(cfg, registry, state: dict, run_dir: Path, report_dir: Path,
              endpoint: Endpoint) -> list[str]:
    source = registry.sources[state["source"]]
    kind = state["resume_kind"]
    runtime = "docker" if kind in RS.DOCKER_KINDS else "subprocess"
    row = dict(state["task"])
    sampling = state.get("sampling") or {"temperature": 0.8}
    if kind == RS.SAME_TASK:
        cmd = eval_cmd(cfg, source, endpoint, state["harness"], [row], run_dir,
                       {"temperature": float(sampling.get("temperature", 0.8))}, runtime=runtime)
        i = cmd.index("--env.agent.max-turns")
        cmd[i + 1] = str(int(state["max_turns"]))
        return cmd
    cmd = eval_cmd(cfg, source, endpoint, RS.HARNESS_ID, [row], run_dir, sampling, runtime=runtime)
    i = cmd.index("--env.agent.max-turns")
    cmd[i + 1] = str(int(state["max_turns"]))
    cmd += ["--env.agent.harness.state-file", state["path"],
            "--env.agent.harness.report-dir", str(report_dir)]
    return cmd


def stem(unit: str, arm: str, k: int) -> str:
    return f"{C.unit_stem(unit)}.{arm}.c{k}"


def run_one(cfg, registry, state: dict, out: Path, env: dict, arm: str, k: int,
            port: int, key_env: str) -> dict:
    unit = RS.unit_key(state)
    st = stem(unit, arm, k)
    run_dir = out / "runs" / st
    report_dir = out / "reports"
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    result: dict = {"state_id": unit, "arm": arm, "continuation": k,
                    "resume_kind": state["resume_kind"], "harness": state["harness"],
                    "source": state["source"], "king_digest": state.get("king_digest"),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    source = registry.sources[state["source"]]
    if source.local_docker_build:
        built = build_local_images([dict(state["task"])])
        if built[1]:
            result.update(status="errored", error="docker_build_failed")
            return result
    cmd = build_cmd(cfg, registry, state, run_dir, report_dir,
                    endpoint_for(port, unit, k, arm, key_env))
    if state["resume_kind"] == RS.SAME_TASK:
        import rollouts.runners.verifiers as rv
        if hasattr(rv, "DOCKERWRAP_DIR"):
            env = dict(env)
            env["PATH"] = rv.DOCKERWRAP_DIR + ":" + env["PATH"]
            env["ROLLOUTS_SUPERVISOR"] = rv.supervisor_id()
            env["ROLLOUTS_BATCH"] = f"coached-{st}"
    t0 = time.time()
    log_path = run_dir / "eval.log"
    with open(log_path, "w") as logf:
        try:
            proc = subprocess.run(cmd, cwd=str(cfg.verifiers_dir), env=env, stdout=logf,
                                  stderr=subprocess.STDOUT, timeout=RS.EVAL_TIMEOUT_S, check=False)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = -2
    result["eval_exit_code"] = code
    result["wall_s"] = round(time.time() - t0, 1)
    traces = run_dir / "traces.jsonl"
    trace = None
    if traces.exists():
        for line in open(traces, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            for t in episode.get("traces") or ([episode] if "nodes" in episode else []):
                trace = t
    if trace is None:
        tail = log_path.read_text(errors="replace")[-1500:]
        result.update(status="errored", error=f"no trace (exit {code}): {tail}")
        return result
    summary = RS.trace_summary(trace)
    result.update(summary)
    report_path = report_dir / f"{summary['trace_id']}.json"
    if report_path.exists():
        result["report"] = json.loads(report_path.read_text())
    result["status"] = "ok" if summary["outcome"] != "errored" else "errored"
    (out / "traces").mkdir(exist_ok=True)
    (out / "traces" / f"{st}.json").write_text(json.dumps(trace))
    return result


def existing(results_dir: Path, unit: str, arm: str) -> dict[int, dict]:
    found: dict[int, dict] = {}
    for p in results_dir.glob(f"{C.unit_stem(unit)}.{arm}.c*.json"):
        r = json.loads(p.read_text())
        found[int(r.get("continuation") or 0)] = r
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--proxy-port", type=int, default=8765)
    ap.add_argument("--key-env", default=RS.TEACHER.key_env,
                    help="env var the eval passes as the API key (the proxy forwards it)")
    ap.add_argument("--kinds", default="textbased,bash,terminus,same_task")
    ap.add_argument("--arms", default="coached,plain")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--continuations", type=int, default=3)
    ap.add_argument("--reuse-plain", action="store_true",
                    help="skip the plain arm for states with a side-table plain_control")
    ap.add_argument("--limit-states", type=int, default=0)
    ap.add_argument("--block", type=int, default=50,
                    help="states per block; a block finishes all its arms before the next starts")
    ap.add_argument("--only", default="", help="comma-separated state ids")
    ap.add_argument("--retry-errored", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--deadline-hours", type=float, default=0.0)
    ap.add_argument("--untag", action="store_true")
    ap.add_argument("--no-reap", action="store_true",
                    help="do not remove leftover recoverable.local/ containers at start / end. "
                         "REQUIRED when another driver shares the pod: the reaper kills every "
                         "container of the namespace, including the other driver's in-flight "
                         "continuations (2026-09-13: a retry pass launched next to the main run "
                         "killed 56 continuations with exit 137)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not os.environ.get(args.key_env):
        sys.exit(f"{args.key_env} missing")
    cfg = load_config()
    registry = load_registry()
    plugin_dir = os.environ.get("RECOVERABLE_PLUGIN", str(RECOV / "plugin"))
    env = dict(os.environ)
    env["PATH"] = f"{Path.home()}/.local/bin:" + env.get("PATH", "")
    env["PYTHONPATH"] = plugin_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    args.out = args.out.resolve()
    args.states = args.states.resolve()
    kinds = set(args.kinds.split(","))
    arms = [a for a in args.arms.split(",") if a in ARMS]
    only = set(args.only.split(",")) if args.only else None
    states = [json.loads(l) for l in open(args.states, encoding="utf-8") if l.strip()]
    for s in states:
        s["path"] = RS.resolve_state_path(s, args.states)
    si, sn = (int(x) for x in args.shard.split("/"))
    states = [s for s in states if s["resume_kind"] in kinds
              and (only is None or s["state_id"] in only) and RS.owns(s["state_id"], si, sn)]
    if args.limit_states:
        states = states[: args.limit_states]
    args.out.mkdir(parents=True, exist_ok=True)
    results_dir = args.out / "results"
    results_dir.mkdir(exist_ok=True)
    units: dict[str, dict] = {}
    for s in states:
        units.setdefault(RS.unit_key(s), s)
    todo: list[tuple[dict, str, int]] = []
    n_have = 0
    for unit, s in units.items():
        for arm in arms:
            if arm == "plain" and args.reuse_plain and s.get("plain_control"):
                continue
            have = existing(results_dir, unit, arm)
            n_have += len(have)
            ok = [k for k, r in have.items() if r.get("status") == "ok"]
            errored = [k for k, r in have.items() if r.get("status") != "ok"]
            need = args.continuations - len(ok) - (0 if args.retry_errored else len(errored))
            ks = list(errored) if args.retry_errored else []
            nk = max(have, default=-1) + 1
            while need > 0:
                ks.append(nk)
                nk += 1
                need -= 1
            todo.extend((s, arm, k) for k in ks)
    # Blocks of --block states complete (all arms, all continuations) before
    # the next block starts, so the first block can be reported early; inside
    # a block, continuation k of every unit runs before k+1 and the two arms
    # interleave (a paired read is possible at any checkpoint).
    order = {u: i for i, u in enumerate(units)}
    todo.sort(key=lambda t: (order[RS.unit_key(t[0])] // args.block, t[2],
                             order[RS.unit_key(t[0])], t[1]))
    log.info("%d states / %d units, %d stored, %d continuation(s) to run (%s), %d workers",
             len(states), len(units), n_have, len(todo), ",".join(arms), args.workers)
    deadline = time.time() + args.deadline_hours * 3600 if args.deadline_hours else None
    skipped: list[str] = []

    def work(item: tuple[dict, str, int]) -> None:
        state, arm, k = item
        unit = RS.unit_key(state)
        if deadline and time.time() > deadline:
            skipped.append(unit)
            return
        try:
            r = run_one(cfg, registry, state, args.out, env, arm, k, args.proxy_port, args.key_env)
        except Exception as e:  # noqa: BLE001 - one unit must not kill the batch
            log.exception("unit %s %s c%d crashed", unit, arm, k)
            r = {"state_id": unit, "arm": arm, "continuation": k, "status": "errored",
                 "resume_kind": state["resume_kind"], "harness": state["harness"],
                 "source": state["source"], "error": f"driver: {type(e).__name__}: {e}"}
        with _lock:
            (results_dir / f"{stem(unit, arm, k)}.json").write_text(json.dumps(r))
            with open(args.out / "results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
        log.info("%s %s c%d -> %s outcome=%s stop=%s turns=%s wall=%ss $%.3f", unit, arm, k,
                 r.get("status"), r.get("outcome"), r.get("stop_condition"), r.get("n_turns"),
                 r.get("wall_s"), float(r.get("cost_usd") or 0))

    if not args.no_reap:
        RS.reap_own_containers()
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
        if skipped:
            log.info("deadline reached: %d continuation(s) not started", len(skipped))
    finally:
        if not args.no_reap:
            RS.reap_own_containers(untag=args.untag)


if __name__ == "__main__":
    main()
