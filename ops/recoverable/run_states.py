"""Run teacher continuations per candidate state on a datagen pod.

For every row of states.jsonl (ops/recoverable/states.py) this launches N
verifiers evals (`--continuations`, default 3; a state row's
`continuations_needed` overrides it) — the task's own taskset, image and
reward, the `recoverable_resume` harness (plugin/) on the state file, the
teacher as the model, `max_turns` = the state's remaining budget — parses
each resulting trace and writes one result per continuation:
results/<state_id>.json (the first) and results/<state_id>.c<k>.json
(k >= 1), plus results.jsonl. Idempotent: existing OK results count toward
the target, so a re-run only adds what is missing. aggregate.py decides
`teacher_solved` / `admit` by majority over a state's continuations (one
continuation at T = 0.8 is a noisy label: the unhinted re-run of "failed"
states recovered 35 % of them, hinted-teacher explorations §E4b).

Runs next to the datagen supervisor without touching it: its own working
directory, its own env file, low container concurrency. The rollouts package
is imported read-only for the source -> eval flag mapping (`eval_cmd`).

  PYTHONPATH=/root/affine:/root/rollouts python run_states.py \
      --states /root/recoverable/states/states.jsonl \
      --out /root/recoverable/out --kinds textbased,bash,terminus --workers 4

Env: ENGY_2 (teacher key), ROLLOUTS_* (sourced from the pod's env files,
read-only), RECOVERABLE_PLUGIN (dir containing the plugin package; default
<this dir>/plugin).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import logging
import os
import re
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
sys.path.insert(0, str(HERE))
from traceutil import primary_score, real_errors, rollout_outcome  # noqa: E402
log = logging.getLogger("recoverable.run")

TEACHER = Endpoint(name="engy2", model="qwen3.8-27b",
                   base_url="https://api.engy.ai/v1", key_env="ENGY_2")
HARNESS_ID = "recoverable_resume"
SHADOW_NS = "recoverable.local"   # plugin/recoverable_resume/shield.py
TRACE_NAME_RE = re.compile(r"^[0-9a-f]{32}$")
DOCKER_KINDS = ("textbased", "bash", "terminus")
EVAL_TIMEOUT_S = 3 * 3600
# Engy list price for qwen3.8-27b (USD per token), 2026-09-11.
PRICE_IN = 0.045e-6
PRICE_OUT = 0.32e-6
PRICE_CACHE = 0.015e-6

_lock = threading.Lock()


def reap_own_containers(untag: bool = False) -> None:
    """Remove containers (and, optionally, the shadow tags) this driver's
    evals created — only the `recoverable.local/` namespace (plugin
    shield.py), never the supervisor's task-image namespaces. verifiers
    removes each container itself at rollout end; this catches what a killed
    driver left behind. Safe because one driver runs per pod at a time."""
    try:
        out = subprocess.run(["docker", "ps", "-a", "--format", "{{.ID}} {{.Names}} {{.Image}}"],
                             capture_output=True, text=True, timeout=60).stdout
        # verifiers names a rollout's container by its 32-hex trace id; a
        # container someone else started from one of our tags keeps a
        # different name and is left alone.
        mine = [l.split()[0] for l in out.splitlines()
                if len(l.split()) == 3 and TRACE_NAME_RE.match(l.split()[1])
                and l.split()[2].startswith(SHADOW_NS + "/")]
        if mine:
            subprocess.run(["docker", "rm", "-f", *mine], capture_output=True, timeout=300)
            log.info("removed %d leftover container(s) of ours", len(mine))
        if untag:
            tags = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                                  capture_output=True, text=True, timeout=60).stdout
            ours = [t for t in tags.splitlines() if t.startswith(SHADOW_NS + "/")]
            if ours:
                subprocess.run(["docker", "rmi", *ours], capture_output=True, timeout=300)
                log.info("untagged %d shadow image tag(s)", len(ours))
    except Exception:  # noqa: BLE001 - cleanup is best-effort
        log.warning("own-container cleanup failed", exc_info=True)


def owns(state_id: str, i: int, n: int) -> bool:
    if n <= 1:
        return True
    h = hashlib.blake2b(state_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % n == i


def first_reply(trace: dict) -> dict | None:
    for nd in trace.get("nodes") or []:
        m = nd.get("message") or {}
        if nd.get("sampled") and m.get("role") == "assistant":
            return {"content": m.get("content"), "tool_calls": m.get("tool_calls")}
    return None


def last_visible_reply(trace: dict) -> str | None:
    """Final sampled reply without a tool call (the wiki answer)."""
    for nd in reversed(trace.get("nodes") or []):
        m = nd.get("message") or {}
        if nd.get("sampled") and m.get("role") == "assistant" and not m.get("tool_calls"):
            return m.get("content")
    return None


def trace_summary(trace: dict) -> dict:
    n_turns = sum(1 for nd in trace.get("nodes") or []
                  if nd.get("sampled") and (nd.get("message") or {}).get("role") == "assistant")
    p = c = cache = 0
    for call in trace.get("calls") or []:
        u = call.get("usage") or {}
        p += int(u.get("prompt_tokens") or 0)
        c += int(u.get("completion_tokens") or 0)
        cache += int(u.get("cached_input_tokens") or 0)
    rewards = trace.get("rewards") or {}
    score = primary_score(trace)
    errs = real_errors(trace)
    return {
        "trace_id": trace.get("id"),
        "outcome": rollout_outcome(trace),
        "reward_score": score,
        "rewards": {k: (v or {}).get("score") for k, v in rewards.items()},
        "stop_condition": trace.get("stop_condition"),
        "error": (errs[0].get("type") + ": " + str(errs[0].get("message"))[:500]) if errs else None,
        "n_turns": n_turns,
        "prompt_tokens": p, "completion_tokens": c, "cached_tokens": cache,
        "cost_usd": round(p * PRICE_IN + cache * PRICE_CACHE + c * PRICE_OUT, 5),
        "first_reply": first_reply(trace),
        "last_visible_reply": last_visible_reply(trace),
    }


def resolve_state_path(state: dict, states_file: Path) -> str:
    """The state JSON: the recorded path when it exists, else the file of the
    same name in `states/` next to states.jsonl (states built on the box,
    shipped to a pod as a directory)."""
    p = Path(state["path"])
    if p.is_file():
        return str(p)
    local = states_file.parent / "states" / p.name
    if local.is_file():
        return str(local)
    raise FileNotFoundError(f"state file for {state['state_id']} not found: {p} / {local}")


def build_cmd(cfg, registry, state: dict, run_dir: Path, report_dir: Path) -> list[str]:
    source = registry.sources[state["source"]]
    kind = state["resume_kind"]
    runtime = "docker" if kind in DOCKER_KINDS else "subprocess"
    row = dict(state["task"])
    cmd = eval_cmd(cfg, source, TEACHER, HARNESS_ID, [row], run_dir,
                   state.get("sampling") or {"temperature": 0.8}, runtime=runtime)
    # eval_cmd sets the pod's default turn cap; the state carries its own.
    i = cmd.index("--env.agent.max-turns")
    cmd[i + 1] = str(int(state["max_turns"]))
    cmd += [
        "--env.agent.harness.state-file", state["path"],
        "--env.agent.harness.report-dir", str(report_dir),
    ]
    return cmd


def file_stem(state_id: str, k: int) -> str:
    """results/<stem>.json: `<sid>` for the first continuation (the pre-N
    layout), `<sid>.c<k>` for the others."""
    sid = state_id.replace(":", "_")
    return sid if k == 0 else f"{sid}.c{k}"


def continuation_index(path: Path, sid: str) -> int:
    m = re.fullmatch(re.escape(sid) + r"(?:\.c(\d+))?", path.stem)
    return int(m.group(1)) if m and m.group(1) else 0


def run_one(cfg, registry, state: dict, out: Path, env: dict, k: int = 0) -> dict:
    stem = file_stem(state["state_id"], k)
    run_dir = out / "runs" / stem
    report_dir = out / "reports"
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    result: dict = {"state_id": state["state_id"], "resume_kind": state["resume_kind"],
                    "continuation": k,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    source = registry.sources[state["source"]]
    if source.local_docker_build:
        # rollouts >= 2026-09-13 also returns the tasks deferred by its
        # per-batch build cap (none here: one task, no cap).
        built = build_local_images([dict(state["task"])])
        failed = built[1]
        if failed:
            result.update(status="errored", error="docker_build_failed")
            return result
    cmd = build_cmd(cfg, registry, state, run_dir, report_dir)
    t0 = time.time()
    log_path = run_dir / "eval.log"
    with open(log_path, "w") as logf:
        try:
            proc = subprocess.run(cmd, cwd=str(cfg.verifiers_dir), env=env,
                                  stdout=logf, stderr=subprocess.STDOUT,
                                  timeout=EVAL_TIMEOUT_S, check=False)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = -2
    result["eval_exit_code"] = code
    result["wall_s"] = round(time.time() - t0, 1)
    traces = run_dir / "traces.jsonl"
    trace = None
    if traces.exists():
        # One episode per line, each carrying `traces: [trace, ...]`.
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
    summary = trace_summary(trace)
    result.update(summary)
    report_path = report_dir / f"{summary['trace_id']}.json"
    if report_path.exists():
        result["report"] = json.loads(report_path.read_text())
    result["status"] = "ok" if summary["outcome"] != "errored" else "errored"
    # Keep the trace for provenance; the run dir (images, logs) is disposable.
    (out / "traces").mkdir(exist_ok=True)
    (out / "traces" / f"{stem}.json").write_text(json.dumps(trace))
    return result


def existing_results(results_dir: Path, state_id: str) -> dict[int, dict]:
    """k -> stored result of this state (both file layouts)."""
    sid = state_id.replace(":", "_")
    found: dict[int, dict] = {}
    for p in [results_dir / f"{sid}.json", *results_dir.glob(f"{sid}.c*.json")]:
        if p.is_file():
            found[continuation_index(p, sid)] = json.loads(p.read_text())
    return found


def plan_continuations(state: dict, have: dict[int, dict], target: int,
                       retry_errored: bool) -> list[int]:
    """Which continuation indices to run for this state.

    Standalone (no `continuations_needed` on the row): fresh indices until
    `target` continuations exist in this out dir; an errored slot counts as
    filled unless --retry-errored re-runs it in place (a 3,600 s timeout
    would otherwise cost an hour per re-run).

    From candidates.py (`continuations_needed` = what the side-table still
    lacks, `table_trace_ids` = the continuations it already counts): run
    that many, minus OK results stored here that the table does not know
    yet (an earlier run that was never merged)."""
    known = set(state.get("table_trace_ids") or [])
    ok_uncounted = [k for k, r in have.items()
                    if r.get("status") == "ok" and r.get("trace_id") not in known]
    errored = sorted(k for k, r in have.items() if r.get("status") != "ok")
    todo = list(errored) if retry_errored else []
    if state.get("continuations_needed") is not None:
        need = int(state["continuations_needed"]) - len(ok_uncounted)
    else:
        need = target - len(ok_uncounted) - (0 if retry_errored else len(errored))
    need -= len(todo)
    next_k = max(have, default=-1) + 1
    while need > 0:
        todo.append(next_k)
        next_k += 1
        need -= 1
    return todo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--kinds", default="textbased,bash,terminus,null")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated state ids")
    ap.add_argument("--retry-errored", action="store_true")
    ap.add_argument("--untag", action="store_true",
                    help="also remove the recoverable.local/ image tags on exit")
    ap.add_argument("--shard", default="0/1",
                    help="i/n: run only states with blake2b(state_id) %% n == i "
                         "(split the work across pods)")
    ap.add_argument("--deadline-hours", type=float, default=0.0,
                    help="start no new continuation after this many hours (0 = "
                         "no deadline); continuations already running finish")
    ap.add_argument("--continuations", type=int, default=3,
                    help="OK continuations wanted per state (a state row's "
                         "`continuations_needed` overrides); existing results count")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not os.environ.get(TEACHER.key_env):
        sys.exit(f"{TEACHER.key_env} missing")
    cfg = load_config()
    registry = load_registry()
    plugin_dir = os.environ.get("RECOVERABLE_PLUGIN", str(HERE / "plugin"))
    env = dict(os.environ)
    env["PATH"] = f"{Path.home()}/.local/bin:" + env.get("PATH", "")
    env["PYTHONPATH"] = plugin_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # The eval runs with cwd = the verifiers checkout; every path it gets
    # must be absolute or it lands in that tree.
    args.out = args.out.resolve()
    args.states = args.states.resolve()
    kinds = set(args.kinds.split(","))
    only = set(args.only.split(",")) if args.only else None
    states = [json.loads(l) for l in open(args.states, encoding="utf-8")]
    for s in states:
        s["path"] = resolve_state_path(s, args.states)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    states = [s for s in states if s["resume_kind"] in kinds
              and (only is None or s["state_id"] in only)
              and owns(s["state_id"], shard_i, shard_n)]
    args.out.mkdir(parents=True, exist_ok=True)
    results_dir = args.out / "results"
    results_dir.mkdir(exist_ok=True)
    # Work items are (state, k): a state's missing continuations, in state
    # order so the priority the candidates file expresses (pivots, first
    # onsets, shallow states first) is kept across states.
    todo: list[tuple[dict, int]] = []
    n_have = n_complete = 0
    for s in states:
        have = existing_results(results_dir, s["state_id"])
        n_have += len(have)
        ks = plan_continuations(s, have, args.continuations, args.retry_errored)
        if not ks:
            n_complete += 1
        todo.extend((s, k) for k in ks)
    if args.limit:
        todo = todo[: args.limit]
    log.info("%d states selected, %d complete, %d stored continuation(s), %d "
             "continuation(s) to run, %d workers, deadline %s h",
             len(states), n_complete, n_have, len(todo), args.workers,
             args.deadline_hours or "none")
    deadline = time.time() + args.deadline_hours * 3600 if args.deadline_hours else None
    skipped = []

    def work(item: tuple[dict, int]) -> None:
        state, k = item
        if deadline and time.time() > deadline:
            skipped.append(state["state_id"])
            return
        try:
            r = run_one(cfg, registry, state, args.out, env, k)
        except Exception as e:  # noqa: BLE001 - one state must not kill the batch
            log.exception("state %s c%d crashed", state["state_id"], k)
            r = {"state_id": state["state_id"], "resume_kind": state["resume_kind"],
                 "continuation": k, "status": "errored",
                 "error": f"driver: {type(e).__name__}: {e}"}
        with _lock:
            (results_dir / f"{file_stem(state['state_id'], k)}.json").write_text(json.dumps(r))
            with open(args.out / "results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
        log.info("%s c%d -> %s outcome=%s stop=%s turns=%s wall=%ss",
                 state["state_id"], k, r.get("status"), r.get("outcome"),
                 r.get("stop_condition"), r.get("n_turns"), r.get("wall_s"))

    reap_own_containers()
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
        if skipped:
            log.info("deadline reached: %d continuation(s) not started", len(skipped))
    finally:
        # Tags stay: another process on the pod may have started a container
        # from one of them; `--untag` at the very end removes them.
        reap_own_containers(untag=args.untag)


if __name__ == "__main__":
    main()
