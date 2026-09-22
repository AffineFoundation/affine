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
import collections
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
# Frontier-arbiter outcome probe (2026-09-20): `--model` swaps the continuing
# model (arm F = glm-5.3); every other knob stays the teacher run's.
ENGY_PRICES = {   # USD per token (in, out, cached) — Engy list, 2026-09-20
    "qwen3.8-27b": (0.045e-6, 0.32e-6, 0.015e-6),
    "glm-5.3": (0.70e-6, 2.8e-6, 0.14e-6),
}
HARNESS_ID = "recoverable_resume"
# plugin/recoverable_resume/shield.py; RECOVERABLE_SHADOW_NS lets a second
# driver run concurrently on the pod with its own containers (2026-09-21).
SHADOW_NS = os.environ.get("RECOVERABLE_SHADOW_NS", "recoverable.local")
TRACE_NAME_RE = re.compile(r"^[0-9a-f]{32}$")
# Same-task proxy (states.SAME_TASK): the teacher replays the whole task under
# the king's own ACP harness (claude_code / pi / kimi_code / hermes_agent) —
# no resume plugin, full turn budget, the task image as datagen runs it.
SAME_TASK = "same_task"
DOCKER_KINDS = ("textbased", "bash", "terminus", SAME_TASK)
EVAL_TIMEOUT_S = 3 * 3600
TURN_CAP_MSG = "rollout stopped: max_turns"
RETRY_DELAY_S = 600
# Errors no re-run fixes (the image / the task's grader), harvest --max-retries skips them.
PERMANENT_ERROR_MARKS = ("docker_build_failed", "pull access denied", "manifest unknown",
                         "checkout HEAD~1 failed", "no such image")
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
            # `reasoning` (the latent thought) rides along since the harvest
            # (2026-09-22): arm Z forces a stored thought without the trace.
            return {"content": m.get("content"), "tool_calls": m.get("tool_calls"),
                    "reasoning": m.get("reasoning_content")}
    return None


def last_visible_reply(trace: dict) -> str | None:
    """Final sampled reply without a tool call (the wiki answer)."""
    for nd in reversed(trace.get("nodes") or []):
        m = nd.get("message") or {}
        if nd.get("sampled") and m.get("role") == "assistant" and not m.get("tool_calls"):
            return m.get("content")
    return None


def trace_summary(trace: dict, model: str = "qwen3.8-27b") -> dict:
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
    outcome = rollout_outcome(trace)
    turn_cap_error = False
    if outcome == "errored" and errs and all(TURN_CAP_MSG in str(e.get("message") or "") for e in errs):
        # The continuation ran its remaining turn budget out: interception
        # refused the next call ("rollout stopped: max_turns") and the
        # harness raised it as a provider error before the stop condition
        # was stamped. The agent did not finish = failed (fold rule,
        # affine.corpus.trace.is_turn_cap_artifact), not an infra error.
        outcome, turn_cap_error = "failed", True
    return {
        "trace_id": trace.get("id"),
        "outcome": outcome,
        "turn_cap_error": turn_cap_error,
        "reward_score": score,
        "rewards": {k: (v or {}).get("score") for k, v in rewards.items()},
        "stop_condition": trace.get("stop_condition"),
        "error": (errs[0].get("type") + ": " + str(errs[0].get("message"))[:500]) if errs else None,
        "n_turns": n_turns,
        "prompt_tokens": p, "completion_tokens": c, "cached_tokens": cache,
        "cost_usd": round(p * ENGY_PRICES.get(model, (PRICE_IN, PRICE_OUT, PRICE_CACHE))[0]
                          + cache * ENGY_PRICES.get(model, (PRICE_IN, PRICE_OUT, PRICE_CACHE))[2]
                          + c * ENGY_PRICES.get(model, (PRICE_IN, PRICE_OUT, PRICE_CACHE))[1], 5),
        "model": model,
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


def unit_key(state: dict) -> str:
    """What one teacher run answers for: the state itself, or — same-task
    proxy — the whole task (`proxy_key`, shared by every state of the
    rollout)."""
    return state.get("proxy_key") or state["state_id"]


def build_cmd(cfg, registry, state: dict, run_dir: Path, report_dir: Path) -> list[str]:
    source = registry.sources[state["source"]]
    kind = state["resume_kind"]
    runtime = "docker" if kind in DOCKER_KINDS else "subprocess"
    row = dict(state["task"])
    sampling = state.get("sampling") or {"temperature": 0.8}
    if kind == SAME_TASK:
        # The stock harness on the task from the start, exactly as the
        # teacher_<harness> datagen policy runs it (T = 0.8, full budget).
        cmd = eval_cmd(cfg, source, TEACHER, state["harness"], [row], run_dir,
                       {"temperature": float(sampling.get("temperature", 0.8))},
                       runtime=runtime)
        i = cmd.index("--env.agent.max-turns")
        cmd[i + 1] = str(int(state["max_turns"]))
        return cmd
    cmd = eval_cmd(cfg, source, TEACHER, HARNESS_ID, [row], run_dir,
                   sampling, runtime=runtime)
    # eval_cmd sets the pod's default turn cap; the state carries its own.
    i = cmd.index("--env.agent.max-turns")
    cmd[i + 1] = str(int(state["max_turns"]))
    cmd += [
        "--env.agent.harness.state-file", state["path"],
        "--env.agent.harness.report-dir", str(report_dir),
    ]
    return cmd


def file_stem(key: str, k: int) -> str:
    """results/<stem>.json: `<key>` for the first continuation (the pre-N
    layout), `<key>.c<k>` for the others. `key` is a state id or a same-task
    proxy key (`task:<rollout_id>`)."""
    sid = key.replace(":", "_")
    return sid if k == 0 else f"{sid}.c{k}"


def continuation_index(path: Path, sid: str) -> int:
    m = re.fullmatch(re.escape(sid) + r"(?:\.c(\d+))?", path.stem)
    return int(m.group(1)) if m and m.group(1) else 0


def run_one(cfg, registry, state: dict, out: Path, env: dict, k: int = 0) -> dict:
    key = unit_key(state)
    stem = file_stem(key, k)
    run_dir = out / "runs" / stem
    report_dir = out / "reports"
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    # result["state_id"] is the unit key: aggregate.py looks a state's
    # results up by its proxy_key when it has one.
    result: dict = {"state_id": key, "resume_kind": state["resume_kind"],
                    "harness": state["harness"], "continuation": k,
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
    if state["resume_kind"] == SAME_TASK:
        # Real task images, no shadow tag: stamp the containers with the
        # supervisor-ownership label (rollouts.runners.verifiers dockerwrap)
        # so the pod's per-batch reaper leaves them alone while this driver
        # lives and clears them if it dies.
        import rollouts.runners.verifiers as rv
        if hasattr(rv, "DOCKERWRAP_DIR"):
            env = dict(env)
            env["PATH"] = rv.DOCKERWRAP_DIR + ":" + env["PATH"]
            env["ROLLOUTS_SUPERVISOR"] = rv.supervisor_id()
            env["ROLLOUTS_BATCH"] = f"recoverable-{stem}"
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
    summary = trace_summary(trace, TEACHER.model)
    result.update(summary)
    report_path = report_dir / f"{summary['trace_id']}.json"
    if report_path.exists():
        result["report"] = json.loads(report_path.read_text())
    result["status"] = "ok" if summary["outcome"] != "errored" else "errored"
    # Keep the trace for provenance; the run dir (images, logs) is disposable.
    (out / "traces").mkdir(exist_ok=True)
    (out / "traces" / f"{stem}.json").write_text(json.dumps(trace))
    return result


def existing_results(results_dir: Path, key: str) -> dict[int, dict]:
    """k -> stored result of this unit (both file layouts)."""
    sid = key.replace(":", "_")
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


def is_permanent(r: dict) -> bool:
    err = str(r.get("error") or "")
    return any(m in err for m in PERMANENT_ERROR_MARKS)


class Adaptive:
    """Harvest scheduler (2026-09-22): Phase A = `phase_a` OK continuations
    per unit; a unit whose Phase A did not solve every continuation is KEPT
    and topped up to `phase_b` (split 0 < s < N and ceiling s = 0 alike);
    all-solved units are dropped. Errored slots are re-run in place up to
    `max_retries` times unless the error is permanent (image / grader);
    a slot past its retries is DEAD and settles the unit's phase like an OK
    result would. Rows with `continuations_needed` (F1 / TX / Z arm files
    the controller ships later) are plain targets. The states file is
    re-read every `watch_minutes` so arm files dropped into it while the
    driver runs are picked up; the driver exits when the deadline passes
    (running continuations finish) or, without --watch, when it runs dry."""

    def __init__(self, cfg, registry, args, env, kinds, only, shard):
        self.cfg, self.registry, self.args, self.env = cfg, registry, args, env
        self.kinds, self.only, self.shard = kinds, only, shard
        self.phase_a, self.phase_b = (int(x) for x in args.adaptive.split("/"))
        self.results_dir = args.out / "results"
        self.units: dict[str, dict] = {}
        self.order: dict[str, int] = {}
        self.pending: dict[tuple[str, int], int] = {}    # (key, k) -> priority
        self.running: dict[cf.Future, tuple[str, int]] = {}
        self.spent = 0.0
        self.deadline = time.time() + args.deadline_hours * 3600 if args.deadline_hours else None
        self.decided: dict[str, str] = {}
        self.not_before: dict[tuple[str, int], float] = {}
        self.n_done = 0

    # -- planning -------------------------------------------------------------
    def load_states(self) -> int:
        try:
            rows = [json.loads(l) for l in open(self.args.states, encoding="utf-8") if l.strip()]
        except (OSError, ValueError):
            log.warning("states file unreadable, keeping the last plan", exc_info=True)
            return 0
        new = 0
        for i, s in enumerate(rows):
            try:
                s["path"] = resolve_state_path(s, self.args.states)
            except FileNotFoundError:
                continue
            if s["resume_kind"] not in self.kinds or (self.only and s["state_id"] not in self.only):
                continue
            shard_i, shard_n = self.shard
            if not owns(s["state_id"], shard_i, shard_n):
                continue
            key = unit_key(s)
            if key not in self.units:
                new += 1
                self.order[key] = i
            self.units[key] = s
        return new

    def target_of(self, s: dict, have: dict[int, dict]) -> tuple[int, str]:
        ok = [r for r in have.values() if r.get("status") == "ok"]
        dead = [r for r in have.values() if r.get("status") != "ok"
                and (is_permanent(r) or int(r.get("attempts") or 1) > self.args.max_retries)]
        if any(is_permanent(r) for r in have.values()):
            # the image / grader is broken for every slot of this unit
            return len(ok), "abandoned_permanent"
        if s.get("continuations_needed") is not None:
            return int(s["continuations_needed"]), "fixed"
        if len(ok) + len(dead) < self.phase_a:
            return self.phase_a, "phase_a"
        if not ok:
            return len(ok), "abandoned"
        solved = sum(r.get("outcome") == "solved" for r in ok)
        if solved == len(ok):
            return len(ok), "dropped_all_solved"
        return self.phase_b, "phase_b"

    def plan(self, key: str) -> None:
        s = self.units[key]
        with _lock:      # workers write result files under the same lock
            have = existing_results(self.results_dir, key)
        target, phase = self.target_of(s, have)
        if self.decided.get(key) != phase:
            self.decided[key] = phase
            if phase not in ("phase_a", "fixed"):
                ok = [r for r in have.values() if r.get("status") == "ok"]
                log.info("unit %s: %s (s/N %d/%d)", key, phase,
                         sum(r.get("outcome") == "solved" for r in ok), len(ok))
        inflight = {k for (kk, k) in list(self.pending) + list(self.running.values()) if kk == key}
        n_ok = sum(1 for r in have.values() if r.get("status") == "ok")
        n_dead = sum(1 for r in have.values() if r.get("status") != "ok"
                     and (is_permanent(r) or int(r.get("attempts") or 1) > self.args.max_retries))
        # a dead slot fills its place: a unit never grows past `target` slots
        need = target - n_ok - n_dead - len(inflight)
        prio = 0 if phase in ("phase_b", "fixed") else 1
        retries = sorted(k for k, r in have.items() if r.get("status") != "ok" and k not in inflight
                         and not is_permanent(r) and int(r.get("attempts") or 1) <= self.args.max_retries)
        for k in retries:
            if need <= 0:
                break
            self.pending[(key, k)] = prio
            # an Engy outage lasts tens of minutes: a fast re-run would burn
            # the slot's retries on the same outage
            self.not_before.setdefault((key, k), time.time() + RETRY_DELAY_S)
            need -= 1
        next_k = max([*have, *inflight], default=-1) + 1
        while need > 0:
            self.pending[(key, next_k)] = prio
            next_k += 1
            need -= 1

    def plan_all(self) -> None:
        for key in list(self.units):
            self.plan(key)

    def pick(self) -> tuple[str, int] | None:
        now = time.time()
        ready = [it for it in self.pending if self.not_before.get(it, 0) <= now]
        if not ready:
            return None
        item = min(ready, key=lambda it: (self.pending[it], self.order.get(it[0], 1 << 30), it[1]))
        del self.pending[item]
        self.not_before.pop(item, None)
        return item

    # -- execution ------------------------------------------------------------
    def work(self, key: str, k: int) -> dict:
        state = self.units[key]
        with _lock:
            prev = existing_results(self.results_dir, key).get(k)
        attempts = int((prev or {}).get("attempts") or (1 if prev else 0)) + 1
        try:
            r = run_one(self.cfg, self.registry, state, self.args.out, self.env, k)
        except Exception as e:  # noqa: BLE001 - one state must not kill the batch
            log.exception("unit %s c%d crashed", key, k)
            r = {"state_id": key, "resume_kind": state["resume_kind"], "harness": state["harness"],
                 "continuation": k, "status": "errored", "error": f"driver: {type(e).__name__}: {e}"}
        r["attempts"] = attempts
        r["arm"] = state.get("arm")
        with _lock:
            self.spent += float(r.get("cost_usd") or 0.0)
            tmp = self.results_dir / f".{file_stem(key, k)}.json.tmp"
            tmp.write_text(json.dumps(r))
            os.replace(tmp, self.results_dir / f"{file_stem(key, k)}.json")
            with open(self.args.out / "results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
        return r

    def host_ok(self) -> bool:
        """Back off on a shared box: no new continuation while the 1-min load
        exceeds RECOVERABLE_MAX_LOAD1 or free disk on / is below
        RECOVERABLE_MIN_DISK_FREE_PCT (both unset = no ceiling). Running
        containers finish; only new starts wait. Requested for
        affine-backfill-4 (live env-backfill driver box), 2026-09-22."""
        max_load = os.environ.get("RECOVERABLE_MAX_LOAD1")
        min_free = os.environ.get("RECOVERABLE_MIN_DISK_FREE_PCT")
        if not max_load and not min_free:
            return True
        try:
            if max_load and os.getloadavg()[0] > float(max_load):
                log.warning("host guard: load1 %.1f > %s, holding new starts", os.getloadavg()[0], max_load)
                return False
            if min_free:
                st = os.statvfs("/")
                free_pct = 100.0 * st.f_bavail / max(st.f_blocks, 1)
                if free_pct < float(min_free):
                    log.warning("host guard: disk free %.1f%% < %s%%, holding new starts", free_pct, min_free)
                    return False
        except OSError:
            return True
        return True

    def can_start(self) -> bool:
        if self.deadline and time.time() > self.deadline:
            return False
        if self.args.budget_usd and self.spent >= self.args.budget_usd:
            return False
        return self.host_ok()

    def run(self) -> None:
        n_new = self.load_states()
        self.plan_all()
        log.info("adaptive %d/%d: %d units (%d new), %d continuation(s) queued, %d workers, deadline %s h, "
                 "budget %s, watch %s min, retries %d", self.phase_a, self.phase_b, len(self.units), n_new,
                 len(self.pending), self.args.workers, self.args.deadline_hours or "none",
                 f"${self.args.budget_usd:.0f}" if self.args.budget_usd else "none",
                 self.args.watch_minutes or "off", self.args.max_retries)
        last_watch = time.time()
        reap_own_containers()
        try:
            with cf.ThreadPoolExecutor(max_workers=self.args.workers) as ex:
                while True:
                    while len(self.running) < self.args.workers and self.can_start():
                        item = self.pick()
                        if item is None:
                            break
                        fut = ex.submit(self.work, *item)
                        self.running[fut] = item
                    if not self.running:
                        # A host-guard hold is a pause, not an exit (2026-09-22: the
                        # bf4 driver quit at 02:57 with 221 continuations queued
                        # when the guard held and nothing was running).
                        if not self.host_ok():
                            time.sleep(60)
                            continue
                        if not self.can_start() or (not self.pending and not self.args.watch_minutes):
                            break
                        time.sleep(30)      # delayed retries / the next watch tick
                        done = set()
                    else:
                        done, _ = cf.wait(list(self.running), timeout=60, return_when=cf.FIRST_COMPLETED)
                    for fut in done:
                        key, k = self.running.pop(fut)
                        r = fut.result()
                        self.n_done += 1
                        log.info("%s c%d -> %s outcome=%s stop=%s turns=%s wall=%ss cost=$%.3f (run $%.2f, "
                                 "%d done, %d queued, %d running)", key, k, r.get("status"), r.get("outcome"),
                                 r.get("stop_condition"), r.get("n_turns"), r.get("wall_s"),
                                 float(r.get("cost_usd") or 0.0), self.spent, self.n_done,
                                 len(self.pending), len(self.running))
                        self.plan(key)
                    if self.args.watch_minutes and time.time() - last_watch > self.args.watch_minutes * 60:
                        last_watch = time.time()
                        n_new = self.load_states()
                        self.plan_all()
                        if n_new:
                            log.info("watch: %d new unit(s), %d queued", n_new, len(self.pending))
                    if self.deadline and time.time() > self.deadline and not self.running:
                        break
            if self.pending:
                log.info("deadline or budget reached: %d continuation(s) not started", len(self.pending))
        finally:
            reap_own_containers(untag=self.args.untag)
        self.write_status()

    def write_status(self) -> None:
        summary = collections.Counter(self.decided.values())
        (self.args.out / "driver_status.json").write_text(json.dumps({
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "units": len(self.units),
            "decided": dict(summary), "done": self.n_done, "spent_usd": round(self.spent, 3),
            "queued": len(self.pending)}))


def main() -> None:
    global TEACHER
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
    ap.add_argument("--model", default=TEACHER.model,
                    help="Engy model that continues the episode (arm F: glm-5.3)")
    ap.add_argument("--budget-usd", type=float, default=0.0,
                    help="start no new continuation once this run's summed "
                         "cost_usd (Engy list price) passes this (0 = no budget)")
    ap.add_argument("--adaptive", default="",
                    help="A/B (harvest 2026-09-22): Phase A = A OK continuations per state, "
                         "kept states (not all solved) topped up to B; see Adaptive")
    ap.add_argument("--watch-minutes", type=float, default=0.0,
                    help="adaptive: re-read --states this often for new arm rows (0 = off)")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="adaptive: re-run an errored slot in place up to this many times")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    TEACHER = Endpoint(name=TEACHER.name, model=args.model, base_url=TEACHER.base_url,
                       key_env=TEACHER.key_env)
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
    if args.adaptive:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "results").mkdir(exist_ok=True)
        shard = tuple(int(x) for x in args.shard.split("/"))
        Adaptive(cfg, registry, args, env, kinds, only, shard).run()
        return
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
    # Work items are (unit, k): a unit's missing continuations, in state
    # order so the priority the candidates file expresses (pivots, first
    # onsets, shallow states first) is kept across units. A unit is a state,
    # or for the same-task proxy the task (`proxy_key`): every state of that
    # rollout shares the run, the first one in the file represents it.
    todo: list[tuple[dict, int]] = []
    units: dict[str, dict] = {}
    for s in states:
        units.setdefault(unit_key(s), s)
    n_have = n_complete = 0
    for key, s in units.items():
        have = existing_results(results_dir, key)
        n_have += len(have)
        ks = plan_continuations(s, have, args.continuations, args.retry_errored)
        if not ks:
            n_complete += 1
        todo.extend((s, k) for k in ks)
    if args.limit:
        todo = todo[: args.limit]
    log.info("%d states / %d units selected, %d complete, %d stored continuation(s), "
             "%d continuation(s) to run, %d workers, deadline %s h, budget %s",
             len(states), len(units), n_complete, n_have, len(todo), args.workers,
             args.deadline_hours or "none",
             f"${args.budget_usd:.0f}" if args.budget_usd else "none")
    deadline = time.time() + args.deadline_hours * 3600 if args.deadline_hours else None
    skipped: list[str] = []
    spent = {"usd": 0.0}

    def work(item: tuple[dict, int]) -> None:
        state, k = item
        key = unit_key(state)
        if (deadline and time.time() > deadline) or \
                (args.budget_usd and spent["usd"] >= args.budget_usd):
            skipped.append(key)
            return
        try:
            r = run_one(cfg, registry, state, args.out, env, k)
        except Exception as e:  # noqa: BLE001 - one state must not kill the batch
            log.exception("unit %s c%d crashed", key, k)
            r = {"state_id": key, "resume_kind": state["resume_kind"],
                 "harness": state["harness"], "continuation": k, "status": "errored",
                 "error": f"driver: {type(e).__name__}: {e}"}
        with _lock:
            spent["usd"] += float(r.get("cost_usd") or 0.0)
            (results_dir / f"{file_stem(key, k)}.json").write_text(json.dumps(r))
            with open(args.out / "results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
        log.info("%s c%d -> %s outcome=%s stop=%s turns=%s wall=%ss cost=$%.3f (run $%.2f)",
                 key, k, r.get("status"), r.get("outcome"),
                 r.get("stop_condition"), r.get("n_turns"), r.get("wall_s"),
                 float(r.get("cost_usd") or 0.0), spent["usd"])

    reap_own_containers()
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
        if skipped:
            log.info("deadline or budget reached: %d continuation(s) not started",
                     len(skipped))
    finally:
        # Tags stay: another process on the pod may have started a container
        # from one of them; `--untag` at the very end removes them.
        reap_own_containers(untag=args.untag)


if __name__ == "__main__":
    main()
