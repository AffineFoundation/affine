"""Streaming SWE bench pipeline: rollout -> immediate eval, no phase barrier.

Differences from the validator's swerunner (deliberate, all for speed):
  * Every rollout runs concurrently (workers = n, capped by config).
  * A watcher polls the agent's incremental preds.json; each finished
    instance is queued for evaluation the moment it lands, so eval work
    overlaps the rollout tail instead of waiting for the whole agent phase.
  * Evals run as micro-batches in isolated working dirs (parallel-safe
    run_ids, no report races).
  * Per-phase and per-instance timings are recorded for optimization work.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import Config
from .pool import image_for, panel_key

log = logging.getLogger("fastbench.runner")


# -- dataset subset -----------------------------------------------------------

def _prepare_subset(pool: dict, instance_ids: list[str], out_dir: Path) -> Path:
    """Materialize the panel as a local HF dataset dir (cached by panel key)."""
    from datasets import Dataset, DatasetDict, load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".ready"
    want = sorted(instance_ids)
    if marker.exists():
        try:
            if json.loads(marker.read_text()).get("instance_ids") == want:
                return out_dir
        except Exception:
            pass
    ds = load_dataset(pool["dataset"], split=pool.get("split", "test"))
    want_set = set(want)
    rows = {r["instance_id"]: r for r in ds if r["instance_id"] in want_set}
    missing = [i for i in want if i not in rows]
    if missing:
        raise RuntimeError(f"instances missing from dataset: {missing[:5]}")
    ordered = [rows[i] for i in want]
    DatasetDict({"test": Dataset.from_list(ordered)}).save_to_disk(str(out_dir))
    marker.write_text(json.dumps({"instance_ids": want}))
    return out_dir


# -- agent config ---------------------------------------------------------------

def _write_agent_config(cfg: Config, model_repo: str, model_port: int,
                        path: Path) -> None:
    """Derive from mini-swe-agent's packaged swebench_backticks.yaml (same
    contract the evalsrv bench uses: text actions, plain vLLM, {{task}})."""
    import importlib.util

    spec = importlib.util.find_spec("minisweagent")
    if spec is None or not spec.origin:
        raise RuntimeError("mini-swe-agent is not installed")
    base = (Path(spec.origin).parent / "config" / "benchmarks"
            / "swebench_backticks.yaml")
    conf = yaml.safe_load(base.read_text())
    conf["agent"]["step_limit"] = cfg.step_limit
    conf["agent"]["cost_limit"] = 0
    model = conf.setdefault("model", {})
    model["model_name"] = f"openai/{model_repo}"
    model["model_class"] = "litellm_textbased"
    model["cost_tracking"] = "ignore_errors"
    kwargs = model.setdefault("model_kwargs", {})
    kwargs.update({
        "api_base": f"http://127.0.0.1:{model_port}/v1",
        "api_key": "local",
        "temperature": 0.0,
        "num_retries": 2,
        "max_tokens": cfg.max_tokens,
    })
    # Also accept the conventional ```bash fence as an action. The default
    # regex takes only the custom ```mswea_bash_command tag, and on n=150,
    # 513 of 518 rejected turns were well-formed "THOUGHT + ```bash" replies
    # thrown away over the fence label alone - which killed 70 tasks with
    # RepeatedFormatError and measured model ability as tag compliance.
    model["action_regex"] = r"```(?:bash|sh|mswea_bash_command)\s*\n(.*?)\n```"
    if cfg.disable_thinking:
        # Chat templates that support thinking honour this flag; the rest
        # ignore an unused template variable.
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}}
    path.write_text(yaml.safe_dump(conf, sort_keys=False))


# -- docker helpers ----------------------------------------------------------------

def _cached_images() -> set[str]:
    try:
        p = subprocess.run(["docker", "images", "--format",
                            "{{.Repository}}:{{.Tag}}"],
                           capture_output=True, text=True, timeout=60)
        return set((p.stdout or "").split())
    except Exception:
        return set()


def prepull_images(instance_ids: list[str], workers: int = 8,
                   timeout_s: int = 3600,
                   abort: threading.Event | None = None) -> float:
    """Pull any panel image not cached yet. No-op on a baked box.

    Pulls run in a killable child and are polled, because a multi-GB pull can
    take minutes and an abort that is only noticed afterwards is not an abort.
    """
    images = [image_for(i) for i in instance_ids]
    missing = [img for img in images if img not in _cached_images()]
    if not missing:
        return 0.0
    log.info("pre-pulling %d/%d images", len(missing), len(images))
    t0 = time.time()
    proc = subprocess.Popen(
        ["xargs", "-P", str(workers), "-n", "1", "docker", "pull"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, text=True, start_new_session=True)
    try:
        assert proc.stdin is not None
        proc.stdin.write("\n".join(missing))
        proc.stdin.close()
        while proc.poll() is None:
            if abort is not None and abort.is_set():
                _killpg(proc)
                break
            if time.time() - t0 > timeout_s:
                log.warning("image pre-pull timed out after %ds", timeout_s)
                _killpg(proc)
                break
            time.sleep(2)
    except OSError:
        log.warning("image pre-pull incomplete", exc_info=True)
    return round(time.time() - t0, 1)


def _exit_status_counts(preds_dir: Path) -> dict[str, int]:
    """Agent exit reasons, e.g. {"Submitted": 4, "RepeatedFormatError": 16}.

    Without this a run that scores near zero because the agent could not parse
    the model's output looks identical to a model that simply failed the
    tasks."""
    files = sorted(preds_dir.glob("exit_statuses_*.yaml"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {}
    try:
        data = yaml.safe_load(files[0].read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    by_status = data.get("instances_by_exit_status") or {}
    return {str(k): len(v or []) for k, v in by_status.items()
            if isinstance(v, list)}


def _reap_stale_containers() -> None:
    """Remove leftover mini-swe-agent containers from aborted runs. Jobs are
    serialized by the server, so nothing live can be named minisweagent-*."""
    try:
        subprocess.run(
            "docker ps -aq --filter name=minisweagent- | xargs -r docker rm -f",
            shell=True, capture_output=True, timeout=180)
    except Exception:
        log.warning("stale container reap failed", exc_info=True)


# -- subprocess plumbing --------------------------------------------------------------

def _spawn(cmd: list[str], env: dict, log_path: Path) -> subprocess.Popen:
    logf = open(log_path, "a")
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=logf,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def _killpg(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# -- eval batches --------------------------------------------------------------------

@dataclass
class _EvalBatch:
    run_id: str          # harness --run_id; also names its report file
    iids: list[str]
    workdir: Path
    proc: subprocess.Popen
    started: float


class _EvalPipeline:
    """Micro-batching eval dispatcher: whenever finished rollouts are pending
    and a slot is free, launch one run_evaluation subprocess for all of them."""

    def __init__(self, cfg: Config, pool: dict, run_root: Path,
                 env: dict, abort: threading.Event):
        self.cfg = cfg
        self.pool = pool
        self.run_root = run_root
        self.env = env
        self.abort = abort
        self.pending: dict[str, dict] = {}   # iid -> prediction record
        self.running: list[_EvalBatch] = []
        self.resolved: set[str] = set()
        self.evaluated: set[str] = set()
        self.errors: list[str] = []
        self.first_eval_start: float | None = None
        self.last_eval_end: float | None = None
        self._n = 0
        # Predictions kept for retries, and per-instance attempt counts. The
        # harness lists containers then inspects each one, so under heavy
        # container churn it can 404 and abort a whole batch; without a retry
        # those instances would silently score as unresolved.
        self._preds: dict[str, dict] = {}
        self._attempts: dict[str, int] = {}
        self._pending_since: float = 0.0

    MAX_ATTEMPTS = 3

    def submit(self, iid: str, pred: dict) -> None:
        if not self.pending:
            self._pending_since = time.time()
        self.pending[iid] = pred
        self._preds[iid] = pred

    def _launch_batch(self) -> None:
        iids = sorted(self.pending)
        preds = [self.pending.pop(i) for i in iids]
        for iid in iids:
            self._attempts[iid] = self._attempts.get(iid, 0) + 1
        self._n += 1
        batch_id = f"b{self._n:03d}"
        workdir = self.run_root / "eval" / batch_id
        workdir.mkdir(parents=True, exist_ok=True)
        preds_path = workdir / "predictions.jsonl"
        with open(preds_path, "w") as f:
            for rec in preds:
                f.write(json.dumps(rec) + "\n")
        run_id = f"fb-{batch_id}-{uuid.uuid4().hex[:8]}"
        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", self.pool["dataset"],
            "--split", self.pool.get("split", "test"),
            "--predictions_path", str(preds_path),
            "--instance_ids", *iids,
            "--max_workers", str(min(len(iids), self.cfg.eval_workers_cap)),
            "--run_id", run_id,
            "--namespace", self.pool.get("namespace", "swerebench"),
            "--cache_level", "instance",
            "--timeout", str(self.cfg.eval_instance_timeout_s),
        ]
        logf = open(workdir / "eval.log", "a")
        proc = subprocess.Popen(cmd, env=self.env, cwd=str(workdir),
                                stdout=logf, stderr=logf,
                                stdin=subprocess.DEVNULL,
                                start_new_session=True)
        if self.first_eval_start is None:
            self.first_eval_start = time.time()
        self.running.append(_EvalBatch(run_id, iids, workdir, proc,
                                       time.time()))
        log.info("eval %s: %d instances", batch_id, len(iids))

    def _requeue(self, iids: list[str], reason: str) -> bool:
        """Re-queue instances whose batch died without a report. Returns True
        if anything was re-queued."""
        again = [i for i in iids if self._attempts.get(i, 0) < self.MAX_ATTEMPTS]
        for iid in again:
            self.pending[iid] = self._preds[iid]
        if again:
            log.warning("re-queueing %d instances after %s", len(again), reason)
        dead = [i for i in iids if i not in again]
        if dead:
            self.errors.append(
                f"{len(dead)} instances unevaluated after "
                f"{self.MAX_ATTEMPTS} attempts ({reason}): {dead[:3]}")
            self.evaluated.update(dead)
        return bool(again)

    def _harvest(self, batch: _EvalBatch) -> None:
        self.last_eval_end = time.time()
        report = None
        for path in sorted(batch.workdir.glob(f"*.{batch.run_id}.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            if isinstance(data, dict) and "resolved_ids" in data:
                report = data
                break
        if report is None:
            self._requeue(batch.iids,
                          f"batch {batch.run_id} exit "
                          f"{batch.proc.returncode} with no report")
            return
        self.evaluated.update(batch.iids)
        self.resolved.update(str(i) for i in report.get("resolved_ids") or [])

    def step(self, drain: bool) -> None:
        """One scheduler tick: harvest finished batches, then launch if a slot
        is free and the pending set is worth a launch.

        Every launch pays a fixed harness startup, so tiny batches waste CPU
        the rollouts need. We launch immediately while nothing is running
        (keeps the pipe primed), and otherwise wait for a full batch or for
        the accumulation window to expire. drain=True means rollouts are
        finished, so latency no longer matters: flush everything."""
        for batch in list(self.running):
            if batch.proc.poll() is None:
                if time.time() - batch.started > self.cfg.eval_batch_timeout_s:
                    log.warning("eval batch %s exceeded %.0fs; killing",
                                batch.run_id, self.cfg.eval_batch_timeout_s)
                    _killpg(batch.proc)
                continue
            self.running.remove(batch)
            self._harvest(batch)
        if not self.pending or len(self.running) >= self.cfg.eval_parallel:
            return
        waited = time.time() - self._pending_since
        if (drain or not self.running
                or len(self.pending) >= self.cfg.eval_batch_min
                or waited >= self.cfg.eval_batch_max_wait_s):
            self._launch_batch()

    def all_preds(self) -> dict[str, dict]:
        return self._preds

    def busy(self) -> bool:
        return bool(self.pending or self.running)

    def kill_all(self) -> None:
        for batch in self.running:
            _killpg(batch.proc)


# -- watcher --------------------------------------------------------------------------

def _read_preds(preds_path: Path) -> dict[str, dict]:
    """Parse the agent's incrementally-updated preds.json, tolerating a
    mid-write partial file (just retry next poll)."""
    try:
        raw = json.loads(preds_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(raw, dict):
        out = {}
        for iid, row in raw.items():
            if isinstance(row, dict):
                rec = dict(row)
                rec.setdefault("instance_id", iid)
                out[iid] = rec
        return out
    if isinstance(raw, list):
        return {str(r.get("instance_id")): r for r in raw
                if isinstance(r, dict) and r.get("instance_id")}
    return {}


# -- main entry --------------------------------------------------------------------------

def run_bench(cfg: Config, pool: dict, instance_ids: list[str],
              model_repo: str, model_port: int,
              abort: threading.Event,
              on_progress=None, engine_counters=None) -> dict:
    """Run the panel against the served model. Returns the full result dict
    including per-instance outcomes and phase timings.

    ``engine_counters`` is an optional callable returning cumulative engine
    totals; it turns the frozen "0 of 25 done" wait into visible motion."""
    n = len(instance_ids)
    t_start = time.time()
    timings: dict = {}
    run_root = cfg.bench_root / f"run-{int(t_start)}-{uuid.uuid4().hex[:6]}"
    preds_dir = run_root / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    base_counters: dict = {}

    def live_stats() -> dict:
        """Steps and tokens produced by this run so far."""
        if engine_counters is None:
            return {}
        now = engine_counters()
        if not now:
            return {}
        return {k: max(0, v - base_counters.get(k, 0)) for k, v in now.items()}

    def progress(phase: str, done: int = 0) -> None:
        if on_progress:
            on_progress(phase, done, n, live_stats())

    # Phase: prepare (dataset subset + agent config + container hygiene)
    progress("prepare")
    t0 = time.time()
    _reap_stale_containers()
    subset_dir = cfg.data_dir / "subsets" / panel_key(instance_ids)
    agent_cfg = run_root / "agent.yaml"
    try:
        _prepare_subset(pool, instance_ids, subset_dir)
        _write_agent_config(cfg, model_repo, model_port, agent_cfg)
    except Exception as e:
        return {"ok": False, "error": f"prepare: {e}"}
    timings["prepare_s"] = round(time.time() - t0, 1)

    if abort.is_set():
        return {"ok": False, "error": "aborted", "aborted": True,
                "timings": timings}

    # Phase: image warm (no-op when the pool is baked)
    progress("image_warm")
    timings["image_warm_s"] = prepull_images(instance_ids, abort=abort)
    if abort.is_set():
        return {"ok": False, "error": "aborted", "aborted": True,
                "timings": timings}

    env = dict(os.environ)
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY") or "local"
    env["FASTBENCH_REPEAT_LIMIT"] = str(cfg.repeat_limit)

    # Phase: rollouts + streaming evals
    workers = min(n, cfg.agent_workers_cap)
    # Same CLI as `mini-extra swebench`, wrapped to install the loop guard.
    agent_cmd = [
        sys.executable, "-m", "fastbench.agent_main",
        "--model", f"openai/{model_repo}",
        "--config", str(agent_cfg),
        "--subset", str(subset_dir),
        "--split", "test",
        "--workers", str(workers),
        "--output", str(preds_dir),
        "--environment-class", "docker",
    ]
    log.info("agent: %s", " ".join(agent_cmd))
    if engine_counters is not None:
        # Baseline before any rollout request, so reported steps count only
        # this run rather than the engine's whole lifetime.
        base_counters = engine_counters() or {}
    progress("rollouts")
    t_agent = time.time()
    agent = _spawn(agent_cmd, env, run_root / "agent.log")
    pipeline = _EvalPipeline(cfg, pool, run_root, env, abort)
    preds_path = preds_dir / "preds.json"
    seen: set[str] = set()
    rollout_done_at: dict[str, float] = {}
    deadline = t_start + cfg.job_timeout_s
    agent_exit: int | None = None
    timed_out = False

    while True:
        if abort.is_set():
            _killpg(agent)
            pipeline.kill_all()
            return {"ok": False, "error": "aborted", "aborted": True,
                    "timings": timings}
        if time.time() > deadline:
            timed_out = True
            _killpg(agent)
            pipeline.kill_all()
            break
        if agent_exit is None and agent.poll() is not None:
            agent_exit = agent.returncode
            timings["agent_wall_s"] = round(time.time() - t_agent, 1)
        # Feed newly finished rollouts to the eval pipeline.
        for iid, rec in _read_preds(preds_path).items():
            if iid not in seen:
                seen.add(iid)
                rollout_done_at[iid] = round(time.time() - t_agent, 1)
                pipeline.submit(iid, rec)
        pipeline.step(drain=agent_exit is not None)
        progress("rollouts" if agent_exit is None else "evals",
                 len(pipeline.evaluated))
        if agent_exit is not None and not pipeline.busy():
            break
        time.sleep(3)

    total_s = round(time.time() - t_start, 1)
    timings.setdefault("agent_wall_s", round(time.time() - t_agent, 1))
    if pipeline.first_eval_start and pipeline.last_eval_end:
        timings["eval_window_s"] = round(
            pipeline.last_eval_end - pipeline.first_eval_start, 1)
        timings["eval_tail_s"] = round(
            max(0.0, pipeline.last_eval_end
                - (t_agent + timings["agent_wall_s"])), 1)
    timings["total_s"] = total_s
    timings["rollout_done_at"] = rollout_done_at

    per_instance = {iid: (iid in pipeline.resolved) for iid in instance_ids
                    if iid in pipeline.evaluated}
    n_resolved = sum(per_instance.values())
    # A run where nothing produced a patch is an infrastructure failure (dead
    # model endpoint, docker trouble), not a model that scored zero. Reporting
    # it as a clean 0.0 would quietly poison comparisons.
    n_patched = sum(1 for rec in pipeline.all_preds().values()
                    if (rec.get("model_patch") or "").strip())
    complete = len(pipeline.evaluated) == n and not timed_out
    result = {
        "ok": complete and not pipeline.errors,
        "score": round(n_resolved / n, 4),
        "n_resolved": n_resolved,
        "n": n,
        "n_evaluated": len(pipeline.evaluated),
        "n_patched": n_patched,
        "exit_statuses": _exit_status_counts(preds_dir),
        "workers": workers,
        "wall_time_s": total_s,
        "timings": timings,
        "per_instance": per_instance,
        "run_root": str(run_root),
    }
    errors = list(pipeline.errors)
    if timed_out:
        errors.append(f"job timeout after {cfg.job_timeout_s}s")
    if agent_exit not in (0, None):
        errors.append(f"agent exit {agent_exit}")
    if n_patched == 0 and n > 0:
        errors.append(
            f"no rollout produced a patch across {n} tasks - the model "
            "endpoint or docker environment is broken, not a 0.0 score")
    missing = [i for i in instance_ids if i not in pipeline.evaluated]
    if missing and not timed_out:
        errors.append(f"{len(missing)} instances produced no prediction "
                      f"(first: {missing[:3]})")
    if errors:
        result["error"] = "; ".join(errors)
    return result
