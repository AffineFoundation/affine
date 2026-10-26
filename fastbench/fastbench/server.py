"""fastbench HTTP service.

  POST /bench           {"model": "org/repo[@rev]", "n": 25, "seed": 0} -> job_id
  GET  /bench/{job_id}  job record (score, timings, per_instance when done)
  POST /bench/{job_id}/abort
  POST /preload         {"model": "org/repo[@rev]"} download weights early
  GET  /health          slot + pool + queue status
  GET  /pool            pool manifest info
  GET  /panel?n=&seed=  the exact task set a (n, seed) resolves to

One bench runs at a time (the model slot owns the GPUs); requests queue FIFO.
Auth: X-Fastbench-Token header when FASTBENCH_TOKEN is set.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import pool as poolmod
from . import runner
from .config import load_config
from .serving import ModelSlot, gpu_count, parse_model

log = logging.getLogger("fastbench")

cfg = load_config()
app = FastAPI(title="fastbench")
_pool = poolmod.load_pool()
_slot = ModelSlot(cfg)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue(maxsize=cfg.queue_max)
_current: dict = {"job_id": None}


def _require_token(x_fastbench_token: str = Header(default="")) -> None:
    if cfg.token and x_fastbench_token != cfg.token:
        raise HTTPException(401, "bad or missing X-Fastbench-Token")


class BenchRequest(BaseModel):
    model: str = Field(description="HF repo, optionally repo@revision")
    n: int = Field(default=25, ge=1)
    seed: int = Field(default=0)


class PreloadRequest(BaseModel):
    model: str = Field(description="HF repo, optionally repo@revision")


def _record_history(job: dict) -> None:
    try:
        cfg.history_path.parent.mkdir(parents=True, exist_ok=True)
        # instance_ids are derivable from (n, seed) + the frozen pool.
        slim = {k: v for k, v in job.items()
                if k not in ("abort", "instance_ids")}
        with open(cfg.history_path, "a") as f:
            f.write(json.dumps(slim) + "\n")
    except Exception:
        log.warning("history append failed", exc_info=True)


def _run_job(job_id: str) -> None:
    job = _jobs[job_id]
    abort: threading.Event = job["abort"]
    _current["job_id"] = job_id
    try:
        job["state"] = "loading_model"
        job["started_at"] = time.time()
        repo, revision = parse_model(job["model"])
        load = _slot.ensure(repo, revision, abort)
        job["model_load"] = {"warm": load["warm"], "load_s": load["load_s"]}
        if not load["ok"]:
            job["state"] = "failed"
            job["error"] = f"model load: {load['error']}"[:1500]
            return
        job["state"] = "running"

        def on_progress(phase: str, done: int, total: int,
                        live: dict | None = None) -> None:
            job["phase"] = phase
            job["progress"] = {"done": done, "total": total, **(live or {})}

        result = runner.run_bench(
            cfg, _pool, job["instance_ids"], repo, _slot.port,
            abort=abort, on_progress=on_progress,
            engine_counters=_slot.engine_counters)
        result["model_load_s"] = load["load_s"]
        result["model_warm"] = load["warm"]
        job["result"] = result
        if result.get("aborted"):
            job["state"] = "aborted"
        else:
            job["state"] = "completed" if result.get("ok") else "failed"
            job["error"] = result.get("error")
    except Exception as e:
        log.exception("job %s crashed", job_id)
        job["state"] = "failed"
        job["error"] = str(e)[:1500]
    finally:
        job["finished_at"] = time.time()
        _current["job_id"] = None
        _record_history(job)


def _worker() -> None:
    while True:
        job_id = _queue.get()
        job = _jobs.get(job_id)
        if job is None or job.get("state") != "queued":
            continue
        _run_job(job_id)


threading.Thread(target=_worker, daemon=True, name="fastbench-worker").start()


@app.on_event("startup")
def _adopt_running_engine() -> None:
    """A redeploy restarts this service but leaves its vLLM child serving;
    adopting it keeps the model warm across deploys."""
    if _slot.adopt_existing():
        log.info("startup: adopted warm slot %s", _slot.status().get("repo"))


@app.get("/health")
def health(_: None = Depends(_require_token)):
    return {
        "ok": True,
        "gpus": gpu_count(),
        "slot": _slot.status(),
        "busy": _current["job_id"] is not None,
        "current_job": _current["job_id"],
        "queued": _queue.qsize(),
        "prefetch": _slot.prefetch_status(),
        "pool_size": len(_pool["instance_ids"]),
        "pool_dataset": _pool["dataset"],
    }


@app.get("/pool")
def pool_info(_: None = Depends(_require_token)):
    return {k: v for k, v in _pool.items() if k != "instance_ids"} | {
        "n_instances": len(_pool["instance_ids"])}


@app.get("/panel")
def panel(n: int = 25, seed: int = 0, _: None = Depends(_require_token)):
    try:
        ids = poolmod.sample(_pool, n, seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"n": n, "seed": seed, "instance_ids": ids}


@app.post("/bench")
def start_bench(req: BenchRequest, _: None = Depends(_require_token)):
    try:
        instance_ids = poolmod.sample(_pool, req.n, req.seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    job_id = f"fb-{uuid.uuid4().hex[:12]}"
    job = {
        "job_id": job_id,
        "state": "queued",
        "model": req.model,
        "n": req.n,
        "seed": req.seed,
        "instance_ids": instance_ids,
        "created_at": time.time(),
        "abort": threading.Event(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    try:
        _queue.put_nowait(job_id)
    except queue.Full:
        with _jobs_lock:
            _jobs.pop(job_id, None)
        raise HTTPException(429, "queue full")
    return {"job_id": job_id, "queued_ahead": max(0, _queue.qsize() - 1)}


@app.post("/preload")
def preload(req: PreloadRequest, _: None = Depends(_require_token)):
    """Fetch a model's weights now so a later /bench on it skips the download.

    Safe to call while a bench is running: it never touches the GPUs."""
    repo, revision = parse_model(req.model)
    return _slot.prefetch(repo, revision)


@app.get("/bench/{job_id}")
def get_bench(job_id: str, _: None = Depends(_require_token)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in job.items() if k != "abort"}


@app.post("/bench/{job_id}/abort")
def abort_bench(job_id: str, _: None = Depends(_require_token)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job["state"] in ("completed", "failed", "aborted"):
        return {"aborting": False, "state": job["state"]}
    job["abort"].set()
    if job["state"] == "queued":
        job["state"] = "aborted"
    return {"aborting": True}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=cfg.port)


if __name__ == "__main__":
    main()
