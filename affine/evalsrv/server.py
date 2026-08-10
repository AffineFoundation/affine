"""Affine eval server — FastAPI app on the GPU pod.

Endpoints (driven by affine/eval_client.py on the root machine, all requiring
the shared X-Affine-Token header when AFFINE_EVAL_TOKEN is set):

  GET  /health              {ok, busy, busy_kind, engine, ...}
  POST /duel                dispatch a duel; 409 when busy (aborts a bench)
  GET  /duel/{id}           poll job record (verdict fallback for SSE races)
  GET  /duel/{id}/stream    SSE: progress + heartbeat + verdict/error
  GET  /duel/{id}/artifact  gzipped full duel record (rollouts + logprobs)
  POST /prefetch            background-download the next challenger's weights
  POST /bench               dispatch a benchmark suite; 409 when busy
  GET  /bench/{id}          poll bench job
  POST /abort_bench         abort a running bench so a duel can take the GPUs

One duel OR one bench at a time (`_busy_lock`): both need miner-serving GPUs.
Duels always win: a `/duel` arriving while a bench holds the lock signals the
bench to abort, which frees the lock within seconds. Heartbeats every 30s keep
the validator's stream-idle watchdog from tripping during cold model
downloads. Fatal CUDA errors self-kill the process; the supervisor
(bootstrap.sh loop) restarts it, and the validator's provisioner replaces the
whole pod if health stays red.
"""

from __future__ import annotations

import asyncio
import gzip
import importlib.metadata
import json
import logging
import os
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from affine.config import load_config
from affine.eval_client import Fault

from . import benchrunner, dueling
from .corpus import CorpusSync
from .engine import Engine
from .vllm_client import ContextLengthError, Served

log = logging.getLogger("evalsrv")

DATA_DIR = Path(os.environ.get("AFFINE_DATA_DIR", "/root/affine_data"))
TURNS_PATH = DATA_DIR / "turns.jsonl"
# Full duel records (rollouts + logprobs), gzipped, fetched by the root
# validator after the verdict and published for miners. Bounded on disk.
ARTIFACTS_DIR = DATA_DIR / "artifacts"
ARTIFACTS_KEEP = 20
EVAL_TOKEN = os.environ.get("AFFINE_EVAL_TOKEN", "")

CUDA_FATAL_MARKERS = ("CUDA error", "device-side assert", "ECC error",
                      "CUDA out of memory", "illegal memory access")

app = FastAPI(title="affine-evalsrv")

_cfg = load_config()
_engine = Engine(_cfg.raw)
_corpus = CorpusSync(_cfg.dataset.corpus_base_url, _cfg.dataset.manifest_key,
                     DATA_DIR)
_busy_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_current: dict = {"kind": None, "job_id": None}
_abort_bench = threading.Event()
_corpus_kick = threading.Event()
_teacher_ready = False
_ROLE = os.environ.get("AFFINE_ROLE", "duel")
_JOBS_RETENTION = _cfg.validator.jobs_retention


def _require_token(x_affine_token: str = Header(default="")) -> None:
    if EVAL_TOKEN and x_affine_token != EVAL_TOKEN:
        raise HTTPException(401, "bad or missing X-Affine-Token")


def _register_job(job_id: str, record: dict) -> None:
    with _jobs_lock:
        _jobs[job_id] = record
        # Prune oldest finished jobs beyond the retention cap.
        if len(_jobs) > _JOBS_RETENTION:
            finished = [(j.get("created_at", 0), jid) for jid, j in _jobs.items()
                        if j.get("state") in ("completed", "failed")]
            finished.sort()
            for _, jid in finished[: len(_jobs) - _JOBS_RETENTION]:
                _jobs.pop(jid, None)


def _save_artifact(job_id: str, req: DuelRequest, verdict: dict,
                   artifact: dict) -> None:
    """Persist the full duel record; failures never affect the verdict."""
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": job_id,
            "request": req.model_dump(),
            "verdict": verdict,
            **artifact,
        }
        path = ARTIFACTS_DIR / f"{job_id}.json.gz"
        path.write_bytes(gzip.compress(json.dumps(payload).encode()))
        files = sorted(ARTIFACTS_DIR.glob("*.json.gz"),
                       key=lambda p: p.stat().st_mtime)
        for old in files[:-ARTIFACTS_KEEP]:
            old.unlink(missing_ok=True)
    except Exception:
        log.warning("artifact save failed for %s", job_id, exc_info=True)


def _schedule_self_kill(reason: str) -> None:
    log.critical("self-kill scheduled: %s", reason)

    def _die():
        time.sleep(5)
        os._exit(70)

    threading.Thread(target=_die, daemon=True).start()


def _self_kill_if_cuda(context: str, err: str) -> None:
    if any(m in err for m in CUDA_FATAL_MARKERS):
        _schedule_self_kill(f"{context}: {err}")


@app.on_event("startup")
def _startup():
    global _teacher_ready
    if _ROLE == "bench":
        # Dedicated bench pod: no teacher, no corpus — healthy as soon as the
        # API is up. Miner weights load per-job into the challenger slot.
        _teacher_ready = True
        log.info("bench role: skipping teacher warmup and corpus refresh")
        return

    def _warm():
        global _teacher_ready
        _teacher_ready = _engine.ensure_teacher()
        if not _teacher_ready:
            _schedule_self_kill("teacher failed to start")
    threading.Thread(target=_warm, daemon=True, name="teacher-warmup").start()

    def _corpus_loop():
        # Refresh between duels only: a manifest swap mid-duel would change
        # turns.jsonl under a running slice. The fixed hourly probe used to
        # starve under a hot queue (observed 15h stale: the instant check
        # almost always found _busy_lock held and slept another full hour), so
        # duel completion now kicks the loop and the inter-duel gap is always
        # used. If the next duel already grabbed the lock, its own completion
        # kicks again.
        while True:
            _corpus_kick.wait(_cfg.dataset.refresh_interval_s)
            _corpus_kick.clear()
            if _busy_lock.locked():
                continue
            _corpus.refresh()
    threading.Thread(target=_corpus_loop, daemon=True,
                     name="corpus-refresh").start()


# -- health ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def _stack_versions() -> dict:
    """Installed serving-stack versions. Disclosed via /health → validator
    state → snapshot API + website, so miners can pre-flight their checkpoint
    against the exact vLLM/transformers build that will load it (pins in
    pyproject are floors; pods install fresh at provision time)."""
    out: dict[str, str | None] = {}
    for pkg in ("vllm", "transformers", "torch"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


@app.get("/health")
def health(_: None = Depends(_require_token)):
    return {
        "ok": _teacher_ready,
        "role": _ROLE,
        "engine": _engine.status(),
        "busy": _busy_lock.locked(),
        "busy_kind": _current["kind"],
        "free_disk_gb": round(_engine.free_disk_gb(), 1),
        "turns_present": (_corpus.ready if _ROLE != "bench"
                          else TURNS_PATH.exists()),
        "corpus": _corpus.info() if _ROLE != "bench" else None,
        "versions": _stack_versions(),
    }


# -- duels ----------------------------------------------------------------------

class DuelFault(RuntimeError):
    """A duel failure the eval server can classify for the root client. The
    `code` (an eval_client.Fault value) is stamped on the SSE error event so
    the validator routes it without parsing free-form text — infra faults
    never burn the miner."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class DuelRequest(BaseModel):
    king_repo: str
    king_revision: str
    challenger_repo: str
    challenger_revision: str
    challenger_hotkey: str
    block_hash: str
    # Servable-weight bytes of the challenger (from the root validator's
    # metadata scan). 0 = unknown → skip the pre-download disk-fit check.
    challenger_weight_bytes: int = 0


def _run_duel_job(job_id: str, req: DuelRequest) -> None:
    job = _jobs[job_id]
    events: Queue = job["events"]
    stop_heartbeat = threading.Event()
    _current.update(kind="duel", job_id=job_id)
    try:
        def hb():
            while not stop_heartbeat.wait(30.0):
                events.put({"type": "heartbeat",
                            "data": {"phase": job.get("phase", "?")}})
        threading.Thread(target=hb, daemon=True).start()

        job["state"] = "running"

        job["phase"] = "ensure_teacher"
        if not _engine.ensure_teacher():
            # The teacher is ours: a dead teacher is a pod fault, not the
            # challenger's — infra code so the miner's slot is never burned.
            raise DuelFault(Fault.TEACHER, "teacher not servable")

        job["phase"] = "ensure_king"
        if not _engine.ensure_king(req.king_repo, req.king_revision):
            # A king that will not *launch* is a transient/pod fault, NOT proof
            # the king is gone: infra code so the miner is not burned and the
            # king is not dethroned. The root proves a king is truly gone with
            # an HF metadata probe before reverting it.
            raise DuelFault(
                Fault.KING_LAUNCH,
                f"king {req.king_repo}@{req.king_revision[:12]} failed to launch")

        job["phase"] = "load_challenger"
        events.put({"type": "progress",
                    "data": {"phase": "load_challenger", "repo": req.challenger_repo}})
        # Fail fast (before a doomed multi-GB download) when the challenger's
        # servable weights cannot fit the pod's serviceable disk. This is our
        # pod being too small for a within-cap repo, so it is an infra fault
        # (no miner burn); the validator defers it and the operator is paged.
        fits, free_gb = _engine.challenger_fits(
            req.challenger_weight_bytes,
            repo=req.challenger_repo, revision=req.challenger_revision)
        if not fits:
            raise DuelFault(
                Fault.POD_CAPACITY,
                f"challenger weights ~{req.challenger_weight_bytes / 1e9:.0f} GB "
                f"do not fit {free_gb:.0f} GB free after cache prune; provision "
                "a larger pod or lower submission.max_model_size_gb")
        if not _engine.load_challenger(req.challenger_repo, req.challenger_revision):
            fault = _engine.diagnose_load_failure()
            if fault == "infra":
                # Our pod is unhealthy (disk/OOM/dead teacher). Emit an infra
                # error so the validator requeues without burning the miner.
                raise DuelFault(Fault.CHALLENGER_INFRA, "challenger could not "
                                "load (pod disk/OOM/teacher unhealthy)")
            # The checkpoint itself is unservable → a real rejection verdict.
            verdict = {"challenger_wins": False, "job_id": job_id,
                       "rejection_reason": "unservable:challenger failed to load in vLLM"}
            job["verdict"] = verdict
            events.put({"type": "verdict", "data": verdict})
            job["state"] = "completed"
            return

        job["phase"] = "scoring"

        def on_progress(miner: str, done: int, total: int):
            events.put({"type": "progress",
                        "data": {"phase": "scoring", "miner": miner,
                                 "done": done, "total": total}})

        try:
            verdict, artifact = asyncio.run(dueling.run_duel(
                _cfg.raw, TURNS_PATH,
                king=Served("king", req.king_repo, req.king_revision,
                            _engine.king_slot.port),
                challenger=Served("challenger", req.challenger_repo,
                                  req.challenger_revision,
                                  _engine.chall_slot.port),
                # All servable teacher replicas — dueling round-robins the
                # echo load across them. Falls back to the primary endpoint
                # if the engine reports none (ensure_teacher just passed).
                teacher=_engine.teacher_serveds() or [
                    Served("teacher", _cfg.teacher.repo, None,
                           _engine.teacher_slot.port,
                           base_url=(_cfg.teacher.base_url or None))],
                block_hash=req.block_hash, hotkey=req.challenger_hotkey,
                corpus_info=_corpus.info(),
                on_progress=on_progress,
                corpus=_corpus))
        except ContextLengthError as e:
            # Serving config / corpus length — requeue without burning miner.
            raise DuelFault(Fault.CONTEXT_LIMIT, str(e)) from e
        verdict["job_id"] = job_id  # lets the root fetch the artifact post-verdict
        _save_artifact(job_id, req, verdict, artifact)

        # Set the verdict on the job and enqueue the event BEFORE flipping the
        # terminal state, so the SSE generator can never observe "completed"
        # with an empty queue while the verdict is still in flight.
        job["verdict"] = verdict
        events.put({"type": "verdict", "data": verdict})
        job["state"] = "completed"
    except Exception as e:
        log.exception("duel %s failed", job_id)
        code = getattr(e, "code", None)
        job["error"] = str(e)
        job["error_code"] = code
        events.put({"type": "error", "data": {"error": str(e), "code": code}})
        job["state"] = "failed"
        _self_kill_if_cuda(f"duel {job_id}", str(e))
    finally:
        stop_heartbeat.set()
        try:
            _engine.unload_challenger()
        except Exception:
            log.warning("challenger unload failed", exc_info=True)
        _current.update(kind=None, job_id=None)
        _busy_lock.release()
        _corpus_kick.set()


@app.post("/duel")
def start_duel(req: DuelRequest, _: None = Depends(_require_token)):
    if _ROLE == "bench":
        raise HTTPException(400, "this pod is AFFINE_ROLE=bench; duels go to the eval machine")
    if not _corpus.ready:
        raise HTTPException(503, "turn corpus not synced yet")
    if not _busy_lock.acquire(blocking=False):
        # Duels own the GPU: if a bench is holding the lock, tell it to abort
        # so the next retry can take over. Still 409 this call.
        if _current["kind"] == "bench":
            _abort_bench.set()
            raise HTTPException(409, "aborting bench for duel; retry")
        raise HTTPException(409, "eval server busy with a duel")
    job_id = f"duel-{uuid.uuid4().hex[:12]}"
    _register_job(job_id, {"state": "queued", "kind": "duel", "events": Queue(),
                           "created_at": time.time(), "request": req.model_dump()})
    threading.Thread(target=_run_duel_job, args=(job_id, req),
                     daemon=True, name=job_id).start()
    return {"job_id": job_id}


@app.get("/duel/{job_id}")
def get_duel(job_id: str, _: None = Depends(_require_token)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in job.items() if k != "events"}


@app.get("/duel/{job_id}/artifact")
def get_duel_artifact(job_id: str, _: None = Depends(_require_token)):
    path = ARTIFACTS_DIR / f"{job_id}.json.gz"
    if not path.is_file():
        raise HTTPException(404, "no artifact for this job")
    return Response(content=path.read_bytes(), media_type="application/gzip")


@app.get("/duel/{job_id}/stream")
async def stream_duel(job_id: str, _: None = Depends(_require_token)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")

    async def generate():
        events: Queue = job["events"]
        while True:
            try:
                ev = events.get_nowait()
            except Empty:
                if job["state"] in ("completed", "failed") and events.empty():
                    break
                await asyncio.sleep(1.0)
                continue
            yield f"data: {json.dumps(ev)}\n\n"
            if ev["type"] in ("verdict", "error"):
                break

    return StreamingResponse(generate(), media_type="text/event-stream")


# -- prefetch ---------------------------------------------------------------------

class PrefetchRequest(BaseModel):
    repo: str
    revision: str
    # Servable-weight bytes from the validator's metadata scan (same source
    # as DuelRequest.challenger_weight_bytes). 0 = unknown → skip disk-fit.
    weight_bytes: int = 0


@app.post("/prefetch")
def prefetch(req: PrefetchRequest, _: None = Depends(_require_token)):
    """Warm the next queued challenger's snapshot while the current duel
    scores (network-bound download overlaps GPU-bound scoring). Purely
    advisory: declining just means the next /duel pays the download."""
    accepted, reason = _engine.start_prefetch(req.repo, req.revision,
                                              req.weight_bytes)
    (log.info if accepted else log.debug)(
        "prefetch %s@%s: %s", req.repo, req.revision[:12], reason)
    return {"accepted": accepted, "reason": reason}


# -- benchmarks -------------------------------------------------------------------

class BenchRequest(BaseModel):
    repo: str
    revision: str
    suite: str
    num_trials: int = 2
    max_concurrency: int = 8
    user_llm: str = "teacher"


def _fail_bench(job: dict, suite: str, detail: str) -> None:
    job["result"] = {"ok": False, "suite": suite, "error": detail[:1500]}
    job["state"] = "failed"


def _run_bench_job(job_id: str, req: BenchRequest) -> None:
    job = _jobs[job_id]
    _current.update(kind="bench", job_id=job_id)
    _abort_bench.clear()
    engine_died = threading.Event()
    watchdog_stop = threading.Event()

    def _watchdog() -> None:
        # A crashed vLLM (e.g. a JIT kernel failing at first request) leaves
        # the agent harness hammering a dead endpoint for the rest of the
        # suite. A dead PROCESS is unambiguous — abort on sight. A failed
        # HTTP probe is not: a saturated API server (12 agents x 16k-token
        # prefills) can blow a short timeout while perfectly healthy, and a
        # false abort wastes a pod-day, so HTTP needs a long timeout and
        # several consecutive misses (~2 min silent) before we call it dead.
        misses = 0
        while not watchdog_stop.wait(30.0):
            if _engine.challenger_process_dead():
                log.error("bench %s: challenger vLLM process died; aborting",
                          job_id)
                engine_died.set()
                _abort_bench.set()
                return
            if _engine.challenger_alive(http_timeout=10.0):
                misses = 0
                continue
            misses += 1
            if misses >= 4:
                log.error("bench %s: challenger vLLM unresponsive for ~2min; "
                          "aborting", job_id)
                engine_died.set()
                _abort_bench.set()
                return

    try:
        job["state"] = "LOADING_MODEL"
        if not _engine.load_challenger(req.repo, req.revision):
            _fail_bench(job, req.suite, _engine.chall_slot.load_error
                        or "model failed to load in vLLM")
            return
        job["state"] = "RUNNING"
        threading.Thread(target=_watchdog, daemon=True,
                         name=f"{job_id}-watchdog").start()
        result = benchrunner.run_suite(
            suite=req.suite, model_repo=req.repo,
            model_port=_engine.chall_slot.port,
            user_llm=req.user_llm,
            teacher_repo=_cfg.teacher.repo,
            teacher_port=_engine.teacher_slot.port,
            num_trials=req.num_trials, max_concurrency=req.max_concurrency,
            abort_event=_abort_bench)
        if engine_died.is_set():
            tail = _engine.challenger_log_tail()
            _fail_bench(job, req.suite, "challenger vLLM died mid-run"
                        + (f" | {tail}" if tail else ""))
            return
        job["result"] = result
        job["state"] = "aborted" if result.get("aborted") else "completed"
    except Exception as e:
        log.exception("bench %s failed", job_id)
        _fail_bench(job, req.suite, str(e))
        _self_kill_if_cuda(f"bench {job_id}", str(e))
    finally:
        watchdog_stop.set()
        try:
            _engine.unload_challenger()
        except Exception:
            log.warning("challenger unload failed", exc_info=True)
        _current.update(kind=None, job_id=None)
        _busy_lock.release()


@app.post("/bench")
def start_bench(req: BenchRequest, _: None = Depends(_require_token)):
    if not _busy_lock.acquire(blocking=False):
        raise HTTPException(409, "eval server busy")
    job_id = f"bench-{uuid.uuid4().hex[:12]}"
    _register_job(job_id, {"state": "queued", "kind": "bench", "events": Queue(),
                           "created_at": time.time(), "request": req.model_dump()})
    threading.Thread(target=_run_bench_job, args=(job_id, req),
                     daemon=True, name=job_id).start()
    return {"job_id": job_id}


@app.get("/bench/{job_id}")
def get_bench(job_id: str, _: None = Depends(_require_token)):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in job.items() if k != "events"}


@app.post("/abort_bench")
def abort_bench(_: None = Depends(_require_token)):
    if _current["kind"] == "bench":
        _abort_bench.set()
        return {"aborting": True, "job_id": _current["job_id"]}
    return {"aborting": False}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    port = int(os.environ.get("AFFINE_EVAL_PORT", "9000"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
