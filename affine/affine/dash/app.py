"""FastAPI app: /api/v1/* + static website/."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import Config, load_config
from .corpus import REFRESH_TTL_S, DatasetView
from .index import DashIndex
from .project import project_duel_summary, project_series, project_turn_detail
from .readers import (
    bench_artifact_index,
    contract_payload,
    duel_log_lines,
    load_audit_detail,
    load_bench_artifact,
    load_eval_artifact,
    load_public_audits,
    load_public_benchmarks,
    snapshot,
)
from .stream import snapshot_events
from .tmc import (
    enrich_snapshot,
    market_for_snapshot,
    reg_history_for_api,
    run_market_stream,
    run_reg_history_loop,
)

log = logging.getLogger("affine.dash")

WEBSITE_DIR = Path(__file__).resolve().parents[2] / "website"


def _etag(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:32]


def _json(payload, *, max_age: int = 5, request: Request | None = None,
          immutable: bool = False) -> JSONResponse:
    tag = f'"{_etag(payload)}"'
    if request is not None and request.headers.get("if-none-match") == tag:
        return JSONResponse(status_code=304, content=None, headers={
            "ETag": tag,
            "Cache-Control": (
                f"public, max-age={max_age}, immutable" if immutable
                else f"public, max-age={max_age}"),
        })
    headers = {
        "ETag": tag,
        "Cache-Control": (
            f"public, max-age={max_age}, immutable" if immutable
            else f"public, max-age={max_age}"),
    }
    return JSONResponse(payload, headers=headers)


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    index = DashIndex(cfg.state_dir)
    dataset = DatasetView(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        app.state.index = index
        index.ingest()

        async def _ingest_loop():
            while True:
                await asyncio.sleep(2.0)
                try:
                    index.ingest()
                except Exception as exc:
                    log.warning("index ingest failed: %s", exc)

        async def _dataset_loop():
            # Manifest + index sync off the event loop; endpoints only read
            # the in-memory caches this builds.
            while True:
                try:
                    await asyncio.to_thread(dataset.refresh)
                except Exception as exc:
                    log.warning("dataset refresh failed: %s", exc)
                await asyncio.sleep(REFRESH_TTL_S)

        ingest_task = asyncio.create_task(_ingest_loop())
        market_task = asyncio.create_task(run_market_stream(cfg))
        reg_task = asyncio.create_task(run_reg_history_loop(cfg))
        dataset_task = asyncio.create_task(_dataset_loop())
        log.info("affine-dash serving state_dir=%s website=%s",
                 cfg.state_dir, WEBSITE_DIR)
        try:
            yield
        finally:
            for task in (ingest_task, market_task, reg_task, dataset_task):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            index.close()

    app = FastAPI(title="Affine dashboard", lifespan=lifespan)

    @app.middleware("http")
    async def _short_cache_static(request: Request, call_next):
        """Keep HTML/JS/CSS out of long Cloudflare browser caches while iterating."""
        response = await call_next(request)
        path = request.url.path or "/"
        if (path == "/" or path.endswith((".html", ".js", ".css"))):
            response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
        return response

    @app.get("/api/v1/snapshot")
    def api_snapshot(request: Request):
        return _json(enrich_snapshot(cfg, snapshot(cfg)), max_age=5,
                     request=request)

    @app.get("/api/v1/history")
    def api_history(
        request: Request,
        limit: int = Query(100, ge=1, le=500),
        cursor: int | None = None,
        q: str = "",
        event: str = "",
    ):
        payload = index.query_history(
            limit=limit, cursor=cursor, q=q.strip(), event=event.strip())
        return _json(payload, max_age=5, request=request)

    @app.get("/api/v1/benchmarks")
    def api_benchmarks(request: Request):
        suites = list(cfg.bench.suites)
        pub = load_public_benchmarks(cfg)
        if pub is not None:
            payload = {**pub, "suites": pub.get("suites") or suites}
            return _json(payload, max_age=5, request=request)
        snap = snapshot(cfg)
        payload = index.benchmarks_payload(snap.get("bench_jobs") or [])
        payload["generated_at"] = snap.get("generated_at")
        payload["suites"] = suites
        return _json(payload, max_age=5, request=request)

    @app.get("/api/v1/contract")
    def api_contract(request: Request):
        return _json(contract_payload(cfg), max_age=60, request=request)

    # -- post-crown exploit audits (verdicts + replayable workspaces) --------
    @app.get("/api/v1/audits")
    def api_audits(request: Request):
        entries = load_public_audits(cfg) or []
        return _json({"items": entries, "count": len(entries)},
                     max_age=15, request=request)

    @app.get("/api/v1/audits/{reign}")
    def api_audit(reign: int, request: Request):
        detail = load_audit_detail(cfg, reign)
        if detail is None:
            return JSONResponse(
                {"error": "not_found", "reign_number": reign},
                status_code=404)
        return _json(detail, max_age=60, request=request)

    # -- dataset browser (corpus D is public on the bucket; these endpoints
    # only make it browsable — stats + turn pages come from the in-memory
    # Parquet index cache, /turn may fetch one chunk on first view) --------------
    @app.get("/api/v1/dataset")
    def api_dataset(request: Request):
        stats = dataset.stats()
        if stats is None:
            return JSONResponse(
                {"error": "corpus_not_synced"}, status_code=503,
                headers={"Retry-After": "60"})
        return _json(stats, max_age=300, request=request)

    @app.get("/api/v1/dataset/turns")
    def api_dataset_turns(
        request: Request,
        limit: int = Query(50, ge=1, le=200),
        cursor: int = Query(0, ge=0),
        source: str = "",
        language: str = "",
        phase: str = "",
        repo: str = "",
        q: str = "",
    ):
        page = dataset.turns_page(
            source=source.strip(), language=language.strip(),
            phase=phase.strip(), repo=repo.strip(), q=q,
            limit=limit, cursor=cursor)
        if page is None:
            return JSONResponse(
                {"error": "corpus_not_synced"}, status_code=503,
                headers={"Retry-After": "60"})
        return _json(page, max_age=300, request=request)

    @app.get("/api/v1/dataset/turn")
    def api_dataset_turn(request: Request,
                         turn_id: str = Query(..., min_length=1)):
        detail = dataset.turn_detail(turn_id)
        if detail is None:
            return JSONResponse(
                {"error": "turn_not_found", "turn_id": turn_id},
                status_code=404)
        # Turn content is immutable once published (append-only corpus).
        return _json(detail, max_age=86400, request=request, immutable=True)

    @app.get("/api/v1/market")
    def api_market(request: Request):
        market = market_for_snapshot(cfg)
        if market is None:
            return JSONResponse({"error": "unavailable", "source": "tmc"},
                                status_code=503)
        return _json(market, max_age=5, request=request)

    @app.get("/api/v1/market/reg-history")
    def api_reg_history(request: Request):
        hist = reg_history_for_api(cfg)
        if hist is None or not hist.get("points"):
            return JSONResponse({"error": "unavailable", "source": "tmc"},
                                status_code=503)
        return _json(hist, max_age=30, request=request)

    # Gunzipping a multi-MB artifact and recomputing per-pair terms is the
    # most expensive request the box serves, the endpoints are public, and
    # the ETag is computed AFTER the work (304s save bandwidth, not CPU).
    # Artifacts are immutable once on disk, so a small LRU turns repeat views
    # into dict lookups. Misses (artifact not written yet) are never cached.
    _artifact_lru: OrderedDict[str, tuple[dict, dict]] = OrderedDict()

    def _artifact_and_series(challenge_id: str) -> tuple[dict, dict] | None:
        hit = _artifact_lru.get(challenge_id)
        if hit is not None:
            _artifact_lru.move_to_end(challenge_id)
            return hit
        art = load_eval_artifact(cfg, challenge_id)
        if art is None:
            return None
        pair = (art, project_series(art))
        _artifact_lru[challenge_id] = pair
        while len(_artifact_lru) > 8:
            _artifact_lru.popitem(last=False)
        return pair

    @app.get("/api/v1/duels/{challenge_id}")
    def api_duel(challenge_id: str, request: Request):
        row = index.get_history_by_challenge(challenge_id)
        cached = _artifact_and_series(challenge_id)
        art, series = cached if cached else (None, None)
        payload = project_duel_summary(row, art, series)
        if row is None and art is None:
            return JSONResponse({"error": "not_found", "challenge_id": challenge_id},
                                status_code=404)
        # NOT immutable: the summary is built from the history row, and history
        # rows can be repaired (2026-08-07: the verdict+crowned row merge left
        # day-long stale browser caches). The artifact-derived /series and
        # /turn endpoints stay immutable; this one revalidates cheaply via
        # ETag + the artifact LRU.
        terminal = payload.get("event") in ("verdict", "crowned", "failed")
        return _json(payload, max_age=300 if terminal else 5, request=request)

    @app.get("/api/v1/duels/{challenge_id}/series")
    def api_duel_series(challenge_id: str, request: Request):
        cached = _artifact_and_series(challenge_id)
        if cached is None:
            return JSONResponse(
                {"error": "not_found", "challenge_id": challenge_id},
                status_code=404)
        # Series is immutable once the artifact exists on disk.
        return _json(cached[1], max_age=86400, request=request, immutable=True)

    @app.get("/api/v1/duels/{challenge_id}/turn")
    def api_duel_turn(challenge_id: str, request: Request,
                      turn_id: str = Query(..., min_length=1)):
        """One turn's full rollout (both sides + teacher refs) from the artifact."""
        cached = _artifact_and_series(challenge_id)
        if cached is None:
            return JSONResponse(
                {"error": "not_found", "challenge_id": challenge_id},
                status_code=404)
        detail = project_turn_detail(cached[0], turn_id)
        if detail is None:
            return JSONResponse(
                {"error": "turn_not_found", "challenge_id": challenge_id,
                 "turn_id": turn_id},
                status_code=404)
        detail["challenge_id"] = challenge_id
        return _json(detail, max_age=86400, request=request, immutable=True)

    @app.get("/api/v1/duels/{challenge_id}/log")
    def api_duel_log(challenge_id: str, request: Request):
        lines = duel_log_lines(challenge_id)
        # Short cache: an in-flight duel keeps appending heartbeat lines.
        return _json({"challenge_id": challenge_id, "lines": lines},
                     max_age=10, request=request)

    # Bench rollout records mirror the duel-artifact pattern: immutable
    # multi-MB gzips on disk, a tiny LRU for repeat views, and the message
    # transcripts only shipped per-instance (the summary strips them).
    _bench_lru: OrderedDict[str, dict] = OrderedDict()

    def _bench_artifact(job_id: str) -> dict | None:
        hit = _bench_lru.get(job_id)
        if hit is not None:
            _bench_lru.move_to_end(job_id)
            return hit
        art = load_bench_artifact(cfg, job_id)
        if art is None:
            return None
        _bench_lru[job_id] = art
        while len(_bench_lru) > 4:
            _bench_lru.popitem(last=False)
        return art

    @app.get("/api/v1/benches")
    def api_benches(request: Request):
        """Manifest of published bench rollout records (newest first)."""
        return _json({"benches": bench_artifact_index(cfg)},
                     max_age=10, request=request)

    @app.get("/api/v1/benches/{job_id}")
    def api_bench(job_id: str, request: Request):
        art = _bench_artifact(job_id)
        if art is None:
            return JSONResponse({"error": "not_found", "job_id": job_id},
                                status_code=404)
        instances = {}
        for iid, inst in (art.get("instances") or {}).items():
            instances[iid] = {
                "resolved": inst.get("resolved"),
                "exit_status": inst.get("exit_status"),
                "model_patch": inst.get("model_patch"),
                "n_messages": len(inst.get("messages") or []),
            }
        payload = {"job_id": art.get("job_id"),
                   "request": art.get("request"),
                   "result": art.get("result"),
                   "instances": instances}
        return _json(payload, max_age=86400, request=request, immutable=True)

    @app.get("/api/v1/benches/{job_id}/trajectory")
    def api_bench_trajectory(job_id: str, request: Request,
                             instance_id: str = Query(..., min_length=1)):
        """One instance's full agent transcript from the bench artifact."""
        art = _bench_artifact(job_id)
        inst = (art or {}).get("instances", {}).get(instance_id)
        if inst is None:
            return JSONResponse(
                {"error": "not_found", "job_id": job_id,
                 "instance_id": instance_id},
                status_code=404)
        payload = {"job_id": job_id, "instance_id": instance_id, **inst}
        return _json(payload, max_age=86400, request=request, immutable=True)

    @app.get("/api/v1/stream")
    async def api_stream():
        return StreamingResponse(
            snapshot_events(cfg),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/v1/health")
    def api_health():
        return {"ok": True, "version": contract_payload(cfg)["version"]}

    # -- public king chat (proxied to the chat pod over its SSH tunnel) ---------
    # The browser never sees the pod or the token; the dash adds both. Freely
    # accessible — abuse control is per-IP rate + a global in-flight cap so
    # one visitor cannot monopolize the single chat GPU pod.
    chat_cfg = cfg.chat
    chat_base = f"http://127.0.0.1:{cfg.chat_machine.port}"
    chat_headers = ({"X-Affine-Token": cfg.secrets.eval_token}
                    if cfg.secrets.eval_token else {})
    chat_rate = int(chat_cfg.get("rate_limit_per_min", 10))
    chat_max_active = int(chat_cfg.get("max_concurrency", 4))
    chat_active = {"n": 0}
    chat_ip_hits: dict[str, list[float]] = {}

    def _chat_ip(request: Request) -> str:
        # Cloudflare fronts affine.io; fall back for the tunnel / local paths.
        return (request.headers.get("cf-connecting-ip")
                or (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
                or (request.client.host if request.client else "unknown"))

    def _chat_rate_ok(ip: str) -> bool:
        now = time.monotonic()
        if len(chat_ip_hits) > 2000:
            for k in [k for k, v in chat_ip_hits.items()
                      if not v or now - v[-1] > 120]:
                chat_ip_hits.pop(k, None)
        hits = [t for t in chat_ip_hits.get(ip, []) if now - t < 60.0]
        if len(hits) >= chat_rate:
            chat_ip_hits[ip] = hits
            return False
        hits.append(now)
        chat_ip_hits[ip] = hits
        return True

    @app.get("/api/v1/chat/status")
    async def api_chat_status():
        """Pod health for the UI: serving / loading / offline + which king."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{chat_base}/health",
                                     headers=chat_headers)
            body = r.json() if r.status_code == 200 else {}
        except Exception:
            body = {}
        payload = {
            "ok": bool(body.get("ok")) and body.get("state") == "serving",
            "state": body.get("state") or "offline",
            "king": body.get("king"),
        }
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/chat")
    async def api_chat(request: Request):
        if not _chat_rate_ok(_chat_ip(request)):
            return JSONResponse(
                {"error": "rate_limited",
                 "detail": f"limit is {chat_rate} messages/minute"},
                status_code=429, headers={"Retry-After": "30"})
        if chat_active["n"] >= chat_max_active:
            return JSONResponse(
                {"error": "busy",
                 "detail": "the model is busy — try again shortly"},
                status_code=429, headers={"Retry-After": "10"})
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "bad_json"}, status_code=400)
        payload = {k: body.get(k) for k in
                   ("messages", "temperature", "max_tokens")
                   if body.get(k) is not None}

        chat_active["n"] += 1
        streaming = False
        client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
        try:
            upstream = await client.send(
                client.build_request("POST", f"{chat_base}/chat",
                                     json=payload, headers=chat_headers),
                stream=True)
            if upstream.status_code != 200:
                detail = (await upstream.aread()).decode("utf-8", "replace")
                await upstream.aclose()
                status = upstream.status_code
                return JSONResponse(
                    {"error": "unavailable" if status == 503 else "upstream",
                     "detail": detail[:300]},
                    status_code=status if status in (400, 503) else 502)

            async def relay():
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                finally:
                    chat_active["n"] -= 1
                    await upstream.aclose()
                    await client.aclose()

            streaming = True
            return StreamingResponse(relay(), media_type="text/event-stream",
                                     headers={
                                         "Cache-Control": "no-cache",
                                         "X-Accel-Buffering": "no",
                                     })
        except Exception as e:
            log.warning("chat proxy failed: %s: %s", type(e).__name__, e)
            return JSONResponse(
                {"error": "offline",
                 "detail": "chat pod unreachable — it may be starting up"},
                status_code=503)
        finally:
            if not streaming:
                chat_active["n"] -= 1
                await client.aclose()

    # -- public OpenAI-compatible wire (same pod, same abuse limits) ------------
    # Any OpenAI client can talk to the king: base_url https://affine.io/v1,
    # model "affine-king", any API key (ignored — limits are per-IP). The dash
    # swaps in the real pod token server-side.

    @app.get("/v1/models")
    async def v1_models():
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{chat_base}/v1/models",
                                     headers=chat_headers)
            return JSONResponse(r.json(), status_code=r.status_code,
                                headers={"Cache-Control": "no-store"})
        except Exception:
            return JSONResponse(
                {"object": "list", "data": [],
                 "error": "chat pod unreachable"}, status_code=503)

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(request: Request):
        if not _chat_rate_ok(_chat_ip(request)):
            return JSONResponse(
                {"error": {"message":
                           f"rate limited: {chat_rate} requests/minute",
                           "type": "rate_limit_exceeded"}},
                status_code=429, headers={"Retry-After": "30"})
        if chat_active["n"] >= chat_max_active:
            return JSONResponse(
                {"error": {"message": "the model is busy — try again shortly",
                           "type": "overloaded"}},
                status_code=429, headers={"Retry-After": "10"})
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "invalid JSON",
                           "type": "invalid_request_error"}}, status_code=400)

        chat_active["n"] += 1
        streaming = False
        client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        try:
            upstream = await client.send(
                client.build_request("POST", f"{chat_base}/v1/chat/completions",
                                     json=body, headers=chat_headers),
                stream=True)
            if not body.get("stream"):
                content = await upstream.aread()
                await upstream.aclose()
                return Response(
                    content=content, status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type",
                                                    "application/json"))

            async def relay():
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                finally:
                    chat_active["n"] -= 1
                    await upstream.aclose()
                    await client.aclose()

            streaming = True
            return StreamingResponse(relay(), media_type="text/event-stream",
                                     status_code=upstream.status_code,
                                     headers={"Cache-Control": "no-cache",
                                              "X-Accel-Buffering": "no"})
        except Exception as e:
            log.warning("v1 proxy failed: %s: %s", type(e).__name__, e)
            return JSONResponse(
                {"error": {"message": "chat pod unreachable",
                           "type": "service_unavailable"}}, status_code=503)
        finally:
            if not streaming:
                chat_active["n"] -= 1
                await client.aclose()

    if WEBSITE_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEBSITE_DIR), html=True),
                  name="website")
    else:
        log.warning("website dir missing: %s", WEBSITE_DIR)

    return app
