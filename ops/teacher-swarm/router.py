#!/usr/bin/env python3
"""Teacher-swarm router: one OpenAI-compatible URL over N vLLM replicas.

Routing: rendezvous (highest-random-weight) hash on the prompt prefix, so
every request for the same duel turn lands on the same replica and reuses
its vLLM prefix cache (the 4 echoes per turn share the turn prefix x).
Backend churn only remaps the keys that lived on the lost backend.

Resilience: per-backend circuit breaker (3 consecutive errors => 60 s out),
one retry on the next-best backend, in-flight overload spill.

Feeds on state/state.json written by manager.py (mtime-watched).
Exposes /v1/models, /v1/completions, /v1/chat/completions, /health, /metrics.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

HERE = Path(__file__).resolve().parent
STATE_JSON = HERE / "state" / "state.json"
ENV_FILE = HERE / ".swarm_env"

BREAKER_TRIP = 3          # consecutive errors before a backend is benched
BREAKER_COOLDOWN_S = 60.0
OVERLOAD_FACTOR = 4       # spill if chosen backend has 4x the min in-flight
# A hung replica must not stall a turn for the engine's full patience:
# echoes (max_tokens<=1) normally finish in 5-30 s, samples in 20-120 s.
ECHO_TIMEOUT_S = 240.0
SAMPLE_TIMEOUT_S = 600.0
AFFINITY_KEY_CHARS = 2048  # prompt prefix length used for affinity hashing


def swarm_key() -> str:
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("SWARM_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"SWARM_KEY missing from {ENV_FILE}")


class Backend:
    def __init__(self, url: str, pod: str, btype: str, weight: float = 1.0):
        self.url = url.rstrip("/")
        self.pod = pod
        self.type = btype
        self.weight = max(weight, 0.01)
        self.in_flight = 0
        self.consecutive_errors = 0
        self.benched_until = 0.0
        self.ok_count = 0
        self.err_count = 0

    @property
    def healthy(self) -> bool:
        return time.monotonic() >= self.benched_until

    def record(self, ok: bool) -> None:
        if ok:
            self.ok_count += 1
            self.consecutive_errors = 0
        else:
            self.err_count += 1
            self.consecutive_errors += 1
            if self.consecutive_errors >= BREAKER_TRIP:
                self.benched_until = time.monotonic() + BREAKER_COOLDOWN_S


class Router:
    def __init__(self):
        self.backends: dict[str, Backend] = {}
        self.model = ""
        self._state_mtime = 0.0
        self.key = swarm_key()
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(SAMPLE_TIMEOUT_S, connect=10.0),
            limits=httpx.Limits(max_connections=512,
                                max_keepalive_connections=512))
        # Rolling completion log for samples/s: (t_done, is_sample).
        self.done: collections.deque[tuple[float, bool]] = \
            collections.deque(maxlen=100_000)
        self.reload_state(force=True)

    def reload_state(self, force: bool = False) -> None:
        try:
            mtime = STATE_JSON.stat().st_mtime
        except OSError:
            return
        if not force and mtime == self._state_mtime:
            return
        self._state_mtime = mtime
        try:
            state = json.loads(STATE_JSON.read_text())
        except (OSError, ValueError):
            return
        self.model = state.get("model") or self.model
        fresh = {}
        for b in state.get("backends", []):
            url = b["url"].rstrip("/")
            existing = self.backends.get(url)
            if existing is None:
                existing = Backend(url, b.get("pod", "?"), b.get("type", "?"),
                                   float(b.get("est_tps") or 1.0))
            else:
                existing.weight = max(float(b.get("est_tps") or 1.0), 0.01)
            fresh[url] = existing
        self.backends = fresh

    # ---- selection --------------------------------------------------------
    def affinity_key(self, payload: dict, path: str) -> str:
        if path.endswith("chat/completions"):
            msgs = payload.get("messages") or []
            raw = json.dumps(msgs[:4], sort_keys=True)
        else:
            p = payload.get("prompt")
            raw = p if isinstance(p, str) else json.dumps(p)
        return (raw or "")[:AFFINITY_KEY_CHARS]

    def ranked(self, key: str) -> list[Backend]:
        """Healthy backends by weighted rendezvous score for this key.

        Weighted HRW: score = -weight / ln(h) with h uniform in (0,1) derived
        from hash(key|url). Traffic lands on each backend proportionally to
        its weight (est_tps), while keeping per-key stickiness for vLLM
        prefix-cache reuse."""
        alive = [b for b in self.backends.values() if b.healthy]
        if not alive:
            alive = list(self.backends.values())  # last resort: try benched

        def score(b: Backend) -> float:
            digest = hashlib.blake2s(f"{key}|{b.url}".encode()).digest()
            h = (int.from_bytes(digest[:8], "big") + 1) / (2**64 + 2)
            return -b.weight / math.log(h)

        ranked = sorted(alive, key=score, reverse=True)
        if len(ranked) >= 2:
            min_if = min(b.in_flight for b in ranked)
            if ranked[0].in_flight > OVERLOAD_FACTOR * (min_if + 1):
                least = min(ranked, key=lambda b: b.in_flight)
                ranked.remove(least)
                ranked.insert(0, least)
        return ranked

    # ---- proxy --------------------------------------------------------------
    async def forward(self, path: str, payload: dict) -> JSONResponse:
        self.reload_state()
        key = self.affinity_key(payload, path)
        candidates = self.ranked(key)[:3]  # primary + two retries
        if not candidates:
            return JSONResponse({"error": "no teacher backends"}, 503)
        is_sample = int(payload.get("max_tokens") or 0) > 1
        timeout = SAMPLE_TIMEOUT_S if is_sample else ECHO_TIMEOUT_S
        last_err = ""
        for b in candidates:
            b.in_flight += 1
            try:
                r = await self.http.post(
                    f"{b.url}{path}", json=payload, timeout=timeout,
                    headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"{r.status_code}", request=r.request, response=r)
                b.record(True)
                self.done.append((time.monotonic(), is_sample))
                return JSONResponse(r.json(), r.status_code)
            except (httpx.HTTPError, ValueError) as e:
                b.record(False)
                last_err = f"{b.pod}: {e!r}"
            finally:
                b.in_flight -= 1
        return JSONResponse({"error": f"all backends failed: {last_err}"}, 502)

    # ---- metrics --------------------------------------------------------------
    def rates(self) -> dict:
        now = time.monotonic()
        windows = {"10s": 10.0, "60s": 60.0, "300s": 300.0}
        out = {}
        for label, w in windows.items():
            recent = [s for t, s in self.done if now - t <= w]
            out[label] = {
                "req_per_s": round(len(recent) / w, 3),
                "samples_per_s": round(sum(recent) / w, 3),
                "echoes_per_s": round((len(recent) - sum(recent)) / w, 3),
            }
        return out


router = Router()
app = FastAPI()


@app.get("/health")
async def health():
    router.reload_state()
    healthy = [b for b in router.backends.values() if b.healthy]
    return {"ok": bool(healthy), "backends": len(router.backends),
            "healthy": len(healthy)}


@app.get("/v1/models")
async def models():
    router.reload_state()
    return {"object": "list",
            "data": [{"id": router.model, "object": "model",
                      "owned_by": "teacher-swarm"}]}


@app.post("/v1/completions")
async def completions(request: Request):
    return await router.forward("/completions", await request.json())


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await router.forward("/chat/completions", await request.json())


@app.get("/metrics")
async def metrics():
    backends = [{
        "url": b.url, "pod": b.pod, "type": b.type,
        "healthy": b.healthy, "in_flight": b.in_flight,
        "ok": b.ok_count, "err": b.err_count,
    } for b in router.backends.values()]
    return {"rates": router.rates(), "backends": backends}


def main() -> int:
    ap = argparse.ArgumentParser()
    # Loopback only: consumers arrive via SSH reverse tunnels, and the proxy
    # would otherwise expose our rented teacher fleet to the internet.
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9100)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
