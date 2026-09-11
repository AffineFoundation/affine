"""Public king-chat server — FastAPI app on the chat pod (AFFINE_ROLE=chat).

Serves the CURRENT KING over a streaming chat endpoint for the affine.io
website. Ops-only: nothing here touches scoring, duels, or the chain.

  GET  /health   {ok, role: "chat", state, king, versions}
  POST /chat     token-gated; {messages:[{role,content}...]} → relays the
                 OpenAI-style SSE chunks from the local vLLM
  GET  /v1/models
  POST /v1/chat/completions
                 OpenAI-compatible wire plane (token via X-Affine-Token OR
                 Authorization: Bearer). The model id "affine-king" is a
                 stable alias that always routes to the current king, so
                 OpenAI clients (arbos, Cursor, curl) survive crowns without
                 reconfiguration. Requests are proxied verbatim to the local
                 vLLM with only the model field rewritten, system messages
                 folded to the front (the Qwen template rejects a system
                 message that is not first — Cursor sends them mid-thread)
                 and output caps applied.

Two keys open /v1: the pod's AFFINE_EVAL_TOKEN (the validator, the dash proxy
— unlimited) and a PUBLIC demo key (AFFINE_CHAT_PUBLIC_KEY, default
DEFAULT_PUBLIC_KEY) that the community plugs into Cursor. Public-key traffic
is rate-limited per client IP (CF-Connecting-IP through the Cloudflare
tunnel) and capped in flight; the GPU is the thing being protected, the key
is not a secret.

Pod-local overrides (AFFINE_CHAT_*) come from /root/affine/.chat_env, written
by ops/king-chat/chatbox.sh; they exist so the public box can serve IDE-sized
contexts without touching the [chat] knobs the website chat is sized for.

King tracking: the pod polls the public dash snapshot ([chat].snapshot_url)
and swaps the vLLM slot when the king's repo@revision changes. /health stays
ok=true throughout (state="loading") — a 65 GB king download takes far longer
than the provisioner's unhealthy_threshold, so readiness must not gate ok or
every crown would get the pod terminated mid-swap.

Engine reuse: the shared Engine (role="chat") gives us the single-slot
layout, the vLLM launch flags Blackwell/GDN kings need, GPU orphan sweeps,
and disk pruning. The king rides the challenger slot; load_challenger()
already does prune → launch → wait-ready.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import threading
import time

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import Response, StreamingResponse

from affine.config import load_config

from .engine import Engine

log = logging.getLogger("chatsrv")

EVAL_TOKEN = os.environ.get("AFFINE_EVAL_TOKEN", "")
# Intentionally public: it is printed in the community announcement. Its
# only job is to make stock OpenAI clients (which insist on a key) work and
# to let the rate limiter tell public traffic from the dash/validator.
DEFAULT_PUBLIC_KEY = "sk-affine-king"
PUBLIC_KEY = os.environ.get("AFFINE_CHAT_PUBLIC_KEY", DEFAULT_PUBLIC_KEY)
PUBLIC_RATE_PER_MIN = int(os.environ.get("AFFINE_CHAT_PUBLIC_RATE_PER_MIN", "60"))
PUBLIC_MAX_CONCURRENCY = int(os.environ.get("AFFINE_CHAT_PUBLIC_MAX_CONCURRENCY", "8"))
UPSTREAM_TIMEOUT_S = 600.0

app = FastAPI(title="affine-chatsrv")

_cfg = load_config()
_chat = _cfg.chat
_engine = Engine(_cfg.raw)

# Single writer (the watcher thread); handlers only read a copied dict.
_state_lock = threading.Lock()
_state: dict = {"state": "waiting_for_king", "repo": "", "revision": "",
                "reign": None, "error": ""}

MAX_MESSAGES = 40


def _set_state(**kv) -> None:
    with _state_lock:
        _state.update(kv)


def _get_state() -> dict:
    with _state_lock:
        return dict(_state)


def _max_output_tokens() -> int:
    return int(os.environ.get("AFFINE_CHAT_MAX_OUTPUT_TOKENS")
               or _chat.get("max_output_tokens", 1024))


def _bearer(authorization: str) -> str:
    return authorization.removeprefix("Bearer ").strip()


def _require_token(x_affine_token: str = Header(default=""),
                   authorization: str = Header(default="")) -> None:
    """Operator gate (/chat): the pod's eval token only, as X-Affine-Token or
    `Authorization: Bearer`."""
    if not EVAL_TOKEN:
        return
    if x_affine_token == EVAL_TOKEN or _bearer(authorization) == EVAL_TOKEN:
        return
    raise HTTPException(401, "bad or missing token")


def _require_v1_key(x_affine_token: str = Header(default=""),
                    authorization: str = Header(default="")) -> bool:
    """/v1 gate. Returns True when the caller used the PUBLIC key (rate
    limited), False for the eval token (dash proxy / operator, unlimited)."""
    if EVAL_TOKEN and (x_affine_token == EVAL_TOKEN
                       or _bearer(authorization) == EVAL_TOKEN):
        return False
    if PUBLIC_KEY and _bearer(authorization) == PUBLIC_KEY:
        return True
    if not EVAL_TOKEN and not PUBLIC_KEY:
        return False
    raise HTTPException(401, "bad or missing API key")


# -- public-traffic limiter ------------------------------------------------------
# Sliding per-IP window + global in-flight cap, public key only. Behind the
# Cloudflare tunnel every connection arrives from 127.0.0.1, so the client
# is identified by CF-Connecting-IP (X-Forwarded-For as a fallback).

_limit_lock = threading.Lock()
_ip_hits: dict[str, list[float]] = {}
_public_active = 0


def _client_ip(request: Request) -> str:
    h = request.headers
    ip = h.get("cf-connecting-ip") or (h.get("x-forwarded-for") or "").split(",")[0]
    return ip.strip() or (request.client.host if request.client else "?")


def _public_admit(request: Request) -> None:
    """Reserve one public in-flight slot or raise 429. Pair with _public_release."""
    global _public_active
    ip = _client_ip(request)
    now = time.monotonic()
    with _limit_lock:
        if len(_ip_hits) > 5000:
            for k in [k for k, v in _ip_hits.items() if not v or now - v[-1] > 60.0]:
                _ip_hits.pop(k, None)
        hits = [t for t in _ip_hits.get(ip, []) if now - t < 60.0]
        if len(hits) >= PUBLIC_RATE_PER_MIN:
            _ip_hits[ip] = hits
            raise HTTPException(
                429, f"rate limited: {PUBLIC_RATE_PER_MIN} requests/minute per client")
        if _public_active >= PUBLIC_MAX_CONCURRENCY:
            raise HTTPException(
                429, f"busy: {PUBLIC_MAX_CONCURRENCY} public requests already in flight")
        hits.append(now)
        _ip_hits[ip] = hits
        _public_active += 1


def _public_release() -> None:
    global _public_active
    with _limit_lock:
        _public_active = max(0, _public_active - 1)


# -- request shaping -------------------------------------------------------------

def _text_of(content) -> str:
    """Flatten OpenAI content (string or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content
                         if isinstance(p, dict) and p.get("type", "text") == "text")
    return "" if content is None else str(content)


def _fold_system_messages(messages: list) -> list:
    """One leading system message. The Qwen chat template raises
    "System message must be at the beginning" for anything else; Cursor (and
    other IDE agents) inject `system`/`developer` messages mid-thread, which
    turned into vLLM 400s on the operator's private king box. Untouched when
    the thread already has at most one system message and it is first."""
    sys_idx = [i for i, m in enumerate(messages)
               if isinstance(m, dict) and m.get("role") in ("system", "developer")]
    if not sys_idx or (sys_idx == [0] and messages[0].get("role") == "system"):
        return messages
    system_text = "\n\n".join(
        t for t in (_text_of(messages[i].get("content")) for i in sys_idx) if t.strip())
    rest = [m for i, m in enumerate(messages) if i not in set(sys_idx)]
    return [{"role": "system", "content": system_text}] + rest


def _stack_versions() -> dict:
    out = {}
    for pkg in ("vllm", "torch", "transformers"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


# -- king watcher ----------------------------------------------------------------

def _fetch_king() -> dict | None:
    """Current king {repo, revision, reign_number} from the public snapshot."""
    url = str(_chat.get("snapshot_url", "https://affine.io/api/v1/snapshot"))
    # affine.io's first byte is often 10-15 s (Cloudflare -> validator box ->
    # dash under cache rebuilds); a 15 s timeout made whole polls fail.
    r = httpx.get(url, timeout=60, follow_redirects=True)
    r.raise_for_status()
    king = (r.json() or {}).get("king") or {}
    if king.get("repo") and king.get("revision"):
        return king
    return None


def _swap_to(king: dict) -> None:
    repo, revision = str(king["repo"]), str(king["revision"])
    reign = king.get("reign_number")
    log.info("loading king reign=%s %s@%s", reign, repo, revision[:12])
    _set_state(state="loading", repo=repo, revision=revision, reign=reign,
               error="")
    if _engine.load_challenger(repo, revision):
        log.info("king serving: %s@%s", repo, revision[:12])
        _set_state(state="serving", error="")
    else:
        err = _engine.chall_slot.load_error or "vllm load failed"
        log.error("king load failed: %s", err[:300])
        _set_state(state="error", error=err[:500])


def _watch_king() -> None:
    interval = int(_chat.get("poll_interval_s", 60))
    while True:
        try:
            king = _fetch_king()
            if king is not None:
                st = _get_state()
                changed = (str(king["repo"]), str(king["revision"])) != \
                          (st["repo"], st["revision"])
                # Also retry a failed load each poll: transient pod trouble
                # (disk, HF hiccup) must not leave chat dark until next crown.
                if changed or st["state"] == "error":
                    _swap_to(king)
                elif st["state"] == "serving" and not _engine.challenger_alive():
                    log.warning("serving vLLM went dark; relaunching")
                    _swap_to(king)
        except Exception as e:
            log.warning("king poll failed: %s: %s", type(e).__name__, e)
        threading.Event().wait(interval)


# -- routes ----------------------------------------------------------------------

@app.get("/health")
def health():
    # Open: nothing here is secret (the king ref is public), and the public
    # hostname needs a status probe the community can read.
    st = _get_state()
    return {
        "ok": True,
        "role": "chat",
        "state": st["state"],
        "king": {"repo": st["repo"], "revision": st["revision"],
                 "reign_number": st["reign"]},
        "error": st["error"],
        "versions": _stack_versions(),
    }


class ChatRequest(BaseModel):
    messages: list[dict]
    temperature: float | None = None
    max_tokens: int | None = None


@app.post("/chat")
async def chat(req: ChatRequest, _: None = Depends(_require_token)):
    st = _get_state()
    if st["state"] != "serving":
        raise HTTPException(503, detail=json.dumps(
            {"state": st["state"], "error": st["error"]}))

    if not req.messages or len(req.messages) > MAX_MESSAGES:
        raise HTTPException(400, "messages must have 1..%d items" % MAX_MESSAGES)
    max_chars = int(_chat.get("max_input_chars", 16000))
    total = 0
    msgs = []
    for m in req.messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        if role not in ("system", "user", "assistant") or not content.strip():
            raise HTTPException(400, "bad message role/content")
        total += len(content)
        msgs.append({"role": role, "content": content})
    if total > max_chars:
        raise HTTPException(400, f"conversation too long (>{max_chars} chars)")

    max_out = _max_output_tokens()
    temperature = req.temperature if req.temperature is not None else 0.7
    payload = {
        "model": st["repo"],
        "messages": msgs,
        "stream": True,
        "max_tokens": min(int(req.max_tokens or max_out), max_out),
        "temperature": max(0.0, min(float(temperature), 1.5)),
    }
    url = f"http://localhost:{_engine.chall_slot.port}/v1/chat/completions"

    async def relay():
        try:
            timeout = httpx.Timeout(300.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload) as r:
                    if r.status_code != 200:
                        body = (await r.aread()).decode("utf-8", "replace")
                        yield ("data: " + json.dumps(
                            {"error": f"upstream {r.status_code}: {body[:300]}"}
                        ) + "\n\n").encode()
                        return
                    async for chunk in r.aiter_bytes():
                        yield chunk
        except Exception as e:
            yield ("data: " + json.dumps(
                {"error": f"{type(e).__name__}: {e}"}) + "\n\n").encode()

    return StreamingResponse(relay(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


# -- OpenAI-compatible wire plane --------------------------------------------
# "affine-king" is the stable public alias; the concrete repo id also works.

KING_ALIAS = "affine-king"


@app.get("/v1/models")
def v1_models(_: bool = Depends(_require_v1_key)):
    st = _get_state()
    data = [{"id": KING_ALIAS, "object": "model", "owned_by": "affine",
             "root": st["repo"] or None}]
    if st["repo"]:
        data.append({"id": st["repo"], "object": "model", "owned_by": "affine"})
    return {"object": "list", "data": data}


def _shape_v1_payload(payload: dict, st: dict) -> dict:
    # The alias (or anything else the client sent) maps to the current king;
    # vLLM only accepts its served ids.
    payload["model"] = st["repo"]
    payload["messages"] = _fold_system_messages(list(payload["messages"]))
    # Output cap. Clients send max_tokens or (newer OpenAI) max_completion_tokens;
    # vLLM treats the latter as an alias, so normalize to one field.
    max_out = _max_output_tokens()
    req_max = payload.pop("max_completion_tokens", None)
    req_max = payload.get("max_tokens") or req_max or max_out
    try:
        req_max = int(req_max)
    except (TypeError, ValueError):
        req_max = max_out
    payload["max_tokens"] = max(1, min(req_max, max_out))
    return payload


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request,
                              public: bool = Depends(_require_v1_key)):
    st = _get_state()
    if st["state"] != "serving":
        raise HTTPException(503, detail=json.dumps(
            {"state": st["state"], "error": st["error"]}))
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "invalid JSON body")
    if not isinstance(payload, dict) or not payload.get("messages"):
        raise HTTPException(400, "messages required")
    payload = _shape_v1_payload(payload, st)

    if public:
        _public_admit(request)
    url = f"http://localhost:{_engine.chall_slot.port}/v1/chat/completions"
    timeout = httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=10.0)
    if payload.get("stream"):
        async def relay():
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("POST", url, json=payload) as r:
                        if r.status_code != 200:
                            body = (await r.aread()).decode("utf-8", "replace")
                            yield ("data: " + json.dumps(
                                {"error": {"message":
                                           f"upstream {r.status_code}: {body[:300]}"}}
                            ) + "\n\n").encode()
                            return
                        async for chunk in r.aiter_bytes():
                            yield chunk
            except Exception as e:
                yield ("data: " + json.dumps(
                    {"error": {"message": f"{type(e).__name__}: {e}"}}
                ) + "\n\n").encode()
            finally:
                if public:
                    _public_release()

        return StreamingResponse(relay(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload)
    finally:
        if public:
            _public_release()
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type",
                                             "application/json"))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    threading.Thread(target=_watch_king, daemon=True,
                     name="king-watcher").start()
    port = int(os.environ.get("AFFINE_EVAL_PORT", "9002"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
