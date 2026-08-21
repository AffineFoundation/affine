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
                 vLLM with only the model field rewritten and output caps
                 applied.

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

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import Response, StreamingResponse

from affine.config import load_config

from .engine import Engine

log = logging.getLogger("chatsrv")

EVAL_TOKEN = os.environ.get("AFFINE_EVAL_TOKEN", "")

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


def _require_token(x_affine_token: str = Header(default=""),
                   authorization: str = Header(default="")) -> None:
    """Shared-secret gate. Accepts the native X-Affine-Token header or an
    OpenAI-style `Authorization: Bearer <token>` so stock OpenAI clients
    work against /v1/* without customization."""
    if not EVAL_TOKEN:
        return
    if x_affine_token == EVAL_TOKEN:
        return
    if authorization.removeprefix("Bearer ").strip() == EVAL_TOKEN:
        return
    raise HTTPException(401, "bad or missing token")


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
    r = httpx.get(url, timeout=15, follow_redirects=True)
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
def health(_: None = Depends(_require_token)):
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

    max_out = int(_chat.get("max_output_tokens", 1024))
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
def v1_models(_: None = Depends(_require_token)):
    st = _get_state()
    data = [{"id": KING_ALIAS, "object": "model", "owned_by": "affine",
             "root": st["repo"] or None}]
    if st["repo"]:
        data.append({"id": st["repo"], "object": "model", "owned_by": "affine"})
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request,
                              _: None = Depends(_require_token)):
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
    # The alias (or anything else the client sent) maps to the current king;
    # vLLM only accepts the exact served repo id.
    payload["model"] = st["repo"]
    max_out = int(_chat.get("max_output_tokens", 1024))
    try:
        req_max = int(payload.get("max_tokens") or max_out)
    except (TypeError, ValueError):
        req_max = max_out
    payload["max_tokens"] = min(req_max, max_out)

    url = f"http://localhost:{_engine.chall_slot.port}/v1/chat/completions"
    if payload.get("stream"):
        async def relay():
            try:
                timeout = httpx.Timeout(600.0, connect=10.0)
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

        return StreamingResponse(relay(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    timeout = httpx.Timeout(600.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
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
