"""Tiny OpenAI-wire proxy in front of vLLM on the private king-chat pod.

One job: fold every `system` / `developer` message of a chat request into
ONE leading system message before vLLM renders the Qwen chat template. That
template raises "System message must be at the beginning" for anything else,
and Cursor (like other IDE agents) injects system messages mid-thread — every
such turn was a vLLM 400 on the operator's Cursor box. Same rule as
`affine/evalsrv/chatsrv.py::_fold_system_messages` on the public chat box.

Two more things Cursor needs: (a) any model id other than the served one is
rewritten to it — Cursor's "Verify" button (and some picker entries) call
/chat/completions with an OpenAI model id such as `gpt-4o`, which vLLM would
404; (b) one access-log line per request (time, method, path, requested
model, user agent, status) on stderr, so "what did Cursor actually send" can
be answered from /root/logs/proxy.log.

Everything else (models, completions, Anthropic /v1/messages, health, the
Authorization header, streaming) passes through byte for byte.

    fold_proxy.py --listen 127.0.0.1:8001 --upstream http://127.0.0.1:8000 \
        --served-model affine-king
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection"}
app = FastAPI()
upstream = "http://127.0.0.1:8000"
served_model = "affine-king"


def access_log(request: Request, status: int, model: str, note: str = "") -> None:
    ua = request.headers.get("user-agent", "-")[:60]
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {request.method} "
          f"{request.url.path} model={model or '-'} ua={ua!r} -> {status} {note}",
          file=sys.stderr, flush=True)


def text_of(content) -> str:
    """OpenAI content is a string or a list of parts; keep the text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text") or "") for p in content
            if isinstance(p, dict) and p.get("type", "text") == "text")
    return ""


def fold_system_messages(messages: list) -> list:
    sys_idx = [i for i, m in enumerate(messages)
               if isinstance(m, dict) and m.get("role") in ("system", "developer")]
    if not sys_idx or (sys_idx == [0] and messages[0].get("role") == "system"):
        return messages
    system_text = "\n\n".join(
        t for t in (text_of(messages[i].get("content")) for i in sys_idx) if t.strip())
    rest = [m for i, m in enumerate(messages) if i not in set(sys_idx)]
    return [{"role": "system", "content": system_text}] + rest


def forward_headers(request: Request) -> dict:
    return {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}


async def relay(request: Request, headers: dict, body: bytes, model: str, note: str):
    """Stream the upstream response back with its status and headers."""
    client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=10.0))
    query = request.url.query
    url = f"{upstream}{request.url.path}" + (f"?{query}" if query else "")
    req = client.build_request(request.method, url, headers=headers, content=body)
    r = await client.send(req, stream=True)
    resp_headers = {k: v for k, v in r.headers.items()
                    if k.lower() not in HOP_HEADERS | {"content-encoding"}}
    if r.status_code >= 400:
        err = (await r.aread())[:300].decode("utf-8", "replace")
        access_log(request, r.status_code, model, f"{note} upstream={err!r}")
        await r.aclose()
        await client.aclose()
        return Response(err.encode(), status_code=r.status_code, headers=resp_headers)
    access_log(request, r.status_code, model, note)

    async def body_iter():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()

    return StreamingResponse(body_iter(), status_code=r.status_code, headers=resp_headers)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str):
    body = await request.body()
    headers = forward_headers(request)
    model, note = "", ""
    if request.method == "POST" and request.url.path.endswith(("/chat/completions", "/completions")):
        try:
            payload = json.loads(body)
        except ValueError:
            access_log(request, 400, "", "invalid JSON body")
            return Response(b'{"error":"invalid JSON body"}', status_code=400,
                            media_type="application/json")
        if isinstance(payload, dict):
            model = str(payload.get("model") or "")
            if model and model != served_model:
                payload["model"] = served_model
                note = f"aliased->{served_model}"
            if isinstance(payload.get("messages"), list):
                payload["messages"] = fold_system_messages(payload["messages"])
            body = json.dumps(payload).encode()
            headers["content-type"] = "application/json"
    return await relay(request, headers, body, model, note)


def main() -> None:
    global upstream, served_model
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--listen", default="127.0.0.1:8001")
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--served-model", default="affine-king",
                    help="every other model id in a request is rewritten to this")
    args = ap.parse_args()
    upstream = args.upstream.rstrip("/")
    served_model = args.served_model
    host, port = args.listen.rsplit(":", 1)
    uvicorn.run(app, host=host, port=int(port), log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
