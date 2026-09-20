"""Tiny OpenAI-wire proxy in front of vLLM on the private king-chat pod.

One job: fold every `system` / `developer` message of a chat request into
ONE leading system message before vLLM renders the Qwen chat template. That
template raises "System message must be at the beginning" for anything else,
and Cursor (like other IDE agents) injects system messages mid-thread — every
such turn was a vLLM 400 on the operator's Cursor box. Same rule as
`affine/evalsrv/chatsrv.py::_fold_system_messages` on the public chat box.

Everything else (models, completions, Anthropic /v1/messages, health, the
Authorization header, streaming) passes through byte for byte.

    fold_proxy.py --listen 127.0.0.1:8001 --upstream http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection"}
app = FastAPI()
upstream = "http://127.0.0.1:8000"


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


async def relay(method: str, path: str, headers: dict, body: bytes, query: str):
    """Stream the upstream response back with its status and headers."""
    client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=10.0))
    url = f"{upstream}{path}" + (f"?{query}" if query else "")
    req = client.build_request(method, url, headers=headers, content=body)
    r = await client.send(req, stream=True)
    resp_headers = {k: v for k, v in r.headers.items()
                    if k.lower() not in HOP_HEADERS | {"content-encoding"}}

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
    if request.method == "POST" and request.url.path.endswith("/chat/completions"):
        try:
            payload = json.loads(body)
        except ValueError:
            return Response(b'{"error":"invalid JSON body"}', status_code=400,
                            media_type="application/json")
        if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
            payload["messages"] = fold_system_messages(payload["messages"])
            body = json.dumps(payload).encode()
            headers["content-type"] = "application/json"
    return await relay(request.method, request.url.path, headers, body, request.url.query)


def main() -> None:
    global upstream
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--listen", default="127.0.0.1:8001")
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    upstream = args.upstream.rstrip("/")
    host, port = args.listen.rsplit(":", 1)
    uvicorn.run(app, host=host, port=int(port), log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
