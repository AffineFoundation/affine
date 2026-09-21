#!/usr/bin/env python3
"""End-to-end checks for the public king chat endpoint (OpenAI wire plane).

    python ops/king-chat/smoke.py [--base https://chat.affine.io/v1] [--key sk-affine-king]

Stdlib only, so it runs from any machine. Each check prints PASS/FAIL and
the script exits nonzero if any check failed. Checks:

  models      GET /v1/models lists the "affine-king" alias
  chat        plain completion: visible text, no <think> leaking into content
  reasoning   a reasoning-heavy prompt returns reasoning AND closes the think
              block (visible content non-empty) — the wvk-13 contract
  tools       a tool-enabled request yields a parsed tool_call (qwen3_xml)
  stream      SSE streaming delivers content deltas and a [DONE] sentinel
  system-mid  a system message in the MIDDLE of the thread (what Cursor
              sends) is accepted — chatsrv folds it to the front
  cursor      the exact Cursor "verify" shape: model=affine-king, a short
              user message, stream=true
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

MODEL = "affine-king"


def _call(base: str, key: str, path: str, body: dict | None = None,
          stream: bool = False, timeout: float = 300.0):
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    # Cloudflare's browser integrity check 403s the default "Python-urllib"
    # agent (error 1010); SDK agents (OpenAI/Python, python-requests, Cursor)
    # pass, so name ourselves.
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json",
                                          "User-Agent": "affine-chat-smoke/1.0"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return resp
    return json.loads(resp.read().decode())


def _reasoning_of(msg: dict) -> str:
    # vLLM versions differ on the field name.
    return str(msg.get("reasoning") or msg.get("reasoning_content") or "")


def check_models(base, key):
    r = _call(base, key, "/models")
    ids = [m["id"] for m in r.get("data", [])]
    assert MODEL in ids, f"alias missing: {ids}"
    return f"ids={ids[:2]}"


def check_chat(base, key):
    r = _call(base, key, "/chat/completions", {
        "model": MODEL, "max_tokens": 400, "temperature": 0.2,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}]})
    msg = r["choices"][0]["message"]
    content = msg.get("content") or ""
    assert content.strip(), f"empty content: {json.dumps(r)[:300]}"
    assert "<think>" not in content and "</think>" not in content, \
        f"think tags leaked into content: {content[:120]!r}"
    return f"content={content.strip()[:60]!r} finish={r['choices'][0].get('finish_reason')}"


def check_reasoning(base, key):
    r = _call(base, key, "/chat/completions", {
        "model": MODEL, "max_tokens": 4000, "temperature": 0.6,
        "messages": [{"role": "user", "content":
                      "A bat and a ball cost $1.10 in total. The bat costs $1.00 "
                      "more than the ball. How much does the ball cost? Think it "
                      "through, then answer in one sentence."}]})
    ch = r["choices"][0]
    msg = ch["message"]
    content = (msg.get("content") or "").strip()
    reasoning = _reasoning_of(msg)
    assert content, (f"no visible text after reasoning (finish={ch.get('finish_reason')}, "
                     f"reasoning_chars={len(reasoning)}) — think block never closed?")
    assert "</think>" not in content, "closing tag leaked into visible content"
    return (f"reasoning_chars={len(reasoning)} content={content[:70]!r} "
            f"finish={ch.get('finish_reason')}")


def check_tools(base, key):
    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    r = _call(base, key, "/chat/completions", {
        "model": MODEL, "max_tokens": 2000, "temperature": 0.2, "tools": tools,
        "tool_choice": "auto",
        "messages": [{"role": "user",
                      "content": "What's the weather in Lisbon right now? Use the tool."}]})
    ch = r["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    assert calls, f"no tool_calls parsed (finish={ch.get('finish_reason')}): " \
                  f"{(ch['message'].get('content') or '')[:200]!r}"
    fn = calls[0]["function"]
    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
    assert fn["name"] == "get_weather" and "city" in args, f"odd call: {calls[0]}"
    return f"tool_call={fn['name']}({args}) finish={ch.get('finish_reason')}"


def check_stream(base, key):
    resp = _call(base, key, "/chat/completions", {
        "model": MODEL, "max_tokens": 300, "stream": True, "temperature": 0.2,
        "messages": [{"role": "user", "content": "Count from 1 to 10, separated by spaces."}]},
        stream=True)
    chunks = content = 0
    done = False
    text = ""
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        j = json.loads(data)
        if "error" in j:
            raise AssertionError(f"stream error: {j['error']}")
        chunks += 1
        delta = (j.get("choices") or [{}])[0].get("delta") or {}
        if delta.get("content"):
            content += 1
            text += delta["content"]
    assert done, "no [DONE] sentinel"
    assert content > 0, f"no content deltas in {chunks} chunks"
    return f"chunks={chunks} content_deltas={content} text={text.strip()[:40]!r}"


def check_system_mid(base, key):
    r = _call(base, key, "/chat/completions", {
        "model": MODEL, "max_tokens": 300, "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hi."},
            {"role": "assistant", "content": "Hi."},
            {"role": "system", "content": "From now on answer in uppercase."},
            {"role": "user", "content": "Say hi again."}]})
    content = (r["choices"][0]["message"].get("content") or "").strip()
    assert content, "empty content"
    return f"content={content[:40]!r}"


def check_cursor(base, key):
    # What Cursor's "Verify" button sends: model id, one user message, stream.
    resp = _call(base, key, "/chat/completions", {
        "model": MODEL, "stream": True, "max_tokens": 64,
        "messages": [{"role": "user", "content": "Test prompt using gpt-4o-mini"}]},
        stream=True)
    first = None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            first = json.loads(line[5:].strip())
            break
    assert first and "choices" in first, f"bad first chunk: {first}"
    return f"first_chunk_model={first.get('model')!r}"


CHECKS = [("models", check_models), ("chat", check_chat),
          ("reasoning", check_reasoning), ("tools", check_tools),
          ("stream", check_stream), ("system-mid", check_system_mid),
          ("cursor", check_cursor)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--base", default="https://chat.affine.io/v1")
    ap.add_argument("--key", default="sk-affine-king")
    ap.add_argument("--only", action="append", help="run only these checks")
    args = ap.parse_args()
    failed = 0
    for name, fn in CHECKS:
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        try:
            info = fn(args.base, args.key)
            print(f"PASS {name:<11} {time.time() - t0:5.1f}s  {info}")
        except urllib.error.HTTPError as e:
            failed += 1
            body = e.read().decode("utf-8", "replace")[:300]
            print(f"FAIL {name:<11} {time.time() - t0:5.1f}s  HTTP {e.code}: {body}")
        except Exception as e:  # noqa: BLE001 — report every failure kind
            failed += 1
            print(f"FAIL {name:<11} {time.time() - t0:5.1f}s  {type(e).__name__}: {e}")
    print(f"{'ALL PASS' if not failed else f'{failed} FAILED'} against {args.base}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
