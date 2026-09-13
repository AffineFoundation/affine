#!/usr/bin/env python
"""Coaching proxy: an OpenAI-/Anthropic-compatible HTTP relay in front of the
teacher endpoint that appends a fresh coach note to every request of a
COACHED continuation and passes PLAIN continuations through untouched.

Route (the harness's `--client.base-url`):
    http://127.0.0.1:PORT/u/<unit_stem>/c<k>/<arm>/v1
    arm = coached | plain ; k = continuation index ; unit_stem = state id with
    ':' -> '_' (research/hints/coached/select_states.py writes one coach
    context per unit under $COACH_CTX_DIR/<unit_stem>.json).

Per coached request:
  1. locate the continuation boundary in the request's messages (the prefix's
     last observation), compress the current run since then;
  2. ask the coach (DeepSeek v4-pro via OpenRouter, T = 0, JSON) for fact /
     plan / action notes given the king's whole failed rollout + outcome +
     post-mortem + the current run;
  3. gate the composed note (grounding in the visible context, no future leak;
     one rewrite attempt when entities are missing); drop it when it fails;
  4. append the note to the last user / tool message and forward upstream;
  5. log everything to $COACH_LOG_DIR/hints.jsonl (one line per request).
Budget guards: COACH_MAX_USD (whole run), COACH_MAX_STEPS (per continuation);
past either, requests pass through unhinted and are logged as such.

  COACH_UPSTREAM=https://api.engy.ai/v1 OPENROUTER_API_KEY=... \
    python coach_proxy.py --port 8765 --ctx-dir CTX --log-dir LOG
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import common as C  # noqa: E402
import hints as H  # noqa: E402

log = logging.getLogger("coach_proxy")
HOP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "keep-alive",
               "accept-encoding"}
COACH_MODEL = os.environ.get("COACH_MODEL", H.DEEPSEEK_MODEL)


class Coach:
    def __init__(self, ctx_dir: Path, log_dir: Path, inject: str, max_usd: float,
                 max_steps: int, upstream: str, teacher_key: str | None):
        self.ctx_dir = ctx_dir
        self.log_dir = log_dir
        self.inject = inject
        self.max_usd = max_usd
        self.max_steps = max_steps
        self.upstream = upstream.rstrip("/")
        self.teacher_key = teacher_key
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        self.lock = threading.Lock()
        self.spent = 0.0
        self.n_calls = 0
        self.steps: dict[str, int] = {}
        self.ctx_cache: dict[str, dict] = {}
        self.http = httpx.Client(timeout=httpx.Timeout(900.0, connect=30.0),
                                 limits=httpx.Limits(max_connections=256))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "hints.jsonl"
        self.calls_path = log_dir / "coach_calls.jsonl"
        for line in self.log_path.read_text().split("\n") if self.log_path.exists() else []:
            if line.strip():
                r = json.loads(line)
                self.spent += float((r.get("coach") or {}).get("cost_usd") or 0)

    def ctx(self, unit_stem: str) -> dict | None:
        with self.lock:
            if unit_stem in self.ctx_cache:
                return self.ctx_cache[unit_stem]
        p = self.ctx_dir / f"{unit_stem}.json"
        if not p.is_file():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        with self.lock:
            self.ctx_cache[unit_stem] = d
        return d

    def write(self, path: Path, row: dict) -> None:
        with self.lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -- the note ----------------------------------------------------------------

    def boundary(self, ctx: dict, messages: list[dict]) -> int:
        if ctx.get("resume_kind") == "same_task":
            return 0
        want = ctx.get("prefix_last_obs_sha")
        if want:
            for i in range(len(messages) - 1, -1, -1):
                m = messages[i]
                if m.get("role") in ("user", "tool") and C.sha(C.message_text(m.get("content"))) == want:
                    return i + 1
        n = int(ctx.get("n_prefix_assistant") or 0)
        seen = 0
        for i, m in enumerate(messages):
            if m.get("role") == "assistant":
                seen += 1
                if seen == n:
                    j = i + 1
                    while j < len(messages) and messages[j].get("role") in ("user", "tool"):
                        j += 1
                    return j
        return len(messages)

    def note_for(self, ctx: dict, messages: list[dict], key: str) -> dict:
        rec: dict = {"injected": False}
        b = self.boundary(ctx, messages)
        cont_text, n_cont = C.compress_wire(messages, b)
        rec["cont_turn"] = n_cont
        rec["boundary_idx"] = b
        with self.lock:
            step = self.steps.get(key, 0)
            self.steps[key] = step + 1
            if self.spent >= self.max_usd:
                rec["reason"] = "budget_exhausted"
                return rec
        if step >= self.max_steps:
            rec["reason"] = "max_steps"
            return rec
        if not self.key:
            rec["reason"] = "no_openrouter_key"
            return rec
        coach_msgs = C.build_coach_messages(ctx, cont_text, n_cont)
        raw = H.openrouter_hints(coach_msgs, self.key, model=COACH_MODEL, max_tokens=1500,
                                 timeout=180.0)
        cost = float(raw.get("cost_usd") or 0)
        with self.lock:
            self.spent += cost
            self.n_calls += 1
        rec["coach"] = {"model": raw.get("model"), "usage": raw.get("usage"), "cost_usd": cost,
                        "ms": raw.get("ms"), "error": raw.get("error"), "finish": raw.get("finish")}
        self.write(self.calls_path, {"key": key, "step": step, "messages": coach_msgs,
                                     "raw": raw})
        levels = H.parse_hints(raw.get("text", ""))
        if not levels:
            rec["reason"] = "coach_unparseable" if not raw.get("error") else "coach_error"
            return rec
        x_text = "\n".join(C.message_text(m.get("content")) for m in messages)
        future = list(ctx.get("king_future_actions") or [])
        note = C.compose_note(levels, self.inject)
        g = C.gate(note, x_text, future)
        rec["levels"] = levels
        rec["gate"] = g
        rec["rewrite"] = False
        if not g["grounding"]["grounded"]:
            # One rewrite with the missing entities named (cheap; the gate is
            # symbolic and strict about paths / identifiers).
            rw = coach_msgs + [{"role": "assistant", "content": json.dumps(levels)},
                               {"role": "user", "content": C.REWRITE_TMPL.format(
                                   missing=", ".join(g["grounding"]["missing"][:12]))}]
            raw2 = H.openrouter_hints(rw, self.key, model=COACH_MODEL, max_tokens=1500,
                                      timeout=180.0)
            cost2 = float(raw2.get("cost_usd") or 0)
            with self.lock:
                self.spent += cost2
                self.n_calls += 1
            rec["coach"]["cost_usd"] = cost + cost2
            rec["coach"]["rewrite_ms"] = raw2.get("ms")
            self.write(self.calls_path, {"key": key, "step": step, "rewrite": True,
                                         "messages": rw[-1:], "raw": raw2})
            lv2 = H.parse_hints(raw2.get("text", ""))
            if lv2:
                note2 = C.compose_note(lv2, self.inject)
                g2 = C.gate(note2, x_text, future)
                rec["rewrite"] = True
                rec["levels_first"] = levels
                rec["gate_first"] = g
                levels, note, g = lv2, note2, g2
                rec["levels"] = levels
                rec["gate"] = g
        if not g["passed"]:
            rec["reason"] = "gate_failed"
            rec["note"] = note
            return rec
        rec["note"] = note
        rec["injected"] = True
        return rec

    # -- request handling ------------------------------------------------------------

    def inject_note(self, body: dict, note: str) -> bool:
        msgs = body.get("messages")
        if not isinstance(msgs, list):
            return False
        for m in reversed(msgs):
            if m.get("role") not in ("user", "tool"):
                continue
            c = m.get("content")
            if isinstance(c, list):
                blocks = c
                # Anthropic: append a text block after the tool results; chat
                # parts: a text part.
                blocks.append({"type": "text", "text": C.HINT_HEADER.strip("\n") + "\n" + note})
                m["content"] = blocks
            else:
                m["content"] = (c or "") + C.HINT_HEADER + note
            return True
        return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    coach: Coach = None  # type: ignore[assignment]

    def log_message(self, fmt, *args):  # quiet
        pass

    def _route(self) -> tuple[str, int, str, str] | None:
        parts = self.path.split("?")[0].strip("/").split("/")
        # u/<unit>/c<k>/<arm>/v1/<rest...>
        if len(parts) < 5 or parts[0] != "u" or not parts[2].startswith("c") or parts[4] != "v1":
            return None
        try:
            k = int(parts[2][1:])
        except ValueError:
            return None
        rest = "/".join(parts[5:])
        return parts[1], k, parts[3], rest

    def _forward_headers(self) -> dict:
        out = {}
        for name, val in self.headers.items():
            if name.lower() in HOP_HEADERS:
                continue
            out[name] = val
        if self.coach.teacher_key:
            if "x-api-key" in {n.lower() for n in out}:
                out = {n: v for n, v in out.items() if n.lower() != "x-api-key"}
                out["x-api-key"] = self.coach.teacher_key
            else:
                out = {n: v for n, v in out.items() if n.lower() != "authorization"}
                out["Authorization"] = f"Bearer {self.coach.teacher_key}"
        return out

    def _send_json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        r = self._route()
        if r is None:
            if self.path.rstrip("/") == "/health":
                c = self.coach
                self._send_json(200, {"ok": True, "spent_usd": round(c.spent, 4),
                                      "coach_calls": c.n_calls})
                return
            self._send_json(404, {"error": "bad route"})
            return
        _, _, _, rest = r
        resp = self.coach.http.get(f"{self.coach.upstream}/{rest}", headers=self._forward_headers())
        self._relay(resp)

    def _relay(self, resp: httpx.Response) -> None:
        data = resp.content
        self.send_response(resp.status_code)
        for n, v in resp.headers.items():
            if n.lower() in HOP_HEADERS or n.lower() == "content-encoding":
                continue
            self.send_header(n, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        r = self._route()
        if r is None:
            self._send_json(404, {"error": "bad route"})
            return
        unit_stem, k, arm, rest = r
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        t0 = time.time()
        row: dict = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "unit": unit_stem,
                     "k": k, "arm": arm, "path": rest, "arm_key": f"{unit_stem}/c{k}/{arm}"}
        body = None
        try:
            body = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("messages"), list):
            msgs = body["messages"]
            row["n_messages"] = len(msgs)
            row["n_assistant"] = sum(1 for m in msgs if m.get("role") == "assistant")
            last = next((m for m in reversed(msgs) if m.get("role") in ("user", "tool")), None)
            row["last_msg_sha"] = C.sha(C.message_text(last.get("content"))) if last else None
            row["request_sha"] = C.sha(raw.decode("utf-8", "replace"))
            row["stream"] = bool(body.get("stream"))
            row["dialect"] = "anthropic" if rest.endswith("messages") else (
                "responses" if rest.endswith("responses") else "chat")
            if arm == "coached" and rest.endswith(("chat/completions", "messages")):
                ctx = self.coach.ctx(unit_stem)
                if ctx is None:
                    row["injected"] = False
                    row["reason"] = "no_ctx"
                else:
                    try:
                        rec = self.coach.note_for(ctx, msgs, row["arm_key"])
                    except Exception as e:  # noqa: BLE001 - the teacher call must go on
                        log.exception("coach failed for %s", row["arm_key"])
                        rec = {"injected": False, "reason": f"coach_exception: {type(e).__name__}: {e}"[:300]}
                    row.update(rec)
                    if rec.get("injected") and rec.get("note"):
                        body = copy.deepcopy(body)
                        if not self.coach.inject_note(body, rec["note"]):
                            row["injected"] = False
                            row["reason"] = "no_user_or_tool_message"
                        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            else:
                row["injected"] = False
                row["reason"] = "plain" if arm == "plain" else "not_a_completion"
        row["coach_ms"] = int((time.time() - t0) * 1000)
        headers = self._forward_headers()
        headers["Content-Type"] = "application/json"
        url = f"{self.coach.upstream}/{rest}"
        stream = bool(isinstance(body, dict) and body.get("stream"))
        try:
            if stream:
                with self.coach.http.stream("POST", url, content=raw, headers=headers) as resp:
                    self.send_response(resp.status_code)
                    for n, v in resp.headers.items():
                        if n.lower() in HOP_HEADERS or n.lower() == "content-encoding":
                            continue
                        self.send_header(n, v)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    nbytes = 0
                    for chunk in resp.iter_bytes():
                        if not chunk:
                            continue
                        self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                        nbytes += len(chunk)
                    self.wfile.write(b"0\r\n\r\n")
                    row["upstream_status"] = resp.status_code
                    row["upstream_bytes"] = nbytes
            else:
                resp = self.coach.http.post(url, content=raw, headers=headers)
                row["upstream_status"] = resp.status_code
                try:
                    d = resp.json()
                    ch = (d.get("choices") or [{}])[0] if isinstance(d, dict) else {}
                    msg = ch.get("message") or {}
                    text = C.message_text(msg.get("content")) if msg else C.message_text(d.get("content"))
                    row["reply_sha"] = C.sha(text)
                    row["reply_head"] = C.norm(text)[:200]
                    row["finish"] = ch.get("finish_reason") or d.get("stop_reason")
                    row["teacher_usage"] = d.get("usage")
                except (ValueError, AttributeError):
                    pass
                self._relay(resp)
        except httpx.HTTPError as e:
            row["upstream_error"] = repr(e)[:300]
            try:
                self._send_json(502, {"error": {"message": f"upstream: {e!r}", "type": "proxy"}})
            except (BrokenPipeError, ConnectionResetError):
                pass
        row["total_ms"] = int((time.time() - t0) * 1000)
        self.coach.write(self.coach.log_path, row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ctx-dir", required=True, type=Path)
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--inject", default=C.INJECT_DEFAULT,
                    help="which hint levels form the note: fact | plan | action | fact+plan ...")
    ap.add_argument("--max-usd", type=float, default=float(os.environ.get("COACH_MAX_USD", "100")))
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get("COACH_MAX_STEPS", "90")))
    ap.add_argument("--upstream", default=os.environ.get("COACH_UPSTREAM", "https://api.engy.ai/v1"))
    ap.add_argument("--teacher-key-env", default=os.environ.get("COACH_TEACHER_KEY_ENV", ""),
                    help="env var holding the upstream key to substitute for the incoming "
                         "Authorization / x-api-key header (empty = forward the incoming header)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    key = os.environ.get(args.teacher_key_env) if args.teacher_key_env else None
    Handler.coach = Coach(args.ctx_dir, args.log_dir, args.inject, args.max_usd, args.max_steps,
                          args.upstream, key)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    log.info("coach proxy on 127.0.0.1:%d -> %s (coach %s, inject %s, cap $%.0f, spent $%.2f)",
             args.port, args.upstream, COACH_MODEL, args.inject, args.max_usd, Handler.coach.spent)
    srv.serve_forever()


if __name__ == "__main__":
    main()
