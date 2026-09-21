#!/usr/bin/env python
"""Tiny local reverse proxy that injects the model endpoint's bearer key.

Why: third-party runners (ARE / Gaia2, UserBench) read ONE OpenAI key from the
environment for every "openai" endpoint they call, but our passes talk to two
endpoints with different keys — the king pod (its own bearer) and a judge /
user simulator (Prime Inference or Engy). Point the agent at this proxy
(http://127.0.0.1:<port>/v1) and give the runner the other key.

  auth_proxy.py --port 18080 --upstream http://host:port/v1 --key-env BENCH_API_KEY
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = ""
KEY = ""
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization", "host", "content-length"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _forward(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        path = self.path
        if path.startswith("/v1"):
            path = path[3:]
        url = UPSTREAM.rstrip("/") + path
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "authorization"}
        headers["Authorization"] = f"Bearer {KEY}"
        if body is not None:
            headers["Content-Length"] = str(len(body))
        req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                data = r.read()
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:  # upstream down: 502 so the runner retries
            data = f'{{"error": "proxy: {type(e).__name__}"}}'.encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    do_GET = do_POST = _forward


def main() -> int:
    global UPSTREAM, KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--upstream", required=True, help="e.g. http://host:port/v1")
    ap.add_argument("--key-env", default="BENCH_API_KEY")
    a = ap.parse_args()
    UPSTREAM, KEY = a.upstream, os.environ.get(a.key_env, "")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"[auth-proxy] 127.0.0.1:{a.port} -> {UPSTREAM}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
