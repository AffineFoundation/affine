#!/usr/bin/env python3
"""Factory sweep dashboard.

Serves one HTML page (index.html, read per request) and /api/data,
which re-reads results.jsonl + fleet.json on every call.
Stdlib only. Bind 127.0.0.1:8177.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
FACTORY = os.path.dirname(HERE)
RESULTS_PATH = os.path.join(FACTORY, "results.jsonl")
FLEET_PATH = os.path.join(FACTORY, "fleet.json")

HOST, PORT = "127.0.0.1", 8177


def read_results():
    rows = []
    try:
        with open(RESULTS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return rows


def read_fleet():
    try:
        with open(FLEET_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def build_data():
    rows = read_results()
    fleet = read_fleet()

    # Dedupe (arm, step): keep the latest ts; file order breaks ties.
    latest = {}
    for r in rows:
        key = (r.get("arm"), r.get("step"))
        prev = latest.get(key)
        if prev is None or r.get("ts", "") >= prev.get("ts", ""):
            latest[key] = r
    deduped = list(latest.values())

    teacher = next((r for r in deduped if r.get("arm") == "teacher"), None)
    base = next((r for r in deduped if r.get("arm") == "base"), None)

    arm_rows = [r for r in deduped if r.get("arm") not in ("teacher", "base")]
    for r in arm_rows:
        exits = r.get("exits") or {}
        n = r.get("n") or 0
        # Degenerate: zero score with the agent stuck repeating itself.
        degenerate = r.get("score") == 0 and exits.get("RepeatedAction", 0) >= max(5, n // 4)
        r["flags"] = ["degenerate"] if degenerate else []
    arm_rows.sort(key=lambda r: (str(r.get("arm")), r.get("step") or 0, r.get("ts", "")))

    pods = fleet.get("pods") or {}
    fleet_price = sum(p.get("price") or 0 for p in pods.values())

    return {
        "teacher": teacher,
        "base": base,
        "results": arm_rows,
        "n_raw_rows": len(rows),
        "fleet": {
            "teacher": fleet.get("teacher"),
            "miner": fleet.get("miner"),
            "pods": pods,
            "milestones_steps": fleet.get("milestones_steps") or [],
            "bench_panel": fleet.get("bench_panel") or {},
            "price_per_h": round(fleet_price, 2),
        },
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    body = f.read()
            except FileNotFoundError:
                self.send_error(404)
                return
            self._send(200, "text/html; charset=utf-8", body)
        elif path == "/api/data":
            body = json.dumps(build_data()).encode()
            self._send(200, "application/json", body)
        else:
            self.send_error(404)

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"factory dashboard on http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
