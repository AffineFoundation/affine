#!/usr/bin/env python3
"""Health endpoint of the env-backfill driver pod (stdlib only).

    GET /health  -> 200 {"ok": true, ...} when the pod can run coverage
                    drivers (docker up, run_backfill.sh present, backfill R2
                    prefix set, disk not full), else 503 with the reasons.
    GET /drivers -> the live `backfill-<digest12>` tmux sessions with the
                    last log line of each.

Started by /post_start.sh (rollouts/scripts/backfill_post_start.sh) on every
container start; listens on BACKFILL_HEALTH_PORT (default 20000, the first
mapped Lium data port). The coverage queue reads it before renting a
serving box; kingctl's watchdog does not touch this pod (its name is not
affine-datagen*).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("BACKFILL_HEALTH_PORT", "20000"))
ROLLOUTS_ENV = "/root/rollouts/.rollouts_env"
STARTED = time.time()


def sh(cmd: str, timeout: int = 15) -> tuple[int, str]:
    try:
        p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or p.stderr).strip()
    except (subprocess.SubprocessError, OSError) as e:
        return 1, repr(e)


def env_value(key: str) -> str:
    try:
        for line in open(ROLLOUTS_ENV):
            line = line.strip().removeprefix("export ").strip()
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def drivers() -> list[dict]:
    rc, out = sh("tmux -f /dev/null ls -F '#{session_name}' 2>/dev/null")
    rows = []
    for name in (out.splitlines() if rc == 0 else []):
        if not name.startswith("backfill-"):
            continue
        d12 = name.removeprefix("backfill-")
        log = f"/root/logs/backfill_{d12}.log"
        _, tail = sh(f"tail -n 1 {log} 2>/dev/null")
        rows.append({"session": name, "digest12": d12, "log": log, "last_line": tail[-300:],
                     "complete": os.path.exists(log) and sh(f"grep -q 'backfill {d12} complete' {log}")[0] == 0})
    return rows


def health() -> tuple[bool, dict]:
    reasons = []
    docker_ok = sh("docker info >/dev/null 2>&1 && echo ok")[1] == "ok"
    if not docker_ok:
        reasons.append("docker not running")
    if not os.access("/root/rollouts/run_backfill.sh", os.X_OK):
        reasons.append("/root/rollouts/run_backfill.sh missing")
    if not os.path.exists("/root/venv/bin/python"):
        reasons.append("/root/venv missing")
    prefix = env_value("ROLLOUTS_R2_PREFIX")
    if prefix != "traces-backfill/":
        reasons.append(f"ROLLOUTS_R2_PREFIX={prefix!r} (must be traces-backfill/)")
    if os.path.exists("/root/rollouts/.king_env"):
        reasons.append("/root/rollouts/.king_env present (live king seat must not be here)")
    if sh("pgrep -f '^/root/venv/bin/python -m rollouts.run' >/dev/null && echo yes")[1] == "yes":
        reasons.append("a live datagen supervisor runs here")
    du = shutil.disk_usage("/root")
    free_gb = du.free / 1e9
    if free_gb < 40:
        reasons.append(f"disk free {free_gb:.0f} GB < 40")
    load = os.getloadavg()
    return not reasons, {
        "ok": not reasons, "reasons": reasons, "role": "env_backfill_driver",
        "pod": os.environ.get("BACKFILL_POD_NAME") or sh("hostname")[1],
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uptime_s": int(time.time() - STARTED),
        "docker": docker_ok, "r2_prefix": prefix, "shard": env_value("ROLLOUTS_SHARD"),
        "disk_free_gb": round(free_gb, 1), "load": [round(x, 1) for x in load],
        "nproc": os.cpu_count(), "drivers": drivers(),
        "containers": int(sh("docker ps -q 2>/dev/null | wc -l")[1] or 0) if docker_ok else None,
    }


class H(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] in ("/", "/health"):
            ok, body = health()
            code = 200 if ok else 503
        elif self.path.split("?")[0] == "/drivers":
            ok, body, code = True, {"drivers": drivers()}, 200
        else:
            body, code = {"error": "not found"}, 404
        data = json.dumps(body, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", PORT), H).serve_forever()
