"""Kingboard web service: static page + stats JSON + a background rebuild loop.

    uvicorn server:app --host 127.0.0.1 --port 8790

The page (static/) polls /api/stats.json every 60 s. A daemon thread runs
build.py as a subprocess every KINGBOARD_REFRESH_S seconds (default 180);
the builder only reads trace chunks it has not seen, so a steady-state
pass is a manifest fetch plus a few new chunks. Running it out of process
keeps the JSON parsing off the server's event loop.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
STATE_DIR = Path(os.environ.get("KINGBOARD_STATE_DIR", HERE / "state"))
STATS_PATH = STATE_DIR / "stats.json"
REFRESH_S = float(os.environ.get("KINGBOARD_REFRESH_S", "180"))
BUILD_TIMEOUT_S = float(os.environ.get("KINGBOARD_BUILD_TIMEOUT_S", "5400"))
BUILD_CMD = [sys.executable, str(HERE / "build.py")]

app = FastAPI(title="affine kingboard", docs_url=None, redoc_url=None)

status: dict = {
    "started_at": time.time(), "runs": 0, "last_start": None, "last_end": None,
    "last_ok": None, "last_error": None, "last_seconds": None, "running": False,
    "refresh_s": REFRESH_S, "log_tail": "",
}
_stats_cache: dict = {"mtime": None, "body": b""}


def run_build_once() -> None:
    status.update(running=True, last_start=time.time())
    try:
        proc = subprocess.run(BUILD_CMD, capture_output=True, text=True,
                              timeout=BUILD_TIMEOUT_S, cwd=str(HERE))
        tail = (proc.stderr or proc.stdout or "")[-4000:]
        status.update(last_ok=proc.returncode == 0, log_tail=tail,
                      last_error=None if proc.returncode == 0
                      else f"exit {proc.returncode}")
    except subprocess.TimeoutExpired:
        status.update(last_ok=False, last_error=f"build timed out after {BUILD_TIMEOUT_S:.0f}s")
    except OSError as e:
        status.update(last_ok=False, last_error=str(e))
    finally:
        end = time.time()
        status.update(running=False, last_end=end,
                      last_seconds=round(end - status["last_start"], 1))
        status["runs"] += 1


def build_loop() -> None:
    while True:
        run_build_once()
        time.sleep(REFRESH_S)


@app.on_event("startup")
def _start_loop() -> None:
    if os.environ.get("KINGBOARD_NO_BUILDER") != "1":
        threading.Thread(target=build_loop, name="kingboard-builder", daemon=True).start()


def stats_bytes() -> bytes | None:
    try:
        mtime = STATS_PATH.stat().st_mtime
    except FileNotFoundError:
        return None
    if _stats_cache["mtime"] != mtime:
        _stats_cache.update(mtime=mtime, body=STATS_PATH.read_bytes())
    return _stats_cache["body"]


@app.get("/api/stats.json")
def api_stats() -> Response:
    body = stats_bytes()
    if body is None:
        return JSONResponse({"error": "first build not finished yet", "builder": status},
                            status_code=503, headers={"Cache-Control": "no-store"})
    return Response(body, media_type="application/json",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def api_health() -> JSONResponse:
    body = stats_bytes()
    generated_at = None
    if body:
        try:
            generated_at = json.loads(body).get("generated_at")
        except ValueError:
            generated_at = None
    return JSONResponse({"ok": body is not None, "stats_generated_at": generated_at,
                         "builder": status}, headers={"Cache-Control": "no-store"})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
