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
MATRIX_PATH = STATE_DIR / "matrix.json"
# affine.io's self-hosted fonts (Inter / IBM Plex Mono): the kingboard shares
# the main page's theme, so it serves the same files under /fonts.
FONTS_DIR = Path(os.environ.get("KINGBOARD_FONTS_DIR",
                                HERE.parents[1] / "affine" / "website" / "fonts"))
# Benchmark-suite scorecards (ops/benchsuite/publish.py writes one JSON per
# run): the "Benchmarks" tab reads them straight from disk.
BENCHSUITE_DIR = Path(os.environ.get("BENCHSUITE_STATE_DIR",
                                     HERE.parents[1] / "affine" / "state" / "benchsuite"))
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
_matrix_cache: dict = {"mtime": None, "body": b""}


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


def _cached_bytes(path: Path, cache: dict) -> bytes | None:
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    if cache["mtime"] != mtime:
        cache.update(mtime=mtime, body=path.read_bytes())
    return cache["body"]


def stats_bytes() -> bytes | None:
    return _cached_bytes(STATS_PATH, _stats_cache)


def matrix_bytes() -> bytes | None:
    return _cached_bytes(MATRIX_PATH, _matrix_cache)


def _json_file_response(body: bytes | None) -> Response:
    if body is None:
        return JSONResponse({"error": "first build not finished yet", "builder": status},
                            status_code=503, headers={"Cache-Control": "no-store"})
    return Response(body, media_type="application/json",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/stats.json")
def api_stats() -> Response:
    return _json_file_response(stats_bytes())


@app.get("/api/matrix")
@app.get("/api/matrix.json")
def api_matrix() -> Response:
    """Model x (benchmark | environment) score matrix (build.py::build_matrix),
    rebuilt by the same refresh pass as stats.json."""
    return _json_file_response(matrix_bytes())


@app.get("/api/benchsuite.json")
def api_benchsuite() -> JSONResponse:
    """Every published benchmark-suite scorecard, newest first (rollout rows
    are not in these files; they live on R2 under research/benchsuite/)."""
    runs = []
    for p in sorted(BENCHSUITE_DIR.glob("*.json")):
        try:
            card = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(card, dict) and card.get("rows") is not None:
            runs.append(card)
    runs.sort(key=lambda c: c.get("created_at") or "", reverse=True)
    return JSONResponse({"runs": runs, "generated_at": time.time()},
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
if FONTS_DIR.is_dir():
    app.mount("/fonts", StaticFiles(directory=FONTS_DIR), name="fonts")
