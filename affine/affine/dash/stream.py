"""SSE stream of snapshot deltas for the live dashboard."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator

from ..config import Config
from .readers import snapshot
from .tmc import enrich_snapshot

log = logging.getLogger("affine.dash.stream")

POLL_S = 1.0
HEARTBEAT_S = 15.0


def _fingerprint(payload: dict) -> str:
    # Stable hash over the live bits that should trigger a UI refresh.
    slim = {
        "generated_at": payload.get("generated_at"),
        "phase": payload.get("phase"),
        "current_eval": payload.get("current_eval"),
        "king": payload.get("king"),
        "queue": payload.get("queue"),
        "intake": payload.get("intake"),
        "stats": payload.get("stats"),
        "reign": payload.get("reign"),
        "market": payload.get("market"),
    }
    raw = json.dumps(slim, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _sse(event: str, data: dict, eid: str) -> str:
    body = json.dumps(data, default=str)
    return f"id: {eid}\nevent: {event}\ndata: {body}\n\n"


async def snapshot_events(cfg: Config) -> AsyncIterator[str]:
    """Yield SSE `data:` lines when the snapshot fingerprint changes."""
    last_fp = ""
    last_beat = 0.0
    try:
        snap = enrich_snapshot(cfg, snapshot(cfg))
        last_fp = _fingerprint(snap)
        yield _sse("snapshot", snap, last_fp)
        last_beat = time.monotonic()
    except Exception as exc:
        log.warning("stream initial snapshot failed: %s", exc)

    while True:
        await asyncio.sleep(POLL_S)
        try:
            snap = enrich_snapshot(cfg, snapshot(cfg))
            fp = _fingerprint(snap)
            now = time.monotonic()
            if fp != last_fp:
                last_fp = fp
                yield _sse("snapshot", snap, fp)
                last_beat = now
                continue
            if now - last_beat >= HEARTBEAT_S:
                yield ": heartbeat\n\n"
                last_beat = now
        except Exception as exc:
            log.warning("stream poll failed: %s", exc)
            yield _sse("error", {"error": str(exc)}, "err")
