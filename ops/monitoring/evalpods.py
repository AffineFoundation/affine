"""Bounded, read-only health panels; credentials and raw payloads never leave probes."""
from __future__ import annotations

import http.client
import json
import math
import os
from pathlib import Path
import queue
import re
import threading
import time
from datetime import datetime, timezone

from . import common

__all__ = ["collect"]

_ENDPOINTS = (("eval", "Eval", 9000), ("bench-engine", "Bench engine", 9001),
              ("chat", "Chat", 9002))
_TIMEOUT = 4.0
_DEADLINE = 5.0
_MAX_BYTES = 65536
_SLOTS = ("teacher", "king", "challenger", "teacher2", "king2", "challenger2")
_VERSION = re.compile(r"[0-9]{1,4}(?:\.[0-9]{1,4}){1,3}(?:(?:a|b|rc|\.post|\.dev)[0-9]{1,6})*(?:\+(?:cu[0-9]{2,4}|cpu|g[0-9a-f]{7,40}))?\Z")


def _token() -> str:
    value = os.environ.get("AFFINE_EVAL_TOKEN")
    if value:
        return value
    try:
        with (Path.home() / ".affine-validator.env").open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line.startswith("export ") or line.startswith("export\t"):
                    line = line[6:].lstrip()
                key, sep, value = line.partition("=")
                if not sep or key.strip() != "AFFINE_EVAL_TOKEN":
                    continue
                value = value.strip()
                if value[:1] in ("'", '"'):
                    quote = value[0]
                    end = value.find(quote, 1)
                    if end < 0 or (value[end + 1:].strip() and
                                   not value[end + 1:].lstrip().startswith("#")):
                        return ""
                    return value[1:end]
                value = value.partition(" #")[0].strip()
                return value if not any(c.isspace() for c in value) else ""
    except (OSError, UnicodeError):
        pass
    return ""


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _number(value):
    if type(value) in (int, float) and 0 <= value <= 1e15 and math.isfinite(value):
        return value
    return None


def _boolean(value, yes="Ready", no="Not ready"):
    return yes if value is True else no if value is False else "Unknown"


def _choice(value, choices):
    return value if isinstance(value, str) and value in choices else "unknown"


def _synced_at(value):
    if _number(value) is not None:
        try:
            return (datetime.fromtimestamp(value, timezone.utc).isoformat()
                    if value > 0 else "Unknown")
        except (ValueError, OverflowError, OSError):
            return "Unknown"
    if not isinstance(value, str) or len(value) > 40:
        return "Unknown"
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            return "Unknown"
        return stamp.astimezone(timezone.utc).isoformat()
    except ValueError:
        return "Unknown"


def _render(key, title, port, health=None, failure=None):
    source = f"http://127.0.0.1:{port}/health"
    data = _mapping(health)
    engine = _mapping(data.get("engine"))
    corpus = _mapping(data.get("corpus"))
    versions = _mapping(data.get("versions"))
    state = _choice(data.get("state"), {"starting", "idle", "loading", "serving", "error"})
    busy_kind = _choice(data.get("busy_kind"), {"duel", "bench", "prefetch", "chat"})
    ready = data.get("ok")
    notes = []
    status = "ok"
    if failure:
        status, readiness = "error", "Unavailable"
        notes.append(failure)
    elif state == "error" or ready is False:
        status, readiness = "error", "Not ready"
        notes.append("Endpoint reports an unhealthy service; diagnostic text withheld.")
    elif key == "chat":
        readiness = "Ready" if ready is True and state == "serving" else "Not ready"
        if readiness != "Ready":
            status = "warn"
    else:
        readiness = _boolean(ready)
        if ready is not True:
            status = "warn"
    metrics = [common.metric("Readiness", readiness, tone=status),
               common.metric("Busy", _boolean(data.get("busy"), "Busy", "Idle"),
                             busy_kind if busy_kind != "unknown" else ""),
               common.metric("State", state.capitalize())]
    if key == "chat" and not failure:
        notes.append("Chat health does not report request occupancy, engine slots, corpus, or disk.")

    slots = []
    slot_ready = []
    for name in _SLOTS:
        if name not in engine and (key == "chat" or name.endswith("2")):
            continue
        slot = _mapping(engine.get(name))
        value = slot.get("ready")
        slots.append([name, _boolean(value)])
        if type(value) is bool:
            slot_ready.append(value)
        if slot.get("state") == "error":
            status = "error"
            notes.append(f"{name.capitalize()} slot reports an error.")
    if not slots:
        slots = [["Engine slots", "Not reported"]]
    bars = []
    if slot_ready:
        bars.append(dict(label="Reported slots ready", value=sum(slot_ready),
                         max=len(slot_ready), detail="Idle slots may be intentionally unloaded.",
                         tone="green" if all(slot_ready) else "warn"))
    if key == "eval" and _mapping(engine.get("teacher")).get("ready") is False:
        status = "error"
        notes.append("Teacher slot is not ready.")

    if not corpus:
        corpus_state = "Not applicable" if key != "eval" and not failure else "Unknown"
    elif corpus.get("state") == "error" or corpus.get("ready") is False:
        corpus_state, status = "Not ready", "error"
    elif corpus.get("stale") is True:
        corpus_state = "Stale"
        if status != "error":
            status = "warn"
    else:
        corpus_state = _boolean(corpus.get("ready"), "Ready", "Not ready")
    if key == "eval" and data.get("turns_present") is False:
        corpus_state, status = "Not ready", "error"
    metrics.append(common.metric("Corpus sync", corpus_state))
    corpus_rows = [["Ready", _boolean(corpus.get("ready"))],
                   ["Stale", _boolean(corpus.get("stale"), "Yes", "No")],
                   ["Last sync (UTC)", _synced_at(corpus.get("synced_at"))]]
    for label, field in (("Epoch", "corpus_epoch"), ("Schema", "schema_version")):
        value = _number(corpus.get(field))
        corpus_rows.append([label, value if value is not None else "Unknown"])

    disk = _number(data.get("free_disk_gb"))
    metrics.append(common.metric("Free disk", f"{disk:,.1f} GB" if disk is not None else "Unknown"))
    docker = _mapping(data.get("docker"))
    if key == "bench-engine":
        metrics.append(common.metric("Docker", _boolean(docker.get("ok"))))
        if docker.get("ok") is False:
            status = "error"
            notes.append("Docker health probe failed; diagnostic text withheld.")
    version_rows = []
    for package in ("vllm", "transformers", "torch"):
        version = versions.get(package)
        safe = isinstance(version, str) and len(version) <= 80 and _VERSION.fullmatch(version)
        version_rows.append([package, version if safe else "Unknown"])
    metrics[0]["tone"] = "green" if status == "ok" else "red" if status == "error" else "warn"
    return common.panel(title, "Read-only local health", status=status, metrics=metrics,
                        sections=[common.table("Slot readiness", ["Slot", "Readiness"], slots),
                                  common.table("Corpus sync", ["Field", "Value"], corpus_rows),
                                  common.table("Versions", ["Package", "Version"], version_rows)],
                        bars=bars, notes=notes, sources=[source])


def _probe(key, title, port, token):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=_TIMEOUT)
    try:
        headers = {"X-Affine-Token": token} if token else {}
        connection.request("GET", "/health", headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            failure = ("Health authentication rejected." if response.status in (401, 403)
                       else "Health endpoint returned a non-success status.")
            return _render(key, title, port, failure=failure)
        raw = response.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            return _render(key, title, port, failure="Health response exceeded the size limit.")
        health = json.loads(raw)
        if not isinstance(health, dict) or type(health.get("ok")) is not bool:
            return _render(key, title, port, failure="Invalid health response.")
        return _render(key, title, port, health)
    except (TimeoutError, OSError):
        return _render(key, title, port, failure="Health endpoint unavailable or timed out.")
    except Exception:
        return _render(key, title, port, failure="Health probe failed or returned invalid data.")
    finally:
        connection.close()


def collect() -> dict:
    """Probe all three endpoints concurrently, returning within a shared deadline.

    Only fixed loopback GETs are issued; no proxies, redirects or config loading.
    Daemon workers keep a slow response from blocking collection or interpreter exit.
    """
    token = _token()
    results = queue.Queue()
    deadline = time.monotonic() + _DEADLINE

    def run(endpoint):
        key, title, port = endpoint
        try:
            result = _probe(key, title, port, token)
        except Exception:
            result = _render(key, title, port, failure="Health probe failed.")
        results.put((key, result))

    for endpoint in _ENDPOINTS:
        threading.Thread(target=run, args=(endpoint,), daemon=True).start()
    panels = {}
    while len(panels) < len(_ENDPOINTS):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            key, result = results.get(timeout=remaining)
            panels[key] = result
        except queue.Empty:
            break
    return {key: panels[key] if key in panels else
            _render(key, title, port, failure="Health probe deadline exceeded.")
            for key, title, port in _ENDPOINTS}
