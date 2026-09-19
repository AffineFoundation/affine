"""Shared helpers for the box-side guards (ops/health, ops/pods, ops/fold).

Everything here is read-only against production except `discord_post`
(one line to the private Arbos channel) and `atomic_write_json` (own
observation files). No secret is ever printed.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE_DIR = REPO / "affine" / "state"
VALIDATOR_ENV = Path.home() / ".affine-validator.env"
REPO_ENV = REPO / ".env"

# Private Arbos ops channel (operator directive 2026-09-12: no automated
# posts in the public SN120 channel). Same channel as evalwatch / kingctl.
DISCORD_CHANNEL_DEFAULT = "1510910974498967613"
DISCORD_TOKEN_ENV = "DISCORD_BOT_TOKEN_ARBOS_BITTENSOR"
# Discord allows ~5 messages / 5 s per channel; several guards may post in
# one tick, so space posts out and honour retry_after on 429.
POST_SPACING_S = 1.1
_last_post = 0.0

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def now() -> float:
    return time.time()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat(timespec="seconds")


def parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    seconds = max(0.0, float(seconds))
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


def log(prefix: str, msg: str) -> None:
    print(f"[{prefix}] {now_iso()} {msg}", flush=True)


def env_file_value(name: str) -> str:
    """One value from the operator env snapshots (validator env first, then
    the repo .env). Process env wins. Missing file / key -> ""."""
    if os.environ.get(name):
        return os.environ[name]
    for path in (VALIDATOR_ENV, REPO_ENV):
        try:
            text = path.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic_write_json(path: Path, obj, *, mode: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    if mode is not None:
        os.chmod(tmp, mode)
    tmp.replace(path)


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def tail_bytes(path: Path, n: int) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - n))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def jsonl_tail(path: Path, n_bytes: int = 4_000_000) -> list[dict]:
    rows = []
    text = tail_bytes(path, n_bytes)
    lines = text.splitlines()
    # the first line of a byte-tail is usually cut in half
    for line in lines[1:] if len(text) >= n_bytes else lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            rows.append(r)
    return rows


def sha256_file(path: Path) -> str | None:
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


# -- pm2 -------------------------------------------------------------------------

def pm2_jlist() -> list[dict] | None:
    """`pm2 jlist` as a list, or None when pm2 itself failed (never [] on
    error: "no processes" and "pm2 broken" must stay distinct)."""
    try:
        raw = subprocess.check_output(["pm2", "jlist"], text=True,
                                      stderr=subprocess.DEVNULL, timeout=40)
    except (subprocess.SubprocessError, OSError):
        return None
    # pm2 sometimes prints a banner line before the JSON
    start = raw.find("[")
    if start < 0:
        return None
    try:
        data = json.loads(raw[start:])
    except ValueError:
        return None
    return data if isinstance(data, list) else None


def pm2_process(procs: list[dict] | None, name: str) -> dict | None:
    """Compact view of one pm2 process: status, restarts, start time, cron,
    script + args, pid."""
    if not procs:
        return None
    for p in procs:
        if p.get("name") != name:
            continue
        env = p.get("pm2_env") or {}
        up = env.get("pm_uptime")
        return {
            "name": name,
            "status": env.get("status"),
            "pid": p.get("pid") or None,
            "restarts": int(env.get("restart_time") or 0),
            "started_at": (float(up) / 1000.0) if up else None,
            "cron": env.get("cron_restart"),
            "script": env.get("pm_exec_path"),
            "args": env.get("args"),
            "cwd": env.get("pm_cwd"),
        }
    return None


def pm2_online(procs: list[dict] | None, name: str) -> bool | None:
    """True/False, or None when pm2 is unreadable."""
    if procs is None:
        return None
    p = pm2_process(procs, name)
    return bool(p and p["status"] == "online")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError):
        return False
    return True


# -- cron ------------------------------------------------------------------------

def cron_interval_seconds(expr: str | None) -> float | None:
    """Nominal interval of a simple cron expression, for "is the job
    overdue" checks. Handles `M H * * *` (daily), `M */N * * *` (every N h),
    `M H1,H2,... * * *` (list of hours -> smallest gap). Unknown -> None."""
    if not expr:
        return None
    parts = expr.split()
    if len(parts) != 5:
        return None
    minute, hour = parts[0], parts[1]
    if not minute.isdigit():
        return None
    if hour == "*":
        return 3600.0
    m = re.fullmatch(r"\*/(\d+)", hour)
    if m:
        return float(m.group(1)) * 3600.0
    if hour.isdigit():
        return 86400.0
    if re.fullmatch(r"\d+(,\d+)+", hour):
        hours = sorted(int(h) for h in hour.split(","))
        gaps = [(b - a) for a, b in zip(hours, hours[1:])] + [24 - hours[-1] + hours[0]]
        return float(max(gaps)) * 3600.0
    return None


# -- discord ---------------------------------------------------------------------

def discord_post(text: str, *, channel: str = DISCORD_CHANNEL_DEFAULT,
                 token_env: str = DISCORD_TOKEN_ENV, dry_run: bool = False,
                 prefix: str = "") -> bool:
    """One line to the private channel. Returns True when posted. Never
    raises; a failure is one log line. `dry_run` logs instead."""
    line = f"{prefix} {text}".strip()
    if dry_run:
        log("discord", f"(dry-run, not posted) {line}")
        return False
    token = env_file_value(token_env)
    if not token or not channel:
        log("discord", f"(no token/channel, not posted) {line}")
        return False
    global _last_post
    for _attempt in range(3):
        gap = time.monotonic() - _last_post
        if gap < POST_SPACING_S:
            time.sleep(POST_SPACING_S - gap)
        try:
            r = requests.post(
                f"https://discord.com/api/v10/channels/{channel}/messages",
                headers={"Authorization": f"Bot {token}"},
                json={"content": line[:1900]}, timeout=20)
        except requests.RequestException as e:
            log("discord", f"post failed: {e!r}")
            return False
        _last_post = time.monotonic()
        if r.status_code == 429:
            try:
                wait = float((r.json() or {}).get("retry_after", 1.0))
            except ValueError:
                wait = 1.0
            time.sleep(min(wait, 5.0) + 0.3)
            continue
        if r.status_code >= 300:
            log("discord", f"HTTP {r.status_code}: {r.text[:120]}")
            return False
        return True
    log("discord", f"rate limited 3x, dropped: {line[:80]}")
    return False


def shell_quote(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def python_bin() -> str:
    venv = REPO / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable
