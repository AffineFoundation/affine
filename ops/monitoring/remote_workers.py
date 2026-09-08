"""Read-only, bounded SSH telemetry for Affine datagen workers.

``collect_workers`` consumes the caller's raw lium inventory; it never runs lium.
Only ``affine-datagen*`` names are selected, in inventory order. Counts describe
supervisors (``python -m rollouts.run``) and their rollout bootstrap shells, not
individual evaluation tasks. Zero counts on a reachable pod mean inactivity.
Unknown counts/timestamps are None. ``error``, when present, is one of ERRORS;
no SSH details, exception text, environment, paths, or log lines are returned.

Cycle timestamps are ISO 8601 *worker-local wall time*, without an invented
UTC offset (Python logging does not stamp one). Log mtimes are ISO 8601 UTC.
The latest recognizable cycle in bounded tails is historical, not a liveness
claim. Missing cycles are not errors. No disk metric is collected.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import re
import selectors
import shlex
import subprocess
import time

__all__ = ["collect_workers"]

ERRORS = frozenset({
    "invalid_ssh", "ssh_unavailable", "timeout", "ssh_failed", "probe_failed",
    "invalid_response", "proc_unreadable", "log_missing", "log_unreadable",
})
_TOTAL_SECONDS = 14.0
_MAX_OUTPUT = 8192
_NAME = re.compile(r"affine-datagen[A-Za-z0-9_-]{0,64}\Z")
_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,79}\Z")

_REMOTE_PROBE = r'''
import datetime as dt
import json
import os
import re
import signal
import stat

signal.alarm(10)
TAIL_BYTES = 65536
MAX_LOGS = 4
CYCLE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) "
    r"rollouts\.run INFO cycle: source=([A-Za-z][A-Za-z0-9_-]{0,79}) "
    r"policy=([A-Za-z][A-Za-z0-9_-]{0,79}) batch=(\d{1,6})(?=\s|$)"
)


def classify(args, cwd):
    if not args:
        return None
    exe = os.path.basename(args[0])
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", exe):
        # Stop at the first execution target, never inspect -c source text.
        for i, arg in enumerate(args[1:], 1):
            if arg == "-m":
                return "rollout_processes" if args[i + 1:i + 2] == ["rollouts.run"] else None
            if arg in ("-c", "-") or not arg.startswith("-"):
                return None
    if exe in ("bash", "sh", "dash"):
        for arg in args[1:]:
            if arg in ("-c", "-lc", "-ic"):
                return None
            if arg.startswith("-"):
                continue
            path = os.path.normpath(os.path.join(cwd, arg))
            return "bootstrap_processes" if path == "/root/rollouts/bootstrap.sh" else None
    return None


def tail(path):
    # O_NONBLOCK prevents a raced FIFO from hanging; do not follow symlinks.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise OSError("not_regular")
        start = max(0, info.st_size - TAIL_BYTES)
        stream.seek(start)
        data = stream.read(TAIL_BYTES)
        if start:
            data = data.partition(b"\n")[2]
        # Ignore an incomplete final record while the writer is appending.
        data = data.rpartition(b"\n")[0]
        return info, data.decode("utf-8", "replace").splitlines()


def probe():
    result = {"rollout_processes": 0, "bootstrap_processes": 0,
              "last_cycle_at": None, "last_cycle_source": None,
              "last_cycle_policy": None, "log_updated_at": None}
    logs = set()
    excluded = {os.getpid(), os.getppid()}
    proc_ok = True
    try:
        with os.scandir("/proc") as entries:
            for entry in entries:
                if not entry.name.isdigit() or int(entry.name) in excluded:
                    continue
                try:
                    with open(entry.path + "/cmdline", "rb") as stream:
                        raw = stream.read(8193)
                    if len(raw) > 8192:
                        continue
                    args = [s.decode("utf-8", "replace") for s in raw.split(b"\0") if s]
                    try:
                        cwd = os.readlink(entry.path + "/cwd")
                    except OSError:
                        cwd = "/"
                    kind = classify(args, cwd)
                    if kind is None:
                        continue
                    result[kind] += 1
                    if kind == "rollout_processes":
                        for number in ("1", "2"):
                            try:
                                path = os.readlink(entry.path + "/fd/" + number)
                                if path.startswith("/") and path.endswith(".log"):
                                    logs.add(path)
                            except OSError:
                                pass
                except FileNotFoundError:
                    pass
                except OSError:
                    proc_ok = False
    except OSError:
        proc_ok = False
    if not proc_ok:
        result.update(rollout_processes=None, bootstrap_processes=None,
                      error="proc_unreadable")
    # Known bootstrap destination also works when no supervisor is running.
    paths = ["/root/logs/rollouts.log"]
    paths.extend(p for p in sorted(logs) if p not in paths)
    read_any = False
    unreadable = False
    latest = None
    for path in paths[:MAX_LOGS]:
        try:
            info, lines = tail(path)
        except FileNotFoundError:
            continue
        except OSError:
            unreadable = True
            continue
        read_any = True
        modified = dt.datetime.fromtimestamp(info.st_mtime, dt.timezone.utc).isoformat()
        if result["log_updated_at"] is None or modified > result["log_updated_at"]:
            result["log_updated_at"] = modified
        for line in reversed(lines):
            match = CYCLE.match(line)
            if not match:
                continue
            timestamp, source, policy, _batch = match.groups()
            try:
                timestamp = dt.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f").isoformat(timespec="milliseconds")
            except ValueError:
                continue
            if latest is None or timestamp > latest:
                latest = timestamp
                result.update(last_cycle_at=timestamp, last_cycle_source=source,
                              last_cycle_policy=policy)
            break
    if unreadable:
        result.setdefault("error", "log_unreadable")
    elif not read_any:
        result.setdefault("error", "log_missing")
    return result


try:
    print(json.dumps(probe(), separators=(",", ":")))
except Exception:
    print('{"error":"probe_failed"}')
'''


def _record(name: str) -> dict:
    return {
        "name": name, "reachable": False,
        "rollout_processes": None, "bootstrap_processes": None,
        "last_cycle_at": None, "last_cycle_source": None,
        "last_cycle_policy": None, "log_updated_at": None,
    }


def _ssh_args(command: str) -> list[str]:
    """Accept inventory connection fields, not shell code or SSH exec options."""
    if not isinstance(command, str) or len(command) > 4096:
        raise ValueError
    parts = shlex.split(command)
    if not parts or parts.pop(0) not in ("ssh", "/usr/bin/ssh"):
        raise ValueError
    host = None
    options = []
    while parts:
        token = parts.pop(0)
        if token in ("-p", "-i", "-l"):
            if not parts:
                raise ValueError
            value = parts.pop(0)
            if not value or value.startswith("-") or any(ord(c) < 32 for c in value):
                raise ValueError
            if token == "-p" and (not value.isascii() or not value.isdigit() or not 0 < int(value) < 65536):
                raise ValueError
            options.extend((token, value))
        elif token.startswith("-") or host is not None:
            raise ValueError
        elif re.fullmatch(r"(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9.:-]*", token):
            host = token
        else:
            raise ValueError
    if host is None:
        raise ValueError
    return [
        "ssh", "-F", "/dev/null", "-T",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
        "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=3",
        "-o", "ServerAliveCountMax=1", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR", "-o", "ClearAllForwardings=yes",
        "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
        *options, host, "python3 -B -c " + shlex.quote(_REMOTE_PROBE),
    ]


def _timestamp(value: object) -> bool:
    if not isinstance(value, str) or len(value) > 40:
        return False
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False


def _sanitize(data: object, result: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError
    result["reachable"] = True
    if data == {"error": "probe_failed"}:
        result["error"] = "probe_failed"
        return result
    error = data.get("error")
    if error is not None and error not in {"proc_unreadable", "log_missing", "log_unreadable"}:
        raise ValueError
    for key in ("rollout_processes", "bootstrap_processes"):
        value = data.get(key)
        if value is None and error == "proc_unreadable":
            continue
        if type(value) is not int or not 0 <= value <= 1000000:
            raise ValueError
        result[key] = value
    modified = data.get("log_updated_at")
    if modified is not None:
        if not _timestamp(modified):
            raise ValueError
        result["log_updated_at"] = modified
    cycle = [data.get(k) for k in ("last_cycle_at", "last_cycle_source", "last_cycle_policy")]
    if any(v is not None for v in cycle):
        if not _timestamp(cycle[0]) or not all(isinstance(v, str) and _LABEL.fullmatch(v) for v in cycle[1:]):
            raise ValueError
        result.update(zip(("last_cycle_at", "last_cycle_source", "last_cycle_policy"), cycle))
    if error:
        result["error"] = error
    return result


def _collect_one(pod: dict, deadline: float) -> dict:
    result = _record(pod["name"])
    try:
        args = _ssh_args(pod.get("ssh_cmd"))
    except (ValueError, TypeError):
        return dict(result, error="invalid_ssh")
    if time.monotonic() >= deadline:
        return dict(result, error="timeout")
    try:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return dict(result, error="ssh_unavailable")
    output = bytearray()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return dict(result, error="timeout")
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fd, 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        break
                    output.extend(chunk)
                    if len(output) > _MAX_OUTPUT:
                        return dict(result, error="invalid_response")
        try:
            code = process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            return dict(result, error="timeout")
        if code:
            return dict(result, reachable=(code != 255),
                        error="ssh_failed" if code == 255 else "probe_failed")
        try:
            return _sanitize(json.loads(output), result.copy())
        except (ValueError, TypeError):
            return dict(result, reachable=True, error="invalid_response")
    except OSError:
        return dict(result, error="ssh_failed")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()


def collect_workers(pods: list[dict]) -> list[dict]:
    """Probe datagen pods concurrently under one shared 14-second deadline.

    No inventory refresh, remote writes, or production process control occurs.
    At most eight SSH connections run concurrently. Queued probes inherit the
    same deadline, so unreachable workers cannot multiply the timeout. SSH
    host keys use non-persisting TOFU, appropriate to ephemeral lium pods.
    """
    deadline = time.monotonic() + _TOTAL_SECONDS
    workers = [p for p in pods if isinstance(p, dict)
               and isinstance(p.get("name"), str) and _NAME.fullmatch(p["name"])]
    if not workers:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(workers))) as executor:
        return list(executor.map(lambda pod: _collect_one(pod, deadline), workers))
