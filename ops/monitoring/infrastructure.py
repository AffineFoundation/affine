"""Read-only infrastructure snapshots for a 120-second monitoring cadence."""
from __future__ import annotations

import json
import math
import re
import subprocess
import time

from .common import duration, metric, panel, table

__all__ = ["collect"]

_NAME = re.compile(r"(?:affine-|swarm-t-)[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_GPU = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _().-]{0,63}\Z")
_CRON = re.compile(r"[0-9*/?,\- ]{5,100}\Z")
_PM2_STATUSES = frozenset({
    "online", "stopped", "errored", "launching", "stopping", "waiting restart",
    "one-launch-status",
})
_POD_STATUSES = frozenset({
    "running", "pending", "starting", "stopping", "stopped", "exited",
    "terminated", "terminating", "deleted", "deleting", "failed", "error",
    "creating", "provisioning", "restarting", "paused", "ready", "active",
})
_UNKNOWN = "Unknown"


def _inventory(command, *, timeout, wrappers=()):
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "Inventory command timed out; current state is unknown."
    except OSError:
        return None, "Inventory command unavailable; current state is unknown."
    if result.returncode:
        return None, "Inventory command failed; current state is unknown."
    try:
        if len(result.stdout) > 8 * 1024 * 1024:
            raise ValueError
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = next((data[k] for k in wrappers if isinstance(data.get(k), list)), None)
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise ValueError
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return None, "Inventory response invalid; current state is unknown."
    return data, None


def _number(value, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
        if not math.isfinite(number) or number < 0 or number > 1e18:
            return None
        if integer:
            return int(number) if number.is_integer() else None
        return number
    except (ValueError, OverflowError):
        return None


def _status(value, allowed):
    if isinstance(value, str) and value.lower() in allowed:
        return value.lower()
    return "unknown"


def _name(value, *, services=False):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        return None
    return value if not services or value.startswith("affine-") else None


def _cron(value):
    if value is None or value is False or value == "":
        return "None"
    if isinstance(value, str) and _CRON.fullmatch(value) and len(value.split()) in (5, 6):
        return value
    return _UNKNOWN


def _services():
    inventory, error = _inventory(["pm2", "jlist"], timeout=10)
    notes = [
        "PM2 process status is not application health. CPU is process percent and may exceed 100%.",
        "Stopped affine-corpus-refresh with autorestart=false and a cron schedule is normal; stopped affine-swarm-enroll is retired, not an outage.",
        "Uptime is time since the current online process started; restart counts are PM2 counters, not interval rates.",
    ]
    rows, cpus = [], []
    online = expected = attention = 0
    now = time.time()
    for item in inventory or []:
        name = _name(item.get("name"), services=True)
        if name is None:
            continue
        env = item.get("pm2_env")
        env = env if isinstance(env, dict) else {}
        monit = item.get("monit")
        monit = monit if isinstance(monit, dict) else {}
        status = _status(env.get("status"), _PM2_STATUSES)
        cron = _cron(env.get("cron_restart"))
        scheduled = (name == "affine-corpus-refresh" and status == "stopped"
                     and env.get("autorestart") is False and cron not in ("None", _UNKNOWN))
        retired = name == "affine-swarm-enroll" and status == "stopped"
        normal = scheduled or retired
        online += status == "online"
        expected += normal
        attention += status != "online" and not normal
        interpretation = "Scheduled idle" if scheduled else "Retired" if retired else (
            "Online" if status == "online" else "Check process"
        )
        pid = _number(item.get("pid"), integer=True)
        cpu = _number(monit.get("cpu"))
        memory = _number(monit.get("memory"))
        restarts = _number(env.get("restart_time"), integer=True)
        started = _number(env.get("pm_uptime"))
        uptime = (now - started / 1000 if status == "online" and started
                  and started / 1000 <= now else None)
        rows.append([
            name, status, interpretation, pid if pid is not None else _UNKNOWN,
            f"{cpu:.1f}%" if cpu is not None else _UNKNOWN,
            f"{memory / 1024 ** 2:.1f} MiB" if memory is not None else _UNKNOWN,
            restarts if restarts is not None else _UNKNOWN,
            duration(uptime) if status == "online" else "Not running",
            cron,
        ])
        if cpu is not None and status == "online":
            cpus.append((name, cpu))
    if error:
        notes.append(error)
    elif not rows:
        notes.append("No affine-* services found in the current PM2 inventory.")
    cpu_max = max([100.0, *(value for _, value in cpus)])
    return panel(
        "Services", "Current project PM2 inventory · 120s cadence",
        status="warn" if error or attention or not rows else "ok",
        metrics=[
            metric("Online", online if not error else _UNKNOWN),
            metric("Expected stopped", expected if not error else _UNKNOWN),
            metric("Needs attention", attention if not error else _UNKNOWN,
                   tone="warn" if attention else ""),
        ],
        sections=[table("Project services", [
            "Service", "Status", "Interpretation", "PID", "CPU", "Memory",
            "Restarts", "Uptime", "Cron",
        ], sorted(rows))],
        bars=[dict(label=name, value=value, max=cpu_max,
                   detail=f"{value:.1f}% process CPU", tone="")
              for name, value in sorted(cpus)],
        notes=notes, sources=["Local PM2 jlist; allowlisted process fields only."],
    )


def _fleet():
    inventory, error = _inventory(
        ["lium", "ps", "--format", "json"], timeout=20, wrappers=("pods", "data"),
    )
    rows, rates = [], []
    running = unknown_status = 0
    running_priced = 0
    for item in inventory or []:
        name = _name(item.get("name") or item.get("pod_name"))
        if name is None:
            continue
        status = _status(item.get("status"), _POD_STATUSES)
        unknown_status += status == "unknown"
        gpu = item.get("gpu_type")
        gpu = gpu if isinstance(gpu, str) and _GPU.fullmatch(gpu) else _UNKNOWN
        count = _number(item.get("gpu_count"), integer=True)
        price = next((value for key in ("price_per_hour", "hourly_price", "price")
                      if (value := _number(item.get(key))) is not None), None)
        rows.append([
            name, status, gpu, count if count is not None else _UNKNOWN,
            f"${price:.2f}/h" if price is not None else _UNKNOWN,
        ])
        if status == "running":
            running += 1
            if price is not None:
                running_priced += 1
                rates.append((name, price))
    notes = [
        "Rates are current inventory estimates in USD, not actual bills or accrued spend; stopped pods may still incur provider charges.",
        "The rate subtotal includes only explicitly running pods with known hourly prices, not historical state or missing rates.",
        "Inventory status is not GPU utilization or application health. No remote probes are performed.",
    ]
    if error:
        notes.append(error)
    elif not rows:
        notes.append("No affine-* or swarm-t-* pods found in the current inventory.")
    if unknown_status:
        notes.append("Some pod statuses are unrecognized and excluded from the running subtotal.")
    if running_priced < running:
        notes.append("Running rate coverage is incomplete; missing prices are unknown, not zero.")
    max_rate = max([1.0, *(value for _, value in rates)])
    return panel(
        "Fleet", "Current project GPU inventory · 120s cadence",
        status="warn" if error or not rows or unknown_status or running_priced < running
        or any(row[1] in ("failed", "error") for row in rows) else "ok",
        metrics=[
            metric("Listed pods", len(rows) if not error else _UNKNOWN),
            metric("Running pods", running if not error else _UNKNOWN),
            metric("Known running rate estimate",
                   f"${sum(value for _, value in rates):.2f}/h" if rates else _UNKNOWN,
                   f"{running_priced}/{running} running pods priced; not an actual bill"),
        ],
        sections=[table("Project pods", [
            "Pod", "Status", "GPU type", "GPU count", "Inventory rate (USD/h)",
        ], sorted(rows))],
        bars=[dict(label=name, value=value, max=max_rate,
                   detail=f"${value:.2f}/h inventory estimate, not actual bill", tone="")
              for name, value in sorted(rates)],
        notes=notes, sources=["Lium ps --format json; current project inventory only."],
    )


def collect():
    """Return independent services/fleet panels; never publish raw CLI output."""
    return {"services": _services(), "fleet": _fleet()}
