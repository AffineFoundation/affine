"""Self-heal for a Lium pod whose container came back without its volume.

Failure this handles (2026-09-21, affine-backfill-4): Lium restarted the
container; the encrypted volume did not remount, so /root was unwritable,
authorized_keys was gone (every ssh key -> "Permission denied (publickey)"),
the health server and every driver died, and Lium kept listing the pod
RUNNING. A `lium reboot` remounted the volume with everything intact.

Rule (used by pipeline-health for the env-backfill driver pod and by kingctl
for the datagen pods and the king-seat box): ssh refused with a publickey
error for longer than DENIED_MIN while the pod is listed RUNNING -> one
`lium reboot <pod>`, at most once per REBOOT_EVERY_H per pod. The caller
then re-runs the pod's post-start hook and relaunches its workers from
state. A pod that is simply unreachable (host down) is NOT rebooted — the
callers' existing dark -> re-rent paths own that.

    from podheal import ssh_state, maybe_reboot
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE = HERE / "state" / "podheal.json"
DENIED_MIN = 20
REBOOT_EVERY_H = 2.0
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]


def ssh_state(host: str, port: int, user: str = "root") -> str:
    """'ok' | 'denied' (sshd answers, key refused) | 'unreachable'."""
    try:
        p = subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"{user}@{host}", "true"],
                           capture_output=True, text=True, timeout=40)
    except (subprocess.SubprocessError, OSError):
        return "unreachable"
    if p.returncode == 0:
        return "ok"
    if "Permission denied" in (p.stderr or ""):
        return "denied"
    return "unreachable"


def _load() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def _save(d: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
    tmp.replace(STATE)


def denied_for(pod: str, denied: bool, now: float | None = None) -> float:
    """Seconds the pod's ssh has been in the denied state (0 when not)."""
    now = now or time.time()
    d = _load()
    rec = d.setdefault(pod, {})
    if denied:
        rec.setdefault("denied_since", now)
    else:
        rec.pop("denied_since", None)
    _save(d)
    return now - float(rec["denied_since"]) if denied else 0.0


def maybe_reboot(pod: str, *, listed_running: bool, denied_s: float, reason: str,
                 now: float | None = None) -> str | None:
    """Reboot when the rule holds and the throttle allows. Returns a one-line
    result string when a reboot was attempted, None otherwise."""
    now = now or time.time()
    if not listed_running or denied_s < DENIED_MIN * 60:
        return None
    d = _load()
    rec = d.setdefault(pod, {})
    if now - float(rec.get("reboot_at", 0)) < REBOOT_EVERY_H * 3600:
        return None
    rec["reboot_at"] = now
    rec["reboot_reason"] = reason
    rec["reboots"] = int(rec.get("reboots", 0)) + 1
    rec.pop("denied_since", None)
    _save(d)
    try:
        p = subprocess.run(["lium", "reboot", pod], input="y\n", capture_output=True, text=True,
                           timeout=180, cwd=str(REPO))
        tail = (p.stdout or p.stderr).strip().splitlines()[-1:] or [""]
        return f"lium reboot {pod}: rc {p.returncode} {tail[0][:100]}"
    except (subprocess.SubprocessError, OSError) as e:
        return f"lium reboot {pod} failed: {e!r}"
