#!/usr/bin/env python3
"""Snatch one 8×B200 pod named affine-teacher for a dedicated GLM-Air teacher.

B200-only (no B300 fallback). Uses Lium HTTP /executors then POST rent,
with CLI fallback. Always TTL. Does not touch mine-* or other affine pods.
"""
from __future__ import annotations

import configparser
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path("/home/const/subnet120")
EXP = ROOT / "ops/teacher-b200"
LOG = EXP / "logs/wait_rent_teacher_b200.log"
PIDF = EXP / "logs/wait_rent_teacher_b200.pid"
STAMP = EXP / "artifacts/rented_affine-teacher.json"

NAME = os.environ.get("POD_NAME", "affine-teacher")
TTL = os.environ.get("TTL", "24h")
MAX_ITERS = int(os.environ.get("MAX_ITERS", "86400"))
EMPTY_SLEEP = float(os.environ.get("EMPTY_SLEEP", "0.5"))
TEMPLATE_ID = os.environ.get(
    "LIUM_TEMPLATE_ID", "da582e38-eb5c-4580-94d6-70cbda7c7c56"
)
SSH_PUBKEY_PATH = Path(
    os.environ.get("LIUM_SSH_PUBKEY", str(Path.home() / ".ssh/id_ed25519.pub"))
)
BASE = os.environ.get("LIUM_BASE_URL", "https://lium.io/api")


def log(msg: str) -> None:
    print(
        f"[teacher-b200] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}",
        flush=True,
    )


def api_key() -> str:
    env = os.environ.get("LIUM_API_KEY")
    if env:
        return env
    cfg = configparser.ConfigParser()
    cfg.read(str(Path.home() / ".lium/config.ini"))
    key = cfg.get("api", "api_key", fallback=cfg.get("default", "api_key", fallback=""))
    if not key:
        raise SystemExit("no Lium API key")
    return key


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "X-API-KEY": api_key(),
            "X-Source": "teacher-b200",
            "X-Lium-Client-Version": "0.0.32",
        }
    )
    return s


def _get_json(
    sess: requests.Session, path: str, params: dict | None = None, retries: int = 6
):
    url = f"{BASE}{path}"
    backoff = 0.5
    for attempt in range(retries):
        try:
            r = sess.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            log(f"GET {path} neterr attempt={attempt+1}: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
            continue
        if r.status_code == 429 or 500 <= r.status_code < 600:
            ra = r.headers.get("Retry-After")
            try:
                wait = float(ra) if ra else backoff
            except ValueError:
                wait = backoff
            wait = max(wait, backoff)
            log(f"GET {path} http={r.status_code} backoff={wait:.2f}s")
            time.sleep(wait)
            backoff = min(backoff * 2, 8.0)
            continue
        if not r.ok:
            log(f"GET {path} http={r.status_code} body={r.text[:160]}")
            return None
        try:
            return r.json()
        except Exception as e:
            log(f"GET {path} jsonerr: {e}")
            return None
    return None


def _is_b200(machine: str) -> bool:
    u = (machine or "").upper()
    return "B200" in u and "B300" not in u


def list_b200_8x(sess: requests.Session) -> list[dict] | None:
    data = _get_json(
        sess,
        "/executors",
        params={"size": 1000, "gpu_count_gte": 8, "gpu_count_lte": 8},
    )
    if not isinstance(data, list):
        return None
    out: list[dict] = []
    for n in data:
        if not isinstance(n, dict) or not n.get("id"):
            continue
        if n.get("has_no_pending_rental") is False:
            continue
        if _is_b200(n.get("machine_name") or ""):
            out.append(n)
    return out


def pods_or_none(sess: requests.Session) -> list[dict] | None:
    data = _get_json(sess, "/pods")
    if not isinstance(data, list):
        return None
    return [p for p in data if isinstance(p, dict)]


def name_live(sess: requests.Session, name: str) -> bool | None:
    pods = pods_or_none(sess)
    if pods is None:
        try:
            raw = subprocess.check_output(
                ["lium", "ps", "--format", "json"], text=True, timeout=60
            )
            pods = json.loads(raw)
            if isinstance(pods, dict):
                pods = pods.get("pods") or pods.get("data") or []
        except Exception as e:
            log(f"live-check fail: {e}")
            return None
    for p in pods or []:
        n = p.get("pod_name") or p.get("name") or ""
        if n == name:
            return True
    return False


def balance_ok() -> bool:
    try:
        raw = subprocess.check_output(["lium", "balance"], text=True, timeout=30)
    except Exception:
        return True
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw.replace(",", ""))
    if not m:
        return True
    return float(m.group(1)) >= 10000.0


def _ttl_hours(ttl: str) -> int:
    m = re.fullmatch(r"\s*(\d+)\s*([hmdHMD]?)\s*", ttl or "")
    if not m:
        return 24
    n = int(m.group(1))
    unit = (m.group(2) or "h").lower()
    if unit == "m":
        return max(1, (n + 59) // 60)
    if unit == "d":
        return n * 24
    return n


def _ssh_pubkey() -> str:
    try:
        return SSH_PUBKEY_PATH.read_text().strip()
    except Exception:
        return ""


def _schedule_ttl(sess: requests.Session, pod_id: str, hours: int) -> None:
    when = datetime.now(timezone.utc) + timedelta(hours=hours)
    iso = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        r = sess.post(
            f"{BASE}/pods/{pod_id}/schedule-removal",
            json={"removal_scheduled_at": iso},
            timeout=30,
        )
        if r.ok:
            log(f"TTL ok pod={pod_id} removal={iso}")
        else:
            log(f"TTL fail http={r.status_code} body={r.text[:160]}")
    except Exception as e:
        log(f"TTL err: {e}")


def try_rent_api(node_id: str, name: str) -> bool:
    body = {
        "pod_name": name,
        "template_id": TEMPLATE_ID,
        "gpu_count": 8,
        "initial_port_count": 12,
        "enable_volume_encryption": True,
        "enable_jupyter": False,
        "termination_hours": _ttl_hours(TTL),
    }
    pk = _ssh_pubkey()
    if pk:
        body["user_public_key"] = pk
    log(f"API rent executor={node_id} name={name} ttl={body['termination_hours']}h")
    sess = make_session()
    try:
        r = sess.post(f"{BASE}/executors/{node_id}/rent", json=body, timeout=60)
    except Exception as e:
        log(f"API rent neterr: {e}")
        return False
    if not r.ok:
        log(f"API rent fail http={r.status_code} body={r.text[:220]}")
        return False
    pod_id = None
    try:
        data = r.json()
        if isinstance(data, dict):
            pod_id = data.get("id") or (data.get("pod") or {}).get("id")
    except Exception:
        data = None
    log(f"API rent ok pod={pod_id or '?'} http={r.status_code}")
    if pod_id:
        _schedule_ttl(sess, str(pod_id), _ttl_hours(TTL))
    return True


def try_rent_cli(node_id: str, name: str) -> bool:
    cmd = [
        "lium", "up", node_id,
        "--name", name,
        "--ttl", TTL,
        "--no-ssh", "-y",
        "--ports", "12",
    ]
    log(f"CLI fallback: {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, timeout=180).returncode == 0
    except Exception as e:
        log(f"CLI rent fail: {e}")
        return False


def write_stamp(gpu_note: str) -> None:
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    try:
        ps = subprocess.check_output(
            ["lium", "ps", "--format", "json"], text=True, timeout=60
        )
    except Exception as e:
        ps = f"err:{e}"
    STAMP.write_text(
        json.dumps(
            {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "name": NAME,
                "gpu": gpu_note,
                "ttl": TTL,
                "purpose": "dedicated GLM-4.5-Air-FP8 teacher (TP experiment)",
                "ps_json": ps[:5000],
            },
            indent=2,
        )
        + "\n"
    )
    log(f"STAMP_OK {STAMP}")


def resolve_ssh(name: str) -> str:
    try:
        raw = subprocess.check_output(
            ["lium", "ps", "--format", "json"], text=True, timeout=60
        )
        pods = json.loads(raw)
        if isinstance(pods, dict):
            pods = pods.get("pods") or []
    except Exception:
        return ""
    for p in pods or []:
        if (p.get("name") or p.get("pod_name")) != name:
            continue
        cmd = p.get("ssh_cmd") or p.get("ssh_connect_cmd") or ""
        m = re.search(r"ssh\s+root@(\S+)\s+-p\s+(\d+)", cmd)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
    return ""


def main() -> int:
    EXP.joinpath("logs").mkdir(parents=True, exist_ok=True)
    EXP.joinpath("artifacts").mkdir(parents=True, exist_ok=True)
    if PIDF.exists():
        try:
            old = int(PIDF.read_text().strip())
        except ValueError:
            old = 0
        if old and old != os.getpid():
            try:
                os.kill(old, 0)
                cmd = Path(f"/proc/{old}/cmdline").read_bytes().replace(b"\0", b" ").decode()
                if "wait_rent_teacher_b200" in cmd:
                    log(f"ABORT already running pid={old}")
                    return 1
            except OSError:
                pass
    PIDF.write_text(str(os.getpid()) + "\n")

    # Tee to log when not already redirected.
    if not os.environ.get("TEACHER_B200_NO_TEE"):
        logf = open(LOG, "a", buffering=1)

        class Tee:
            def write(self, s):
                sys.__stdout__.write(s)
                logf.write(s)
                return len(s)

            def flush(self):
                sys.__stdout__.flush()
                logf.flush()

        sys.stdout = Tee()  # type: ignore[assignment]
        sys.stderr = sys.stdout  # type: ignore[assignment]

    log(f"start name={NAME} ttl={TTL} max_iters={MAX_ITERS} empty_sleep={EMPTY_SLEEP}")
    if not NAME.startswith("affine-"):
        log(f"REFUSE unexpected name={NAME} (want affine-*)")
        return 2

    sess = make_session()
    live = name_live(sess, NAME)
    if live is True:
        log("already live — stamp and exit")
        write_stamp("already-live")
        ssh = resolve_ssh(NAME)
        if ssh:
            log(f"SSH {ssh}")
        return 0

    for i in range(1, MAX_ITERS + 1):
        if not balance_ok():
            log("balance < $10k — stop")
            return 3
        nodes = list_b200_8x(sess)
        if nodes is None:
            log(f"iter={i} executors fail — sleep")
            time.sleep(2.0)
            continue
        if not nodes:
            if i == 1 or i % 60 == 0:
                log(f"iter={i} 8×B200 stock=0")
            time.sleep(EMPTY_SLEEP)
            continue
        log(f"iter={i} sighted {len(nodes)}× 8×B200 — renting")
        for n in nodes:
            nid = str(n["id"])
            mn = n.get("machine_name") or "?"
            price = n.get("price_per_hour") or n.get("price") or "?"
            log(f"try {mn} id={nid[:12]}… ${price}/h")
            if try_rent_api(nid, NAME) or try_rent_cli(nid, NAME):
                # Confirm appearance
                for _ in range(30):
                    if name_live(make_session(), NAME):
                        write_stamp(f"B200 {mn}")
                        ssh = resolve_ssh(NAME)
                        log(f"RENTED ok ssh={ssh or 'pending'}")
                        return 0
                    time.sleep(2)
                log("rent reported ok but name not in ps yet — keep trying")
            # Name may have been taken mid-flight
            if name_live(make_session(), NAME):
                write_stamp(f"B200 {mn}")
                log("name live after race")
                return 0
        time.sleep(EMPTY_SLEEP)
    log("max iters exhausted")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
