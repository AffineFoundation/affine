"""Thin Lium HTTP client for the teacher swarm (rent / list / remove).

Mirrors the proven request shapes in ops/teacher-b200/wait_rent_teacher_b200.py.
"""
from __future__ import annotations

import configparser
import json
import os
import re
import subprocess
import time
from pathlib import Path

import requests

BASE = os.environ.get("LIUM_BASE_URL", "https://lium.io/api")


def _api_key() -> str:
    env = os.environ.get("LIUM_API_KEY")
    if env:
        return env
    cfg = configparser.ConfigParser()
    cfg.read(str(Path.home() / ".lium/config.ini"))
    key = cfg.get("api", "api_key",
                  fallback=cfg.get("default", "api_key", fallback=""))
    if not key:
        raise SystemExit("no Lium API key")
    return key


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "X-API-KEY": _api_key(),
        "X-Source": "teacher-swarm",
        "X-Lium-Client-Version": "0.0.32",
    })
    return s


def get_json(sess: requests.Session, path: str, params: dict | None = None,
             retries: int = 5):
    backoff = 0.5
    for _ in range(retries):
        try:
            r = sess.get(f"{BASE}{path}", params=params, timeout=30)
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
            continue
        if r.status_code == 429 or 500 <= r.status_code < 600:
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
            continue
        if not r.ok:
            return None
        try:
            return r.json()
        except ValueError:
            return None
    return None


def executors(sess: requests.Session) -> list[dict]:
    data = get_json(sess, "/executors", params={"size": 2000})
    if not isinstance(data, list):
        return []
    return [n for n in data
            if isinstance(n, dict) and n.get("id")
            and n.get("has_no_pending_rental") is not False]


def pods(sess: requests.Session) -> list[dict] | None:
    """None on API failure (caller must not treat as 'all pods gone')."""
    data = get_json(sess, "/pods")
    if not isinstance(data, list):
        return None
    return [p for p in data if isinstance(p, dict)]


def rent(sess: requests.Session, executor_id: str, pod_name: str,
         gpu_count: int, template_id: str, ttl_hours: int,
         ssh_pubkey: str) -> str | None:
    """Returns pod id on success, None on failure."""
    body = {
        "pod_name": pod_name,
        "template_id": template_id,
        "gpu_count": gpu_count,
        "initial_port_count": 12,
        "enable_volume_encryption": True,
        "enable_jupyter": False,
        "termination_hours": ttl_hours,
    }
    if ssh_pubkey:
        body["user_public_key"] = ssh_pubkey
    try:
        r = sess.post(f"{BASE}/executors/{executor_id}/rent", json=body,
                      timeout=60)
    except requests.RequestException as e:
        print(f"[lium] rent network error: {e}", flush=True)
        return None
    if r.status_code == 429:
        return "RATE_LIMITED"
    if not r.ok:
        print(f"[lium] rent http={r.status_code} body={r.text[:200]}",
              flush=True)
        return None
    # 2xx means the pod is being created even when the body carries no id.
    try:
        data = r.json()
        return data.get("id") or (data.get("pod") or {}).get("id") or "?"
    except ValueError:
        return "?"


def remove(pod_name: str, prefix: str) -> bool:
    """Delete a pod by name via the CLI (proven path). Prefix-guarded."""
    if not pod_name.startswith(prefix):
        raise ValueError(f"refuse to rm non-swarm pod {pod_name!r}")
    p = subprocess.run(["lium", "rm", pod_name, "-y"],
                       capture_output=True, text=True, timeout=180)
    return p.returncode == 0


def balance_usd() -> float | None:
    try:
        raw = subprocess.check_output(["lium", "balance"], text=True,
                                      timeout=30)
    except (subprocess.SubprocessError, OSError):
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw.replace(",", ""))
    return float(m.group(1)) if m else None


def parse_ssh(pod: dict) -> tuple[str, int] | None:
    """(host, port) from a pod record's ssh command string."""
    cmd = pod.get("ssh_cmd") or pod.get("ssh_connect_cmd") or ""
    m = re.search(r"ssh\s+root@(\S+)\s+-p\s+(\d+)", cmd)
    if m:
        return m.group(1), int(m.group(2))
    ip = pod_ip(pod)
    port = (pod.get("ports_mapping") or pod.get("ports") or {}).get("22")
    if ip and port:
        return ip, int(port)
    return None


def pod_ip(pod: dict) -> str:
    cmd = pod.get("ssh_cmd") or pod.get("ssh_connect_cmd") or ""
    m = re.search(r"root@(\S+)", cmd)
    if m:
        return m.group(1)
    ex = pod.get("executor") or {}
    return str(pod.get("ip") or ex.get("executor_ip_address") or "")


def data_ports(pod: dict) -> dict[int, int]:
    """{internal: external} minus ssh/jupyter."""
    out = {}
    for k, v in (pod.get("ports_mapping") or pod.get("ports") or {}).items():
        try:
            ki, vi = int(k), int(v)
        except (TypeError, ValueError):
            continue
        if ki in (22, 8888):
            continue
        out[ki] = vi
    return out


def pod_price(pod: dict) -> float:
    for field in ("price", "price_per_hour"):
        try:
            return float(pod.get(field))
        except (TypeError, ValueError):
            continue
    return 0.0


def pod_name(pod: dict) -> str:
    return str(pod.get("name") or pod.get("pod_name") or "")


def dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)
