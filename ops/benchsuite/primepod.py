#!/usr/bin/env python
"""Prime Intellect pod helper for the benchmark suite (create / wait / ssh /
terminate), thin wrapper over the `prime` CLI + the pods API.

  python primepod.py create --name affine-benchsuite-<run_id>   # per [prime] in suite.toml
  python primepod.py wait <pod_id>                               # prints "user@ip port" when ACTIVE + sshable
  python primepod.py terminate <pod_id>
  python primepod.py list

Requires PRIME_API_KEY in the environment and the SSH private key whose
public half is registered on the Prime account (PRIME_SSH_KEY, default
~/.ssh/prime_bench). Never prints a secret.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
API = "https://api.primeintellect.ai/api/v1"
SSH_KEY = os.environ.get("PRIME_SSH_KEY", str(Path.home() / ".ssh" / "prime_bench"))
SSH_OPTS = ["-i", SSH_KEY, "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=" + str(HERE / "state" / "prime_known_hosts"),
            "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]


def log(msg: str) -> None:
    print(f"[primepod] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def headers() -> dict:
    key = os.environ.get("PRIME_API_KEY")
    if not key:
        raise SystemExit("PRIME_API_KEY not set")
    return {"Authorization": f"Bearer {key}"}


def availability(gpu_type: str, gpu_count: int) -> list[dict]:
    r = httpx.get(f"{API}/availability/", headers=headers(),
                  params={"gpu_type": gpu_type, "gpu_count": gpu_count}, timeout=60)
    r.raise_for_status()
    out = []
    for _, rows in (r.json() or {}).items():
        for row in rows:
            if "navailable" not in str(row.get("stockStatus")):
                out.append(row)
    out.sort(key=lambda x: (x.get("prices") or {}).get("onDemand") or 1e9)
    return out


def cmd_create(a: argparse.Namespace) -> int:
    cfg = SUITE["prime"]
    stock = availability(cfg["gpu_type"], int(cfg["gpu_count"]))
    if not stock:
        log(f"no stock for {cfg['gpu_count']}x {cfg['gpu_type']}")
        return 2
    pick = next((s for s in stock if s.get("cloudId") == cfg.get("cloud_id")), stock[0])
    price = (pick.get("prices") or {}).get("onDemand")
    if price and price > float(cfg.get("max_usd_per_hour", 40.0)):
        log(f"cheapest {price}/h exceeds max_usd_per_hour")
        return 2
    image = cfg["image"] if cfg["image"] in (pick.get("images") or [cfg["image"]]) else (pick.get("images") or [None])[0]
    cmd = ["prime", "--plain", "pods", "create", "--cloud-id", pick["cloudId"],
           "--gpu-type", cfg["gpu_type"], "--gpu-count", str(cfg["gpu_count"]),
           "--name", a.name, "--image", image, "-y"]
    if pick.get("provider"):
        pass  # provider is implied by cloud-id
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    out = p.stdout + p.stderr
    pod_id = None
    for tok in out.split():
        if len(tok) == 32 and all(c in "0123456789abcdef" for c in tok):
            pod_id = tok
    if p.returncode != 0 or not pod_id:
        log(f"create failed: {out[-400:]}")
        return 1
    log(f"created {pod_id} ({pick.get('provider')} {cfg['gpu_count']}x {cfg['gpu_type']} ${price}/h)")
    print(json.dumps({"pod_id": pod_id, "provider": pick.get("provider"), "usd_per_hour": price,
                      "gpu": f"{cfg['gpu_count']}x {cfg['gpu_type']}", "cloud_id": pick["cloudId"],
                      "image": image}))
    return 0


def status(pod_id: str) -> dict:
    p = subprocess.run(["prime", "--plain", "pods", "status", pod_id, "-o", "json"],
                       capture_output=True, text=True, timeout=60)
    try:
        return json.loads(p.stdout)
    except ValueError:
        return {}


def cmd_wait(a: argparse.Namespace) -> int:
    deadline = time.time() + a.timeout_min * 60
    while time.time() < deadline:
        st = status(a.pod_id)
        ssh = st.get("ssh") or ""
        if st.get("status") == "ACTIVE" and "@" in ssh:
            user_host, _, port = ssh.partition(" -p ")
            port = port.strip() or "22"
            p = subprocess.run(["ssh", *SSH_OPTS, "-p", port, user_host, "echo ok"],
                               capture_output=True, text=True, timeout=40)
            if p.returncode == 0:
                print(f"{user_host} {port}")
                return 0
        time.sleep(20)
    log("timeout waiting for the pod")
    return 3


def cmd_terminate(a: argparse.Namespace) -> int:
    p = subprocess.run(["prime", "--plain", "pods", "terminate", a.pod_id, "-y"],
                       capture_output=True, text=True, timeout=120)
    log(f"terminate {a.pod_id}: rc={p.returncode} {(p.stdout + p.stderr).strip()[-200:]}")
    return p.returncode


def cmd_list(_: argparse.Namespace) -> int:
    subprocess.run(["prime", "--plain", "pods", "list"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    w = sub.add_parser("wait")
    w.add_argument("pod_id")
    w.add_argument("--timeout-min", type=int, default=40)
    sub.add_parser("terminate").add_argument("pod_id")
    sub.add_parser("list")
    a = ap.parse_args()
    return {"create": cmd_create, "wait": cmd_wait, "terminate": cmd_terminate,
            "list": cmd_list}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
