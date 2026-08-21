"""Enroll mining-fleet teacher engines as swarm router backends.

The mining pods each keep a production teacher replica (GLM-4.5-Air-FP8,
port 8000, TP2) warm for their own n80 duels. This daemon lets those
replicas moonlight in the chain-duel teacher pool at zero extra rent:

  every REFRESH_S:
    1. list live mine-* pods from Lium
    2. keep one ssh -L tunnel per pod (local 127.0.0.1:<PORT_BASE+i> -> pod :8000)
    3. probe /v1/models through each tunnel; healthy pods become backends
    4. atomically rewrite state/state.json for router.py (mtime-watched)

Pod churn (the mining loop reaps/re-rents constantly) is handled by steps
1-3; a vanished pod's tunnel dies, its probe fails, and the next rewrite
drops it. The router's own circuit breaker covers mid-window deaths.

est_tps weights are set below the old dedicated-swarm replicas (0.5-0.63)
so chain traffic soaks idle teacher cycles without starving the mining
loop's own n80 evals, which share these engines.

Run under pm2 (affine-swarm-enroll). Requires ~/.lium/config.ini API key.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lium_api

HERE = Path(__file__).resolve().parent
STATE_JSON = HERE / "state" / "state.json"
KNOWN_HOSTS = HERE / "state" / "known_hosts"
MODEL = "zai-org/GLM-4.5-Air-FP8"
TEACHER_PORT = 8000
PORT_BASE = 9210
REFRESH_S = 120
PROBE_TIMEOUT_S = 4

# Gentle weights by GPU class (dedicated swarm replicas ran 0.5-0.63).
def est_tps(gpu: str) -> float:
    g = gpu.lower()
    if "b300" in g or "b200" in g:
        return 0.50
    if "h200" in g:
        return 0.45
    return 0.35


class Tunnel:
    def __init__(self, pod_name: str, host: str, ssh_port: int, local_port: int):
        self.pod_name, self.host, self.ssh_port = pod_name, host, ssh_port
        self.local_port = local_port
        self.proc: subprocess.Popen | None = None

    def ensure(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        self.proc = subprocess.Popen(
            ["ssh", "-N",
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
             "-o", "ConnectTimeout=10",
             "-o", "ServerAliveInterval=15",
             "-o", "ServerAliveCountMax=3",
             "-o", "ExitOnForwardFailure=yes",
             "-L", f"127.0.0.1:{self.local_port}:127.0.0.1:{TEACHER_PORT}",
             "-p", str(self.ssh_port), f"root@{self.host}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def probe(local_port: int) -> bool:
    try:
        r = requests.get(f"http://127.0.0.1:{local_port}/v1/models",
                         timeout=PROBE_TIMEOUT_S)
        return r.ok and MODEL in r.text
    except requests.RequestException:
        return False


def log(msg: str) -> None:
    print(f"[enroll] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}",
          flush=True)


def main() -> None:
    tunnels: dict[str, Tunnel] = {}      # pod name -> tunnel
    ports: dict[str, int] = {}           # pod name -> local port
    next_port = PORT_BASE
    last_backends: list[str] = []

    while True:
        sess = lium_api.session()
        pods = lium_api.pods(sess) or []
        mine = {}
        for p in pods:
            name = lium_api.pod_name(p)
            if not name.startswith("mine-"):
                continue
            hp = lium_api.parse_ssh(p)
            if not hp:
                continue
            ex = p.get("executor") or {}
            mine[name] = (hp, str(ex.get("machine_name") or ""))

        # Reap tunnels for pods that no longer exist.
        for name in list(tunnels):
            if name not in mine:
                tunnels.pop(name).kill()
                log(f"dropped {name} (pod gone)")

        # Ensure a tunnel per live pod.
        for name, ((host, ssh_port), _gpu) in mine.items():
            if name not in ports:
                ports[name] = next_port
                next_port += 1
            t = tunnels.get(name)
            if t is None or (t.host, t.ssh_port) != (host, ssh_port):
                if t:
                    t.kill()
                t = tunnels[name] = Tunnel(name, host, ssh_port, ports[name])
            t.ensure()

        time.sleep(5)  # let fresh tunnels establish before probing

        backends = []
        for name, (_hp, gpu) in sorted(mine.items()):
            lp = ports[name]
            if probe(lp):
                backends.append({
                    "url": f"http://127.0.0.1:{lp}/v1",
                    "pod": name,
                    "type": "miner-tk",
                    "est_tps": est_tps(gpu),
                })

        state = {
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": MODEL,
            "spend_usd_hr": 0.0,   # riding already-rented mining pods
            "source": "enroll_miners",
            "backends": backends,
        }
        tmp = STATE_JSON.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, STATE_JSON)

        names = [b["pod"] for b in backends]
        if names != last_backends:
            log(f"{len(backends)}/{len(mine)} miner teachers enrolled: "
                + ", ".join(names))
            last_backends = names
        time.sleep(REFRESH_S)


if __name__ == "__main__":
    main()
