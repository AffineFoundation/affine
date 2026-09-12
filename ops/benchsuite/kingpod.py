#!/usr/bin/env python
"""Rent, bootstrap, probe and release a dedicated pod that serves one Affine
model (the king) for the benchmark suite.

Reuses the king-seat pieces (ops/king-datagen/bootstrap_king.sh for the
vLLM 0.28 + nginx stack with the chat-box parsers, ops/teacher-swarm/
lium_api.py for the Lium API). Pods are named `bench-king-<digest12>-<hex4>`
so neither kingctl (`king-dg-`) nor the swarm manager touches them.

  python kingpod.py rent --plan h200-1x --digest <sha256>   # rent + launch bootstrap
  python kingpod.py wait <pod-name>                          # block until /v1/models + canary answer
  python kingpod.py status                                   # every bench pod + Lium listing state
  python kingpod.py endpoint <pod-name>                      # base_url + model name (key stays in state)
  python kingpod.py release <pod-name>                       # lium rm

State (per-box bearer included) lives in ops/benchsuite/state/pods.json,
mode 0600. Never prints a secret.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "ops" / "teacher-swarm"))
sys.path.insert(0, str(REPO / "ops" / "king-datagen"))

import kingctl  # noqa: E402  (ssh helpers, env_file_value)
import lium_api  # noqa: E402

STATE_DIR = HERE / "state"
PODS_JSON = STATE_DIR / "pods.json"
KNOWN_HOSTS = STATE_DIR / "known_hosts"
POD_PREFIX = "bench-king-"
REPLICA_PORT_BASE = 31001
SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
    "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR",
]


def log(msg: str) -> None:
    print(f"[kingpod] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}",
          flush=True)


def load_suite() -> dict:
    return tomllib.loads((HERE / "suite.toml").read_text())


def load_pods() -> dict:
    if PODS_JSON.exists():
        return json.loads(PODS_JSON.read_text())
    return {}


def save_pods(pods: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PODS_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(pods, indent=1, sort_keys=True))
    os.chmod(tmp, 0o600)
    tmp.replace(PODS_JSON)


def ssh_run(host: str, port: int, cmd: str, *, input_text: str | None = None,
            timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd],
                          input=input_text, capture_output=True, text=True,
                          timeout=timeout)


def scp_put(host: str, port: int, local: Path, remote: str) -> bool:
    p = subprocess.run(["scp", *SSH_OPTS, "-P", str(port), str(local),
                        f"root@{host}:{remote}"], capture_output=True, text=True,
                       timeout=180)
    return p.returncode == 0


def plan_of(name: str) -> dict:
    plans = load_suite()["pods"]["plans"]
    if name not in plans:
        raise SystemExit(f"unknown plan {name!r}; have {sorted(plans)}")
    return dict(plans[name], name=name)


def match_stock(stock: list[dict], plan: dict) -> list[dict]:
    out = []
    for n in stock:
        if int(n.get("gpu_count") or 0) != plan["gpu_count"]:
            continue
        avail = n.get("available_gpu_count")
        if avail is not None and int(avail) < plan["gpu_count"]:
            continue
        mn = (n.get("machine_name") or "").upper()
        if plan["match"].upper() not in mn:
            continue
        if plan["match"].upper() == "B200" and "B300" in mn:
            continue
        price = (n.get("price_per_gpu") or 1e9) * plan["gpu_count"]
        if price > plan["max_price"]:
            continue
        out.append(n)
    out.sort(key=lambda n: n.get("price_per_gpu") or 1e9)
    return out


def find_pod(sess, name: str) -> dict | None:
    for p in lium_api.pods(sess) or []:
        if lium_api.pod_name(p) == name:
            return p
    return None


def cmd_rent(args: argparse.Namespace) -> int:
    suite = load_suite()
    cfg = suite["pods"]
    plan = plan_of(args.plan)
    digest = args.digest
    name = f"{POD_PREFIX}{digest[:12]}-{secrets.token_hex(2)}"
    sess = lium_api.session()
    pubkey = (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()
    stock = match_stock(lium_api.executors(sess), plan)
    if not stock:
        log(f"no stock for plan {plan['name']} under ${plan['max_price']}/h")
        return 2
    for cand in stock:
        price = (cand.get("price_per_gpu") or 0) * plan["gpu_count"]
        res = lium_api.rent(sess, cand["id"], name, plan["gpu_count"],
                            cfg["template_id"], int(cfg["ttl_hours"]), pubkey)
        if res in (None, "RATE_LIMITED"):
            log(f"rent on {str(cand['id'])[:12]} failed ({res}); next candidate")
            continue
        pods = load_pods()
        pods[name] = {
            "digest": digest, "served": f"king-{digest[:12]}", "plan": plan,
            "executor_id": str(cand["id"]), "machine": cand.get("machine_name"),
            "price": price, "rented_at": time.time(), "key": secrets.token_hex(24),
            "state": "rented",
        }
        save_pods(pods)
        log(f"rented {name}: {plan['name']} {cand.get('machine_name')} "
            f"${price:.2f}/h executor={str(cand['id'])[:12]}")
        print(name)
        return 0
    log("every candidate refused the rent")
    return 2


def bootstrap(name: str, mem: dict, pod: dict, cfg: dict) -> bool:
    ssh = lium_api.parse_ssh(pod)
    ports = lium_api.data_ports(pod)
    if not ssh or not ports:
        return False
    host, port = ssh
    plan = mem["plan"]
    front = sorted(ports)[0]
    specs = []
    for i in range(plan["replicas"]):
        gpus = ",".join(str(g) for g in range(i * plan["tp"], (i + 1) * plan["tp"]))
        specs.append(f"{REPLICA_PORT_BASE + i}:{gpus}:{plan['tp']}")
    lines = [
        f'KING_KEY="{mem["key"]}"', f'SERVED_NAME="{mem["served"]}"',
        f'FRONT_PORT="{front}"', f'REPLICAS="{";".join(specs)}"',
        f'MAX_MODEL_LEN="{cfg["max_model_len"]}"',
        f'GPU_UTIL="{cfg["gpu_memory_utilization"]}"',
        f'BATCHED_TOKENS="{cfg["max_num_batched_tokens"]}"',
        f'MAX_NUM_SEQS="{cfg["max_num_seqs"]}"',
        f'VLLM_VERSION="{cfg["vllm_version"]}"', f'DIGEST="{mem["digest"]}"',
    ]
    subprocess.run(["ssh-keygen", "-R", f"[{host}]:{port}", "-f", str(KNOWN_HOSTS)],
                   capture_output=True)
    try:
        p = ssh_run(host, port, "mkdir -p /root/king && umask 077 && cat > /root/king/env",
                    input_text="\n".join(lines) + "\n", timeout=40)
    except subprocess.SubprocessError as e:
        log(f"{name}: env push error {e!r}")
        return False
    if p.returncode != 0:
        log(f"{name}: env push failed: {p.stderr.strip()[:160]}")
        return False
    if not scp_put(host, port, REPO / "ops/king-datagen/bootstrap_king.sh",
                   "/root/king/bootstrap.sh"):
        log(f"{name}: bootstrap upload failed")
        return False
    launch = ("if [ -f /root/king/boot.pid ] && kill -0 $(cat /root/king/boot.pid) "
              "2>/dev/null; then echo already-running; else setsid nohup bash "
              "/root/king/bootstrap.sh >> /root/king/bootstrap.log 2>&1 < /dev/null & "
              "echo $! > /root/king/boot.pid; echo started; fi")
    p = ssh_run(host, port, launch)
    if p.returncode != 0:
        log(f"{name}: launch failed: {p.stderr.strip()[:160]}")
        return False
    mem.update(state="booting", boot_started=time.time(), ssh_host=host, ssh_port=port,
               base_url=f"http://{lium_api.pod_ip(pod)}:{ports[front]}/v1")
    log(f"{name}: bootstrap launched ({len(specs)} replicas) -> {mem['base_url']}")
    return True


def probe(mem: dict, canary: bool = True) -> bool:
    headers = {"Authorization": f"Bearer {mem['key']}"}
    base = mem.get("base_url")
    if not base:
        return False
    try:
        r = httpx.get(f"{base}/models", headers=headers, timeout=8.0)
        r.raise_for_status()
        if mem["served"] not in {m.get("id") for m in r.json().get("data", [])}:
            return False
        if not canary:
            return True
        r = httpx.post(f"{base}/chat/completions", headers=headers, timeout=180.0,
                       json={"model": mem["served"], "max_tokens": 32, "temperature": 0,
                             "messages": [{"role": "user", "content": "Say OK."}]})
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return bool(msg.get("content") or msg.get("reasoning_content"))
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return False


def cmd_wait(args: argparse.Namespace) -> int:
    cfg = load_suite()["pods"]
    pods = load_pods()
    mem = pods.get(args.name)
    if mem is None:
        raise SystemExit(f"unknown pod {args.name}")
    sess = lium_api.session()
    deadline = time.time() + int(cfg["bootstrap_timeout_min"]) * 60
    while time.time() < deadline:
        pod = find_pod(sess, args.name)
        if mem["state"] == "rented" and pod is not None:
            if bootstrap(args.name, mem, pod, cfg):
                save_pods(pods)
        elif mem["state"] == "booting":
            if probe(mem):
                mem.update(state="ready", ready_at=time.time())
                save_pods(pods)
                log(f"{args.name}: READY at {mem['base_url']}")
                print(mem["base_url"])
                return 0
            if pod is not None:
                ssh = lium_api.parse_ssh(pod)
                if ssh:
                    p = ssh_run(ssh[0], ssh[1],
                                "cat /root/king/bootstrap.failed 2>/dev/null; "
                                "tail -n 1 /root/king/bootstrap.log 2>/dev/null",
                                timeout=30)
                    tail = p.stdout.strip().splitlines()
                    if tail:
                        log(f"{args.name}: {tail[-1][:160]}")
                    if p.stdout.strip() and "FATAL" in p.stdout:
                        log(f"{args.name}: bootstrap FAILED")
                        mem["state"] = "failed"
                        save_pods(pods)
                        return 3
        time.sleep(30)
    log(f"{args.name}: timeout waiting for READY")
    return 3


def cmd_status(_: argparse.Namespace) -> int:
    sess = lium_api.session()
    listing = {lium_api.pod_name(p): p for p in (lium_api.pods(sess) or [])}
    for name, mem in load_pods().items():
        pod = listing.get(name)
        up = probe(mem, canary=False) if mem.get("base_url") else False
        age = (time.time() - mem["rented_at"]) / 3600
        print(f"{name}: {mem['plan']['name']} ${mem['price']:.2f}/h {mem['state']} "
              f"listed={'yes' if pod else 'no'} up={up} age={age:.1f}h "
              f"url={mem.get('base_url')} spent≈${age * mem['price']:.2f}")
    return 0


def cmd_endpoint(args: argparse.Namespace) -> int:
    mem = load_pods()[args.name]
    print(json.dumps({"base_url": mem["base_url"], "model": mem["served"],
                      "key_env": "BENCH_KING_KEY"}))
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    pods = load_pods()
    mem = pods.get(args.name)
    if mem is None:
        raise SystemExit(f"unknown pod {args.name}")
    ok = lium_api.remove(args.name, POD_PREFIX)
    age = (time.time() - mem["rented_at"]) / 3600
    log(f"{args.name}: released={ok} after {age:.2f}h ≈ ${age * mem['price']:.2f}")
    mem.update(state="released", released_at=time.time(),
               cost_usd=round(age * mem["price"], 2))
    save_pods(pods)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rent")
    r.add_argument("--plan", required=True)
    r.add_argument("--digest", required=True)
    for c in ("wait", "endpoint", "release"):
        sub.add_parser(c).add_argument("name")
    sub.add_parser("status")
    args = ap.parse_args()
    return {"rent": cmd_rent, "wait": cmd_wait, "status": cmd_status,
            "endpoint": cmd_endpoint, "release": cmd_release}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
