#!/usr/bin/env python
"""Rent, bootstrap, watch and release the research teacher boxes on Lium.

Research-only twin of ops/teacher-swarm/manager.py: one vLLM 0.28.0 replica
of the frozen teacher (Qwen/Qwen3.8-27B) per box, the echo-cache plugin
installed, no router, never advertised to production. Every rent / release
is appended to a ledger (pod name, type, $/h, timestamps) so spend can be
audited after the fact.

Secrets: LIUM_API_KEY and HF_TOKEN from the environment (never printed). Per-box
state (including the bearer the replica enforces) lives in
$HINTS_STATE_DIR/pods.json, mode 0600.

  python pods.py rent --type h200-1x [--name hints-a]
  python pods.py bootstrap             # push env + script to every pod we own
  python pods.py status                # /v1/models + canary per pod
  python pods.py wait --timeout-min 50 # block until every pod serves
  python pods.py release --all | --name NAME
  python pods.py ledger                # spend so far
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "ops" / "teacher-swarm"))

import lium_api  # noqa: E402

STATE_DIR = Path(os.environ.get("HINTS_STATE_DIR", "/tmp/hints-secrets"))
STATE_JSON = STATE_DIR / "pods.json"
KNOWN_HOSTS = STATE_DIR / "pod_known_hosts"
POD_KEY = STATE_DIR / "pod_key"
LEDGER = Path(os.environ.get("HINTS_LEDGER", str(HERE / "state" / "ledger.jsonl")))
SWARM_DIR = REPO / "ops" / "teacher-swarm"
BOOTSTRAP = SWARM_DIR / "bootstrap_pod.sh"
ECHO_PLUGIN = SWARM_DIR / "echo_cache_plugin"

POD_PREFIX = "hints-"
TEMPLATE_ID = "345273fa-4818-46f7-a8fa-32f0e331713c"   # Pytorch (Cuda+DinD) cu13.0.2, same as the swarm
MODEL = "Qwen/Qwen3.8-27B"
VLLM_VERSION = "0.28.0"
MAX_MODEL_LEN = 131072
BATCHED_TOKENS = 8192
TTL_HOURS = 24
BUDGET_USD_HR = 60.0
# Match the eval pod's serving flags for the GDN layers.
VLLM_EXTRA = "--additional-config {\"gdn_prefill_backend\":\"triton\"}"


@dataclass(frozen=True)
class TypePlan:
    name: str
    match: str
    gpu_count: int
    tp: int
    gpu_util: float
    max_price: float


TYPES = {
    "h200-1x": TypePlan("h200-1x", "H200", 1, 1, 0.80, 5.0),
    "b200-1x": TypePlan("b200-1x", "B200", 1, 1, 0.75, 7.0),
    "b300-1x": TypePlan("b300-1x", "B300", 1, 1, 0.75, 8.5),
    "h100-8x": TypePlan("h100-8x", "H100", 8, 2, 0.85, 24.0),
}

SSH_OPTS = [
    "-i", str(POD_KEY),
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
    "-o", "ConnectTimeout=15",
    "-o", "BatchMode=yes",
    "-o", "LogLevel=ERROR",
]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    print(f"[pods] {now_iso()} {msg}", flush=True)


def load_state() -> dict:
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())
    return {"pods": {}}


def save_state(st: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
    os.chmod(tmp, 0o600)
    tmp.replace(STATE_JSON)


def ledger_add(event: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"at": now_iso(), **event}) + "\n")


def ssh_run(host: str, port: int, cmd: str, *, input_text: str | None = None,
            timeout: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd],
                          input=input_text, capture_output=True, text=True,
                          timeout=timeout)


def scp_put(host: str, port: int, local: Path, remote: str) -> bool:
    p = subprocess.run(["scp", *SSH_OPTS, "-P", str(port), str(local),
                        f"root@{host}:{remote}"], capture_output=True, text=True,
                       timeout=180)
    return p.returncode == 0


def my_pods(sess) -> dict[str, dict]:
    pods = lium_api.pods(sess) or []
    return {lium_api.pod_name(p): p for p in pods
            if lium_api.pod_name(p).startswith(POD_PREFIX)}


def match_stock(stock: list[dict], plan: TypePlan) -> list[dict]:
    out = []
    for n in stock:
        if int(n.get("gpu_count") or 0) != plan.gpu_count:
            continue
        avail = n.get("available_gpu_count")
        if avail is not None and int(avail) < plan.gpu_count:
            continue
        mn = (n.get("machine_name") or "").upper()
        if plan.match.upper() not in mn:
            continue
        if plan.match.upper() == "B200" and "B300" in mn:
            continue
        price = (n.get("price_per_gpu") or 1e9) * plan.gpu_count
        if price > plan.max_price:
            continue
        out.append(n)
    out.sort(key=lambda n: n.get("price_per_gpu") or 1e9)
    return out


def cmd_rent(args) -> None:
    plan = TYPES[args.type]
    sess = lium_api.session()
    st = load_state()
    mine = my_pods(sess)
    spend = sum(lium_api.pod_price(p) for p in mine.values())
    stock = lium_api.executors(sess)
    cands = match_stock(stock, plan)
    if not cands:
        raise SystemExit(f"no {plan.name} stock under ${plan.max_price}/h")
    pubkey = Path(str(POD_KEY) + ".pub").read_text().strip()
    name = args.name or f"{POD_PREFIX}{plan.name}-{secrets.token_hex(2)}"
    for cand in cands:
        price = (cand.get("price_per_gpu") or 0) * plan.gpu_count
        if spend + price > BUDGET_USD_HR:
            raise SystemExit(f"${spend:.2f}+${price:.2f} > ${BUDGET_USD_HR}/h cap")
        res = lium_api.rent(sess, cand["id"], name, plan.gpu_count, TEMPLATE_ID,
                            TTL_HOURS, pubkey)
        if res in (None, "RATE_LIMITED"):
            log(f"rent on {str(cand['id'])[:8]} failed ({res}); next candidate")
            continue
        st["pods"][name] = {
            "type": plan.name, "executor_id": str(cand["id"]),
            "machine": cand.get("machine_name"), "price": price,
            "rented_at": time.time(), "key": secrets.token_hex(24),
            "pod_id": res,
        }
        save_state(st)
        ledger_add({"event": "rent", "pod": name, "type": plan.name,
                    "machine": cand.get("machine_name"), "usd_per_hour": price,
                    "executor": str(cand["id"])})
        log(f"rented {name}: {cand.get('machine_name')} ${price:.2f}/h")
        return
    raise SystemExit("every candidate refused the rent")


def _bootstrap_one(name: str, pod: dict, mem: dict) -> bool:
    ssh = lium_api.parse_ssh(pod)
    ports = lium_api.data_ports(pod)
    if not ssh or not ports:
        log(f"{name}: no ssh / data ports listed yet")
        return False
    host, port = ssh
    plan = TYPES[mem["type"]]
    internal = sorted(ports)[0]
    gpus = ",".join(str(g) for g in range(plan.tp))
    lines = [
        f'HF_TOKEN="{os.environ.get("HF_TOKEN", "")}"',
        f'SWARM_KEY="{mem["key"]}"',
        f'MODEL="{MODEL}"',
        f'VLLM_VERSION="{VLLM_VERSION}"',
        f'REPLICAS="{internal}:{gpus}:{plan.tp}"',
        f'MAX_MODEL_LEN="{MAX_MODEL_LEN}"',
        f'GPU_UTIL="{plan.gpu_util}"',
        f'BATCHED_TOKENS="{BATCHED_TOKENS}"',
        f"EXTRA_VLLM_ARGS='{VLLM_EXTRA}'",
    ]
    try:
        p = ssh_run(host, port, "mkdir -p /root/swarm/echo_cache_plugin && umask 077 "
                                "&& cat > /root/swarm/env",
                    input_text="\n".join(lines) + "\n")
    except subprocess.SubprocessError as e:
        log(f"{name}: ssh not ready ({e!r})")
        return False
    if p.returncode != 0:
        log(f"{name}: env push failed: {p.stderr.strip()[:160]}")
        return False
    ok = scp_put(host, port, BOOTSTRAP, "/root/swarm/bootstrap.sh")
    for fname in ("pyproject.toml", "affine_vllm_echo_cache.py"):
        ok = ok and scp_put(host, port, ECHO_PLUGIN / fname,
                            f"/root/swarm/echo_cache_plugin/{fname}")
    if not ok:
        log(f"{name}: upload failed")
        return False
    launch = ("if [ -f /root/swarm/boot.pid ] && kill -0 $(cat /root/swarm/boot.pid) "
              "2>/dev/null; then echo already-running; else setsid nohup bash "
              "/root/swarm/bootstrap.sh >> /root/swarm/bootstrap.log 2>&1 < /dev/null & "
              "echo $! > /root/swarm/boot.pid; echo started; fi")
    p = ssh_run(host, port, launch)
    log(f"{name}: {p.stdout.strip() or p.stderr.strip()[:120]}")
    mem["boot_started"] = mem.get("boot_started") or time.time()
    mem["ssh"] = [host, port]
    mem["internal_port"] = internal
    mem["base_url"] = f"http://{lium_api.pod_ip(pod)}:{ports[internal]}/v1"
    return p.returncode == 0


def cmd_bootstrap(args) -> None:
    sess = lium_api.session()
    st = load_state()
    mine = my_pods(sess)
    for name, mem in st["pods"].items():
        if name not in mine:
            log(f"{name}: not in the Lium listing yet")
            continue
        _bootstrap_one(name, mine[name], mem)
    save_state(st)


def probe(mem: dict) -> dict:
    base = mem.get("base_url")
    if not base:
        return {"up": False}
    headers = {"Authorization": f"Bearer {mem['key']}"}
    try:
        r = httpx.get(f"{base}/models", headers=headers, timeout=8.0)
        if r.status_code != 200:
            return {"up": False, "http": r.status_code}
        c = httpx.post(f"{base}/completions", headers=headers, timeout=60.0, json={
            "model": MODEL, "prompt": "1, 2, 3, 4,", "max_tokens": 4,
            "temperature": 0})
        text = c.json()["choices"][0]["text"]
        return {"up": True, "canary": text.strip()[:12]}
    except (httpx.HTTPError, KeyError, ValueError) as e:
        return {"up": False, "err": type(e).__name__}


def cmd_status(args) -> None:
    st = load_state()
    sess = lium_api.session()
    mine = my_pods(sess)
    for name, mem in st["pods"].items():
        listed = name in mine
        pr = probe(mem) if listed else {"up": False}
        age_h = (time.time() - mem["rented_at"]) / 3600
        print(f"{name:28s} {mem['type']:9s} ${mem['price']:.2f}/h "
              f"age={age_h:.1f}h listed={listed} {pr}")


def cmd_wait(args) -> None:
    deadline = time.time() + args.timeout_min * 60
    st = load_state()
    while time.time() < deadline:
        sess = lium_api.session()
        mine = my_pods(sess)
        pending = []
        for name, mem in st["pods"].items():
            if name not in mine:
                pending.append(name)
                continue
            if not mem.get("base_url"):
                _bootstrap_one(name, mine[name], mem)
                save_state(st)
            pr = probe(mem)
            if pr.get("up"):
                if not mem.get("ready_at"):
                    mem["ready_at"] = time.time()
                    save_state(st)
                    ledger_add({"event": "ready", "pod": name,
                                "boot_min": round((time.time() - mem.get("boot_started", time.time())) / 60, 1)})
                    log(f"{name}: READY canary={pr.get('canary')!r}")
            else:
                pending.append(name)
        if not pending:
            log("all pods serve")
            return
        log(f"waiting on {pending}")
        time.sleep(60)
    raise SystemExit("timeout waiting for pods")


def api_remove(sess, name: str, mine: dict[str, dict]) -> bool:
    """DELETE /pods/{id}; verified by the pod leaving the listing."""
    if not name.startswith(POD_PREFIX):
        raise ValueError(f"refuse to rm non-hints pod {name!r}")
    pod = mine.get(name)
    if pod is None:
        return True
    pid = pod.get("id") or pod.get("pod_id")
    try:
        r = sess.delete(f"{lium_api.BASE}/pods/{pid}", timeout=60)
    except Exception as e:  # noqa: BLE001 — network error, report and retry later
        log(f"{name}: delete error {e!r}")
        return False
    if not r.ok:
        log(f"{name}: delete http={r.status_code} {r.text[:120]}")
        return False
    time.sleep(5)
    return name not in my_pods(sess)


def cmd_release(args) -> None:
    st = load_state()
    sess = lium_api.session()
    mine = my_pods(sess)
    names = list(st["pods"]) if args.all else [args.name]
    for name in names:
        if not name:
            continue
        mem = st["pods"].get(name, {})
        ok = api_remove(sess, name, mine)
        hours = (time.time() - mem.get("rented_at", time.time())) / 3600
        ledger_add({"event": "release", "pod": name, "ok": ok,
                    "hours": round(hours, 2),
                    "usd": round(hours * mem.get("price", 0.0), 2)})
        log(f"released {name}: ok={ok} {hours:.2f}h ≈ ${hours * mem.get('price', 0):.2f}")
        if ok:
            st["pods"].pop(name, None)
    save_state(st)


def cmd_ledger(args) -> None:
    total = 0.0
    live = {}
    for line in LEDGER.read_text().splitlines() if LEDGER.exists() else []:
        e = json.loads(line)
        if e["event"] == "rent":
            live[e["pod"]] = (e["usd_per_hour"], e["at"])
        elif e["event"] == "release":
            total += e.get("usd", 0.0)
            live.pop(e["pod"], None)
    for name, (rate, at) in live.items():
        t0 = time.mktime(time.strptime(at, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
        hrs = (time.time() - t0) / 3600
        total += hrs * rate
        print(f"live {name} ${rate:.2f}/h {hrs:.2f}h")
    print(f"GPU spend so far ≈ ${total:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rent")
    r.add_argument("--type", required=True, choices=sorted(TYPES))
    r.add_argument("--name")
    sub.add_parser("bootstrap")
    sub.add_parser("status")
    w = sub.add_parser("wait")
    w.add_argument("--timeout-min", type=int, default=50)
    rl = sub.add_parser("release")
    rl.add_argument("--all", action="store_true")
    rl.add_argument("--name")
    sub.add_parser("ledger")
    args = ap.parse_args()
    {"rent": cmd_rent, "bootstrap": cmd_bootstrap, "status": cmd_status,
     "wait": cmd_wait, "release": cmd_release, "ledger": cmd_ledger}[args.cmd](args)


if __name__ == "__main__":
    main()
