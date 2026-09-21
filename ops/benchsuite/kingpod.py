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
import contextlib
import fcntl
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

import lium_api  # noqa: E402

sys.path.insert(0, str(REPO / "ops" / "pods"))
import registry as pod_registry  # noqa: E402

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


@contextlib.contextmanager
def pods_lock():
    """Serialise read-modify-write of pods.json: two passes renting at once
    (challengers run in parallel) clobbered each other's entry on 2026-09-14
    and one pod went untracked."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / "pods.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def save_pods(pods: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PODS_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(pods, indent=1, sort_keys=True))
    os.chmod(tmp, 0o600)
    tmp.replace(PODS_JSON)


def update_pod(name: str, **fields) -> dict:
    """Atomic merge of `fields` into one pod's record (re-reads under the lock)."""
    with pods_lock():
        pods = load_pods()
        mem = pods.setdefault(name, {})
        mem.update(fields)
        save_pods(pods)
        return mem


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


BLACKLIST = STATE_DIR / "blacklist.txt"   # executor ids that bootstrapped too slowly or failed


def blacklist_ids() -> set[str]:
    if not BLACKLIST.exists():
        return set()
    return {l.split()[0] for l in BLACKLIST.read_text().splitlines() if l.strip()}


def match_stock(stock: list[dict], plan: dict) -> list[dict]:
    out = []
    bl = blacklist_ids()
    for n in stock:
        if str(n.get("id")) in bl:
            continue
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
    if args.r2 and not (os.environ.get("AFFINE_EVAL_R2_ACCESS_KEY_ID") and os.environ.get("AFFINE_EVAL_R2_SECRET_ACCESS_KEY")):
        raise SystemExit("--r2 needs AFFINE_EVAL_R2_ACCESS_KEY_ID / AFFINE_EVAL_R2_SECRET_ACCESS_KEY in the environment")
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
        update_pod(name, digest=digest, served=f"king-{digest[:12]}", plan=plan, r2=args.r2 or "", hf=args.hf or "",
                   executor_id=str(cand["id"]), machine=cand.get("machine_name"), price=price,
                   rented_at=time.time(), key=secrets.token_hex(24), state="rented")
        # coverage audit 2026-09-19: a bench box lives <= 14 h; the pass log that
        # names it (or a live process) is its owner, the reaper releases the rest
        pod_registry.register(name, purpose="bench", owner="passlog:ops/benchsuite/state",
                              expected_hours=14, price_usd_h=price, ttl_hours=int(cfg["ttl_hours"]),
                              meta={"digest": digest[:12], "plan": plan["name"], "r2": bool(args.r2), "hf": bool(args.hf)})
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
        f'VLLM_VERSION="{cfg["vllm_version"]}"',
    ]
    if mem.get("hf"):
        # genesis: an HF repo at a pinned revision (bootstrap_king.sh's HF_MODEL path);
        # DIGEST stays empty so the public-copy download path is not taken
        repo, _, rev = mem["hf"].partition("@")
        lines += ['DIGEST=""', f'HF_MODEL="{repo}"', f'HF_REV="{rev or "main"}"',
                  f'HF_TOKEN="{os.environ.get("HF_TOKEN", "")}"']
    else:
        lines.append(f'DIGEST="{mem["digest"]}"')
    if mem.get("r2"):
        # a private (challenger) ref: the pod downloads with the eval pods' read-only key
        endpoint = os.environ.get("AFFINE_EVAL_R2_ENDPOINT") or os.environ.get("R2_ENDPOINT") or ""
        lines += [f'KING_R2="{mem["r2"]}"', f'AFFINE_EVAL_R2_ENDPOINT="{endpoint}"',
                  f'AFFINE_EVAL_R2_ACCESS_KEY_ID="{os.environ.get("AFFINE_EVAL_R2_ACCESS_KEY_ID", "")}"',
                  f'AFFINE_EVAL_R2_SECRET_ACCESS_KEY="{os.environ.get("AFFINE_EVAL_R2_SECRET_ACCESS_KEY", "")}"']
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
               front_internal=front, front_external=ports[front],
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
                       json={"model": mem["served"], "max_tokens": 64, "temperature": 0,
                             "messages": [{"role": "user", "content": "Say OK."}]})
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        # vLLM 0.28's qwen3 reasoning parser returns the thinking as `reasoning`
        # (older builds: `reasoning_content`); a short canary may be all thinking.
        return bool(msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning"))
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return False


def cmd_wait(args: argparse.Namespace) -> int:
    cfg = load_suite()["pods"]
    pods = load_pods()
    mem = pods.get(args.name)
    if mem is None:
        raise SystemExit(f"unknown pod {args.name}")
    sess = lium_api.session()
    if mem.get("state") == "ready":
        # an already-serving box (2026-09-21: start_env_backfill re-runs on a live box waited the full
        # hour here, then released the box as "never became ready")
        if probe(mem):
            print(mem["base_url"]); return 0
        mem["state"] = "booting"
    deadline = time.time() + int(cfg["bootstrap_timeout_min"]) * 60
    while time.time() < deadline:
        pod = find_pod(sess, args.name)
        if mem["state"] == "rented" and pod is not None:
            if bootstrap(args.name, mem, pod, cfg):
                update_pod(args.name, **mem)
        elif mem["state"] == "booting":
            if probe(mem):
                mem.update(state="ready", ready_at=time.time())
                update_pod(args.name, **mem)
                log(f"{args.name}: READY at {mem['base_url']}")
                print(mem["base_url"])
                return 0
            if pod is not None:
                ssh = lium_api.parse_ssh(pod)
                if ssh:
                    try:
                        p = ssh_run(ssh[0], ssh[1],
                                    "cat /root/king/bootstrap.failed 2>/dev/null; "
                                    "tail -n 1 /root/king/bootstrap.log 2>/dev/null",
                                    timeout=30)
                    except subprocess.SubprocessError as e:   # a slow ssh is not a failed pod (2026-09-19: two serving pods marked "never served")
                        log(f"{args.name}: bootstrap log unreadable ({type(e).__name__}); keep waiting")
                        time.sleep(30)
                        continue
                    tail = p.stdout.strip().splitlines()
                    if tail:
                        log(f"{args.name}: {tail[-1][:160]}")
                    if p.stdout.strip() and "FATAL" in p.stdout:
                        log(f"{args.name}: bootstrap FAILED")
                        update_pod(args.name, state="failed")
                        return 3
        time.sleep(30)
    log(f"{args.name}: timeout waiting for READY")
    return 3


def cmd_stock(args: argparse.Namespace) -> int:
    """How many executors match a plan right now (price cap + blacklist applied)."""
    n = len(match_stock(lium_api.executors(lium_api.session()), plan_of(args.plan)))
    print(n)
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    sess = lium_api.session()
    listing = {lium_api.pod_name(p): p for p in (lium_api.pods(sess) or [])}
    for name, mem in load_pods().items():
        pod = listing.get(name)
        up = probe(mem, canary=False) if mem.get("base_url") else False
        age = ((mem.get("released_at") or time.time()) - mem["rented_at"]) / 3600
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
    if args.strike and mem.get("executor_id"):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with BLACKLIST.open("a") as fh:
            fh.write(f"{mem['executor_id']} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {args.strike}\n")
        log(f"{args.name}: executor {mem['executor_id'][:12]} blacklisted ({args.strike})")
    age = (time.time() - mem["rented_at"]) / 3600
    log(f"{args.name}: released={ok} after {age:.2f}h ≈ ${age * mem['price']:.2f}")
    update_pod(args.name, state="released", released_at=time.time(),
               cost_usd=round(age * mem["price"], 2))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rent")
    r.add_argument("--plan", required=True)
    r.add_argument("--digest", required=True)
    r.add_argument("--r2", default="", help="private r2://bucket/prefix/ ref (challenger); needs AFFINE_EVAL_R2_* in the env")
    r.add_argument("--hf", default="", help="Hugging Face repo@revision instead of a public digest (genesis)")
    for c in ("wait", "endpoint"):
        sub.add_parser(c).add_argument("name")
    rel = sub.add_parser("release")
    rel.add_argument("name")
    rel.add_argument("--strike", default="", help="also blacklist the executor, with this reason")
    st = sub.add_parser("stock", help="count of executors matching a plan")
    st.add_argument("--plan", required=True)
    sub.add_parser("status")
    args = ap.parse_args()
    return {"rent": cmd_rent, "wait": cmd_wait, "status": cmd_status, "stock": cmd_stock,
            "endpoint": cmd_endpoint, "release": cmd_release}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
