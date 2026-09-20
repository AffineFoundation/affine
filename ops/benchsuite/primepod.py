#!/usr/bin/env python
"""Prime Intellect GPU pods as a second pod provider for the benchsuite launchers
(2026-09-20, Jacob: wall-clock is the constraint; Lium stock is ~20 GPUs).

kingpod-compatible commands — the launchers (fast_pass.sh, cap_backfill.sh) pick
this tool for any plan whose name starts with "prime-" and read the same
state/pods.json record (key, served, ssh_*, front_internal, base_url):

  primepod.py rent --plan prime-a100-2x --digest <sha256> [--hf repo@rev] [--r2 r2://...]
  primepod.py wait <name>          # create -> ssh up -> pod_bootstrap.sh -> /root/bench/ready -> probe
  primepod.py release <name>       # prime pods terminate
  primepod.py stock --plan <plan>  # offers in stock under the plan's price cap
  primepod.py status

Serving: ops/benchsuite/pod_bootstrap.sh (the 2026-09-12 Prime pass's bootstrap:
vLLM 0.28.0 in /root/venv, qwen3 parsers, nginx on loopback). A digest king is
served as king-<digest12> on 127.0.0.1:8001; an --hf ref (teacher, genesis) goes
through the bootstrap's TEACHER_HF path and is served as "teacher" on :8002 —
the served name is only the API model id, the cell's model label comes from the
launcher. Prime pods expose no data ports, so base_url is pod-local
(http://127.0.0.1:<port>/v1): fine for the chat cells, which run ON the pod;
box-side jobs (Harbor / ARE) need a Lium pod. Chat cells on a Prime pod use the
`prime` sandbox runtime (no docker on the image); the launchers switch on
provider == "prime".

Plans live in suite.toml [pods.prime_plans.<name>]: gpu_type, gpu_count, tp,
max_price (USD/h for the whole pod), image, vllm_cuda (cu129 for the 570-series
drivers of the A100 hosts). The old create/wait/terminate/list commands (pod_id
based) are kept for pod_bootstrap.sh-era scripts.

Requires PRIME_API_KEY and the SSH key registered on the account
(PRIME_SSH_KEY, default ~/.ssh/prime_bench). Never prints a secret.
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
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
API = "https://api.primeintellect.ai/api/v1"
SSH_KEY = os.environ.get("PRIME_SSH_KEY", str(Path.home() / ".ssh" / "prime_bench"))
SSH_OPTS = ["-i", SSH_KEY, "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=" + str(HERE / "state" / "prime_known_hosts"),
            "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]
POD_PREFIX = "bench-prime-"
KING_PORT, TEACHER_PORT = 8001, 8002

sys.path.insert(0, str(HERE))
import kingpod  # noqa: E402  (pods.json helpers: load_pods / update_pod, log format)


def log(msg: str) -> None:
    print(f"[primepod] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def headers() -> dict:
    key = os.environ.get("PRIME_API_KEY") or os.environ.get("PRIME")
    if not key:
        raise SystemExit("PRIME_API_KEY not set")
    os.environ["PRIME_API_KEY"] = key
    return {"Authorization": f"Bearer {key}"}


def availability(gpu_type: str, gpu_count: int) -> list[dict]:
    r = httpx.get(f"{API}/availability/", headers=headers(),
                  params={"gpu_type": gpu_type, "gpu_count": gpu_count}, timeout=60)
    r.raise_for_status()
    out = []
    for _, rows in (r.json() or {}).items():
        for row in rows:
            if "navailable" not in str(row.get("stockStatus")) and int(row.get("gpuCount") or 0) == gpu_count:
                out.append(row)
    out.sort(key=lambda x: (x.get("prices") or {}).get("onDemand") or 1e9)
    return out


def wallet() -> float | None:
    try:
        r = httpx.get(f"{API}/billing/wallet", headers=headers(), timeout=30)
        return float(r.json().get("balance_usd"))
    except Exception:
        return None


def plan_of(name: str) -> dict:
    plans = (SUITE.get("pods") or {}).get("prime_plans") or {}
    if name not in plans:
        raise SystemExit(f"unknown prime plan {name}; known: {sorted(plans)}")
    return {"name": name, **plans[name]}


def offers(plan: dict) -> list[dict]:
    return [o for o in availability(plan["gpu_type"], int(plan["gpu_count"]))
            if ((o.get("prices") or {}).get("onDemand") or 1e9) <= float(plan["max_price"])
            and plan.get("image", "ubuntu_22_cuda_12") in (o.get("images") or [plan.get("image", "ubuntu_22_cuda_12")])]


def prime_status(pod_id: str) -> dict:
    p = subprocess.run(["prime", "--plain", "pods", "status", pod_id, "-o", "json"],
                       capture_output=True, text=True, timeout=60, env={**os.environ, "PRIME_DISABLE_VERSION_CHECK": "1"})
    try:
        return json.loads(p.stdout)
    except ValueError:
        return {}


def ssh_run(mem: dict, cmd: str, *, input_text: str | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_OPTS, "-p", str(mem["ssh_port"]), f"{mem['ssh_user']}@{mem['ssh_host']}", cmd],
                          input=input_text, capture_output=True, text=True, timeout=timeout)


def scp_put(mem: dict, local: Path, remote: str) -> bool:
    p = subprocess.run(["scp", *SSH_OPTS, "-P", str(mem["ssh_port"]), str(local), f"{mem['ssh_user']}@{mem['ssh_host']}:{remote}"],
                       capture_output=True, text=True, timeout=120)
    return p.returncode == 0


# ------------------------------------------------------------------ commands
def cmd_rent(a: argparse.Namespace) -> int:
    plan = plan_of(a.plan)
    digest = a.digest
    name = f"{POD_PREFIX}{digest[:12]}-{secrets.token_hex(2)}"
    if a.r2 and not (os.environ.get("AFFINE_EVAL_R2_ACCESS_KEY_ID") and os.environ.get("AFFINE_EVAL_R2_SECRET_ACCESS_KEY")):
        raise SystemExit("--r2 needs AFFINE_EVAL_R2_ACCESS_KEY_ID / AFFINE_EVAL_R2_SECRET_ACCESS_KEY in the environment")
    bal = wallet()
    if bal is not None and bal < float((SUITE.get("pods") or {}).get("prime_min_balance_usd", 50)):
        log(f"Prime balance ${bal:.0f} below the floor; not renting")
        return 2
    stock = offers(plan)
    if not stock:
        log(f"no Prime stock for {plan['gpu_count']}x {plan['gpu_type']} under ${plan['max_price']}/h")
        return 2
    image = plan.get("image", "ubuntu_22_cuda_12")
    for pick in stock:
        price = (pick.get("prices") or {}).get("onDemand")
        cmd = ["prime", "--plain", "pods", "create", "--cloud-id", pick["cloudId"], "--gpu-type", plan["gpu_type"],
               "--gpu-count", str(plan["gpu_count"]), "--name", name, "--image", image, "-y"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env={**os.environ, "PRIME_DISABLE_VERSION_CHECK": "1"})
        out = p.stdout + p.stderr
        pod_id = next((tok for tok in out.split() if len(tok) == 32 and all(c in "0123456789abcdef" for c in tok)), None)
        if p.returncode != 0 or not pod_id:
            log(f"create on {pick.get('provider')} {pick['cloudId']} failed: {out[-200:].strip()}; next offer")
            continue
        kingpod.update_pod(name, provider="prime", pod_id=pod_id, digest=digest, plan={"name": plan["name"], "gpu_count": int(plan["gpu_count"]),
                           "tp": int(plan.get("tp", plan["gpu_count"])), "replicas": 1, "gpu_type": plan["gpu_type"], "vllm_cuda": plan.get("vllm_cuda", "cu130")},
                           served=("teacher" if a.hf else f"king-{digest[:12]}"), r2=a.r2 or "", hf=a.hf or "",
                           executor_id=f"prime:{pick.get('provider')}:{pick['cloudId']}", machine=f"{plan['gpu_count']}x {plan['gpu_type']} ({pick.get('provider')})",
                           price=float(price or 0), rented_at=time.time(), key=secrets.token_hex(24), state="rented", ssh_user="root")
        log(f"rented {name}: {plan['name']} on {pick.get('provider')} {pick['cloudId']} ${price}/h pod_id={pod_id} (balance ${bal if bal is None else round(bal)})")
        print(name)
        return 0
    log("every offer refused the create")
    return 2


def push_env_and_launch(name: str, mem: dict) -> bool:
    cfg = SUITE["pods"]
    plan = mem["plan"]
    gpus = ",".join(str(i) for i in range(int(plan["gpu_count"])))
    lines = [f'API_KEY="{mem["key"]}"', f'MAX_MODEL_LEN="{cfg["max_model_len"]}"', f'GPU_UTIL="{cfg["gpu_memory_utilization"]}"',
             f'BATCHED_TOKENS="{cfg["max_num_batched_tokens"]}"', f'MAX_NUM_SEQS="{cfg["max_num_seqs"]}"',
             f'VLLM_VERSION="{cfg["vllm_version"]}"', f'VLLM_CUDA="{plan.get("vllm_cuda", "cu130")}"']
    if mem.get("hf"):
        repo, _, rev = mem["hf"].partition("@")
        lines += ['KING_REPLICAS=""', f'TEACHER_HF="{repo}"', f'TEACHER_REV="{rev or "main"}"',
                  f'TEACHER_REPLICAS="32001:{gpus}:{plan["tp"]}"', f'HF_TOKEN="{os.environ.get("HF_TOKEN", "")}"']
        front = TEACHER_PORT
    else:
        lines += [f'KING_DIGEST="{mem["digest"]}"', f'KING_REPLICAS="31001:{gpus}:{plan["tp"]}"', 'TEACHER_REPLICAS=""']
        front = KING_PORT
    if mem.get("r2"):
        endpoint = os.environ.get("AFFINE_EVAL_R2_ENDPOINT") or os.environ.get("R2_ENDPOINT") or ""
        lines += [f'KING_R2="{mem["r2"]}"', f'AFFINE_EVAL_R2_ENDPOINT="{endpoint}"',
                  f'AFFINE_EVAL_R2_ACCESS_KEY_ID="{os.environ.get("AFFINE_EVAL_R2_ACCESS_KEY_ID", "")}"',
                  f'AFFINE_EVAL_R2_SECRET_ACCESS_KEY="{os.environ.get("AFFINE_EVAL_R2_SECRET_ACCESS_KEY", "")}"']
    try:
        p = ssh_run(mem, "mkdir -p /root/bench && umask 077 && cat > /root/bench/env", input_text="\n".join(lines) + "\n", timeout=40)
    except subprocess.SubprocessError as e:
        log(f"{name}: env push error {e!r}")
        return False
    if p.returncode != 0:
        log(f"{name}: env push failed: {p.stderr.strip()[:160]}")
        return False
    if not scp_put(mem, HERE / "pod_bootstrap.sh", "/root/bench/bootstrap.sh"):
        log(f"{name}: bootstrap upload failed")
        return False
    launch = ("if [ -f /root/bench/boot.pid ] && kill -0 $(cat /root/bench/boot.pid) 2>/dev/null; then echo already-running; "
              "else setsid nohup bash /root/bench/bootstrap.sh >> /root/bench/bootstrap.log 2>&1 < /dev/null & echo $! > /root/bench/boot.pid; echo started; fi")
    p = ssh_run(mem, launch)
    if p.returncode != 0:
        log(f"{name}: launch failed: {p.stderr.strip()[:160]}")
        return False
    mem.update(state="booting", boot_started=time.time(), front_internal=front, base_url=f"http://127.0.0.1:{front}/v1")
    log(f"{name}: bootstrap launched -> {mem['base_url']} (pod-local)")
    return True


def probe(mem: dict) -> bool:
    try:
        p = ssh_run(mem, f"curl -sf -m 8 -H 'Authorization: Bearer {mem['key']}' http://127.0.0.1:{mem['front_internal']}/v1/models", timeout=30)
        if p.returncode != 0 or mem["served"] not in p.stdout:
            return False
        body = json.dumps({"model": mem["served"], "max_tokens": 64, "temperature": 0, "messages": [{"role": "user", "content": "Say OK."}]})
        p = ssh_run(mem, f"curl -sf -m 170 -H 'Authorization: Bearer {mem['key']}' -H 'Content-Type: application/json' "
                         f"-d '{body}' http://127.0.0.1:{mem['front_internal']}/v1/chat/completions", timeout=190)
        msg = json.loads(p.stdout)["choices"][0]["message"]
        return bool(msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning"))
    except (subprocess.SubprocessError, KeyError, TypeError, ValueError):
        return False


def cmd_wait(a: argparse.Namespace) -> int:
    cfg = SUITE["pods"]
    mem = kingpod.load_pods().get(a.name)
    if mem is None:
        raise SystemExit(f"unknown pod {a.name}")
    headers()
    deadline = time.time() + int(cfg["bootstrap_timeout_min"]) * 60
    while time.time() < deadline:
        if mem["state"] == "rented":
            st = prime_status(mem["pod_id"])
            ssh = st.get("ssh") or ""
            if st.get("status") == "ACTIVE" and "@" in ssh:
                user_host, _, port = ssh.partition(" -p ")
                user, _, host = user_host.partition("@")
                mem.update(ssh_user=user or "root", ssh_host=host.strip(), ssh_port=int(port.strip() or 22))
                try:
                    ok = ssh_run(mem, "echo ok", timeout=40).returncode == 0
                except subprocess.SubprocessError:
                    ok = False
                if ok and push_env_and_launch(a.name, mem):
                    kingpod.update_pod(a.name, **mem)
            elif st.get("status") in ("ERROR", "TERMINATED", "FAILED"):
                log(f"{a.name}: prime status {st.get('status')}")
                kingpod.update_pod(a.name, state="failed")
                return 3
            else:
                log(f"{a.name}: prime status {st.get('status') or '?'}")
        elif mem["state"] == "booting":
            try:
                p = ssh_run(mem, "if [ -f /root/bench/bootstrap.failed ]; then echo FAILED: $(cat /root/bench/bootstrap.failed); fi; "
                                 "test -f /root/bench/ready && echo READY; tail -n 1 /root/bench/bootstrap.log 2>/dev/null", timeout=30)
            except subprocess.SubprocessError as e:
                log(f"{a.name}: bootstrap log unreadable ({type(e).__name__}); keep waiting")
                time.sleep(30)
                continue
            lines = p.stdout.strip().splitlines()
            failed = [l for l in lines if l.startswith("FAILED:")]
            if failed:
                log(f"{a.name}: bootstrap {failed[0][:160]}")
                kingpod.update_pod(a.name, state="failed")
                return 3
            if "READY" in lines and probe(mem):
                mem.update(state="ready", ready_at=time.time())
                kingpod.update_pod(a.name, **mem)
                log(f"{a.name}: READY at {mem['base_url']} (pod-local; ssh {mem['ssh_user']}@{mem['ssh_host']}:{mem['ssh_port']})")
                print(mem["base_url"])
                return 0
            tail = [l for l in lines if l.startswith("[bench-boot]")]
            if tail:
                log(f"{a.name}: {tail[-1][:160]}")
        time.sleep(30)
    log(f"{a.name}: timeout waiting for READY")
    return 3


def cmd_release(a: argparse.Namespace) -> int:
    mem = kingpod.load_pods().get(a.name)
    if mem is None:
        raise SystemExit(f"unknown pod {a.name}")
    headers()
    p = subprocess.run(["prime", "--plain", "pods", "terminate", mem["pod_id"], "-y"], capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PRIME_DISABLE_VERSION_CHECK": "1"})
    ok = p.returncode == 0
    age = (time.time() - mem["rented_at"]) / 3600
    log(f"{a.name}: terminated={ok} after {age:.2f}h ≈ ${age * mem['price']:.2f}")
    kingpod.update_pod(a.name, state="released", released_at=time.time(), cost_usd=round(age * mem["price"], 2))
    return 0 if ok else 1


def cmd_stock(a: argparse.Namespace) -> int:
    plan = plan_of(a.plan)
    n = len(offers(plan))
    print(n)
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    headers()
    bal = wallet()
    print(f"Prime balance: ${bal:.2f}" if bal is not None else "Prime balance: ?")
    for name, mem in sorted(kingpod.load_pods().items()):
        if mem.get("provider") != "prime" or mem.get("state") == "released":
            continue
        age = (time.time() - mem["rented_at"]) / 3600
        print(f"{name}: {mem['plan']['name']} ${mem['price']:.2f}/h {mem['state']} age={age:.1f}h url={mem.get('base_url')} spent≈${age * mem['price']:.2f}")
    return 0


# ------------------------------------------------------------- legacy (pod_id based)
def cmd_create(a: argparse.Namespace) -> int:
    cfg = SUITE["prime"]
    stock = availability(cfg["gpu_type"], int(cfg["gpu_count"]))
    if not stock:
        log(f"no stock for {cfg['gpu_count']}x {cfg['gpu_type']}")
        return 2
    pick = next((s for s in stock if s.get("cloudId") == cfg.get("cloud_id")), stock[0])
    price = (pick.get("prices") or {}).get("onDemand")
    image = cfg["image"] if cfg["image"] in (pick.get("images") or [cfg["image"]]) else (pick.get("images") or [None])[0]
    cmd = ["prime", "--plain", "pods", "create", "--cloud-id", pick["cloudId"], "--gpu-type", cfg["gpu_type"],
           "--gpu-count", str(cfg["gpu_count"]), "--name", a.name, "--image", image, "-y"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    out = p.stdout + p.stderr
    pod_id = next((tok for tok in out.split() if len(tok) == 32 and all(c in "0123456789abcdef" for c in tok)), None)
    if p.returncode != 0 or not pod_id:
        log(f"create failed: {out[-400:]}")
        return 1
    print(json.dumps({"pod_id": pod_id, "provider": pick.get("provider"), "usd_per_hour": price, "cloud_id": pick["cloudId"], "image": image}))
    return 0


def cmd_wait_id(a: argparse.Namespace) -> int:
    deadline = time.time() + a.timeout_min * 60
    while time.time() < deadline:
        st = prime_status(a.pod_id)
        ssh = st.get("ssh") or ""
        if st.get("status") == "ACTIVE" and "@" in ssh:
            user_host, _, port = ssh.partition(" -p ")
            port = port.strip() or "22"
            p = subprocess.run(["ssh", *SSH_OPTS, "-p", port, user_host, "echo ok"], capture_output=True, text=True, timeout=40)
            if p.returncode == 0:
                print(f"{user_host} {port}")
                return 0
        time.sleep(20)
    return 3


def cmd_terminate(a: argparse.Namespace) -> int:
    p = subprocess.run(["prime", "--plain", "pods", "terminate", a.pod_id, "-y"], capture_output=True, text=True, timeout=120)
    return p.returncode


def cmd_list(_: argparse.Namespace) -> int:
    subprocess.run(["prime", "--plain", "pods", "list"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rent")
    r.add_argument("--plan", required=True)
    r.add_argument("--digest", required=True)
    r.add_argument("--r2", default="")
    r.add_argument("--hf", default="")
    sub.add_parser("wait").add_argument("name")
    rel = sub.add_parser("release")
    rel.add_argument("name")
    rel.add_argument("--strike", default="")     # accepted for launcher symmetry; Prime offers are not blacklisted
    sub.add_parser("stock").add_argument("--plan", required=True)
    sub.add_parser("status")
    c = sub.add_parser("create"); c.add_argument("--name", required=True)
    w = sub.add_parser("wait-id"); w.add_argument("pod_id"); w.add_argument("--timeout-min", type=int, default=40)
    sub.add_parser("terminate").add_argument("pod_id")
    sub.add_parser("list")
    a = ap.parse_args()
    return {"rent": cmd_rent, "wait": cmd_wait, "release": cmd_release, "stock": cmd_stock, "status": cmd_status,
            "create": cmd_create, "wait-id": cmd_wait_id, "terminate": cmd_terminate, "list": cmd_list}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
