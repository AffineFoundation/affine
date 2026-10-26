#!/usr/bin/env python3
"""Rent + provision + manage the fastbench box on Lium.

Standalone (no affine imports) so it runs from the operator machine with just
the lium CLI configured (~/.lium/config.ini or LIUM_API_KEY).

    python ops/rent.py rent          # pick executor, rent, provision
    python ops/rent.py provision     # (re)upload + bootstrap existing pod
    python ops/rent.py status        # pod + /health through the tunnel
    python ops/rent.py tunnel        # print (and exec) the SSH tunnel command
    python ops/rent.py push-model DIR [--bench 25]   # upload local weights
    python ops/rent.py rm            # terminate the pod

State is kept in ops/state.json (pod name, ssh, price).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

POD_NAME = os.environ.get("FASTBENCH_POD_NAME", "fastbench")
# Pytorch Cuda + Docker-in-Docker (cu13) — same template the mining fleet
# uses; rollouts need a working docker daemon inside the pod.
TEMPLATE_ID = os.environ.get(
    "FASTBENCH_LIUM_TEMPLATE", "345273fa-4818-46f7-a8fa-32f0e331713c")
GPU_TYPES = (os.environ.get("FASTBENCH_GPU_TYPES") or "B200,B300,H200").split(",")
GPU_COUNT = int(os.environ.get("FASTBENCH_GPU_COUNT", "8"))
MAX_PRICE = float(os.environ.get("FASTBENCH_MAX_PRICE", "70"))
PORT = int(os.environ.get("FASTBENCH_PORT", "9100"))
REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = Path(__file__).resolve().parent / "state.json"
# Where pushed checkpoints live on the box; bench them by absolute path.
MODEL_DIR = os.environ.get("FASTBENCH_MODEL_DIR", "/root/models")

SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=20"]


def _run(cmd: list[str], timeout: int = 120, input_text: str | None = None
         ) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, input=input_text)


def _ssh(ssh: str, command: str, timeout: int = 120
         ) -> subprocess.CompletedProcess:
    return _run(["ssh", *SSH_OPTS, *shlex.split(ssh), command],
                timeout=timeout)


def _scp(ssh: str, local: Path, remote: str, timeout: int = 1200
         ) -> subprocess.CompletedProcess:
    parts = shlex.split(ssh)  # "root@host -p PORT"
    host = parts[0]
    port = parts[parts.index("-p") + 1] if "-p" in parts else "22"
    return _run(["scp", *SSH_OPTS, "-P", port, str(local),
                 f"{host}:{remote}"], timeout=timeout)


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=1))


# -- rent -----------------------------------------------------------------------

def _rank_executors(raw: str) -> list[tuple[str, float, int, str]]:
    """(huid, price, gpu_count, gpu_type) under the cap, cheapest first."""
    try:
        rows = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    out = []
    for r in rows if isinstance(rows, list) else []:
        # Prefer the UUID: `lium up <huid>` has been observed to 404 even for
        # huids that `lium ls` itself just printed; the UUID always works.
        ident = r.get("id") or r.get("huid")
        price = r.get("price_per_hour") or r.get("price") or 0
        count = int(r.get("gpu_count") or r.get("count") or 0)
        gpu = str(r.get("gpu_type") or r.get("gpu") or "?")
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue
        if ident and price <= MAX_PRICE and count >= GPU_COUNT:
            out.append((str(ident), price, count, gpu))
    return sorted(out, key=lambda x: x[1])


def _resolve_ssh(name: str) -> str | None:
    p = _run(["lium", "ps", "--format", "json"])
    if p.returncode != 0:
        return None
    try:
        rows = json.loads(p.stdout or "[]")
    except json.JSONDecodeError:
        return None
    for row in rows if isinstance(rows, list) else []:
        if row.get("name") != name and row.get("huid") != name:
            continue
        cmd = (row.get("ssh_cmd") or "").strip()
        if cmd.startswith("ssh "):
            return cmd[4:].strip()
        ip, ports = row.get("ip"), row.get("ports") or {}
        port = ports.get("22") or ports.get(22)
        if ip and port:
            return f"root@{ip} -p {port}"
    return None


def cmd_rent(args) -> int:
    tried: set[str] = set()
    for attempt in range(4):
        pick = None
        for gpu in GPU_TYPES:
            p = _run(["lium", "ls", "--gpu", gpu.strip(), "--format", "json"])
            if p.returncode != 0:
                print(f"lium ls {gpu} failed: {p.stderr[-300:]}")
                continue
            for huid, price, count, gtype in _rank_executors(p.stdout):
                if huid not in tried:
                    pick = (huid, price, count, gtype)
                    break
            if pick:
                break
        if pick is None:
            print(f"no executor: {GPU_TYPES} x{GPU_COUNT} under ${MAX_PRICE}/hr")
            return 1
        huid, price, count, gtype = pick
        tried.add(huid)
        print(f"renting {gtype} x{count} @ ${price:.2f}/hr ({huid}) "
              f"[attempt {attempt + 1}]")
        _run(["lium", "rm", POD_NAME, "-y"])
        up = ["lium", "up", huid, "--name", POD_NAME, "--yes"]
        if TEMPLATE_ID:
            up += ["--template_id", TEMPLATE_ID]
        try:
            _run(up, timeout=240)
        except subprocess.TimeoutExpired:
            print("lium up timed out; checking if the pod appeared anyway")
        ssh = None
        for _ in range(12):
            ssh = _resolve_ssh(POD_NAME)
            if ssh:
                break
            time.sleep(10)
        if not ssh:
            print("no ssh after up; trying next executor")
            _run(["lium", "rm", POD_NAME, "-y"])
            continue
        ok = False
        for _ in range(9):
            probe = _ssh(ssh, "echo SSH_OK", timeout=45)
            if probe.returncode == 0 and "SSH_OK" in (probe.stdout or ""):
                ok = True
                break
            time.sleep(10)
        if not ok:
            print(f"ssh probe failed ({ssh}); trying next executor")
            _run(["lium", "rm", POD_NAME, "-y"])
            continue
        # Verify GPUs, CPUs, disk — the speed spec depends on all three.
        facts = _ssh(ssh, "nvidia-smi -L | wc -l; nproc; "
                          "df -B1G --output=avail / | tail -1", timeout=60)
        vals = (facts.stdout or "").split()
        n_gpu = int(vals[0]) if vals else 0
        n_cpu = int(vals[1]) if len(vals) > 1 else 0
        disk_gb = int(vals[2]) if len(vals) > 2 else 0
        print(f"pod facts: {n_gpu} GPUs, {n_cpu} vCPU, {disk_gb}GB free disk")
        if n_gpu < GPU_COUNT:
            print("pod exposes fewer GPUs than promised; releasing")
            _run(["lium", "rm", POD_NAME, "-y"])
            continue
        state = {"pod": POD_NAME, "executor": huid, "gpu": gtype,
                 "price_per_hour": price, "ssh": ssh, "n_cpu": n_cpu,
                 "disk_gb": disk_gb, "rented_at": time.time()}
        _save_state(state)
        print(f"rented: {json.dumps(state, indent=1)}")
        return cmd_provision(args)
    return 1


# -- provision -----------------------------------------------------------------------

def _pack_tar() -> Path:
    fd, name = tempfile.mkstemp(suffix=".tar.gz", prefix="fastbench-src-")
    os.close(fd)
    path = Path(name)
    with tarfile.open(path, "w:gz") as tar:
        for item in ("fastbench", "scripts", "bootstrap.sh", "pyproject.toml"):
            src = REPO_ROOT / item
            tar.add(src, arcname=item,
                    filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return path


def _secret(name: str) -> str:
    val = os.environ.get(name, "")
    if val:
        return val
    try:
        p = _run(["doppler", "secrets", "get", name, "--plain"], timeout=30)
        if p.returncode == 0:
            return (p.stdout or "").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def cmd_provision(args) -> int:
    state = _load_state()
    ssh = state.get("ssh") or _resolve_ssh(POD_NAME)
    if not ssh:
        print("no pod; run `rent` first")
        return 1
    state["ssh"] = ssh
    _save_state(state)

    tar_path = _pack_tar()
    try:
        p = _scp(ssh, tar_path, "/tmp/fastbench-src.tar.gz")
        if p.returncode != 0:
            print(f"scp failed: {(p.stderr or '')[-400:]}")
            return 1
    finally:
        tar_path.unlink(missing_ok=True)
    extract = ("mkdir -p /root/fastbench && "
               "tar xzf /tmp/fastbench-src.tar.gz -C /root/fastbench && "
               "rm -f /tmp/fastbench-src.tar.gz && "
               "test $(find /root/fastbench/fastbench -name '*.py' | wc -l) "
               "-ge 5 && echo EXTRACT_OK")
    p = _ssh(ssh, extract, timeout=300)
    if "EXTRACT_OK" not in (p.stdout or ""):
        print(f"extract failed: {(p.stderr or p.stdout or '')[-400:]}")
        return 1

    hf_token = _secret("HF_TOKEN")
    fb_token = _secret("FASTBENCH_TOKEN")
    env = (f"HF_TOKEN={hf_token}\n"
           f"FASTBENCH_TOKEN={fb_token}\n"
           f"FASTBENCH_PORT={PORT}\n")
    with tempfile.NamedTemporaryFile("w", delete=False,
                                     prefix="fastbench-env-") as f:
        f.write(env)
        env_path = Path(f.name)
    try:
        os.chmod(env_path, 0o600)
        p = _scp(ssh, env_path, "/root/fastbench/.env", timeout=120)
        if p.returncode != 0:
            print(f"env upload failed: {(p.stderr or '')[-300:]}")
            return 1
    finally:
        env_path.unlink(missing_ok=True)

    # Kill any previous bootstrap/server/prepull, then detach a fresh
    # bootstrap. [b]racket patterns so the pkill wrapper shell (whose own
    # cmdline contains these strings) never matches and suicides mid-chain.
    _ssh(ssh, "pkill -f '[f]astbench.server' || true; "
              "pkill -f '[b]ash bootstrap.sh' || true; "
              "pkill -f '[p]repull_images' || true", timeout=60)
    boot = ("cd /root/fastbench && (nohup bash bootstrap.sh </dev/null "
            ">> /root/bootstrap.log 2>&1 &) && echo BOOTSTRAP_STARTED")
    p = _ssh(ssh, boot, timeout=60)
    if "BOOTSTRAP_STARTED" not in (p.stdout or ""):
        print(f"bootstrap launch failed: {(p.stdout or '')[-300:]} "
              f"{(p.stderr or '')[-300:]}")
        return 1
    print("provisioned; bootstrap running. Tail with:")
    print(f"  ssh {ssh} tail -f /root/bootstrap.log")
    print("tunnel with:")
    print(f"  {' '.join(_tunnel_cmd(ssh))}")
    return 0


def _tunnel_cmd(ssh: str) -> list[str]:
    return ["ssh", "-N", *SSH_OPTS,
            "-o", "ServerAliveInterval=30",
            "-o", "ExitOnForwardFailure=yes",
            "-L", f"{PORT}:127.0.0.1:{PORT}", *shlex.split(ssh)]


def cmd_tunnel(args) -> int:
    state = _load_state()
    ssh = state.get("ssh") or _resolve_ssh(POD_NAME)
    if not ssh:
        print("no pod")
        return 1
    cmd = _tunnel_cmd(ssh)
    print(" ".join(cmd))
    if args.exec_tunnel:
        os.execvp(cmd[0], cmd)
    return 0


def cmd_status(args) -> int:
    state = _load_state()
    ssh = state.get("ssh") or _resolve_ssh(POD_NAME)
    print(json.dumps(state, indent=1))
    if not ssh:
        print("no pod")
        return 1
    p = _ssh(ssh, f"curl -s -m 5 http://127.0.0.1:{PORT}/health "
                  f"-H \"X-Fastbench-Token: $(grep FASTBENCH_TOKEN "
                  f"/root/fastbench/.env | cut -d= -f2)\" || echo DOWN",
             timeout=45)
    print(f"health: {(p.stdout or '').strip()[-2000:]}")
    return 0


def _checkpoint_problems(path: Path) -> list[str]:
    """Reject a checkpoint here rather than after a 68 GB upload.

    vLLM needs a config, weight shards and a tokenizer; a directory missing any
    of them fails at engine boot, which on this box costs a model load to find
    out."""
    problems = []
    if not (path / "config.json").exists():
        problems.append("no config.json")
    if not (list(path.glob("*.safetensors")) or list(path.glob("*.bin"))):
        problems.append("no .safetensors or .bin weight shards")
    if not any((path / f).exists() for f in
               ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")):
        problems.append("no tokenizer files")
    return problems


def _dir_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1e9


def _remote_free_gb(ssh: str) -> float | None:
    p = _ssh(ssh, "df -B1 --output=avail /root | tail -1", timeout=45)
    try:
        return int((p.stdout or "").strip()) / 1e9
    except (ValueError, AttributeError):
        return None


def _box_token_curl(method: str, route: str, payload: str | None = None) -> str:
    """A curl run on the box itself, so no tunnel is needed."""
    token = ("$(grep FASTBENCH_TOKEN /root/fastbench/.env | cut -d= -f2)")
    cmd = (f"curl -s -m 60 -X {method} http://127.0.0.1:{PORT}{route} "
           f'-H "X-Fastbench-Token: {token}"')
    if payload is not None:
        cmd += f" -H 'Content-Type: application/json' -d {shlex.quote(payload)}"
    return cmd


def cmd_push_model(args) -> int:
    """Copy a local checkpoint to the box and print the model string for it."""
    local = Path(args.path).expanduser().resolve()
    if not local.is_dir():
        print(f"not a directory: {local}")
        return 1
    problems = _checkpoint_problems(local)
    if problems and not args.force:
        print(f"not a usable checkpoint ({', '.join(problems)});"
              " pass --force to copy anyway")
        return 1

    state = _load_state()
    ssh = state.get("ssh") or _resolve_ssh(POD_NAME)
    if not ssh:
        print("no pod")
        return 1

    name = args.name or local.name
    remote_dir = f"{MODEL_DIR}/{name}"
    size_gb = _dir_size_gb(local)
    free_gb = _remote_free_gb(ssh)
    print(f"pushing {local} ({size_gb:.1f} GB) -> {remote_dir}")
    if free_gb is not None:
        print(f"box free disk: {free_gb:.0f} GB")
        if free_gb < size_gb * 1.05:
            print("not enough free disk on the box")
            return 1

    parts = shlex.split(ssh)
    host = parts[0]
    port = parts[parts.index("-p") + 1] if "-p" in parts else "22"
    _ssh(ssh, f"mkdir -p {shlex.quote(remote_dir)}", timeout=45)
    t0 = time.time()
    # Inherits stdio so the transfer shows progress, and resumes on retry.
    rc = subprocess.run(
        ["rsync", "-a", "--partial", "--info=progress2",
         "-e", " ".join(["ssh", "-p", port, *SSH_OPTS]),
         f"{local}/", f"{host}:{remote_dir}/"]).returncode
    if rc != 0:
        print(f"rsync failed (exit {rc})")
        return rc
    print(f"copied in {(time.time() - t0) / 60:.1f} min")
    print(f"model string: {remote_dir}")

    if args.bench:
        payload = json.dumps({"model": remote_dir, "n": args.bench,
                              "seed": args.seed})
        p = _ssh(ssh, _box_token_curl("POST", "/bench", payload), timeout=90)
        print(f"bench: {(p.stdout or p.stderr or '').strip()[-300:]}")
    return 0


def cmd_rm(args) -> int:
    p = _run(["lium", "rm", POD_NAME, "-y"])
    print((p.stdout or p.stderr or "").strip()[-300:])
    STATE_PATH.unlink(missing_ok=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rent")
    sub.add_parser("provision")
    t = sub.add_parser("tunnel")
    t.add_argument("--exec", dest="exec_tunnel", action="store_true")
    sub.add_parser("status")
    m = sub.add_parser("push-model")
    m.add_argument("path", help="local checkpoint directory")
    m.add_argument("--name", help=f"directory name under {MODEL_DIR}")
    m.add_argument("--bench", type=int, default=0,
                   help="submit a bench of this many tasks once copied")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--force", action="store_true",
                   help="copy even if required model files are missing")
    sub.add_parser("rm")
    args = ap.parse_args()
    fn = {"rent": cmd_rent, "provision": cmd_provision, "tunnel": cmd_tunnel,
          "status": cmd_status, "push-model": cmd_push_model,
          "rm": cmd_rm}[args.cmd]
    sys.exit(fn(args))


if __name__ == "__main__":
    main()
