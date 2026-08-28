#!/usr/bin/env python3
"""Bench a set of generator LoRA checkpoints in parallel, one GPU each.

Every checkpoint runs against the identical instance list, and every run keeps
its per-instance resolved flags, so any two runs can be compared with a paired
test rather than by their headline rates.

Each parallel run is isolated: its own GPU, port, container, harness working
directory and run id. Sharing any of those makes the harness runs collide.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg, path=None):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} | {msg}"
    print(line, flush=True)
    if path:
        try:
            with open(path, "a") as fh:
                fh.write(line + "\n")
        except Exception:
            pass


def serve(base, port, gpu, tag, ckpt, serve_sh, max_len):
    cmd = [serve_sh, base, str(port), str(gpu),
           "--enable-lora", "--max-lora-rank", "16", "--max-loras", "2",
           "--lora-modules", f"{tag}={ckpt}"]
    env = dict(os.environ, MAX_MODEL_LEN=str(max_len))
    subprocess.run(cmd, env=env, check=True, capture_output=True, timeout=300)


def wait_up(port, tag, timeout=900):
    import requests
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            if r.status_code < 300:
                names = [m["id"] for m in r.json().get("data", [])]
                if tag in names:
                    return True
        except Exception:
            pass
        time.sleep(10)
    return False


def container_name(base, port):
    return "vllm_" + base.replace("/", "_").replace(".", "_") + f"_{port}"


def run_one(job, args, statuslog):
    ckpt, gpu, port = job["ckpt"], job["gpu"], job["port"]
    tag = job["tag"]
    cname = container_name(args.base, port)
    try:
        serve(args.base, port, gpu, tag, ckpt, args.serve_sh, args.max_len)
        if not wait_up(port, tag):
            log(f"BENCH {tag} | server never came up on :{port}", statuslog)
            return
        out = os.path.join(args.results, f"{tag}.json")
        work = os.path.join(args.workroot, tag)
        subprocess.run(
            ["python3", os.path.join(args.scripts, "swe_bench_run.py"),
             "--model", tag, "--port", str(port), "--panel", args.panel,
             "--workers", str(args.workers), "--timeout", str(args.timeout),
             "--work", work, "--tag", tag, "--out", out],
            check=False, capture_output=True, timeout=args.timeout + 1200)
        if os.path.exists(out):
            with open(out) as fh:
                r = json.load(fh)
            log(f"RESULT {tag} | score={r.get('score')} "
                f"resolved={r.get('n_resolved')}/{r.get('n_instances')} "
                f"per_instance={len(r.get('per_instance') or {})} "
                f"wall={r.get('wall_time_s')}", statuslog)
        else:
            log(f"BENCH {tag} | produced no result file", statuslog)
    except Exception as e:
        log(f"BENCH {tag} | failed {type(e).__name__}: {str(e)[:120]}", statuslog)
    finally:
        subprocess.run(["docker", "rm", "-f", cname],
                       capture_output=True, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--gpus", default="1,2,5,7")
    ap.add_argument("--base", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--panel", default="panels/panel_full.json")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=25000)
    ap.add_argument("--max-len", type=int, default=131072)
    ap.add_argument("--results", default="/root/work/results")
    ap.add_argument("--workroot", default="/opt/bench")
    ap.add_argument("--scripts", default="/root/work")
    ap.add_argument("--serve-sh", default="/root/work/pod_serve_docker.sh")
    ap.add_argument("--base-port", type=int, default=8100)
    ap.add_argument("--tag-prefix", default="g_",
                    help="distinguishes arms so result files do not collide")
    ap.add_argument("--status-log", default=None)
    args = ap.parse_args()

    os.makedirs(args.results, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    jobs = queue.Queue()
    for ck in args.ckpts:
        jobs.put(ck)

    log(f"BENCH wave | {len(args.ckpts)} checkpoints over gpus={gpus} "
        f"workers={args.workers} each", args.status_log)

    def worker(slot, gpu):
        port = args.base_port + slot
        while True:
            try:
                ck = jobs.get_nowait()
            except queue.Empty:
                return
            tag = args.tag_prefix + os.path.basename(ck.rstrip("/"))
            run_one({"ckpt": ck, "gpu": gpu, "port": port, "tag": tag},
                    args, args.status_log)
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(i, g), daemon=True)
               for i, g in enumerate(gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("BENCH wave complete", args.status_log)


if __name__ == "__main__":
    main()
