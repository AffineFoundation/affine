#!/usr/bin/env python3
"""Unattended driver: bench generator checkpoints as the loop produces them.

Waits for milestone rounds to appear on disk, then benches them in waves across
the free GPUs. Every run uses the identical panel, so the resulting sequence of
per-instance result files is a paired trajectory.

The loop's checkpoint retention keeps rounds that are multiples of five, so
milestones are chosen from those.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg, path):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} | {msg}"
    print(line, flush=True)
    try:
        with open(path, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/opt/ckpt/gadA")
    ap.add_argument("--milestones", default="5,10,20,30,45,60,80,100,120,150")
    ap.add_argument("--gpus", default="1,2,5,7")
    ap.add_argument("--panel", default="panels/panel_full.json")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--wave-size", type=int, default=4)
    ap.add_argument("--results", default="/root/work/results")
    ap.add_argument("--status-log", default="/root/work/bench_wave.log")
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--prefix", default="g_",
                    help="result-file prefix, one per experiment arm")
    args = ap.parse_args()

    miles = [int(x) for x in args.milestones.split(",") if x.strip()]
    done, t0 = set(), time.time()
    log(f"DRIVER start | milestones={miles} gpus={args.gpus} panel={args.panel}",
        args.status_log)

    while time.time() - t0 < args.max_hours * 3600:
        ready = []
        for m in miles:
            if m in done:
                continue
            ck = os.path.join(args.ckpt_dir, f"round{m:04d}")
            out = os.path.join(args.results, f"{args.prefix}round{m:04d}.json")
            if os.path.exists(out):
                done.add(m)
                continue
            if os.path.isdir(ck) and os.path.exists(
                    os.path.join(ck, "adapter_model.safetensors")):
                ready.append((m, ck))
        if not ready:
            time.sleep(args.poll)
            continue

        wave = ready[: args.wave_size]
        log(f"DRIVER wave | rounds={[m for m, _ in wave]}", args.status_log)
        cmd = ["python3", os.path.join(HERE, "bench_ckpts.py"),
               "--ckpts"] + [c for _, c in wave] + [
              "--gpus", args.gpus, "--panel", args.panel,
              "--workers", str(args.workers), "--results", args.results,
              "--status-log", args.status_log,
              "--tag-prefix", args.prefix]
        try:
            subprocess.run(cmd, check=False, timeout=6 * 3600)
        except Exception as e:
            log(f"DRIVER wave error {type(e).__name__}", args.status_log)
        for m, _ in wave:
            done.add(m)

    log(f"DRIVER done | benched={sorted(done)}", args.status_log)


if __name__ == "__main__":
    main()
