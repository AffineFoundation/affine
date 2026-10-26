#!/usr/bin/env python3
"""Submit a shipped checkpoint to fastbench and record the verdict row.

usage: factory_bench.py --model-path /root/incoming/witness_step250 --arm witness --step 250
       factory_bench.py --model Qwen/Qwen3-8B --arm base --step 0
Polls to completion and appends one row to research/factory/results.jsonl.
Serializes politely: waits while the box is busy with someone else's job.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import requests

URL = os.environ.get("FASTBENCH_URL", "http://127.0.0.1:9310")
LEDGER = "/home/const/subnet120/research/factory/results.jsonl"


def token():
    for line in open(os.path.expanduser("~/.fastbench.env")):
        if line.startswith("FASTBENCH_TOKEN="):
            return line.strip().split("=", 1)[1]
    raise SystemExit("no FASTBENCH_TOKEN in ~/.fastbench.env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", help="local dir on the bench box")
    ap.add_argument("--model", help="HF repo name (baselines)")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-queue", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--think", action="store_true",
                    help="serve with the model's thinking channel enabled")
    ap.add_argument("--instances",
                    help="explicit panel: comma-separated ids or @file.json")
    ap.add_argument("--panel-tag", default="",
                    help="label recorded in the ledger for this panel")
    args = ap.parse_args()
    model = args.model_path or args.model
    if not model:
        raise SystemExit("need --model-path or --model")
    hdr = {"X-Fastbench-Token": token()}

    while True:
        h = requests.get(f"{URL}/health", headers=hdr, timeout=30).json()
        if (h.get("queued") or 0) < args.max_queue:
            break
        print(f"queue depth {h['queued']}; waiting", flush=True)
        time.sleep(60)

    payload = {"model": model, "n": args.n, "seed": args.seed,
               "temperature": args.temperature,
               "enable_thinking": args.think}
    if args.instances:
        if args.instances.startswith("@"):
            payload["instance_ids"] = json.load(open(args.instances[1:]))
        else:
            payload["instance_ids"] = args.instances.split(",")
    r = requests.post(f"{URL}/bench", headers=hdr, timeout=60, json=payload)
    r.raise_for_status()
    job = r.json()["job_id"]
    print(f"job {job} submitted for {args.arm} step{args.step}", flush=True)

    while True:
        time.sleep(30)
        j = requests.get(f"{URL}/bench/{job}", headers=hdr, timeout=30).json()
        st = j.get("state")
        if st in ("running", "queued", "loading_model"):
            pr = j.get("progress") or {}
            print(f"  {st} {j.get('phase','')} {pr.get('done','')}/"
                  f"{pr.get('total','')}", flush=True)
            continue
        res = j.get("result") or {}
        row = {"ts": time.strftime("%FT%TZ", time.gmtime()),
               "arm": args.arm, "step": args.step, "model": model,
               "temperature": args.temperature, "think": args.think,
               "panel": args.panel_tag or None,
               "state": st, "score": res.get("score"),
               "n_resolved": res.get("n_resolved"), "n": res.get("n"),
               "wall_s": res.get("wall_time_s"),
               "exits": res.get("exit_statuses"), "job": job}
        with open(LEDGER, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print("RESULT", json.dumps(row), flush=True)
        break


if __name__ == "__main__":
    main()
