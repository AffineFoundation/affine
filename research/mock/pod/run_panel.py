#!/usr/bin/env python3
"""Run one SWE panel (proxy 16 or full 150) against a served checkpoint,
using Track A's swerunner harness, and append the score to the status log."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/work")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", required=True, help="panel json (instance_ids)")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--port", type=int, default=8006)
    ap.add_argument("--tag", required=True, help="e.g. proxy_r1 / full_r1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--bench-dir", default="/dshare/bench")
    ap.add_argument("--work", default="/dshare/koth")
    args = ap.parse_args()

    os.environ["AFFINE_BENCH_DIR"] = f"{args.bench_dir}/{args.tag.split('_')[0]}"
    from evalsrv import swerunner
    swerunner.IDS_PATH = Path(args.pin)
    swerunner.BENCH_DATA_DIR = Path(os.environ["AFFINE_BENCH_DIR"])

    def status(line):
        with open(f"{args.work}/status.log", "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                     f"[eval] {line}\n")
        print(line, flush=True)

    n = len(json.load(open(args.pin))["instance_ids"])
    status(f"BENCH start tag={args.tag} model={args.model} n={n}")
    res = swerunner.run_swe_lite(args.model, args.port,
                                 workers=args.workers,
                                 timeout_s=args.timeout)
    res.pop("_artifact", None)
    out = f"{args.work}/bench_{args.tag}.json"
    json.dump(res, open(out, "w"))
    if res.get("ok"):
        status(f"BENCH tag={args.tag} score={res['score']} "
               f"resolved={res['n_resolved']}/{res['n_instances']} "
               f"wall={res['wall_time_s']}s (baselines: base35B=0.1333 "
               f"teacher27B=0.3133 on full panel)")
    else:
        status(f"BENCH tag={args.tag} FAILED: {str(res.get('error'))[:200]}")


if __name__ == "__main__":
    main()
