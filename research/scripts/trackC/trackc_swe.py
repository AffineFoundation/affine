#!/usr/bin/env python3
"""Run the pinned swe_rebench_lite 25-task suite against a served model.

Thin driver over evalsrv/swerunner.py (uploaded to /root/work/affine/evalsrv,
layout preserved so its data/ pin file resolves). Persists the full
per-instance artifact next to the summary, as the protocol requires.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

SWERUNNER = "/opt/work/affine/evalsrv/swerunner.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=10800)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/root/work/trackC/swe")
    args = ap.parse_args()

    run_dir = os.path.join(args.out, args.tag)
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)  # swebench harness writes reports into CWD

    spec = importlib.util.spec_from_file_location("swerunner", SWERUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swerunner"] = mod
    spec.loader.exec_module(mod)

    t0 = time.time()
    res = mod.run_swe_lite(args.model, args.port, workers=args.workers,
                           timeout_s=args.timeout)
    art = res.pop("_artifact", {}) or {}
    with open(os.path.join(run_dir, "instances.json"), "w") as fh:
        json.dump(art, fh)
    res["tag"] = args.tag
    res["total_wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
