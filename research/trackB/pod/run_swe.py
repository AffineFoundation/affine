#!/usr/bin/env python3
"""Run the pinned swe_rebench_lite 25-task suite against a served model and
persist the full result (including per-instance artifacts). Runs under
/root/benchenv; swerunner.py lives at /root/work/evalsrv/swerunner.py."""
import argparse
import importlib.util
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=10800)
    ap.add_argument("--tag", required=True, help="label for output files")
    ap.add_argument("--out-dir", default="/dshare/gad/swe")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_cwd = os.path.join(args.out_dir, f"cwd_{args.tag}")
    os.makedirs(run_cwd, exist_ok=True)
    os.chdir(run_cwd)  # swebench harness writes its report into CWD

    spec = importlib.util.spec_from_file_location(
        "swerunner", "/root/work/evalsrv/swerunner.py")
    swerunner = importlib.util.module_from_spec(spec)
    sys.modules["swerunner"] = swerunner
    spec.loader.exec_module(swerunner)

    t0 = time.time()
    res = swerunner.run_swe_lite(args.model, args.port,
                                 workers=args.workers, timeout_s=args.timeout)
    art = res.pop("_artifact", {}) or {}
    res["tag"] = args.tag
    res["total_wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out_dir, f"swe_{args.tag}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    with open(os.path.join(args.out_dir, f"swe_{args.tag}_instances.json"), "w") as fh:
        json.dump(art, fh)
    print(json.dumps({k: v for k, v in res.items()
                      if k in ("ok", "score", "n_resolved", "n_instances",
                               "wall_time_s", "error", "tag")}), flush=True)


if __name__ == "__main__":
    main()
