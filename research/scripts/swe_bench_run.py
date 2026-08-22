#!/usr/bin/env python3
"""Run a SWE panel against a served model, keeping per-instance results.

Differences from the validator's own bench path, all of them deliberate:

* Arbitrary panel. The validator pins 25 instances; one task is worth 4 points
  there, so a 7/25 -> 9/25 move is inside the noise. We pass our own ~150
  instance panel instead.

* Per-instance results are kept. run_swe_lite returns them under
  result["_artifact"]["instances"], and they are only stripped when the result
  travels over the HTTP API. Calling the function directly keeps them, which is
  what makes a paired McNemar comparison between checkpoints possible.

* Safe to run several at once. Two things in the validator path are unsafe in
  parallel and are neutralised here:
    - _reap_stale_containers() deletes every minisweagent-* container at the
      start of a run, which would murder a concurrent bench's containers. We
      disable it per-run and reap once, centrally, before launching a batch.
    - run_id is derived from int(time.time()), so simultaneous launches can
      collide in the harness's own log namespace. We pass an explicit unique
      run_id via a stable clock shim.

AFFINE_BENCH_DIR is read by swerunner at import time, so it is set from --work
before the module is imported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--panel", required=True, help="panel json with instance_ids")
    ap.add_argument("--workers", type=int, default=48,
                    help="agent-phase concurrency; the harness eval phase is "
                         "capped at min(workers,8) inside run_swe_lite")
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--work", required=True,
                    help="private AFFINE_BENCH_DIR for this run (must be unique "
                         "per concurrent bench)")
    ap.add_argument("--tag", required=True, help="label for the output file")
    ap.add_argument("--out", required=True, help="where to write the result json")
    ap.add_argument("--allow-reap", action="store_true",
                    help="permit container reaping (only safe when nothing else "
                         "is benching)")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    os.environ["AFFINE_BENCH_DIR"] = args.work  # must precede the import
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from evalsrv import swerunner

    panel = json.load(open(args.panel))
    swerunner._load_pin = lambda: panel

    if not args.allow_reap:
        swerunner._reap_stale_containers = lambda: None

    # Unique run_id even when several benches start in the same second.
    uniq = f"{int(time.time())}-{args.tag}-{os.getpid()}"
    real_time = time.time
    swerunner.time.time = lambda: real_time()  # keep timing behaviour intact

    print(f"[{args.tag}] model={args.model} port={args.port} "
          f"instances={len(panel['instance_ids'])} workers={args.workers}",
          flush=True)
    t0 = time.time()
    result = swerunner.run_swe_lite(args.model, args.port,
                                   workers=args.workers, timeout_s=args.timeout)
    wall = time.time() - t0

    art = result.pop("_artifact", None) or {}
    instances = art.get("instances") or {}
    per = {iid: bool(v.get("resolved")) for iid, v in instances.items()}

    out = {
        "tag": args.tag,
        "model": args.model,
        "panel": os.path.basename(args.panel),
        "panel_n": len(panel["instance_ids"]),
        "run_uid": uniq,
        "workers": args.workers,
        "wall_time_s": round(wall, 1),
        "ok": bool(result.get("ok")),
        "score": result.get("score"),
        "n_resolved": result.get("n_resolved"),
        "n_instances": result.get("n_instances"),
        "error": result.get("error"),
        # the payload that makes paired testing possible
        "per_instance": per,
        "n_with_result": len(per),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    print(f"[{args.tag}] ok={out['ok']} score={out['score']} "
          f"resolved={out['n_resolved']}/{out['n_instances']} "
          f"per_instance_kept={out['n_with_result']} "
          f"wall={out['wall_time_s']}s", flush=True)
    if out["error"]:
        print(f"[{args.tag}] error: {str(out['error'])[:400]}", flush=True)


if __name__ == "__main__":
    main()
