#!/usr/bin/env python3
"""Build a larger SWE evaluation panel, plus a small proxy panel.

Why a bigger panel: swe_rebench_lite pins 25 instances, so one task is worth 4
points and a 7/25 -> 9/25 move is inside the noise. A ~150 instance panel makes
real movement detectable, and because every checkpoint runs the IDENTICAL
instance set the results are pairable, which is what McNemar's test needs.

The 25 lite instances are always included first, so anything measured here stays
comparable to the network's own bench_history.

Two files are written:
  panel_full.json   the verdict panel (~150 instances)
  panel_proxy.json  a small subset for fast in-loop sanity checks only

The proxy panel is deliberately a SUBSET of the full panel, so a proxy result is
never measuring something the full panel does not also measure.
"""
from __future__ import annotations

import argparse
import json
import os
import random

LITE_IDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "evalsrv", "data", "swe_rebench_lite_ids.json")


def image_for(iid: str) -> str:
    """Same naming rule the harness and evalsrv prepull use."""
    return ("swerebench/sweb.eval.x86_64."
            f"{iid.replace('__', '_1776_').lower()}:latest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="full panel size")
    ap.add_argument("--proxy-n", type=int, default=16, help="fast subset size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lite", default=LITE_IDS_PATH)
    ap.add_argument("--out-dir", default="/root/work/panels")
    ap.add_argument("--print-images", action="store_true")
    args = ap.parse_args()

    pin = json.load(open(args.lite))
    dataset, split = pin["dataset"], pin.get("split", "test")
    namespace = pin.get("namespace", "swerebench")
    lite_ids = list(pin["instance_ids"])

    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    available = [r["instance_id"] for r in ds]
    avail_set = set(available)
    print(f"dataset {dataset}:{split} has {len(available)} instances")

    keep = [i for i in lite_ids if i in avail_set]
    missing = [i for i in lite_ids if i not in avail_set]
    print(f"lite pin: {len(keep)}/{len(lite_ids)} still present"
          + (f" (missing: {missing})" if missing else ""))

    # deterministic top-up from everything not already taken
    rest = sorted(i for i in avail_set if i not in set(keep))
    random.Random(args.seed).shuffle(rest)
    panel = keep + rest[: max(0, args.n - len(keep))]
    panel = sorted(set(panel))
    print(f"full panel: {len(panel)} instances (lite {len(keep)} + "
          f"{len(panel) - len(keep)} sampled, seed={args.seed})")

    # proxy = subset of the panel, biased to include lite ids for continuity
    rng = random.Random(args.seed + 1)
    proxy_pool = [i for i in panel if i in set(keep)]
    others = [i for i in panel if i not in set(keep)]
    rng.shuffle(others)
    proxy = sorted(set((proxy_pool + others)[: args.proxy_n]))
    print(f"proxy panel: {len(proxy)} instances (subset of full)")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, ids in (("panel_full", panel), ("panel_proxy", proxy)):
        p = os.path.join(args.out_dir, f"{name}.json")
        json.dump({"dataset": dataset, "split": split, "namespace": namespace,
                   "n": len(ids), "seed": args.seed, "instance_ids": ids},
                  open(p, "w"), indent=1)
        print(f"  wrote {p}")

    if args.print_images:
        with open(os.path.join(args.out_dir, "images_full.txt"), "w") as fh:
            for i in panel:
                fh.write(image_for(i) + "\n")
        print(f"  wrote {os.path.join(args.out_dir, 'images_full.txt')}")


if __name__ == "__main__":
    main()
