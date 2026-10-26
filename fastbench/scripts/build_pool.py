#!/usr/bin/env python3
"""Build the frozen pool manifest (fastbench/data/pool.json).

Snapshots every instance id in the SWE-rebench leaderboard set. Optionally
verifies that each instance's docker image actually exists on the registry
(--verify-images; requires docker + network) so sampled panels never hit a
missing image at bench time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_DATASET = "nebius/SWE-rebench-leaderboard"
OUT_PATH = (Path(__file__).resolve().parent.parent
            / "fastbench" / "data" / "pool.json")


def image_for(iid: str) -> str:
    return ("swerebench/sweb.eval.x86_64."
            f"{iid.replace('__', '_1776_').lower()}:latest")


def image_exists(iid: str) -> tuple[str, bool]:
    p = subprocess.run(["docker", "manifest", "inspect", image_for(iid)],
                       capture_output=True, timeout=120)
    return iid, p.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="test")
    ap.add_argument("--namespace", default="swerebench")
    ap.add_argument("--verify-images", action="store_true")
    ap.add_argument("--verify-workers", type=int, default=16)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    ids = sorted({r["instance_id"] for r in ds})
    print(f"{args.dataset}:{args.split} -> {len(ids)} instances")

    dropped: list[str] = []
    if args.verify_images:
        print(f"verifying images with {args.verify_workers} workers ...")
        keep: list[str] = []
        with ThreadPoolExecutor(max_workers=args.verify_workers) as ex:
            for i, (iid, ok) in enumerate(ex.map(image_exists, ids), 1):
                (keep if ok else dropped).append(iid)
                if i % 100 == 0:
                    print(f"  {i}/{len(ids)} checked, {len(dropped)} missing")
        ids = keep
        print(f"kept {len(ids)}, dropped {len(dropped)} without images")

    pool = {
        "dataset": args.dataset,
        "split": args.split,
        "namespace": args.namespace,
        "built_at": int(time.time()),
        "n_instances": len(ids),
        "dropped_no_image": dropped,
        "instance_ids": ids,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, indent=1))
    print(f"wrote {out} ({len(ids)} instances)")


if __name__ == "__main__":
    main()
