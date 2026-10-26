#!/usr/bin/env python3
"""Pre-pull docker images for the pool (bake step) or one sampled panel.

Run on the bench box, ideally in the background right after provisioning:
    python scripts/prepull_images.py --workers 12
    python scripts/prepull_images.py --n 150 --seed 0   # one panel only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastbench.pool import image_for, load_pool, sample  # noqa: E402


def cached_images() -> set[str]:
    p = subprocess.run(["docker", "images", "--format",
                        "{{.Repository}}:{{.Tag}}"],
                       capture_output=True, text=True, timeout=120)
    return set((p.stdout or "").split())


# Docker Hub throttles anonymous pulls (~100 per 6h per IP), and a throttled
# pull looks identical to a missing image unless the error is read. Back off
# and retry instead of writing the instance off as imageless.
RETRY_SLEEPS = (120, 420, 900)


def pull(img: str) -> tuple[str, bool, float]:
    t0 = time.time()
    last = ""
    for attempt in range(len(RETRY_SLEEPS) + 1):
        p = subprocess.run(["docker", "pull", img], capture_output=True,
                           text=True, timeout=1800)
        if p.returncode == 0:
            return img, True, time.time() - t0
        last = ((p.stderr or "") + (p.stdout or "")).lower()
        if attempt == len(RETRY_SLEEPS):
            break
        throttled = ("toomanyrequests" in last or "rate limit" in last
                     or "429" in last)
        time.sleep(RETRY_SLEEPS[attempt] if throttled else 20)
    return img, False, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n", type=int, default=0,
                    help="panel size; 0 = whole pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=200.0,
                    help="stop pulling when docker disk drops below this")
    args = ap.parse_args()

    pool = load_pool()
    ids = (sample(pool, args.n, args.seed) if args.n
           else sorted(pool["instance_ids"]))
    images = [image_for(i) for i in ids]
    have = cached_images()
    todo = [img for img in images if img not in have]
    print(f"{len(images)} images, {len(images) - len(todo)} cached, "
          f"{len(todo)} to pull")

    failed: list[str] = []
    stopped = False
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (img, ok, dt) in enumerate(ex.map(pull, todo), 1):
            if not ok:
                failed.append(img)
            free_gb = shutil.disk_usage("/var/lib/docker").free / 1e9 \
                if Path("/var/lib/docker").exists() else 1e9
            if i % 20 == 0 or not ok:
                print(f"  {i}/{len(todo)} ({'ok' if ok else 'FAIL'}) "
                      f"{dt:.0f}s {img.split('.')[-1][:40]} "
                      f"free={free_gb:.0f}GB", flush=True)
            if free_gb < args.min_free_gb:
                print(f"STOP: free disk {free_gb:.0f}GB < "
                      f"{args.min_free_gb}GB floor", flush=True)
                stopped = True
                break

    print(f"done in {(time.time() - t0) / 60:.1f} min; "
          f"{len(failed)} failed" + (" (disk stop)" if stopped else ""))
    for img in failed[:20]:
        print(f"  FAILED {img}")


if __name__ == "__main__":
    main()
