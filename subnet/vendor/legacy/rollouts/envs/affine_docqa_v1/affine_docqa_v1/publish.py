#!/usr/bin/env python3
"""Push a generated docqa epoch (tasks + bundles + spend) to data.affine.io.

Objects land under `envs/affine_docqa/e<epoch>/` in the public corpus bucket
(ROLLOUTS_R2_* env vars, same as rollouts/r2mirror.py); the taskset's
GenTaskStore fetches from `https://data.affine.io/envs/affine_docqa/e<epoch>/`.
Immutable by convention: a regenerated set goes to a new epoch.

  python -m affine_docqa_v1.publish --epoch 1
"""

from __future__ import annotations

import argparse
import os
import sys

import boto3

from affine_docqa_v1.taskset import PACKAGE_DIR, SOURCE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--bucket", default=os.environ.get("ROLLOUTS_R2_BUCKET", "affine-data"))
    a = ap.parse_args()
    endpoint = os.environ.get("ROLLOUTS_R2_ENDPOINT", "").rstrip("/")
    key_id, secret = os.environ.get("ROLLOUTS_R2_ACCESS_KEY_ID"), os.environ.get("ROLLOUTS_R2_SECRET_ACCESS_KEY")
    if not (endpoint and key_id and secret):
        sys.exit("ROLLOUTS_R2_ENDPOINT / ROLLOUTS_R2_ACCESS_KEY_ID / ROLLOUTS_R2_SECRET_ACCESS_KEY are not set")
    s3 = boto3.client("s3", endpoint_url=endpoint, region_name="auto", aws_access_key_id=key_id, aws_secret_access_key=secret)
    root = PACKAGE_DIR / "data" / f"e{a.epoch}"
    n = 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            key = f"envs/{SOURCE}/e{a.epoch}/{path.relative_to(root).as_posix()}"
            try:
                s3.head_object(Bucket=a.bucket, Key=key)
                continue                      # immutable: never overwrite
            except Exception:  # noqa: BLE001
                pass
            s3.put_object(Bucket=a.bucket, Key=key, Body=path.read_bytes(),
                          ContentType="application/gzip" if path.suffix == ".gz" else "application/json")
            n += 1
    print(f"uploaded {n} objects to s3://{a.bucket}/envs/{SOURCE}/e{a.epoch}/")


if __name__ == "__main__":
    main()
