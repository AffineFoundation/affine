"""Stream stdin into an object on the affine-data R2 bucket (multipart).

Used by publish_uv_cache.sh: `ssh pod 'tar | zstd' | python upload_stream.py`.
Env: DATA_R2_ENDPOINT / DATA_R2_ACCESS_KEY_ID / DATA_R2_SECRET_ACCESS_KEY,
WH_KEY (object key). Refuses (and deletes) uploads under 100 MB — a broken
pod-side pipeline must not leave a truncated wheelhouse behind.
"""
import os
import sys

import boto3
from boto3.s3.transfer import TransferConfig

MIN_BYTES = 100 * 1024 * 1024

s3 = boto3.client(
    "s3", endpoint_url=os.environ["DATA_R2_ENDPOINT"],
    aws_access_key_id=os.environ["DATA_R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["DATA_R2_SECRET_ACCESS_KEY"],
    region_name="auto")
key = os.environ["WH_KEY"]
cfg = TransferConfig(multipart_chunksize=64 * 1024 * 1024, max_concurrency=4)
s3.upload_fileobj(sys.stdin.buffer, "affine-data", key,
                  ExtraArgs={"ContentType": "application/zstd"}, Config=cfg)
size = s3.head_object(Bucket="affine-data", Key=key)["ContentLength"]
if size < MIN_BYTES:
    s3.delete_object(Bucket="affine-data", Key=key)
    raise SystemExit(f"upload too small ({size} bytes) — deleted; check the pod-side tar")
print(f"uploaded {size} bytes")
