"""Mirror the legacy Hippius corpus (`turns/**`) into the [data_r2] bucket.

    source .venv/bin/activate && source .env
    python affine/scripts/corpus_mirror_r2.py [--prefix turns/] [--verify-only]

Keys are copied unchanged so every published verdict's
`slice.manifest_sha256` keeps resolving at
<public_base_url>/turns/manifests/{sha}.json after the base URL moves.
Idempotent: an object already on R2 with the same sha256 is skipped. Every
object copied (or already present) is read back through the public domain
and its sha256 compared with the Hippius bytes; the run fails if any differ.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import boto3
import httpx
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from affine.config import load_config  # noqa: E402

CONTENT_TYPES = {".json": "application/json", ".gz": "application/gzip",
                 ".parquet": "application/vnd.apache.parquet"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--prefix", default="turns/")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    sec = cfg.secrets
    src_bucket = cfg.hippius["bucket"]
    dst_bucket = cfg.data_r2["bucket"]
    public = cfg.data_r2["public_base_url"].rstrip("/")
    src = boto3.client("s3", endpoint_url=cfg.hippius["endpoint"],
                       region_name="us-east-1",
                       config=Config(signature_version=UNSIGNED))
    dst = boto3.client("s3", endpoint_url=sec.data_r2_endpoint,
                       aws_access_key_id=sec.data_r2_access_key_id,
                       aws_secret_access_key=sec.data_r2_secret_access_key,
                       region_name="auto", config=Config(signature_version="s3v4"))

    keys: list[str] = []
    for page in src.get_paginator("list_objects_v2").paginate(
            Bucket=src_bucket, Prefix=args.prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    print(f"{len(keys)} objects under {src_bucket}/{args.prefix}")

    copied = skipped = 0
    bad: list[str] = []
    with httpx.Client(timeout=120) as http:
        for key in keys:
            body = src.get_object(Bucket=src_bucket, Key=key)["Body"].read()
            sha = _sha(body)
            present = False
            try:
                head = dst.head_object(Bucket=dst_bucket, Key=key)
                present = head.get("Metadata", {}).get("sha256") == sha
            except ClientError as e:
                if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                    raise
            if present:
                skipped += 1
            elif args.verify_only:
                bad.append(f"{key}: missing on R2")
                continue
            else:
                ext = Path(key).suffix
                cache = ("no-cache" if key.endswith("manifest.json")
                         else "public, max-age=31536000, immutable")
                dst.put_object(Bucket=dst_bucket, Key=key, Body=body,
                               ContentType=CONTENT_TYPES.get(ext, "application/octet-stream"),
                               CacheControl=cache, Metadata={"sha256": sha})
                copied += 1
            r = http.get(f"{public}/{key}")
            if r.status_code != 200 or _sha(r.content) != sha:
                bad.append(f"{key}: public read {r.status_code}, sha mismatch")
            print(f"{'ok  ' if not bad or key not in bad[-1] else 'BAD '}{key} "
                  f"{len(body):>12,} B  {sha[:12]}")
    print(f"copied {copied}, already present {skipped}, bad {len(bad)}")
    if bad:
        print("\n".join(bad))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
