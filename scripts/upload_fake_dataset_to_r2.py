import os
import sys
import json
import math
import time
import asyncio
import base64
import argparse
from typing import Any, List

from dotenv import load_dotenv
from aiobotocore.session import get_session
from botocore.config import Config

load_dotenv()


def _dataset_root(dataset_name: str, config: str, split: str) -> str:
    cfg = config or "default"
    spl = split or "train"
    return f"datasets/{dataset_name}/{cfg}/{spl}"


def _dataset_meta_key(root: str) -> str:
    return f"{root}/meta.json"


def _dataset_page_key(root: str, page_index: int) -> str:
    return f"{root}/pages/page-{page_index:08d}.jsonl"


def _dumps(o: Any) -> bytes:
    try:
        import orjson as _oj  # type: ignore
        return _oj.dumps(o)
    except Exception:
        return json.dumps(o, separators=(",", ":")).encode()


async def _get_client():
    endpoint = os.getenv("HIPPIUS_ENDPOINT", "https://s3.hippius.com")
    region = os.getenv("HIPPIUS_REGION", "decentralized")
    seed = os.getenv("HIPPIUS_SEED_PHRASE", "")
    if not seed:
        raise RuntimeError("HIPPIUS_SEED_PHRASE is required")
    access = base64.b64encode(seed.encode("utf-8")).decode("utf-8")
    return get_session().create_client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access,
        aws_secret_access_key=seed,
        config=Config(max_pool_connections=64),
    )


async def _ensure_bucket(c, bucket: str, region: str) -> None:
    try:
        await c.head_bucket(Bucket=bucket)
    except Exception:
        try:
            await c.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
        except Exception:
            pass


async def _put_jsonl_page(c, *, bucket: str, key: str, rows: List[dict]) -> None:
    buf = bytearray()
    for r in rows:
        buf += _dumps(r) + b"\n"
    await c.put_object(Bucket=bucket, Key=key, Body=bytes(buf), ContentType="application/json")


async def _write_meta(c, *, bucket: str, key: str, dataset_name: str, config: str, split: str, page_size: int, total: int, keys: List[str]) -> None:
    meta = {
        "dataset_name": dataset_name,
        "config": config,
        "split": split,
        "page_size": int(page_size),
        "total": int(total),
        "keys": keys,
        "version": int(time.time()),
    }
    await c.put_object(Bucket=bucket, Key=key, Body=json.dumps(meta, separators=(",", ":")).encode(), ContentType="application/json")


def _gen_fake_rows(n: int) -> List[dict]:
    rows = []
    for i in range(n):
        rows.append({
            "program": f"print({i})",
            "inputs": "",
            "output": f"{i}\n",
            "index": i,
        })
    return rows


async def main_async(dataset_name: str, config: str, split: str, rows_n: int, page_size: int) -> None:
    region = os.getenv("HIPPIUS_REGION", "decentralized")
    bucket = os.getenv("HIPPIUS_BUCKET", os.getenv("AFFINE_BUCKET", "affine"))
    root = _dataset_root(dataset_name, config, split)
    meta_key = _dataset_meta_key(root)

    rows = _gen_fake_rows(rows_n)
    total = len(rows)
    keys = list(rows[0].keys()) if rows else []
    num_pages = math.ceil(total / page_size) if total > 0 else 0

    async with await _get_client() as c:
        await _ensure_bucket(c, bucket, region)
        # Write pages
        for page_idx in range(num_pages):
            start = page_idx * page_size
            end = min(total, start + page_size)
            page_rows = rows[start:end]
            key = _dataset_page_key(root, page_idx)
            await _put_jsonl_page(c, bucket=bucket, key=key, rows=page_rows)
        # Write meta
        await _write_meta(c, bucket=bucket, key=meta_key, dataset_name=dataset_name, config=config, split=split, page_size=page_size, total=total, keys=keys)
    print({"ok": True, "dataset": dataset_name, "pages": num_pages, "total": total})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a fake dataset to Hippius in Affine layout")
    p.add_argument("dataset_name", type=str, nargs="?", default="local/fake-sample")
    p.add_argument("--dbconfig", dest="config", type=str, default="default")
    p.add_argument("--split", dest="split", type=str, default="train")
    p.add_argument("--rows", dest="rows", type=int, default=10)
    p.add_argument("--page-size", dest="page_size", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args.dataset_name, args.config, args.split, args.rows, args.page_size))
    except Exception as e:
        print(f"Upload fake dataset failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


