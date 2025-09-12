import os
import sys
import json
import base64
import argparse
import asyncio
import logging
from typing import List

from aiobotocore.session import get_session
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()


logger = logging.getLogger("hippius_dataset_tester")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(_h)


def _dataset_root(dataset_name: str, config: str, split: str) -> str:
    cfg = config or "default"
    spl = split or "train"
    return f"datasets/{dataset_name}/{cfg}/{spl}"


def _dataset_meta_key(root: str) -> str:
    return f"{root}/meta.json"


def _dataset_page_key(root: str, page_index: int) -> str:
    return f"{root}/pages/page-{page_index:08d}.jsonl"


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


async def _read_meta(c, bucket: str, key: str) -> dict:
    r = await c.get_object(Bucket=bucket, Key=key)
    body = await r["Body"].read()
    return json.loads(body.decode())


async def _read_first_n_from_page(c, bucket: str, key: str, n: int) -> List[dict]:
    r = await c.get_object(Bucket=bucket, Key=key)
    data = await r["Body"].read()
    out: List[dict] = []
    for line in data.splitlines():
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
        if len(out) >= n:
            break
    return out


async def main_async(dataset_name: str, config: str, split: str, sample_n: int, page_index: int) -> int:
    bucket = os.getenv("HIPPIUS_BUCKET", os.getenv("AFFINE_BUCKET", "affine"))
    root = _dataset_root(dataset_name, config, split)
    meta_key = _dataset_meta_key(root)
    page_key = _dataset_page_key(root, page_index)

    async with await _get_client() as c:
        # Meta
        logger.info(f"Fetching meta: s3://{bucket}/{meta_key}")
        meta = await _read_meta(c, bucket, meta_key)
        total = int(meta.get("total", 0))
        page_size = int(meta.get("page_size", 0))
        keys = list(meta.get("keys", []))
        if total <= 0 or page_size <= 0:
            logger.error(f"Invalid meta: total={total} page_size={page_size}")
            return 2
        logger.info(f"Meta ok: total={total} page_size={page_size} keys={keys}")

        # Page
        logger.info(f"Fetching page: s3://{bucket}/{page_key}")
        rows = await _read_first_n_from_page(c, bucket, page_key, max(1, sample_n))
        if not rows:
            logger.error("No rows found in first page")
            return 3
        logger.info(f"Read {len(rows)} rows from page {page_index}")
        # Validate key coverage
        missing_any = False
        for i, r in enumerate(rows[:min(3, len(rows))]):
            missing = [k for k in keys if k not in r]
            logger.info(f"Row[{i}] sample keys present={len(keys)-len(missing)}/{len(keys)} missing={missing}")
            if missing:
                missing_any = True
        # Print one sample row
        logger.info(f"Sample row: {json.dumps(rows[0], ensure_ascii=False)[:512]}")
        return 0 if not missing_any else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test reading a dataset from Hippius S3")
    p.add_argument("dataset_name", type=str, nargs="?", default="satpalsr/rl-python")
    p.add_argument("--dbconfig", dest="config", type=str, default="default")
    p.add_argument("--split", dest="split", type=str, default="train")
    p.add_argument("--page", dest="page", type=int, default=0)
    p.add_argument("--n", dest="n", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        rc = asyncio.run(main_async(args.dataset_name, args.config, args.split, args.n, args.page))
    except Exception as e:
        logger.error(f"Test failed: {e}")
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()


