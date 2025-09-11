import os
import sys
import json
import math
import asyncio
import argparse
from typing import Any, List

import datasets as hf_ds
import aiohttp
from aiobotocore.session import get_session
from botocore.config import Config

# Minimal self-contained R2 helpers (avoid importing affine package)
FOLDER = os.getenv("R2_FOLDER", "affine")
BUCKET = os.getenv("R2_BUCKET_ID", "80f15715bb0b882c9e967c13e677ed7d")
ACCESS = os.getenv("R2_WRITE_ACCESS_KEY_ID", "ff3f4f078019b064bfb6347c270bee4d")
SECRET = os.getenv("R2_WRITE_SECRET_ACCESS_KEY", "a94b20516013519b2959cbbb441b9d1ec8511dce3c248223d947be8e85ec754d")
ENDPOINT = f"https://{BUCKET}.r2.cloudflarestorage.com"
APPEND_CONCURRENCY = int(os.getenv("R2_MAX_CONCURRENCY", "16"))

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

get_client_ctx = lambda: get_session().create_client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS,
    aws_secret_access_key=SECRET,
    config=Config(max_pool_connections=256),
)


async def _upload_page(c, *, bucket: str, key: str, rows: List[dict]) -> None:
    # Encode rows as JSONL with trailing newline
    buf_parts = []
    for r in rows:
        buf_parts.append(_dumps(r))
        buf_parts.append(b"\n")
    body = b"".join(buf_parts)
    await c.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


async def _try_load_hf_dataset(dataset_name: str, config: str, split: str):
    def _load():
        name_arg = None if config == "default" else config
        return hf_ds.load_dataset(dataset_name, name=name_arg, split=split)
    return await asyncio.to_thread(_load)


HTTP_MAX_LENGTH = int(os.getenv("HF_HTTP_MAX_LENGTH", "100"))


async def _iter_http_rows(dataset_name: str, config: str, split: str, chunk_len: int = HTTP_MAX_LENGTH):
    base = "https://datasets-server.huggingface.co/rows"
    total = None
    async with aiohttp.ClientSession() as sess:
        offset = 0
        while True:
            length = max(1, min(int(chunk_len), HTTP_MAX_LENGTH))
            params = {
                "dataset": dataset_name,
                "config": config,
                "split": split,
                "offset": str(offset),
                "length": str(length),
            }
            async with sess.get(base, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            rows = []
            for item in data.get("rows", []):
                if isinstance(item, dict) and "row" in item:
                    rows.append(item["row"])
                elif isinstance(item, dict):
                    rows.append(item)
            if total is None:
                total = int(data.get("num_rows_total", len(rows)))
            yield offset, rows, total
            if not rows:
                break
            offset += len(rows)
            if total is not None and offset >= total:
                break


async def upload_dataset_to_r2(dataset_name: str, config: str, split: str, page_size: int) -> None:
    root = _dataset_root(dataset_name, config, split)
    meta_key = _dataset_meta_key(root)
    # Try HF library first; on failure, fallback to HTTP API
    ds = None
    try:
        ds = await _try_load_hf_dataset(dataset_name, config, split)
    except Exception:
        ds = None

    async with get_client_ctx() as c:
        if ds is not None:
            total = int(len(ds))
            num_pages = math.ceil(total / page_size) if total > 0 else 0
            sem = asyncio.Semaphore(APPEND_CONCURRENCY)
            tasks = []
            for page_idx in range(num_pages):
                start = page_idx * page_size
                end = min(total, start + page_size)
                rows: List[dict] = [dict(ds[i]) for i in range(start, end)]
                key = _dataset_page_key(root, page_idx)
                async def _worker(k: str, rws: List[dict]):
                    async with sem:
                        await _upload_page(c, bucket=FOLDER, key=k, rows=rws)
                tasks.append(asyncio.create_task(_worker(key, rows)))
            if tasks:
                await asyncio.gather(*tasks)
            sample_keys: List[str] = []
            try:
                if total > 0:
                    sample_keys = list(dict(ds[0]).keys())
            except Exception:
                sample_keys = []
        else:
            # HTTP fallback; stream rows in chunks (<=100) and accumulate into pages
            total = 0
            seen = 0
            page_idx = 0
            buffer: List[dict] = []
            sample_keys = []
            async for _offset, rows, http_total in _iter_http_rows(dataset_name, config, split):
                if not sample_keys and rows:
                    sample_keys = list(rows[0].keys())
                if http_total is not None:
                    total = int(http_total)
                for r in rows:
                    buffer.append(r)
                    seen += 1
                    if len(buffer) >= page_size:
                        key = _dataset_page_key(root, page_idx)
                        await _upload_page(c, bucket=FOLDER, key=key, rows=buffer)
                        page_idx += 1
                        buffer = []
            # Flush remaining rows
            if buffer:
                key = _dataset_page_key(root, page_idx)
                await _upload_page(c, bucket=FOLDER, key=key, rows=buffer)
            if total == 0:
                total = seen

        meta = {
            "dataset_name": dataset_name,
            "config": config,
            "split": split,
            "page_size": int(page_size),
            "total": int(total),
            "keys": sample_keys,
            "version": 1,
        }
        await c.put_object(Bucket=FOLDER, Key=meta_key, Body=json.dumps(meta, separators=(",", ":")).encode(), ContentType="application/json")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a Hugging Face dataset to R2 in Affine layout")
    p.add_argument("dataset_name", type=str, help="HF dataset name, e.g. satpalsr/rl-python")
    p.add_argument("--config", dest="config", type=str, default="default", help="HF config name (default: default)")
    p.add_argument("--split", dest="split", type=str, default="train", help="HF split (default: train)")
    p.add_argument("--page-size", dest="page_size", type=int, default=1000, help="Rows per page file (default: 1000)")
    return p.parse_args(argv)


async def _amain(argv: List[str]) -> None:
    args = parse_args(argv)
    await upload_dataset_to_r2(args.dataset_name, args.config, args.split, args.page_size)


def main() -> None:
    asyncio.run(_amain(sys.argv[1:]))


if __name__ == "__main__":
    main()


