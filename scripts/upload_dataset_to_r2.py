import os
import sys
import json
import math
import asyncio
import argparse
import logging
import time
from typing import Any, List

# optional: datasets will be imported lazily inside _try_load_hf_dataset
import aiohttp
from aiobotocore.session import get_session
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv(".env.local")


"""Uploader targets Hippius S3-compatible storage."""
import base64

# Hippius
ENDPOINT = os.getenv("HIPPIUS_ENDPOINT", "https://s3.hippius.com")
REGION = os.getenv("HIPPIUS_REGION", "decentralized")
SEED = os.getenv("HIPPIUS_SEED_PHRASE", "")
FOLDER = os.getenv("HIPPIUS_BUCKET", os.getenv("AFFINE_BUCKET", "affine"))
ACCESS = base64.b64encode(SEED.encode("utf-8")).decode("utf-8") if SEED else ""
SECRET = SEED
APPEND_CONCURRENCY = int(os.getenv("R2_MAX_CONCURRENCY", "16"))

# Logger
logger = logging.getLogger("hippius_uploader")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

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
    region_name=REGION,
    aws_access_key_id=ACCESS,
    aws_secret_access_key=SECRET,
    config=Config(
        s3={"addressing_style": "path"},
        max_pool_connections=256,
    ),
)


PART_SIZE = int(os.getenv("HIPPIUS_PART_SIZE", str(10 * 1024 * 1024)))  # 10 MiB default


async def _upload_page(c, *, bucket: str, key: str, rows: List[dict]) -> None:
    """Upload a JSONL page using MPU for large payloads to maximize compatibility."""
    # Stream rows into parts of size PART_SIZE
    parts = []
    upload_id = None
    buf = bytearray()
    part_no = 1
    page_start = time.perf_counter()
    try:
        for r in rows:
            buf += _dumps(r) + b"\n"
            if len(buf) >= PART_SIZE:
                if upload_id is None:
                    mpu = await c.create_multipart_upload(Bucket=bucket, Key=key, ContentType="application/json")
                    upload_id = mpu["UploadId"]
                part_bytes = bytes(buf)
                t0 = time.perf_counter()
                up = await c.upload_part(Bucket=bucket, Key=key, PartNumber=part_no, UploadId=upload_id, Body=part_bytes)
                dt = time.perf_counter() - t0
                logger.info(f"uploaded part {part_no} size={len(part_bytes)}B time={dt:.2f}s speed={(len(part_bytes)/(1024*1024))/max(dt,1e-6):.2f} MiB/s -> {key}")
                parts.append({"ETag": up["ETag"], "PartNumber": part_no})
                part_no += 1
                buf.clear()
        # Final flush
        if upload_id is None:
            # Small object, single PUT
            final_bytes = bytes(buf)
            t0 = time.perf_counter()
            await c.put_object(Bucket=bucket, Key=key, Body=final_bytes, ContentType="application/json")
            dt = time.perf_counter() - t0
            logger.info(f"uploaded single PUT size={len(final_bytes)}B time={dt:.2f}s -> {key}")
        else:
            if buf:
                part_bytes = bytes(buf)
                t0 = time.perf_counter()
                up = await c.upload_part(Bucket=bucket, Key=key, PartNumber=part_no, UploadId=upload_id, Body=part_bytes)
                dt = time.perf_counter() - t0
                logger.info(f"uploaded part {part_no} size={len(part_bytes)}B time={dt:.2f}s speed={(len(part_bytes)/(1024*1024))/max(dt,1e-6):.2f} MiB/s -> {key}")
                parts.append({"ETag": up["ETag"], "PartNumber": part_no})
            t0 = time.perf_counter()
            await c.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id, MultipartUpload={"Parts": parts})
            dt = time.perf_counter() - t0
            logger.info(f"completed MPU with {len(parts)} parts in {dt:.2f}s -> {key}")
        logger.info(f"page uploaded total_time={time.perf_counter() - page_start:.2f}s -> {key}")
    except Exception:
        if upload_id is not None:
            try:
                await c.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
            except Exception:
                pass
        raise


async def _try_load_hf_dataset(dataset_name: str, config: str, split: str):
    def _load():
        try:
            import datasets as hf_ds  # type: ignore
        except Exception:
            return None
        name_arg = None if config == "default" else config
        try:
            return hf_ds.load_dataset(dataset_name, name=name_arg, split=split)
        except Exception:
            return None
    return await asyncio.to_thread(_load)


HTTP_MAX_LENGTH = int(os.getenv("HF_HTTP_MAX_LENGTH", "100"))
HF_HTTP_DELAY_S = float(os.getenv("HF_HTTP_DELAY_S", "0.2"))
HF_HTTP_MAX_RETRIES = int(os.getenv("HF_HTTP_MAX_RETRIES", "8"))
HF_HTTP_BACKOFF_BASE = float(os.getenv("HF_HTTP_BACKOFF_BASE", "0.5"))


async def _iter_http_rows(dataset_name: str, config: str, split: str, chunk_len: int = HTTP_MAX_LENGTH, start_offset: int = 0):
    base = "https://datasets-server.huggingface.co/rows"
    total = None
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        offset = int(start_offset)
        while True:
            length = max(1, min(int(chunk_len), HTTP_MAX_LENGTH))
            params = {
                "dataset": dataset_name,
                "config": config,
                "split": split,
                "offset": str(offset),
                "length": str(length),
            }
            # Retry loop for transient server/network errors
            attempt = 0
            while True:
                try:
                    t0 = time.perf_counter()
                    async with sess.get(base, params=params) as resp:
                        status = resp.status
                        if status >= 500 or status == 429:
                            raise aiohttp.ClientResponseError(
                                resp.request_info, resp.history, status=status, message=f"status={status}", headers=resp.headers
                            )
                        resp.raise_for_status()
                        data = await resp.json()
                    dt = time.perf_counter() - t0
                    break
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    attempt += 1
                    if attempt > HF_HTTP_MAX_RETRIES:
                        logger.error(f"http fetch failed permanently at offset={offset}: {e}")
                        raise
                    backoff = HF_HTTP_BACKOFF_BASE * (2 ** (attempt - 1))
                    jitter = 0.1 * backoff
                    sleep_s = backoff + (jitter * 0.5)
                    logger.warning(f"http fetch retry {attempt}/{HF_HTTP_MAX_RETRIES} offset={offset} in {sleep_s:.2f}s due to {e}")
                    await asyncio.sleep(sleep_s)
            rows = []
            for item in data.get("rows", []):
                if isinstance(item, dict) and "row" in item:
                    rows.append(item["row"])
                elif isinstance(item, dict):
                    rows.append(item)
            if total is None:
                total = int(data.get("num_rows_total", len(rows)))
            logger.info(f"fetched http rows offset={offset} len={len(rows)} total={total} time={dt:.2f}s")
            yield offset, rows, total
            if not rows:
                break
            offset += len(rows)
            if total is not None and offset >= total:
                break
            # Gentle pacing to avoid rate limits
            if HF_HTTP_DELAY_S > 0:
                await asyncio.sleep(HF_HTTP_DELAY_S)


async def _get_hf_total(dataset_name: str, config: str, split: str) -> int:
    # Make a single lightweight request to get num_rows_total
    try:
        async for _offset, rows, total in _iter_http_rows(dataset_name, config, split, chunk_len=1, start_offset=0):
            return int(total or 0)
    except Exception:
        return 0
    return 0


async def _object_exists(c, bucket: str, key: str) -> bool:
    try:
        await c.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


async def _count_lines_in_object(c, bucket: str, key: str) -> int:
    try:
        r = await c.get_object(Bucket=bucket, Key=key)
        count = 0
        async for chunk in r["Body"]:
            count += chunk.count(b"\n")
        return count
    except Exception:
        return 0


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


async def upload_dataset_to_r2(dataset_name: str, config: str, split: str, page_size: int) -> None:
    root = _dataset_root(dataset_name, config, split)
    meta_key = _dataset_meta_key(root)
    logger.info(f"start upload dataset={dataset_name} config={config} split={split} page_size={page_size}")
    # Try HF library first; on failure, fallback to HTTP API
    ds = None
    try:
        ds = await _try_load_hf_dataset(dataset_name, config, split)
    except Exception:
        ds = None

    async with get_client_ctx() as c:
        # Ensure bucket exists (best-effort)
        try:
            await c.head_bucket(Bucket=FOLDER)
        except Exception:
            try:
                await c.create_bucket(Bucket=FOLDER, CreateBucketConfiguration={"LocationConstraint": REGION})
            except Exception:
                pass
        # Test minimal write to validate credentials early
        try:
            t0 = time.perf_counter()
            health_key = f"datasets/_health_{os.getpid()}.txt"
            await c.put_object(Bucket=FOLDER, Key=health_key, Body=b"ok", ContentType="text/plain")
            logger.info(f"bucket health write ok key={health_key} time={time.perf_counter()-t0:.2f}s")
        except Exception as e:
            raise RuntimeError(f"Failed to write to Hippius bucket '{FOLDER}': {e}")
        # Discover total rows via HTTP to write consistent meta and to allow skipping existing pages
        total_rows_http = await _get_hf_total(dataset_name, config, split)
        if ds is not None:
            total = int(len(ds))
            num_pages = math.ceil(total / page_size) if total > 0 else 0
            sem = asyncio.Semaphore(APPEND_CONCURRENCY)
            logger.info(f"using HF library total_rows={total} pages={num_pages}")
            tasks = []
            sample_keys: List[str] = []
            for page_idx in range(num_pages):
                start = page_idx * page_size
                end = min(total, start + page_size)
                rows: List[dict] = [dict(ds[i]) for i in range(start, end)]
                key = _dataset_page_key(root, page_idx)
                # Skip if page already exists with equal or more rows
                if await _object_exists(c, FOLDER, key):
                    existing = await _count_lines_in_object(c, FOLDER, key)
                    if existing >= len(rows):
                        logger.info(f"skip existing page idx={page_idx} rows={existing} -> {key}")
                        if not sample_keys and rows:
                            sample_keys = list(rows[0].keys())
                        # Update meta continuously
                        await _write_meta(c, bucket=FOLDER, key=meta_key, dataset_name=dataset_name, config=config, split=split, page_size=page_size, total=total_rows_http or total, keys=sample_keys)
                        continue
                async def _worker(k: str, rws: List[dict]):
                    async with sem:
                        t0w = time.perf_counter()
                        await _upload_page(c, bucket=FOLDER, key=k, rows=rws)
                        logger.info(f"uploaded page idx={k.split('/')[-1]} rows={len(rws)} time={time.perf_counter()-t0w:.2f}s")
                        # Update meta after each page upload
                        if not sample_keys and rws:
                            sample_keys.extend(list(rws[0].keys()))
                        await _write_meta(c, bucket=FOLDER, key=meta_key, dataset_name=dataset_name, config=config, split=split, page_size=page_size, total=total_rows_http or total, keys=sample_keys)
                tasks.append(asyncio.create_task(_worker(key, rows)))
            if tasks:
                await asyncio.gather(*tasks)
            if not sample_keys and total > 0:
                try:
                    sample_keys = list(dict(ds[0]).keys())
                except Exception:
                    sample_keys = []
        else:
            # HTTP fallback; stream rows in chunks (<=100) and accumulate into pages
            total = int(total_rows_http)
            # Pre-scan existing pages to avoid downloading them again from HF
            seen = 0
            page_idx = 0
            while True:
                key = _dataset_page_key(root, page_idx)
                if not await _object_exists(c, FOLDER, key):
                    break
                existing = await _count_lines_in_object(c, FOLDER, key)
                if existing <= 0:
                    break
                logger.info(f"found existing page idx={page_idx} rows={existing}; skipping HF fetch for this range")
                seen += existing
                page_idx += 1
            buffer: List[dict] = []
            sample_keys = []
            async for _offset, rows, http_total in _iter_http_rows(dataset_name, config, split, start_offset=seen):
                if not sample_keys and rows:
                    sample_keys = list(rows[0].keys())
                if http_total is not None:
                    total = int(http_total)
                for r in rows:
                    buffer.append(r)
                    seen += 1
                    if len(buffer) >= page_size:
                        key = _dataset_page_key(root, page_idx)
                        # Upload page (by construction this should be the first missing page)
                        t0w = time.perf_counter()
                        await _upload_page(c, bucket=FOLDER, key=key, rows=buffer)
                        logger.info(f"uploaded page idx={page_idx} rows={len(buffer)} time={time.perf_counter()-t0w:.2f}s")
                        await _write_meta(c, bucket=FOLDER, key=meta_key, dataset_name=dataset_name, config=config, split=split, page_size=page_size, total=total or total_rows_http, keys=sample_keys)
                        page_idx += 1
                        buffer = []
            # Flush remaining rows
            if buffer:
                key = _dataset_page_key(root, page_idx)
                t0w = time.perf_counter()
                await _upload_page(c, bucket=FOLDER, key=key, rows=buffer)
                logger.info(f"uploaded final page idx={page_idx} rows={len(buffer)} time={time.perf_counter()-t0w:.2f}s")
                await _write_meta(c, bucket=FOLDER, key=meta_key, dataset_name=dataset_name, config=config, split=split, page_size=page_size, total=total or total_rows_http, keys=sample_keys)
            if total == 0:
                total = seen

        # Final meta write (idempotent)
        t0m = time.perf_counter()
        await _write_meta(c, bucket=FOLDER, key=meta_key, dataset_name=dataset_name, config=config, split=split, page_size=page_size, total=total or total_rows_http, keys=sample_keys)
        logger.info(f"wrote meta total_rows={total or total_rows_http} page_size={page_size} time={time.perf_counter()-t0m:.2f}s -> {meta_key}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a Hugging Face dataset to R2 in Affine layout")
    p.add_argument("dataset_name", type=str, help="HF dataset name, e.g. satpalsr/rl-python")
    p.add_argument("--dbconfig", dest="config", type=str, default="default", help="HF config name (default: default)")
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
