from __future__ import annotations
import os
import json
import uuid
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# NOTE: Avoid hard dependency at import time to prevent side‑effects (e.g., bittensor config)
try:
    import affine as af  # type: ignore
except Exception:  # pragma: no cover - safe fallback for utility-only usage
    class _NullLogger:
        def __getattr__(self, _name):
            def _noop(*_a, **_kw):
                return None
            return _noop
    class _AFStub:  # minimal stub for logger usage in optional paths
        logger = _NullLogger()
    af = _AFStub()  # type: ignore
from aiobotocore.session import get_session
from botocore.config import Config
from botocore.exceptions import ClientError

# --------------------------------------------------------------------------- #
#                       R2-backed multi-writer shard store                    #
# --------------------------------------------------------------------------- #

# Stream path layout
# streams/{miner}/{envver}/
#   meta/
#     active.json
#     next_seq.json
#     catalog.jsonl   (append-only)
#   shards/
#     shard-{seq:08d}.jsonl
#   leases/
#     shard-{seq:08d}.lease.json

# Tunables
POOL = int(os.getenv("R2_POOL", "10"))
MAX_SHARD = 32 * 1024 * 1024  # 32 MiB
MIN_COPY_PART = 8 * 1024 * 1024
LEASE_TTL_S = int(os.getenv("R2_LEASE_TTL", "30"))
APPEND_CONCURRENCY = int(os.getenv("R2_MAX_CONCURRENCY", "16"))
TAIL_BLOCKS_DEFAULT = int(os.getenv("AFFINE_TAIL", "20000"))

# Legacy keys for compatibility (unused by new design)
WINDOW: int = int(os.getenv("AFFINE_WINDOW", "20"))
RESULT_PREFIX = "affine/results/"
INDEX_KEY = "affine/index.json"

FOLDER = os.getenv("R2_FOLDER", "affine")
BUCKET = os.getenv("R2_BUCKET_ID", "80f15715bb0b882c9e967c13e677ed7d")
ACCESS = os.getenv("R2_WRITE_ACCESS_KEY_ID", "ff3f4f078019b064bfb6347c270bee4d")
SECRET = os.getenv("R2_WRITE_SECRET_ACCESS_KEY", "a94b20516013519b2959cbbb441b9d1ec8511dce3c248223d947be8e85ec754d")
ENDPOINT = f"https://{BUCKET}.r2.cloudflarestorage.com"

get_client_ctx = lambda: get_session().create_client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS,
    aws_secret_access_key=SECRET,
    config=Config(max_pool_connections=256),
)

CACHE_DIR = Path(os.getenv("AFFINE_CACHE_DIR", Path.home() / ".cache" / "affine" / "blocks"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#                             Utility helpers                                  #
# --------------------------------------------------------------------------- #
def _stream_root(miner: str, env_name: str, env_version: Optional[str]) -> str:
    ev = env_version or "0"
    envver = f"{env_name}+{ev}"
    return f"streams/{miner}/{envver}"

def _meta_keys(root: str) -> Tuple[str, str, str]:
    return (
        f"{root}/meta/active.json",
        f"{root}/meta/next_seq.json",
        f"{root}/meta/catalog.jsonl",
    )

def _shard_key(root: str, seq: int) -> str:
    return f"{root}/shards/shard-{seq:08d}.jsonl"

def _lease_key(root: str, seq: int) -> str:
    return f"{root}/leases/shard-{seq:08d}.lease.json"

def _now() -> int:
    return int(time.time())

def _loads(b: bytes) -> Any:
    try:
        import orjson as _oj
        return _oj.loads(b)
    except Exception:
        return json.loads(b.decode())

def _dumps(o: Any) -> bytes:
    try:
        import orjson as _oj
        return _oj.dumps(o)
    except Exception:
        return json.dumps(o, separators=(",", ":")).encode()


# --------------------------------------------------------------------------- #
#                         Dataset object-store layout                          #
# --------------------------------------------------------------------------- #
def _dataset_root(dataset_name: str, config: str, split: str) -> str:
    cfg = config or "default"
    spl = split or "train"
    # Use URL-safe dataset name path as provided
    return f"datasets/{dataset_name}/{cfg}/{spl}"

def _dataset_meta_key(root: str) -> str:
    return f"{root}/meta.json"

def _dataset_page_key(root: str, page_index: int) -> str:
    return f"{root}/pages/page-{page_index:08d}.jsonl"


# --------------------------------------------------------------------------- #
#                      CAS helpers (ETag-based optimistic)                     #
# --------------------------------------------------------------------------- #
async def _get_json(c, *, bucket: str, key: str) -> Tuple[Optional[str], Optional[dict]]:
    try:
        r = await c.get_object(Bucket=bucket, Key=key)
        body = await r["Body"].read()
        return r.get("ETag"), _loads(body)
    except ClientError as e:
        if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
            return None, None
        raise

async def _put_json(c, *, bucket: str, key: str, body: dict, if_none_match: bool = False, match_etag: Optional[str] = None) -> bool:
    try:
        params = {"Bucket": bucket, "Key": key, "Body": _dumps(body), "ContentType": "application/json"}
        if if_none_match:
            params["IfNoneMatch"] = "*"
        if match_etag:
            params["IfMatch"] = match_etag
        await c.put_object(**params)
        return True
    except ClientError as e:
        code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in (412, 409):
            return False
        if code == 404 and match_etag:
            return False
        raise

async def _append_jsonl(c, *, bucket: str, key: str, line_obj: dict) -> None:
    # Atomic append using MPU compose: copy old + upload tail, promote with If-Match on old ETag
    tail = _dumps(line_obj) + b"\n"
    try:
        h = await c.head_object(Bucket=bucket, Key=key)
        old_len = int(h["ContentLength"])
        old_etag = h.get("ETag")
    except ClientError as e:
        if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
            # Create fresh
            await c.put_object(Bucket=bucket, Key=key, Body=tail, ContentType="application/json")
            return
        raise

    tmp = f"{key}.tmp.{uuid.uuid4().hex}"
    mpu = await c.create_multipart_upload(Bucket=bucket, Key=tmp, ContentType="application/json")
    upload_id = mpu["UploadId"]
    parts = []
    try:
        if old_len > 0:
            if old_len <= 5 * 1024 * 1024 * 1024:
                cp = await c.upload_part_copy(
                    Bucket=bucket,
                    Key=tmp,
                    PartNumber=1,
                    UploadId=upload_id,
                    CopySource={"Bucket": bucket, "Key": key},
                    CopySourceIfMatch=old_etag,
                )
                parts.append({"ETag": cp["CopyPartResult"]["ETag"], "PartNumber": 1})
            else:
                # Chunked copy in ranges (rare given 32 MiB shards)
                part_no = 1
                for start in range(0, old_len, MIN_COPY_PART):
                    end = min(old_len - 1, start + MIN_COPY_PART - 1)
                    cp = await c.upload_part_copy(
                        Bucket=bucket,
                        Key=tmp,
                        PartNumber=part_no,
                        UploadId=upload_id,
                        CopySource={"Bucket": bucket, "Key": key},
                        CopySourceIfMatch=old_etag,
                        CopySourceRange=f"bytes={start}-{end}",
                    )
                    parts.append({"ETag": cp["CopyPartResult"]["ETag"], "PartNumber": part_no})
                    part_no += 1
        # Tail as last part
        up = await c.upload_part(Bucket=bucket, Key=tmp, PartNumber=len(parts) + 1, UploadId=upload_id, Body=tail)
        parts.append({"ETag": up["ETag"], "PartNumber": len(parts) + 1})
        await c.complete_multipart_upload(Bucket=bucket, Key=tmp, UploadId=upload_id, MultipartUpload={"Parts": parts})
        # Promote with If-Match against current destination ETag (old_etag)
        await c.copy_object(
            Bucket=bucket,
            Key=key,
            CopySource={"Bucket": bucket, "Key": tmp},
            IfMatch=old_etag,
            MetadataDirective="REPLACE",
        )
        await c.delete_object(Bucket=bucket, Key=tmp)
    except ClientError:
        try:
            await c.abort_multipart_upload(Bucket=bucket, Key=tmp, UploadId=upload_id)
        except Exception:
            pass
        raise


# --------------------------------------------------------------------------- #
#                           Pool / lease operations                            #
# --------------------------------------------------------------------------- #
async def _ensure_pool(c, *, root: str) -> List[int]:
    active_key, next_key, catalog_key = _meta_keys(root)
    # Read active
    etag_a, active = await _get_json(c, bucket=FOLDER, key=active_key)
    if not active:
        # Initialize next_seq
        ok = await _put_json(c, bucket=FOLDER, key=next_key, body={"ver": 0, "next": 1}, if_none_match=True)
        if not ok:
            # another writer created; proceed
            pass
        etag_a, active = (None, {"ver": 0, "pool": POOL, "seqs": []})

    seqs = list(active.get("seqs", []))
    if len(seqs) >= POOL:
        return seqs

    need = POOL - len(seqs)
    # Mint new seqs
    while True:
        etag_n, nxt = await _get_json(c, bucket=FOLDER, key=next_key)
        if not nxt:
            await _put_json(c, bucket=FOLDER, key=next_key, body={"ver": 0, "next": 1}, if_none_match=True)
            continue
        start = int(nxt.get("next", 1))
        minted = list(range(start, start + need))
        new_next = {"ver": int(nxt.get("ver", 0)) + 1, "next": start + need}
        ok = await _put_json(c, bucket=FOLDER, key=next_key, body=new_next, match_etag=etag_n)
        if not ok:
            # retry on CAS fail
            await asyncio.sleep(0.05)
            continue
        # Append open events to catalog (best-effort)
        try:
            for m in minted:
                await _append_jsonl(c, bucket=FOLDER, key=catalog_key, line_obj={"ts": _now(), "op": "open", "seq": m})
        except Exception:
            pass
        # Update active
        while True:
            etag_a, active = await _get_json(c, bucket=FOLDER, key=active_key)
            if not active:
                active = {"ver": 0, "pool": POOL, "seqs": []}
            merged = list(active.get("seqs", [])) + minted
            new_active = {"ver": int(active.get("ver", 0)) + 1, "pool": POOL, "seqs": merged}
            ok2 = await _put_json(c, bucket=FOLDER, key=active_key, body=new_active, match_etag=etag_a)
            if ok2:
                return merged
            await asyncio.sleep(0.05)


async def _acquire_lease(c, *, root: str, seqs: List[int], holder: str, ttl_s: int) -> Tuple[Optional[int], Optional[str]]:
    for attempt in range(2):
        order = list(seqs)
        # simple deterministic jitter shuffle
        seed = hash((holder, attempt, _now())) & 0xFFFFFFFF
        if seed % 2:
            order.reverse()
        for seq in order:
            lk = _lease_key(root, seq)
            body = {"holder": holder, "until": _now() + ttl_s}
            ok = await _put_json(c, bucket=FOLDER, key=lk, body=body, if_none_match=True)
            if ok:
                # Fetch etag for renewal
                et, _ = await _get_json(c, bucket=FOLDER, key=lk)
                return seq, et
            # Try steal if stale
            et, lease = await _get_json(c, bucket=FOLDER, key=lk)
            try:
                if lease and int(lease.get("until", 0)) < _now():
                    ok2 = await _put_json(c, bucket=FOLDER, key=lk, body=body, match_etag=et)
                    if ok2:
                        et2, _ = await _get_json(c, bucket=FOLDER, key=lk)
                        return seq, et2
            except Exception:
                pass
        await asyncio.sleep(0.1)
    return None, None

async def _release_lease(c, *, root: str, seq: int) -> None:
    try:
        await c.delete_object(Bucket=FOLDER, Key=_lease_key(root, seq))
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#                                 Sink                                         #
# --------------------------------------------------------------------------- #
async def sink(*, wallet: Any, results: List["af.Result"], block: Optional[int] = None) -> None:
    if not results:
        return
    # Sign first (remote signer preferred)
    _, signed = await af.sign_results(wallet, results)

    # Group results by stream root
    by_stream: Dict[str, List[Dict[str, Any]]] = {}
    for r in signed:
        env_obj = getattr(r.challenge, "env", None)
        env_name = getattr(env_obj, "name", str(env_obj))
        env_version = getattr(env_obj, "__version__", None)
        miner = getattr(r, "miner", None)
        miner_hk = getattr(miner, "hotkey", None) or getattr(miner, "uid", "unknown")
        root = _stream_root(str(miner_hk), str(env_name), env_version)
        doc = r.model_dump(mode="json")
        # Denormalized fields for fast reads
        doc.setdefault("_dn", {}).update({
            "env_name": env_name,
            "env_version": env_version,
            "revision": getattr(miner, "revision", None),
            "score": getattr(r.evaluation, "score", None),
            "success": getattr(r.response, "success", None),
        })
        by_stream.setdefault(root, []).append(doc)

    async with get_client_ctx() as c:
        # For each stream, ensure pool and append
        for root, docs in by_stream.items():
            seqs = await _ensure_pool(c, root=root)
            holder = f"runner-{uuid.uuid4().hex[:8]}"
            # Serialize each doc append to avoid over-greed size conflicts within this process
            for d in docs:
                # Acquire lease over pool
                seq, _etag = await _acquire_lease(c, root=root, seqs=seqs, holder=holder, ttl_s=LEASE_TTL_S)
                if seq is None:
                    # Could not acquire any; small backoff
                    await asyncio.sleep(0.1)
                    continue
                shard_key = _shard_key(root, seq)
                # Check size and append or rotate
                try:
                    try:
                        h = await c.head_object(Bucket=FOLDER, Key=shard_key)
                        old_len = int(h["ContentLength"])
                    except ClientError as e:
                        if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                            old_len = 0
                        else:
                            raise
                    tail = _dumps(d) + b"\n"
                    if old_len + len(tail) > MAX_SHARD:
                        # Close & replace: remove seq from active and top up
                        active_key, _next_key, catalog_key = _meta_keys(root)
                        while True:
                            et_a, active = await _get_json(c, bucket=FOLDER, key=active_key)
                            if not active:
                                break
                            seqs_now = list(active.get("seqs", []))
                            if seq not in seqs_now:
                                break
                            seqs_new = [x for x in seqs_now if x != seq]
                            new_active = {"ver": int(active.get("ver", 0)) + 1, "pool": POOL, "seqs": seqs_new}
                            ok = await _put_json(c, bucket=FOLDER, key=active_key, body=new_active, match_etag=et_a)
                            if ok:
                                try:
                                    await _append_jsonl(c, bucket=FOLDER, key=catalog_key, line_obj={"ts": _now(), "op": "close", "seq": seq})
                                except Exception:
                                    pass
                                break
                            await asyncio.sleep(0.05)
                        # Top up pool
                        seqs = await _ensure_pool(c, root=root)
                        # Release and continue to next document (new lease will pick a fresh shard)
                        await _release_lease(c, root=root, seq=seq)
                        continue
                    # Append
                    await _append_jsonl(c, bucket=FOLDER, key=shard_key, line_obj=d)
                finally:
                    await _release_lease(c, root=root, seq=seq)


# --------------------------------------------------------------------------- #
#                         Reader-side aggregations                             #
# --------------------------------------------------------------------------- #
async def _load_catalog_and_active(c, *, root: str) -> Tuple[List[int], List[int]]:
    active_key, _next_key, catalog_key = _meta_keys(root)
    # Active
    _ea, active = await _get_json(c, bucket=FOLDER, key=active_key)
    seqs_active = list((active or {}).get("seqs", []))
    # Catalog (append-only jsonl)
    seqs_open: set[int] = set()
    seqs_close: set[int] = set()
    try:
        r = await c.get_object(Bucket=FOLDER, Key=catalog_key)
        async for raw in r["Body"]:
            try:
                line = raw.rstrip(b"\n")
                if not line:
                    continue
                ev = _loads(line)
                if ev.get("op") == "open":
                    seqs_open.add(int(ev.get("seq")))
                elif ev.get("op") == "close":
                    seqs_close.add(int(ev.get("seq")))
            except Exception:
                continue
    except ClientError:
        pass
    # Known shards = open - close, plus current active (for safety)
    known = sorted((seqs_open - seqs_close) | set(seqs_active))
    return known, seqs_active


async def _cache_shard(key: str, sem: asyncio.Semaphore) -> Path:
    name = Path(key).name
    out = CACHE_DIR / f"{name}.jsonl"
    mod = out.with_suffix(".modified")
    async with sem, get_client_ctx() as c:
        try:
            h = await c.head_object(Bucket=FOLDER, Key=key)
        except ClientError as e:
            if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                raise
            raise
        if out.exists() and mod.exists():
            if h["LastModified"].isoformat() == mod.read_text().strip():
                return out
        o = await c.get_object(Bucket=FOLDER, Key=key)
        body = await o["Body"].read()
        lm = h["LastModified"].isoformat()
    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as f:
        # Ensure newline termination
        f.write(body if body.endswith(b"\n") else (body + b"\n"))
    os.replace(tmp, out)
    mod.write_text(lm)
    return out

async def _jsonl(p: Path):
    try:
        import aiofiles
        async with aiofiles.open(p, "rb") as f:
            async for l in f:
                yield l.rstrip(b"\n")
    except ModuleNotFoundError:
        def _read():
            with p.open("rb") as f:
                return f.read().splitlines()
        for l in await asyncio.to_thread(_read):
            yield l


async def _aggregate_for_pairs(root: str, *, env_name: str, pairs: List[Tuple[str, str]], env_version: Optional[str], success_only: bool) -> Dict[str, Dict[str, float]]:
    async with get_client_ctx() as c:
        known, _active = await _load_catalog_and_active(c, root=root)
    if not known:
        return {}
    hotkeys = list({hk for hk, _rv in pairs})
    rev_by_hk: Dict[str, set] = {}
    for hk, rv in pairs:
        rev_by_hk.setdefault(hk, set()).add(rv)
    sem = asyncio.Semaphore(APPEND_CONCURRENCY)
    keys = [_shard_key(root, seq) for seq in known]
    tasks = [asyncio.create_task(_cache_shard(k, sem)) for k in keys]
    paths: List[Path] = []
    for t in tasks:
        try:
            paths.append(await t)
        except Exception:
            pass

    n_total: Dict[str, int] = {hk: 0 for hk in hotkeys}
    sum_score: Dict[str, float] = {hk: 0.0 for hk in hotkeys}
    sum_sq: Dict[str, float] = {hk: 0.0 for hk in hotkeys}

    for path in paths:
        # deduce hk from stream path is not reliable (multiple miners can share?); use content
        async for raw in _jsonl(path):
            try:
                obj = _loads(raw)
            except Exception:
                continue
            dn = obj.get("_dn") or {}
            e_env = dn.get("env_name") or (obj.get("evaluation") or {}).get("env") or (obj.get("challenge") or {}).get("env")
            if not e_env or e_env != env_name:
                continue
            e_ver = dn.get("env_version")
            if env_version is not None and (e_ver != env_version):
                continue
            miner = obj.get("miner") or {}
            hk = miner.get("hotkey") or miner.get("uid")
            if hk not in rev_by_hk:
                continue
            rev = dn.get("revision") or miner.get("revision")
            allowed_revs = rev_by_hk.get(hk, set())
            if (None not in allowed_revs) and (rev not in allowed_revs):
                continue
            succ = dn.get("success")
            if success_only and not succ:
                continue
            score = dn.get("score")
            if score is None:
                try:
                    score = float((obj.get("evaluation") or {}).get("score"))
                except Exception:
                    score = None
            if score is None:
                continue
            n_total[hk] += 1
            s = float(score)
            sum_score[hk] += s
            sum_sq[hk] += s * s

    if success_only:
        out: Dict[str, Dict[str, float]] = {}
        for hk in hotkeys:
            if n_total[hk] > 0:
                out[hk] = {"n_success": float(n_total[hk]), "sum_score": float(sum_score[hk])}
        return out
    else:
        out = {}
        for hk in hotkeys:
            if n_total[hk] > 0:
                out[hk] = {
                    "n_total": float(n_total[hk]),
                    "sum_score": float(sum_score[hk]),
                    "sum_sq_score": float(sum_sq[hk]),
            }
        return out
    

# ---------------------- Dataset local cache helpers (R2) --------------------- #
async def _cache_dataset_page(root: str, page_index: int, sem: asyncio.Semaphore) -> Path:
    key = _dataset_page_key(root, page_index)
    # Build dataset-specific cache directory
    safe_root = root.replace("/", "__")
    out_dir = CACHE_DIR / "datasets" / safe_root / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"page-{page_index:08d}.jsonl"
    mod = out.with_suffix(".modified")
    async with sem, get_client_ctx() as c:
        try:
            h = await c.head_object(Bucket=FOLDER, Key=key)
        except ClientError as e:
            if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                raise
            raise
        if out.exists() and mod.exists():
            if h["LastModified"].isoformat() == mod.read_text().strip():
                return out
        o = await c.get_object(Bucket=FOLDER, Key=key)
        body = await o["Body"].read()
        lm = h["LastModified"].isoformat()
    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as f:
        f.write(body if body.endswith(b"\n") else (body + b"\n"))
    os.replace(tmp, out)
    mod.write_text(lm)
    return out

# Public aggregations consumed by validator
async def aggregate_success_by_env(*, env_name: str, pairs: List[Tuple[str, str]], env_version: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    if not pairs:
        return {}
    # Aggregate per miner stream root(s) discovered from pairs' hotkeys
    # We don't need to list streams; we can derive roots by miner hk and env
    # but to avoid double-scan, compute once per miner
    results: Dict[str, Dict[str, float]] = {}
    # Build per-miner roots
    miner_roots = {_stream_root(hk, env_name, env_version) for hk, _ in pairs}
    for root in miner_roots:
        agg = await _aggregate_for_pairs(root, env_name=env_name, pairs=pairs, env_version=env_version, success_only=True)
        for hk, vals in agg.items():
            r = results.setdefault(hk, {"n_success": 0.0, "sum_score": 0.0})
            r["n_success"] += vals.get("n_success", 0.0)
            r["sum_score"] += vals.get("sum_score", 0.0)
    return results

async def aggregate_scores_by_env(*, env_name: str, pairs: List[Tuple[str, str]], env_version: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    if not pairs:
        return {}
    results: Dict[str, Dict[str, float]] = {}
    miner_roots = {_stream_root(hk, env_name, env_version) for hk, _ in pairs}
    for root in miner_roots:
        agg = await _aggregate_for_pairs(root, env_name=env_name, pairs=pairs, env_version=env_version, success_only=False)
        for hk, vals in agg.items():
            r = results.setdefault(hk, {"n_total": 0.0, "sum_score": 0.0, "sum_sq_score": 0.0})
            r["n_total"] += vals.get("n_total", 0.0)
            r["sum_score"] += vals.get("sum_score", 0.0)
            r["sum_sq_score"] += vals.get("sum_sq_score", 0.0)
    return results


# --------------------------------------------------------------------------- #
#                         Compatibility shims for CLI                          #
# --------------------------------------------------------------------------- #
dataset_rows = None  # type: ignore

async def _get_engine():  # type: ignore
    return None

def _sm():  # type: ignore
    raise RuntimeError("Database engine is not available in R2-backed mode")

_HF_CACHE: dict[tuple[str, str, str], Any] = {}
_HF_CACHE_LOCK = asyncio.Lock()

async def _get_hf_dataset_split(dataset_name: str, config: str, split: str):
    key = (dataset_name, config, split)
    if key in _HF_CACHE:
        return _HF_CACHE[key]
    async with _HF_CACHE_LOCK:
        if key in _HF_CACHE:
            return _HF_CACHE[key]
        def _load():
            from datasets import load_dataset
            name_arg = None if config == "default" else config
            return load_dataset(dataset_name, name=name_arg, split=split)
        try:
            ds = await asyncio.to_thread(_load)
        except Exception as e:
            try:
                af.logger.warning(f"HF dataset load failed for {dataset_name} [{config}/{split}]: {e}; using fallback samples if available")
            except Exception:
                pass
            # Minimal built-in fallback for known datasets
            if dataset_name == "satpalsr/rl-python":
                ds = [
                    {
                        "program": "a=int(input()); b=int(input()); print(a+b)",
                        "inputs": "2\n3\n",
                        "output": "5\n",
                    },
                    {
                        "program": "print(input())",
                        "inputs": "hello\n",
                        "output": "hello\n",
                    },
                    {
                        "program": "a=int(input()); print(a*2)",
                        "inputs": "7\n",
                        "output": "14\n",
                    },
                ]
            else:
                ds = []
        _HF_CACHE[key] = ds
        return ds

async def select_dataset_rows(*, dataset_name: str, config: str = "default", split: str = "train", limit: int = 1000, offset: int = 0, include_index: bool = False) -> List[Dict[str, Any]]:  # noqa: E501
    """Return dataset rows, preferring R2 object-store if present, else HF.

    If include_index is True, attach __row_index__ with the global row index.
    """
    # Try R2 first
    try:
        out = await _select_dataset_rows_r2(dataset_name=dataset_name, config=config, split=split, limit=limit, offset=offset, include_index=include_index)
        if out is not None:
            return out
    except Exception:
        pass
    # HF fallback
    ds = await _get_hf_dataset_split(dataset_name, config, split)
    n = len(ds)
    if offset >= n or limit <= 0:
        return []
    start = max(0, int(offset))
    end = min(n, start + int(limit))
    rows: List[Dict[str, Any]] = []
    for idx in range(start, end):
        rec = dict(ds[int(idx)])
        if include_index:
            rec["__row_index__"] = idx
        rows.append(rec)
    return rows

async def get_dataset_size(*, dataset_name: str, config: str = "default", split: str = "train") -> int:
    """Return total number of rows for a dataset from R2 if available, else HF."""
    # R2
    try:
        async with get_client_ctx() as c:
            root = _dataset_root(dataset_name, config, split)
            _etag, meta = await _get_json(c, bucket=FOLDER, key=_dataset_meta_key(root))
            if meta and isinstance(meta.get("total"), int):
                return int(meta["total"])
    except Exception:
        pass
    # HF fallback
    ds = await _get_hf_dataset_split(dataset_name, config, split)
    try:
        return int(len(ds))
    except Exception:
        return 0

# ----------------------------- R2 dataset reads ----------------------------- #
async def _select_dataset_rows_r2(*, dataset_name: str, config: str, split: str, limit: int, offset: int, include_index: bool) -> Optional[List[Dict[str, Any]]]:
    if limit <= 0:
        return []
    root = _dataset_root(dataset_name, config, split)
    async with get_client_ctx() as c:
        _etag, meta = await _get_json(c, bucket=FOLDER, key=_dataset_meta_key(root))
        if not meta:
            return None
        total = int(meta.get("total", 0))
        page_size = int(meta.get("page_size", 1000))
        if offset >= total:
            return []
        # Compute page range and slice bounds
        start = max(0, int(offset))
        end = min(total, start + int(limit))
        first_page = start // page_size
        last_page = (end - 1) // page_size if end > 0 else first_page
    sem = asyncio.Semaphore(APPEND_CONCURRENCY)
    # Cache needed pages
    keys = list(range(first_page, last_page + 1))
    tasks = [asyncio.create_task(_cache_dataset_page(root, pidx, sem)) for pidx in keys]
    pages: Dict[int, Path] = {}
    for pidx, t in zip(keys, tasks):
        try:
            pages[pidx] = await t
        except Exception:
            # Missing page should not happen if meta exists; treat as empty
            pass
    # Iterate and collect rows within bounds
    out: List[Dict[str, Any]] = []
    global_index_base = first_page * page_size
    current_index_start = start
    for pidx in range(first_page, last_page + 1):
        p = pages.get(pidx)
        if not p:
            continue
        page_start_global = pidx * page_size
        page_end_global = page_start_global + page_size
        sel_start = max(start, page_start_global)
        sel_end = min(end, page_end_global)
        if sel_end <= sel_start:
            continue
        # Relative line indices within the page
        rel_start = sel_start - page_start_global
        rel_end = sel_end - page_start_global
        rel_idx = 0
        async for raw in _jsonl(p):
            if rel_idx >= rel_end:
                break
            if rel_idx >= rel_start:
                try:
                    rec = _loads(raw)
                except Exception:
                    rec = None
                if isinstance(rec, dict):
                    if include_index:
                        rec["__row_index__"] = page_start_global + rel_idx
                    out.append(rec)
            rel_idx += 1
    return out

async def count(**filters: Any) -> int:
    raise RuntimeError("Count is not supported in R2-backed mode")

async def select_rows(*, limit: int = 1000, order: str = "r2_last_modified", ascending: bool = False, **filters: Any) -> List[Dict[str, Any]]:  # noqa: E501
    raise RuntimeError("Select is not supported in R2-backed mode")

async def get_env_counts(*, pairs: List[Tuple[str, str]], env_version: Optional[str] = None) -> Dict[str, Dict[Tuple[str, str], int]]:
    # Build counts by summing success across envs via aggregate_success_by_env; caller rarely uses this.
    if not pairs:
        return {}
    out: Dict[str, Dict[Tuple[str, str], int]] = {}
    for e in af.ENVS:
        agg = await aggregate_success_by_env(env_name=str(e), pairs=pairs, env_version=env_version)
        env_map: Dict[Tuple[str, str], int] = {}
        for hk, vals in agg.items():
            for _hk, rev in pairs:
                if _hk == hk:
                    env_map[(hk, rev)] = int(vals.get("n_success", 0))
        out[str(e)] = env_map
    return out


