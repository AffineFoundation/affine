from __future__ import annotations
import os
import json
import uuid
import time
import asyncio
import base64
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
#                      Hippius S3 Configuration                               #
# --------------------------------------------------------------------------- #

# Load configuration from environment variables
HIPPIUS_ENDPOINT = os.getenv("HIPPIUS_ENDPOINT", "https://s3.hippius.com")
HIPPIUS_REGION = os.getenv("HIPPIUS_REGION", "decentralized")
HIPPIUS_SEED_PHRASE = os.getenv("HIPPIUS_SEED_PHRASE")
HIPPIUS_BUCKET = os.getenv("HIPPIUS_BUCKET")

if not HIPPIUS_SEED_PHRASE:
    raise ValueError("HIPPIUS_SEED_PHRASE must be set in your .env file.")
if not HIPPIUS_BUCKET:
    raise ValueError("HIPPIUS_BUCKET must be set in your .env file.")

FOLDER = HIPPIUS_BUCKET
STREAMS_PREFIX = os.getenv("AFFINE_STREAMS_PREFIX", "rollouts")

def _hippius_access_from_seed(seed: str) -> Tuple[str, str]:
    """Generates S3 credentials from a Hippius seed phrase."""
    access_key = base64.b64encode(seed.encode("utf-8")).decode("utf-8")
    secret_key = seed
    return access_key, secret_key

# Generate credentials once
ACCESS_KEY, SECRET_KEY = _hippius_access_from_seed(HIPPIUS_SEED_PHRASE)

# S3 Client Factory using a context manager
def get_client_ctx():
    """Returns an async context manager for a configured S3 client."""
    return get_session().create_client(
        "s3",
        endpoint_url=HIPPIUS_ENDPOINT,
        region_name=HIPPIUS_REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(
            s3={"addressing_style": "path"},
            max_pool_connections=256,
        ),
    )

# --------------------------------------------------------------------------- #
#                       R2-backed multi-writer shard store                    #
# --------------------------------------------------------------------------- #

# Tunables (remain the same)
POOL = int(os.getenv("R2_POOL", "10"))
MAX_SHARD = 32 * 1024 * 1024  # 32 MiB
MIN_COPY_PART = 8 * 1024 * 1024
LEASE_TTL_S = int(os.getenv("R2_LEASE_TTL", "30"))
APPEND_CONCURRENCY = int(os.getenv("R2_MAX_CONCURRENCY", "16"))
TAIL_BLOCKS_DEFAULT = int(os.getenv("AFFINE_TAIL", "20000"))
AGG_CONCURRENCY = int(os.getenv("AFFINE_AGG_CONCURRENCY", "32"))
SINK_LOG = os.getenv("AFFINE_SINK_LOG", "0") == "1"

CACHE_DIR = Path(os.getenv("AFFINE_CACHE_DIR", Path.home() / ".cache" / "affine" / "blocks"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#                             Utility helpers                                  #
# --------------------------------------------------------------------------- #
def _stream_root(miner: str, env_name: str, env_version: Optional[str]) -> str:
    from urllib.parse import quote as _quote
    ev = env_version or "0"
    envver = _quote(f"{env_name}+{ev}", safe="")
    return f"{STREAMS_PREFIX}/{miner}/{envver}"

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
    return f"datasets/{dataset_name}/{cfg}/{spl}"

def _dataset_meta_key(root: str) -> str:
    return f"{root}/meta.json"

def _dataset_page_key(root: str, page_index: int) -> str:
    return f"{root}/pages/page-{page_index:08d}.jsonl"


# --------------------------------------------------------------------------- #
#                      CAS helpers (ETag-based optimistic)                     #
# --------------------------------------------------------------------------- #
async def _get_json(c, *, bucket: str, key: str) -> Tuple[Optional[str], Optional[dict]]:
    """Fetch JSON with retries; treat 404/403 as missing and tolerate transient errors.

    Some providers return transient errors or empty ClientError payloads. We retry
    a few times, and if we can't determine existence, we fall back to HEAD to decide.
    """
    for attempt in range(8):
        try:
            r = await c.get_object(Bucket=bucket, Key=key)
            body = await r["Body"].read()
            return r.get("ETag"), _loads(body)
        except ClientError as e:
            meta = e.response.get("ResponseMetadata", {}) if isinstance(e.response, dict) else {}
            http = meta.get("HTTPStatusCode")
            msg = (e.response.get("Error", {}).get("Message", "") if isinstance(e.response, dict) else "") or str(e)
            low = msg.lower()
            # Missing
            if http in (404, 403):
                return None, None
            # Unknown/empty code: try HEAD to detect existence
            if http is None:
                try:
                    await c.head_object(Bucket=bucket, Key=key)
                except ClientError as e2:
                    h2 = (e2.response.get("ResponseMetadata", {}) if isinstance(e2.response, dict) else {}).get("HTTPStatusCode")
                    if h2 in (404, 403):
                        return None, None
                # If HEAD succeeded or inconclusive, retry get
                if attempt < 7:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
            # Transient service conditions
            if http in (500, 503) or "publish is in progress" in low:
                if attempt < 7:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
            raise

async def _put_json(c, *, bucket: str, key: str, body: dict, if_none_match: bool = False, match_etag: Optional[str] = None) -> bool:
    """Robust JSON write with optional existence/CAS semantics.

    Falls back to explicit HEAD checks to avoid providers that don't support
    conditional headers on PutObject.
    """
    # If-None-Match semantics: only create if object does not exist
    if if_none_match:
        try:
            await c.head_object(Bucket=bucket, Key=key)
            return False  # already exists
        except ClientError as e:
            code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code not in (404, 403):
                raise
            # does not exist or not visible → proceed
    # If-Match semantics: only write if current ETag matches
    if match_etag:
        try:
            h = await c.head_object(Bucket=bucket, Key=key)
            curr = h.get("ETag")
            if not curr or curr != match_etag:
                return False
        except ClientError as e:
            code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in (404, 403):
                return False
            raise
    # Unconditional write with retries for transient provider errors
    payload = _dumps(body)
    for attempt in range(8):
        try:
            await c.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/json")
            return True
        except ClientError as e:
            meta = e.response.get("ResponseMetadata", {}) if isinstance(e.response, dict) else {}
            http = meta.get("HTTPStatusCode")
            msg = (e.response.get("Error", {}).get("Message", "") if isinstance(e.response, dict) else "") or str(e)
            low = msg.lower()
            if http in (500, 503) or "publish is in progress" in low or http is None:
                if attempt < 7:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
            raise

async def _append_jsonl(c, *, bucket: str, key: str, line_obj: dict) -> None:
    tail = _dumps(line_obj) + b"\n"
    try:
        h = await c.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in (404, 403):
            # Create fresh object with first line
            await c.put_object(Bucket=bucket, Key=key, Body=tail, ContentType="application/json")
            if SINK_LOG:
                af.logger.info(f"sink: created new shard key={key} bytes={len(tail)}")
            return
        raise

    # Attempt Hippius append with version-based CAS
    for attempt in range(8):
        # Ensure object is readable (HEAD 200); tolerate transient 202/503
        try:
            h = await c.head_object(Bucket=bucket, Key=key)
            http_status = (h.get("ResponseMetadata", {}) or {}).get("HTTPStatusCode", 200)
        except ClientError as e:
            http_status = (e.response or {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
        if http_status != 200:
            await asyncio.sleep(0.1 * (attempt + 1))
            continue

        headers = (h.get("ResponseMetadata", {}) or {}).get("HTTPHeaders", {}) if isinstance(h, dict) else {}
        ver_str = (headers or {}).get("x-amz-meta-append-version", "0")
        try:
            ver = int(str(ver_str))
        except Exception:
            ver = 0
        meta = {
            "append": "true",
            "append-if-version": str(ver),
            "append-id": uuid.uuid4().hex,
        }
        try:
            await c.put_object(Bucket=bucket, Key=key, Body=tail, ContentType="application/json", Metadata=meta)
            if SINK_LOG:
                af.logger.info(f"sink: hippus-append ok bytes={len(tail)} -> {key} v={ver}")
            return
        except ClientError as e:
            resp = e.response or {}
            http = (resp.get("ResponseMetadata", {}) or {}).get("HTTPStatusCode")
            code = (resp.get("Error", {}) or {}).get("Code")
            msg = (resp.get("Error", {}) or {}).get("Message", "").lower()
            # Retry on CAS failure or temporary unavailability
            if http in (412, 503) or "precondition" in (msg or "") or "not yet ready" in (msg or ""):
                await asyncio.sleep(0.1 * (attempt + 1))
                continue
            # If server doesn't recognize append (InvalidRequest/NotImplemented), fail fast
            if code in ("InvalidRequest", "NotImplemented") or http in (400, 501):
                raise
            # Transient publish window
            if http in (500,) or "publish is in progress" in msg:
                await asyncio.sleep(0.1 * (attempt + 1))
                continue
            raise
    # If we exit the loop without returning, all retries failed
    raise RuntimeError("hippus-append failed after retries")


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
        # Ensure bucket exists (best-effort) before writing leases/shards
        try:
            await c.head_bucket(Bucket=FOLDER)
        except Exception:
            try:
                await c.create_bucket(Bucket=FOLDER, CreateBucketConfiguration={"LocationConstraint": HIPPIUS_REGION})
            except Exception:
                # If it fails (e.g., already exists), proceed anyway. The object writes will fail if it's truly inaccessible.
                af.logger.warning(f"Could not create or confirm bucket '{FOLDER}'. Writes may fail.")

        # Optional simplified path: disable leases/pool for single-writer environments
        no_lease = os.getenv("AFFINE_NO_LEASE", "0") == "1"
        if no_lease:
            # Write to fixed shard-00000001 and maintain minimal meta so reader can discover it
            for root, docs in by_stream.items():
                shard_key = _shard_key(root, 1)
                active_key, _next_key, catalog_key = _meta_keys(root)
                for d in docs:
                    await _append_jsonl(c, bucket=FOLDER, key=shard_key, line_obj=d)
                # best-effort active pool with seq=1
                try:
                    await _put_json(c, bucket=FOLDER, key=active_key, body={"ver": 1, "pool": 1, "seqs": [1]})
                except Exception:
                    pass
                # best-effort catalog open event
                try:
                    await _append_jsonl(c, bucket=FOLDER, key=catalog_key, line_obj={"ts": _now(), "op": "open", "seq": 1})
                except Exception:
                    pass
            return
        # For each stream, ensure pool and append (pooled shards with leases)
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
                        if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") in (404, 403):
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
    t0 = time.perf_counter()
    _ea, active = await _get_json(c, bucket=FOLDER, key=active_key)
    af.logger.debug(f"s3: read active key={active_key} time={time.perf_counter()-t0:.3f}s ok={bool(active)}")
    seqs_active = list((active or {}).get("seqs", []))
    # Catalog (append-only jsonl)
    seqs_open: set[int] = set()
    seqs_close: set[int] = set()
    try:
        t1 = time.perf_counter()
        af.logger.debug(f"s3: get catalog key={catalog_key} ...")
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
        af.logger.debug(f"s3: catalog read done key={catalog_key} time={time.perf_counter()-t1:.3f}s open={len(seqs_open)} close={len(seqs_close)}")
    except ClientError:
        pass
    # Fallback discovery: if nothing known/active, probe first shard
    if not seqs_open and not seqs_active:
        try:
            first_key = _shard_key(root, 1)
            await c.head_object(Bucket=FOLDER, Key=first_key)
            af.logger.debug(f"s3: fallback discovered shard=1 for root={root}")
            return [1], []
        except Exception:
            pass
    # Known shards = open - close, plus current active (for safety)
    known = sorted((seqs_open - seqs_close) | set(seqs_active))
    af.logger.info(f"s3: root={root} known_shards={len(known)} active={len(seqs_active)}")
    return known, seqs_active


async def _cache_shard(key: str, sem: asyncio.Semaphore) -> Path:
    name = Path(key).name
    out = CACHE_DIR / f"{name}.jsonl"
    mod = out.with_suffix(".modified")
    async with sem, get_client_ctx() as c:
        try:
            t0 = time.perf_counter()
            h = await c.head_object(Bucket=FOLDER, Key=key)
            af.logger.debug(f"s3: head shard key={key} time={time.perf_counter()-t0:.3f}s size={h.get('ContentLength')}")
        except ClientError as e:
            if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") in (404, 403):
                raise
            raise
        if out.exists() and mod.exists():
            if h["LastModified"].isoformat() == mod.read_text().strip():
                return out
        t1 = time.perf_counter()
        o = await c.get_object(Bucket=FOLDER, Key=key)
        body = await o["Body"].read()
        af.logger.debug(f"s3: get shard key={key} time={time.perf_counter()-t1:.3f}s bytes={len(body)}")
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
    

# ---------------------- Dataset local cache helpers (S3) --------------------- #
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
            if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") in (404, 403):
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
    results: Dict[str, Dict[str, float]] = {}
    miner_roots = list({_stream_root(hk, env_name, env_version) for hk, _ in pairs})
    sem = asyncio.Semaphore(AGG_CONCURRENCY)
    async def worker(rt: str):
        async with sem:
            try:
                return await _aggregate_for_pairs(rt, env_name=env_name, pairs=pairs, env_version=env_version, success_only=True)
            except Exception:
                return {}
    tasks = [asyncio.create_task(worker(rt)) for rt in miner_roots]
    for t in tasks:
        agg = await t
        for hk, vals in (agg or {}).items():
            r = results.setdefault(hk, {"n_success": 0.0, "sum_score": 0.0})
            r["n_success"] += float(vals.get("n_success", 0.0))
            r["sum_score"] += float(vals.get("sum_score", 0.0))
    return results

async def aggregate_scores_by_env(*, env_name: str, pairs: List[Tuple[str, str]], env_version: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    if not pairs:
        return {}
    results: Dict[str, Dict[str, float]] = {}
    miner_roots = list({_stream_root(hk, env_name, env_version) for hk, _ in pairs})
    sem = asyncio.Semaphore(AGG_CONCURRENCY)
    async def worker(rt: str):
        async with sem:
            try:
                return await _aggregate_for_pairs(rt, env_name=env_name, pairs=pairs, env_version=env_version, success_only=False)
            except Exception:
                return {}
    tasks = [asyncio.create_task(worker(rt)) for rt in miner_roots]
    for t in tasks:
        agg = await t
        for hk, vals in (agg or {}).items():
            r = results.setdefault(hk, {"n_total": 0.0, "sum_score": 0.0, "sum_sq_score": 0.0})
            r["n_total"] += float(vals.get("n_total", 0.0))
            r["sum_score"] += float(vals.get("sum_score", 0.0))
            r["sum_sq_score"] += float(vals.get("sum_sq_score", 0.0))
    return results


# --------------------------------------------------------------------------- #
#                         Compatibility shims for CLI                          #
# --------------------------------------------------------------------------- #
# NOTE: The postgres-based CLI commands are removed as they are incompatible
# with a pure S3-based storage model for datasets.
dataset_rows = None  # type: ignore
async def _get_engine(): return None
def _sm(): raise RuntimeError("Database engine is not available in S3-backed mode")


# ----------------------------- S3 dataset reads ----------------------------- #
async def select_dataset_rows(*, dataset_name: str, config: str = "default", split: str = "train", limit: int = 1000, offset: int = 0, include_index: bool = False) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    root = _dataset_root(dataset_name, config, split)
    async with get_client_ctx() as c:
        _etag, meta = await _get_json(c, bucket=FOLDER, key=_dataset_meta_key(root))
        if not meta:
            return []
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
    keys = list(range(first_page, last_page + 1))
    tasks = [asyncio.create_task(_cache_dataset_page(root, pidx, sem)) for pidx in keys]
    pages: Dict[int, Path] = {}
    for pidx, t in zip(keys, tasks):
        try:
            pages[pidx] = await t
        except Exception:
            pass
    # Iterate and collect rows within bounds
    out: List[Dict[str, Any]] = []
    for pidx in range(first_page, last_page + 1):
        p = pages.get(pidx)
        if not p: continue
        page_start_global = pidx * page_size
        page_end_global = page_start_global + page_size
        sel_start = max(start, page_start_global)
        sel_end = min(end, page_end_global)
        if sel_end <= sel_start: continue
        rel_start = sel_start - page_start_global
        rel_end = sel_end - page_start_global
        rel_idx = 0
        async for raw in _jsonl(p):
            if rel_idx >= rel_end: break
            if rel_idx >= rel_start:
                try: rec = _loads(raw)
                except Exception: rec = None
                if isinstance(rec, dict):
                    if include_index: rec["__row_index__"] = page_start_global + rel_idx
                    out.append(rec)
            rel_idx += 1
    return out

async def get_dataset_size(*, dataset_name: str, config: str = "default", split: str = "train") -> int:
    """Return total number of rows for a dataset from Hippius meta."""
    async with get_client_ctx() as c:
        root = _dataset_root(dataset_name, config, split)
        _etag, meta = await _get_json(c, bucket=FOLDER, key=_dataset_meta_key(root))
        if meta and isinstance(meta.get("total"), int):
            return int(meta["total"])
    return 0

# These are not supported in S3-only mode
async def count(**filters: Any) -> int:
    raise RuntimeError("Count is not supported in S3-backed mode")
async def select_rows(*, limit: int = 1000, order: str = "r2_last_modified", ascending: bool = False, **filters: Any) -> List[Dict[str, Any]]:
    raise RuntimeError("Select is not supported in S3-backed mode")
async def get_env_counts(*, pairs: List[Tuple[str, str]], env_version: Optional[str] = None) -> Dict[str, Dict[Tuple[str, str], int]]:
    raise RuntimeError("get_env_counts is not supported in S3-backed mode")
