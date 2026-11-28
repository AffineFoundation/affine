#!/usr/bin/env python3
"""Fetch the current, public SN120 official-active pool (standard library only).

    python v2/miner/live_pool.py --out tasks.jsonl

``mining.json.round.pool`` is the authoritative pointer. The summary is only a
fallback when that round has no pool. Every active-list download is checked
against ``active_sha``. Reserve manifests and withdrawal pointers are refreshed;
only immutable, hash-addressed reserve shards are reused across calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sn120-standalone-source-path

from sn120_v2 import layout
from sn120_v2.envs.reserve import atomic_write, load_reserve
from sn120_v2.errors import ConfigError, V2Error, VerifyError

PUBLIC_URL = "https://affine.io"
CLIENT = {"User-Agent": "sn120-v2-live-pool/1", "Cache-Control": "no-cache", "Pragma": "no-cache"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ATTEMPTS = 3
TIMEOUT_S = 30


class PoolError(V2Error):
    """The public pool could not be fetched or did not verify."""


def _location(public_url: str, key: str) -> str:
    base = public_url.rstrip("/") + "/"
    if not urllib.parse.urlsplit(base).scheme:
        raise PoolError("public URL must be an http(s) URL or local object root")
    return urllib.parse.urljoin(base, key)


def _uncached(location: str) -> str:
    parts = urllib.parse.urlsplit(location)
    if parts.scheme not in ("http", "https"):
        raise PoolError("public URL must be an http(s) URL")
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query.append(("_", f"{time.time_ns()}-{os.getpid()}"))
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def read_bytes(public_url: str, key: str, *, attempts: int = ATTEMPTS) -> bytes:
    """Read local fixtures directly; HTTP reads use the production cache-busting client."""
    location = _location(public_url, key)
    if not location.startswith(("http://", "https://")):
        path = Path(urllib.request.url2pathname(urllib.parse.urlsplit(location).path))
        try:
            return path.read_bytes()
        except OSError as exc:
            raise PoolError(f"could not read {path}: {exc}") from exc
    return fetch_bytes(public_url, key, attempts=attempts)


def fetch_bytes(public_url: str, key: str, *, attempts: int = ATTEMPTS) -> bytes:
    """GET fresh bytes with a named client, no-cache headers and bounded retries."""
    location = _location(public_url, key)
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(_uncached(location), headers=CLIENT)
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as reply:
                return reply.read()
        except (OSError, urllib.error.URLError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(0.25 * 2**attempt)
    raise PoolError(f"could not fetch {location} after {attempts} attempts: {type(last).__name__}: {last}") from last


def _json(raw: bytes, name: str) -> Any:
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise PoolError(f"{name} is not valid UTF-8 JSON: {exc}") from exc


def _pointer(public_url: str) -> tuple[dict[str, Any], bytes, Mapping[str, Any]]:
    mining = _json(read_bytes(public_url, layout.MINING_JSON), layout.MINING_JSON)
    if not isinstance(mining, dict):
        raise PoolError("mining.json is not an object")
    pool = (mining.get("round") or {}).get("pool")
    if not isinstance(pool, Mapping) or not pool.get("active_sha"):
        # summary.json can lag mining.json. Use it only when the round has no pool.
        pool = _json(read_bytes(public_url, layout.POOL_SUMMARY), layout.POOL_SUMMARY)
        if not isinstance(pool, Mapping):
            raise PoolError("pool summary is not an object")
    sha = str(pool.get("active_sha", ""))
    if not SHA256.fullmatch(sha):
        raise PoolError("the public pool has no valid active_sha")
    raw = read_bytes(public_url, layout.active_pool(sha))
    found = hashlib.sha256(raw).hexdigest()
    if found != sha:
        raise PoolError(f"active pool SHA256 mismatch: expected {sha}, got {found}")
    return dict(pool), raw, mining


def active_ids(raw: bytes) -> tuple[str, ...]:
    """Parse the already SHA-verified active object, using the validator's shape rules."""
    ids = _json(raw, "active pool")
    if not isinstance(ids, list) or any(not isinstance(task, str) for task in ids) or ids != sorted(ids) or len(set(ids)) != len(ids):
        raise PoolError("active pool is not a sorted, unique list of task IDs")
    return tuple(ids)


def fetch_active_ids(public_url: str = PUBLIC_URL) -> tuple[str, ...]:
    """Return the live official-active task IDs after verifying the active object."""
    _, raw, _ = _pointer(public_url)
    return active_ids(raw)


def _cached_shard(cache: Path, public_url: str, name: str, sha: str) -> bytes:
    if not SHA256.fullmatch(sha):
        raise PoolError(f"reserve shard {name} has an invalid SHA256")
    path = cache / "shards" / f"{sha}.jsonl"
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() == sha:
            return raw
    except OSError:
        pass
    raw = read_bytes(public_url, layout.reserve_shard(name, sha))
    if hashlib.sha256(raw).hexdigest() != sha:
        raise PoolError(f"reserve shard {name} does not match its manifest SHA256")
    atomic_write(path, raw)
    return raw


def _put_fresh(path: Path, raw: bytes) -> None:
    """Refresh a pointer without replacing a concurrently-read good file in place."""
    atomic_write(path, raw)


def _catalog(public_url: str, pool: Mapping[str, Any], mining: Mapping[str, Any], cache: Path,
             active: Sequence[str] = ()):
    rotation: Mapping[str, Any] = pool
    if "reserve_manifest_sha" not in pool:
        round_id = str(pool.get("round_id", ""))
        if not round_id:
            raise PoolError("pool pointer lacks reserve_manifest_sha and round_id")
        rotation = _json(read_bytes(public_url, layout.pool_rotation(round_id)), "pool rotation")
        if not isinstance(rotation, Mapping):
            raise PoolError("pool rotation is not an object")
    expected = str(rotation.get("reserve_manifest_sha", ""))
    if not SHA256.fullmatch(expected):
        raise PoolError("the pool rotation has no valid reserve_manifest_sha")
    manifest = read_bytes(public_url, layout.RESERVE_MANIFEST)
    if hashlib.sha256(manifest).hexdigest() != expected:
        raise PoolError("reserve manifest does not match the round's reserve_manifest_sha")
    withdrawn = read_bytes(public_url, layout.RESERVE_WITHDRAWN)
    withdrawn_sha = hashlib.sha256(withdrawn).hexdigest()
    root = cache / "catalog" / f"{expected}-{withdrawn_sha}"
    _put_fresh(root / layout.RESERVE_MANIFEST.rsplit("/", 1)[-1], manifest)
    _put_fresh(root / "withdrawn.jsonl", withdrawn)
    lines = _json_lines(manifest, "reserve manifest")
    if not lines or not isinstance(lines[-1], dict) or lines[-1].get("withdrawn", {}).get("sha256") != withdrawn_sha:
        raise PoolError("reserve manifest's final withdrawal pointer does not match withdrawn.jsonl")
    (root / "shards").mkdir(parents=True, exist_ok=True)
    for row in lines[:-1]:
        if not isinstance(row, Mapping) or not isinstance(row.get("shard"), str) or not isinstance(row.get("sha256"), str):
            raise PoolError("reserve manifest has a malformed shard row")
        name, sha = row["shard"], row["sha256"]
        raw = _cached_shard(cache, public_url, name, sha)
        target = root / "shards" / name
        try:
            if target.read_bytes() != raw:
                _put_fresh(target, raw)
        except OSError:
            _put_fresh(target, raw)
    for key, name in ((layout.RETIRED_POOL, "retired.json"), (layout.RETIRED_TASKS, "retired_tasks.json")):
        # These pointers can change between active sets; never use stale copies.
        _put_fresh(root / name, read_bytes(public_url, key))
    allow_retired = rotation.get("rule_version", (mining.get("params") or {}).get("rotation_rule")) in ("affine-pool-rotation-v2", "info")
    _, catalog = load_reserve(root, with_catalog=True, allow_retired=allow_retired)
    if not active:
        return catalog
    from sn120_v2.envs.math_pool import MathPool
    return MathPool(tuple(catalog.task(task_id) for task_id in active), catalog.sha256)


def _json_lines(raw: bytes, name: str) -> list[Any]:
    try:
        return [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, ValueError) as exc:
        raise PoolError(f"{name} is not valid UTF-8 JSONL: {exc}") from exc


def active_tasks(public_url: str = PUBLIC_URL, cache: str | Path | None = None) -> list[dict[str, str]]:
    """Resolve every live ID to ``task_id``, ``problem``, ``answer`` and official ``prompt``.

    Base ``dapo:``/``math:`` bodies come from the checked-in base catalog.
    Generator ``gen:`` and other reserve bodies come from the verified public
    manifest and its content-addressed shards. The bundled JSONL is not treated
    as the current active set.
    """
    cache = Path(cache).expanduser() if cache is not None else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "affine" / "live-pool"
    pool, raw, mining = _pointer(public_url)
    ids = active_ids(raw)
    if pool.get("size") is not None and pool["size"] != len(ids):
        raise PoolError("active pool size differs from its public pointer")
    catalog = _catalog(public_url, pool, mining, cache, ids)
    return tasks_for_ids(catalog, ids)


def tasks_for_ids(catalog: Any, ids: Sequence[str]) -> list[dict[str, str]]:
    """Resolve IDs through a verified public catalog and render official batch prompts."""
    try:
        tasks = [catalog.task(task_id) for task_id in ids]
    except V2Error as exc:
        raise PoolError(f"an active task has no verified public body: {exc}") from exc
    return [{"task_id": task.task_id, "problem": task.problem, "answer": task.answer, "prompt": task.batch_prompt()} for task in tasks]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--public-url", default=PUBLIC_URL)
    parser.add_argument("--cache", help="public-pool cache (immutable shards are reused; pointers are refreshed)")
    parser.add_argument("--out", required=True, help="write task_id/problem/answer/prompt JSONL here")
    args = parser.parse_args(argv)
    try:
        tasks = active_tasks(args.public_url, args.cache)
        raw = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in tasks).encode("utf-8")
        atomic_write(Path(args.out).expanduser(), raw)
    except (V2Error, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"live pool failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"out": args.out, "tasks": len(tasks), "source": "live official-active pool"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
