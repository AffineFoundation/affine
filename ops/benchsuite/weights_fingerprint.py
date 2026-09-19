#!/usr/bin/env python
"""Cheap weight-identity check for the watcher's "identical weights" guard.

Question answered: are the new king's weights the same tensors as the last
benchmarked king's? If yes, a pass would measure noise for ~$350 (reign 13 was
a byte-copy of reign 12 with new file hashes). The check must not delay the
pass: the first version streamed every shard (~70 GB, 90 min on 2026-09-14);
this one finishes in well under two minutes and never pulls a shard.

Two tiers, both against the PUBLIC copy (https://models.affine.io/models/
sha256/<digest>/, Cloudflare R2, HTTP range requests honoured):

1. Manifest file hashes. The signed manifest lists sha256 + size per file. Equal
   sets of safetensors file hashes => byte-identical checkpoints => identical
   weights. No weight bytes read.
2. Sampled tensor fingerprint (the re-sharded / re-serialised case). Read each
   shard's safetensors header with a range request (8-byte length + JSON), then
   for every tensor hash a FIXED sample of its bytes: five 4 KiB windows at
   0 / 25 / 50 / 75 / 100 % of the tensor (the whole tensor when it is
   <= 32 KiB), fetched as range requests in a thread pool. The sample plan is a
   function of (name, nbytes) only, so it is independent of sharding, file names
   and header layout — the same tensor gives the same sample hash wherever it
   lives. The fingerprint is the set of (name, dtype, shape, nbytes, sample
   sha256); ~1,000 tensors x 5 windows = ~5k small GETs, ~20 MB, tens of seconds.

A sampled match is identity "with overwhelming probability", not a proof (a
model built to match the king's samples while differing elsewhere would pass);
this guard only decides whether to spend a monitoring pass, so that is fine.
Full-weight identity for admission lives in the validator (PR #20), not here.

Cache: ops/benchsuite/state/fingerprints/<digest12>.sampled.json (one per king).
The old full-tensor files <digest12>.json are left in place and not used.

  python weights_fingerprint.py <digest>                      # compute (or read the cache), print a summary
  python weights_fingerprint.py <digest> --compare <digest2>  # exit 0 if identical weights, 1 if different
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
CACHE = HERE / "state" / "fingerprints"
PUBLIC = "https://models.affine.io/models/sha256/{digest}/"

WINDOW = 4096                       # bytes per sample window
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
WHOLE_BELOW = 32 * 1024             # tensors this small are hashed in full
WORKERS = 48
FINGERPRINT_VERSION = 2             # bump when the sample plan changes (caches are per version)


def log(msg: str) -> None:
    print(f"[fingerprint] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def get_range(client: httpx.Client, url: str, start: int, end_inclusive: int) -> bytes:
    """One HTTP range GET with retries; the server must honour it (206)."""
    last: Exception | None = None
    for attempt in range(6):
        try:
            r = client.get(url, headers={"Range": f"bytes={start}-{end_inclusive}"})
            if r.status_code == 206 and len(r.content) == end_inclusive - start + 1:
                return r.content
            last = RuntimeError(f"range {start}-{end_inclusive}: status {r.status_code}, {len(r.content)} bytes")
        except httpx.HTTPError as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"range GET failed for {url}: {last!r}")


def shard_header(client: httpx.Client, url: str) -> tuple[dict, int]:
    """(safetensors header JSON, byte offset where tensor data starts)."""
    (n,) = struct.unpack("<Q", get_range(client, url, 0, 7))
    if not 8 <= n <= 256 << 20:
        raise RuntimeError(f"{url}: implausible safetensors header length {n}")
    header = json.loads(get_range(client, url, 8, 8 + n - 1))
    return header, 8 + n


def sample_plan(nbytes: int) -> list[tuple[int, int]]:
    """Fixed (offset, length) windows inside a tensor of `nbytes` bytes."""
    if nbytes <= WHOLE_BELOW:
        return [(0, nbytes)] if nbytes else []
    plan = []
    for f in FRACTIONS:
        off = min(int(f * nbytes), nbytes - WINDOW)
        plan.append((off, WINDOW))
    # de-duplicate overlapping windows on small tensors, keep order
    seen, out = set(), []
    for w in plan:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def manifest(digest: str, client: httpx.Client) -> dict:
    return client.get(PUBLIC.format(digest=digest) + "manifest.json").raise_for_status().json()


def file_sha_set(man: dict) -> list[str]:
    return sorted(f["sha256"] for f in man["files"] if f["path"].endswith(".safetensors") and f.get("sha256"))


def fingerprint(digest: str, workdir: Path | None = None) -> dict:
    """Sampled tensor fingerprint of the public copy of `digest` (cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{digest[:12]}.sampled.json"
    if cached.exists():
        fp = json.loads(cached.read_text())
        if fp.get("version") == FINGERPRINT_VERSION:
            return fp
    t0 = time.time()
    base = PUBLIC.format(digest=digest)
    with httpx.Client(timeout=60, http2=False,
                      limits=httpx.Limits(max_connections=WORKERS, max_keepalive_connections=WORKERS)) as client:
        man = manifest(digest, client)
        shards = sorted((f for f in man["files"] if f["path"].endswith(".safetensors")), key=lambda f: f["path"])
        # tensor -> (url, absolute data offset, nbytes, dtype, shape)
        tensors: dict[str, tuple[str, int, int, str, list]] = {}
        for f in shards:
            url = base + f["path"]
            header, data0 = shard_header(client, url)
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                start, end = meta["data_offsets"]
                if name in tensors:
                    raise RuntimeError(f"{digest[:12]}: tensor {name} appears in two shards")
                tensors[name] = (url, data0 + start, end - start, meta["dtype"], meta["shape"])
        log(f"{digest[:12]}: {len(shards)} shards, {len(tensors)} tensors; sampling "
            f"{sum(len(sample_plan(t[2])) for t in tensors.values())} windows")

        def one(item: tuple[str, tuple]) -> dict:
            name, (url, off, nbytes, dtype, shape) = item
            h = hashlib.sha256(f"{name}|{nbytes}|v{FINGERPRINT_VERSION}".encode())
            for w_off, w_len in sample_plan(nbytes):
                h.update(get_range(client, url, off + w_off, off + w_off + w_len - 1))
            return {"name": name, "dtype": dtype, "shape": shape, "nbytes": nbytes, "sample_sha256": h.hexdigest()}

        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            rows = list(pool.map(one, sorted(tensors.items())))
    rows.sort(key=lambda t: t["name"])
    set_hash = hashlib.sha256("\n".join(
        f"{t['name']}|{t['dtype']}|{t['shape']}|{t['nbytes']}|{t['sample_sha256']}" for t in rows).encode()).hexdigest()
    fp = {"version": FINGERPRINT_VERSION, "digest": digest, "n_tensors": len(rows), "n_shards": len(shards),
          "bytes": sum(f["size"] for f in shards), "file_sha256_set": file_sha_set(man),
          "tensor_set_sha256": set_hash, "sample_plan": {"window": WINDOW, "fractions": FRACTIONS,
                                                          "whole_below": WHOLE_BELOW},
          "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "seconds": round(time.time() - t0, 1), "tensors": rows}
    cached.write_text(json.dumps(fp))
    log(f"{digest[:12]}: {len(rows)} tensors, set sha {set_hash[:16]} in {fp['seconds']} s")
    return fp


def identical(a: dict, b: dict) -> bool:
    return a["tensor_set_sha256"] == b["tensor_set_sha256"]


def same_weights(digest_a: str, digest_b: str) -> tuple[bool, str]:
    """(identical?, how it was decided). Tier 1 = manifest file hashes (no weight
    bytes read); tier 2 = sampled tensor fingerprints."""
    if digest_a == digest_b:
        return True, "same digest"
    with httpx.Client(timeout=60) as client:
        fa, fb = file_sha_set(manifest(digest_a, client)), file_sha_set(manifest(digest_b, client))
    if fa and fa == fb:
        return True, f"manifest: same {len(fa)} safetensors file hashes"
    a, b = fingerprint(digest_a), fingerprint(digest_b)
    if identical(a, b):
        return True, f"sampled tensor fingerprint: same {a['n_tensors']} tensors ({a['tensor_set_sha256'][:16]})"
    differ = len({t["sample_sha256"] for t in a["tensors"]} ^ {t["sample_sha256"] for t in b["tensors"]}) // 2
    return False, f"sampled tensor fingerprint: {differ} of {a['n_tensors']} tensors differ"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("digest")
    ap.add_argument("--compare", default="")
    a = ap.parse_args()
    if a.compare:
        t0 = time.time()
        same, how = same_weights(a.digest, a.compare)
        print(json.dumps({"identical": same, "how": how, "seconds": round(time.time() - t0, 1)}))
        return 0 if same else 1
    fa = fingerprint(a.digest)
    print(json.dumps({k: v for k, v in fa.items() if k != "tensors"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
