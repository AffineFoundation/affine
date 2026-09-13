#!/usr/bin/env python
"""Tensor-level fingerprint of a king's weights, and the "identical weights"
guard for the watcher.

A fingerprint is the SET of (tensor name, dtype, shape, sha256 of the tensor's
raw bytes) over every tensor in every safetensors shard — independent of how
the checkpoint is sharded, what the files are called, or their order. Two
kings with the same set are the same weights (reign 13 was a byte-copy of
reign 12 with new file hashes), and re-benchmarking them would measure noise
for ~$350.

Shards are streamed from the PUBLIC copy (https://models.affine.io/models/
sha256/<digest>/, one shard on disk at a time, ~66 GB total download for a
35B-A3B) and the result is cached in ops/benchsuite/state/fingerprints/
<digest12>.json, so each king is hashed once.

  python weights_fingerprint.py <digest>                      # compute (or read the cache), print a summary
  python weights_fingerprint.py <digest> --compare <digest2>  # exit 0 if identical tensors, 1 if different
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
CACHE = HERE / "state" / "fingerprints"
PUBLIC = "https://models.affine.io/models/sha256/{digest}/"


def log(msg: str) -> None:
    print(f"[fingerprint] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def tensor_hashes(shard: Path) -> list[dict]:
    """(name, dtype, shape, sha256) for every tensor in one safetensors file."""
    out = []
    with shard.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            fh.seek(base + start)
            h = hashlib.sha256()
            remaining = end - start
            while remaining > 0:
                chunk = fh.read(min(1 << 24, remaining))
                if not chunk:
                    raise IOError(f"short read in {shard.name} for {name}")
                h.update(chunk)
                remaining -= len(chunk)
            out.append({"name": name, "dtype": meta["dtype"], "shape": meta["shape"],
                        "sha256": h.hexdigest()})
    return out


def fingerprint(digest: str, workdir: Path | None = None) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{digest[:12]}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    base = PUBLIC.format(digest=digest)
    man = httpx.get(base + "manifest.json", timeout=60).raise_for_status().json()
    shards = [f for f in man["files"] if f["path"].endswith(".safetensors")]
    tensors: list[dict] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="fp-", dir=str(workdir) if workdir else None))
    try:
        for i, f in enumerate(sorted(shards, key=lambda x: x["path"])):
            dst = tmpdir / Path(f["path"]).name
            log(f"{digest[:12]}: shard {i + 1}/{len(shards)} {f['path']} ({f['size'] / 1e9:.1f} GB)")
            subprocess.run(["curl", "-sSL", "--http1.1", "--retry", "8", "--retry-all-errors",
                            "-o", str(dst), base + f["path"]], check=True)
            if f.get("sha256"):
                h = hashlib.sha256()
                with dst.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 24), b""):
                        h.update(chunk)
                if h.hexdigest() != f["sha256"]:
                    raise SystemExit(f"shard sha mismatch {f['path']}")
            tensors += tensor_hashes(dst)
            dst.unlink()
    finally:
        for p in tmpdir.glob("*"):
            p.unlink()
        tmpdir.rmdir()
    tensors.sort(key=lambda t: t["name"])
    set_hash = hashlib.sha256("\n".join(f"{t['name']}|{t['dtype']}|{t['shape']}|{t['sha256']}"
                                        for t in tensors).encode()).hexdigest()
    fp = {"digest": digest, "n_tensors": len(tensors), "n_shards": len(shards),
          "bytes": sum(f["size"] for f in shards), "tensor_set_sha256": set_hash,
          "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tensors": tensors}
    cached.write_text(json.dumps(fp))
    log(f"{digest[:12]}: {len(tensors)} tensors, set sha {set_hash[:16]}")
    return fp


def identical(a: dict, b: dict) -> bool:
    return a["tensor_set_sha256"] == b["tensor_set_sha256"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("digest")
    ap.add_argument("--compare", default="")
    ap.add_argument("--workdir", default=os.environ.get("BENCHSUITE_FP_WORKDIR", ""))
    a = ap.parse_args()
    wd = Path(a.workdir) if a.workdir else None
    fa = fingerprint(a.digest, wd)
    print(json.dumps({k: v for k, v in fa.items() if k != "tensors"}, indent=1))
    if not a.compare:
        return 0
    fb = fingerprint(a.compare, wd)
    same = identical(fa, fb)
    diff_names = {t["sha256"] for t in fa["tensors"]} ^ {t["sha256"] for t in fb["tensors"]}
    print(json.dumps({"identical": same, "tensors_differing": len(diff_names) // 2 if not same else 0,
                      "a": fa["tensor_set_sha256"][:16], "b": fb["tensor_set_sha256"][:16]}))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
