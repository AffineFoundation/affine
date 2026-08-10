#!/usr/bin/env python
"""One-shot migration: active v1 turn shards → schema_version=2 (dual-publish).

Downloads every active v1 shard from the public Hippius bucket, packs into
trajectory chunks + Parquet index, runs a golden seed-parity check against
the flat corpus, then publishes epoch N+1 with schema_version=2 and a
compat flat shard for miner cutover.

Deploy eval pods with the v2 CorpusSync/dueling code BEFORE flipping the
manifest pointer (this script is the flip).

  source .venv/bin/activate && source .env
  # HIPPIUS_* from validator environ or doppler
  python affine/scripts/migrate_corpus_v2.py [--work-dir DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "affine"))

from affine.corpus.pack import pack_turns_to_chunks  # noqa: E402
from evalsrv.dueling import sample_slice, turn_id  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "corpus_push", REPO / "affine" / "scripts" / "corpus_push.py")
corpus_push = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(corpus_push)


def log(msg: str) -> None:
    print(msg, flush=True)


def hippius_env_from_validator() -> None:
    if os.environ.get("HIPPIUS_ACCESS_KEY") and os.environ.get("HIPPIUS_SECRET_KEY"):
        return
    import subprocess
    p = subprocess.run(["pm2", "jlist"], capture_output=True, text=True,
                       timeout=60)
    if p.returncode != 0:
        sys.exit("pm2 jlist failed; set HIPPIUS_* or start affine-validator")
    pid = next((proc["pid"] for proc in json.loads(p.stdout)
                if proc["name"] == "affine-validator"
                and proc["pm2_env"]["status"] == "online"), None)
    if not pid:
        sys.exit("affine-validator not online; set HIPPIUS_* env vars")
    env = {}
    for entry in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        k, _, v = entry.partition(b"=")
        env[k.decode()] = v.decode()
    for key in ("HIPPIUS_ACCESS_KEY", "HIPPIUS_SECRET_KEY"):
        if not env.get(key):
            sys.exit(f"{key} missing from validator environ")
        os.environ[key] = env[key]
    log("hippius credentials sourced from validator process env")


def golden_parity(flat_path: Path, pack) -> None:
    """sample_slice turn_ids from flat turns must equal index-row sampling."""
    with open(flat_path, encoding="utf-8") as f:
        flat = [json.loads(line) for line in f if line.strip()]
    import pyarrow.parquet as pq
    table = pq.read_table(pack.index_path)
    cols = {n: table.column(n).to_pylist() for n in table.column_names}
    index_rows = [{k: cols[k][i] for k in cols} for i in range(table.num_rows)]
    if len(flat) != len(index_rows):
        sys.exit(f"golden: flat {len(flat)} != index {len(index_rows)}")
    for seed in (0, 1, 42, 123456789, 0xDEADBEEF):
        a = [turn_id(t) for t in sample_slice(flat, 80, seed)]
        b = [turn_id(t) for t in sample_slice(index_rows, 80, seed)]
        if a != b:
            sys.exit(f"golden FAIL seed={seed}: first mismatch at "
                     f"{next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)}")
    log(f"golden parity OK (5 seeds, n=80) over {len(flat):,} turns")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="pack + golden check only; do not publish")
    ap.add_argument("--chunk-trajs", type=int, default=1000)
    args = ap.parse_args()

    hippius_env_from_validator()
    corpus = corpus_push.Corpus()
    m = corpus._require_manifest()
    if int(m.get("schema_version") or 1) >= 2:
        sys.exit("manifest already schema_version>=2; nothing to migrate")

    active = [s for s in m["shards"] if s.get("active")]
    if not active:
        sys.exit("no active shards")
    epoch = int(m["corpus_epoch"]) + 1
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="corpus_v2_"))
    work.mkdir(parents=True, exist_ok=True)
    log(f"work dir: {work} → target epoch {epoch}")

    base = corpus.cfg.dataset.corpus_base_url.rstrip("/")
    flat_all = work / "all_turns.jsonl"
    with open(flat_all, "wb") as out:
        for s in active:
            log(f"fetch {s['key']}")
            r = httpx.get(f"{base}/{s['key']}", timeout=600,
                          follow_redirects=True)
            r.raise_for_status()
            raw = gzip.decompress(r.content)
            got = hashlib.sha256(raw).hexdigest()
            if got != s["sha256"]:
                sys.exit(f"sha mismatch for {s['key']}: {got} != {s['sha256']}")
            out.write(raw)
            if raw and not raw.endswith(b"\n"):
                out.write(b"\n")

    pack_dir = work / "pack"
    pack = pack_turns_to_chunks(
        [flat_all], pack_dir, epoch=epoch,
        chunk_trajs=args.chunk_trajs, write_compat=True)
    golden_parity(flat_all, pack)
    log(f"packed: {pack.n_trajectories:,} trajs, {pack.n_turns:,} turns, "
        f"{len(pack.chunk_paths)} chunks")

    if args.dry_run:
        log("dry-run: not publishing")
        return

    manifest = corpus.cmd_add_v2(pack, retire_previous=True)
    raw = json.dumps(manifest, indent=2, sort_keys=True).encode()
    mhash = hashlib.sha256(raw).hexdigest()
    log(f"PUBLISHED epoch={manifest['corpus_epoch']} schema=2 "
        f"manifest={mhash}")
    log(f"compat: {[c['key'] for c in manifest.get('compat_shards', [])]}")


if __name__ == "__main__":
    main()
