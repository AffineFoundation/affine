"""Pack duel_turns@v4 view records into chunks + a Parquet turn index.

Same discipline as pack.py for v2: chunk files are uncompressed JSONL
locally (sha256 over those bytes is the manifest sha; gzip happens at
upload), one record per line, and the index has one row per scorable turn
pointing at (chunk_key, traj_line, node_id).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .materialize import materialize_turn, stratum_key
from .pack import PackResult

log = logging.getLogger("affine.corpus.viewpack")

DEFAULT_CHUNK_RECORDS = 1000
ROUNDTRIP_SAMPLE = 64
FORMAT = "view_v4"

INDEX_SCHEMA = pa.schema([
    ("turn_id", pa.string()),
    ("traj_id", pa.string()),
    ("rollout_id", pa.string()),
    ("turn_idx", pa.int32()),
    ("node_id", pa.int32()),
    ("stratum", pa.string()),
    ("phase", pa.string()),
    ("source", pa.string()),
    ("language", pa.string()),
    ("action_kind", pa.string()),
    ("chunk_key", pa.string()),
    ("traj_line", pa.int32()),
    ("n_prefix_chars", pa.int32()),
])


def index_rows(record: dict, chunk_key: str, line: int) -> list[dict]:
    rows = []
    for meta in record["turns"]:
        rows.append({
            "turn_id": f"{record['traj_id']}:{meta['turn_idx']}",
            "traj_id": record["traj_id"],
            "rollout_id": record.get("rollout_id") or "",
            "turn_idx": int(meta["turn_idx"]),
            "node_id": int(meta["node_id"]),
            "stratum": stratum_key({"traj_id": record["traj_id"],
                                    "stratum": record.get("stratum")}),
            "phase": str(meta.get("phase") or ""),
            "source": str(record.get("source") or ""),
            "language": str(record.get("language") or ""),
            "action_kind": str(meta.get("action_kind")
                               or record.get("action_kind") or "bash"),
            "chunk_key": chunk_key,
            "traj_line": line,
            "n_prefix_chars": int(meta.get("n_prefix_chars") or 0),
        })
    return rows


def _roundtrip(records: list[dict], n_sample: int = ROUNDTRIP_SAMPLE) -> None:
    """Every sampled turn must materialize (graph intact, prefix ends on
    user, reply is an assistant node)."""
    checked = 0
    for rec in records:
        for meta in rec["turns"]:
            materialize_turn(rec, meta)
            checked += 1
            if checked >= n_sample:
                return


def pack_view_records(records: list[dict], out_dir: Path, *, epoch: int,
                      view_spec: str,
                      chunk_records: int = DEFAULT_CHUNK_RECORDS) -> PackResult:
    """Chunk keys: views/{view_spec}/chunks/view_{epoch:04d}_{i:04d}.jsonl.gz;
    index key: views/{view_spec}/index/turns_{epoch:04d}.parquet (the
    publisher uploads it under that name)."""
    if not records:
        raise ValueError("no view records to pack")
    out_dir.mkdir(parents=True, exist_ok=True)
    _roundtrip(records)
    chunks_prefix = f"views/{view_spec}/chunks"
    chunk_paths: list[Path] = []
    chunk_meta: list[dict] = []
    rows: list[dict] = []
    for ci in range(0, len(records), chunk_records):
        batch = records[ci:ci + chunk_records]
        i = ci // chunk_records
        local = out_dir / f"view_{epoch:04d}_{i:04d}.jsonl"
        raw = ("\n".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False)
                         for r in batch) + "\n").encode("utf-8")
        local.write_bytes(raw)
        key = f"{chunks_prefix}/view_{epoch:04d}_{i:04d}.jsonl.gz"
        chunk_paths.append(local)
        chunk_meta.append({
            "key": key,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "n_trajectories": len(batch),
            "n_turns": sum(len(r["turns"]) for r in batch),
            "format": FORMAT,
            "active": True,
        })
        for line, rec in enumerate(batch):
            rows.extend(index_rows(rec, key, line))

    index_path = out_dir / f"turns_{epoch:04d}.parquet"
    table = pa.Table.from_pylist(rows, schema=INDEX_SCHEMA)
    pq.write_table(table, index_path, compression="zstd")
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    log.info("packed %d turns / %d records -> %d chunks + index (%s)",
             len(rows), len(records), len(chunk_paths), index_sha[:12])
    return PackResult(chunk_paths=chunk_paths, chunk_meta=chunk_meta,
                      index_path=index_path, index_sha256=index_sha,
                      n_turns=len(rows), n_trajectories=len(records))
