"""Build the compact SHA-256(normalise(problem)) -> source/license prior index.

Run with the prior-index tooling extra installed:
    python -m sn120_v2.envs.tools.prior_index PRIOR_DIR OUTPUT.json
PRIOR_DIR has the eight release parquet files and MANIFEST.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ...canonical import canonical_json, file_sha256, sha256_hex
from ..reserve import normalise
from ..reserve.generator import POOL, POOL_SHA256

COLUMNS = ["id", "problem", "answer", "answer_type", "source", "source_id", "license", "subsource", "topic", "difficulty_meta", "also_in", "in_sn120_pool"]


def build(root: Path) -> dict:
    manifest_raw = (root / "MANIFEST.json").read_bytes()
    if sha256_hex(manifest_raw) != POOL_SHA256:
        raise ValueError("prior MANIFEST.json does not match the pinned pool sha256")
    manifest = json.loads(manifest_raw)
    if manifest.get("rows") != sum(shard["rows"] for shard in manifest["shards"]):
        raise ValueError("prior manifest row count mismatch")
    for shard in manifest["shards"]:
        if file_sha256(root / shard["file"]) != shard["sha256"]:
            raise ValueError(f"prior shard {shard['file']} does not match its published sha256")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("prior index builder requires pyarrow") from exc
    index = {}
    rows = 0
    for shard in manifest["shards"]:
        parquet = pq.ParquetFile(root / shard["file"])
        if parquet.schema_arrow.names != COLUMNS or parquet.metadata.num_rows != shard["rows"]:
            raise ValueError(f"unexpected schema or rows in {shard['file']}")
        for batch in parquet.iter_batches(columns=["problem", "source", "license"], batch_size=8192):
            for row in batch.to_pylist():
                key = sha256_hex(normalise(row["problem"]).encode())
                entry = {"source": row["source"], "license": row["license"]}
                if index.setdefault(key, entry) != entry:
                    raise ValueError("prior has conflicting provenance for a normalized problem")
                rows += 1
    if rows != manifest["rows"]:
        raise ValueError("prior decoded row count mismatch")
    return {"schema": "sn120-math-prior-index-v1", "pool": POOL, "rows": len(index), "source_rows": rows, "problems": index}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    document = build(args.prior_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(canonical_json(document) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({key: value for key, value in document.items() if key != "problems"}, indent=2))
    print(f"prior index size: {len(document['problems']):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
