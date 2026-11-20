"""Ranked reserve reload after a displaced task returns through its archive."""
from __future__ import annotations

import json
from pathlib import Path

from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.reserve import load_reserve


def test_ranked_reload_keeps_retired_shard_history_without_duplicate_admission(tmp_path):
    root = tmp_path / "reserve"
    (root / "shards").mkdir(parents=True)
    row = {"id": "omni:0", "env": "affine_math", "problem": "UNIQUE-DISPLACED-RETURN",
           "answer": "1", "source": "test@rev:0", "license": "CC0-1.0", "added_at": "2026-11-20",
           "audit": {"verdict": "keep", "method": "test", "ref": "test"}}
    raw = canonical_json(row) + b"\n"
    shard = "0001-test.jsonl"
    (root / "shards" / shard).write_bytes(raw)
    (root / "MANIFEST.jsonl").write_bytes(canonical_json({"shard": shard, "sha256": sha256_hex(raw), "count": 1, "added_at": "2026-11-20"}) + b"\n" +
                                         canonical_json({"withdrawn": {"sha256": sha256_hex(b""), "count": 0}}) + b"\n")
    (root / "withdrawn.jsonl").write_bytes(b"")
    retired = {"id": "omni:0", "round_id": "r000070", "reason": "displaced", "event_seq": 0}
    task = {"id": "omni:0", "env": "affine_math", "problem": row["problem"], "answer": row["answer"]}
    (root / "retired.json").write_bytes(canonical_json([retired]))
    (root / "retired_tasks.json").write_bytes(canonical_json([task]))

    reserve, catalog = load_reserve(root, with_catalog=True, allow_retired=True)
    assert reserve.queue == ("omni:0",)  # historical queue evidence is retained
    assert catalog.task("omni:0").problem == row["problem"]
    # The archive remains the source of the task; the original row is not a
    # second candidate for ranked admission.
    assert catalog.task_ids.count("omni:0") == 1
    assert len(catalog.task_ids) == len(set(catalog.task_ids))
