"""Ranked reserve reload after a displaced task returns through its archive."""
from __future__ import annotations

import json

import pytest

from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.reserve import candidate_metadata, load_reserve
from sn120_v2.errors import ConfigError, VerifyError


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


def test_content_addressed_public_shards_match_the_live_reserve(tmp_path):
    row = {"id": "omni:0", "env": "affine_math", "problem": "UNIQUE-PUBLIC-SHARD-READ",
           "answer": "1", "source": "test@rev:0", "license": "CC0-1.0", "added_at": "2026-11-20",
           "audit": {"verdict": "keep", "method": "test", "ref": "test"}}
    raw = canonical_json(row) + b"\n"
    digest = sha256_hex(raw)
    shard = "0001-test.jsonl"
    manifest = (canonical_json({"shard": shard, "sha256": digest, "count": 1, "added_at": "2026-11-20"}) + b"\n" +
                canonical_json({"withdrawn": {"sha256": sha256_hex(b""), "count": 0}}) + b"\n")
    for name in ("live", "public"):
        root = tmp_path / name
        (root / "shards").mkdir(parents=True)
        (root / "MANIFEST.jsonl").write_bytes(manifest)
        (root / "withdrawn.jsonl").write_bytes(b"")
    live, public = tmp_path / "live", tmp_path / "public"
    (live / "shards" / shard).write_bytes(raw)
    (public / "shards" / f"{shard}.{digest}.jsonl").write_bytes(raw)

    expected = load_reserve(live, with_catalog=True, allow_retired=True)
    actual = load_reserve(public, with_catalog=True, allow_retired=True)
    assert actual[0] == expected[0]
    assert actual[1].task_ids == expected[1].task_ids
    assert actual[1].task("omni:0") == expected[1].task("omni:0")
    assert candidate_metadata(public, *actual) == candidate_metadata(live, *expected)

    (public / "shards" / f"{shard}.{digest}.jsonl").unlink()
    with pytest.raises(ConfigError, match=f"reserve shard {shard} cannot be read"):
        load_reserve(public, with_catalog=True, allow_retired=True)
    with pytest.raises(VerifyError, match=f"reserve shard {shard} cannot be read"):
        candidate_metadata(public, *actual)
