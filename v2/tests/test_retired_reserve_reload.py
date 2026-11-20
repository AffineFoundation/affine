"""Ranked reserve reload preserves retired original-shard history."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.reserve import load_reserve
from sn120_v2.errors import ConfigError
from sn120_v2.params import Params
from sn120_v2.pool_runtime import PoolRuntime
from sn120_v2.rotation import Rotation, active_sha
from sn120_v2.state import State
from sn120_v2.store import LocalStore


def reserve_fixture(root: Path, *, retired_reason="displaced", task_id="omni:0") -> dict:
    (root / "shards").mkdir(parents=True)
    row = {"id": task_id, "env": "affine_math", "problem": f"UNIQUE-DISPLACED-RETURN {task_id}",
           "answer": "1", "source": "test@rev:0", "license": "CC0-1.0", "added_at": "2026-11-20",
           "audit": {"verdict": "keep", "method": "test", "ref": "test"}}
    raw = canonical_json(row) + b"\n"
    shard = "0001-test.jsonl"
    (root / "shards" / shard).write_bytes(raw)
    (root / "MANIFEST.jsonl").write_bytes(canonical_json({"shard": shard, "sha256": sha256_hex(raw), "count": 1, "added_at": "2026-11-20"}) + b"\n" +
                                         canonical_json({"withdrawn": {"sha256": sha256_hex(b""), "count": 0}}) + b"\n")
    (root / "withdrawn.jsonl").write_bytes(b"")
    retired = {"id": task_id, "round_id": "r000070", "reason": retired_reason, "event_seq": 0}
    task = {"id": task_id, "env": row["env"], "problem": row["problem"], "answer": row["answer"]}
    (root / "retired.json").write_bytes(canonical_json([retired]))
    (root / "retired_tasks.json").write_bytes(canonical_json([task]))
    return row


def test_ranked_reload_keeps_retired_shard_history_without_duplicate_admission(tmp_path):
    root = tmp_path / "reserve"
    row = reserve_fixture(root)

    reserve, catalog = load_reserve(root, with_catalog=True, allow_retired=True)
    assert reserve.queue == ("omni:0",)  # original shard history remains replay evidence
    assert catalog.task("omni:0").problem == row["problem"]
    assert catalog.task_ids.count("omni:0") == 1
    assert len(catalog.task_ids) == len(set(catalog.task_ids))

    params = Params(rotation_enabled=True, rotation_rule="info")
    store = LocalStore(tmp_path / "store")
    store.put(layout.RESERVE_MANIFEST, (root / "MANIFEST.jsonl").read_bytes())
    store.put(layout.RESERVE_WITHDRAWN, b"")
    store.put(layout.RETIRED_POOL, (root / "retired.json").read_bytes())
    store.put(layout.RETIRED_TASKS, (root / "retired_tasks.json").read_bytes())
    store.put(layout.reserve_shard("0001-test.jsonl", sha256_hex((root / "shards" / "0001-test.jsonl").read_bytes())),
              (root / "shards" / "0001-test.jsonl").read_bytes())
    archive = json.loads((root / "retired.json").read_bytes())[0]
    previous = Rotation(round_id="r000001", active_ids=("omni:0",), active_sha=active_sha(("omni:0",)),
                        prev_active_sha="", admitted=(), retired=(), reserve_manifest_sha="a" * 64,
                        queue_remaining=0, next_up=(), reserve_history={}, ledger={"omni:0": (0, 0)},
                        retired_archive=({**archive, "round_id": "r000000"},),
                        rule_version="affine-pool-rotation-v2", candidate_ranking=({},))
    store.put(layout.pool_rotation_state(previous.digest()), previous.to_json())
    pool = PoolRuntime(store, params, State(params=params, rotation_state_sha=previous.digest()), root=root)
    assert pool.reserve.queue == ("omni:0",)
    assert pool._rule_reserve().queue == ()


@pytest.mark.parametrize("reason", ["high", "low", "excess"])
def test_other_retirement_reasons_also_keep_original_history(tmp_path, reason):
    root = tmp_path / "reserve"
    reserve_fixture(root, retired_reason=reason)
    reserve, catalog = load_reserve(root, with_catalog=True, allow_retired=True)
    assert reserve.queue == ("omni:0",)
    assert catalog.task("omni:0").task_id == "omni:0"


def test_off_mode_dedupe_still_rejects_retired_original_shard(tmp_path):
    root = tmp_path / "reserve"
    reserve_fixture(root, retired_reason="high")
    shard = root / "shards" / "0001-test.jsonl"
    row = json.loads(shard.read_bytes())
    row["id"] = "omni:1"
    raw = canonical_json(row) + b"\n"
    shard.write_bytes(raw)
    second = root / "shards" / "0002-test.jsonl"
    second.write_bytes(raw)
    manifest = [json.loads(line) for line in (root / "MANIFEST.jsonl").read_bytes().splitlines()]
    manifest[0]["sha256"] = sha256_hex(raw)
    manifest[-1:-1] = [{"shard": second.name, "sha256": sha256_hex(raw), "count": 1, "added_at": "2026-11-20"}]
    (root / "MANIFEST.jsonl").write_bytes(b"".join(canonical_json(row) + b"\n" for row in manifest))
    with pytest.raises(ConfigError, match="repeats a task id"):
        load_reserve(root, with_catalog=True)


def test_retired_shard_still_requires_canonical_bytes(tmp_path):
    root = tmp_path / "reserve"
    reserve_fixture(root)
    (root / "shards" / "0001-test.jsonl").write_bytes(b"{}")
    with pytest.raises(ConfigError, match="does not match its manifest sha256"):
        load_reserve(root, with_catalog=True, allow_retired=True)
