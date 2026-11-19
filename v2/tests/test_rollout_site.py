"""Live rollout keys and the hand-made public/rollouts directory coexist."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

from sn120_v2 import layout
from sn120_v2.store import LocalStore

sys.path.insert(0, str(Path(__file__).parents[1] / "site"))
import pull as site_pull


def test_layout_accepts_exact_live_keys_and_rejects_backfill_and_lookalikes():
    assert site_pull.layout_key(layout.public_rollout_bench("genesis-a", "math"))
    assert site_pull.layout_key(layout.public_rollout_bench_meta("genesis-a", "math"))
    assert site_pull.layout_key(layout.public_rollout_gate("r000001"))
    assert site_pull.layout_key(layout.public_rollout_gate("r000001", "winner"))
    assert site_pull.layout_key(layout.public_rollout_gate_meta("r000001"))
    for key in ("public/rollouts/index.json", "public/rollouts/backfill_index.json", "public/rollouts/SCHEMA.md",
                "public/rollouts/bench/model.per_task.jsonl.gz", "public/rollouts/gate/head_outcomes.r000001-r000068.jsonl.gz",
                "public/rollouts/gate/tasks.jsonl.gz", "public/rollouts/bench/m/math/deep.jsonl.gz",
                "public/rollouts/bench/m/math.jsonl", "public/rollouts/gate/r000001/loser.jsonl.gz",
                "public/rollouts/gate/other/meta.json"):
        assert not site_pull.layout_key(key), key


def test_pull_streams_verifies_adopts_index_and_leaves_backfill_files_alone(tmp_path):
    site = tmp_path / "site"
    backfill = {
        "public/rollouts/index.json": b'{"hand":"written"}',
        "public/rollouts/backfill_index.json": b'{"schema":"backfill","rounds":[1]}',
        "public/rollouts/SCHEMA.md": b"old schema",
        "public/rollouts/bench/model.per_task.jsonl.gz": b"old bench",
        "public/rollouts/gate/head_outcomes.r000001-r000068.jsonl.gz": b"old gate",
        "public/rollouts/gate/tasks.jsonl.gz": b"old tasks",
    }
    for key, data in backfill.items():
        path = site / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    store = LocalStore(tmp_path / "store")
    data = gzip.compress(b'{"model_id":"genesis-a","completion_text":"public"}\n', mtime=0)
    sha = hashlib.sha256(data).hexdigest()
    bench_key = layout.public_rollout_bench("genesis-a", "math")
    store.put(bench_key, data)
    meta = dict(schema="sn120-v2-rollouts-bench-1", model_id="genesis-a", benchmark_id="math", key=bench_key,
                rows=1, bytes=len(data), sha256=sha, bench_version="math/v1", finished_at="1970-01-01T00:00:00Z")
    store.put(layout.public_rollout_bench_meta("genesis-a", "math"), json.dumps(meta).encode())
    head = gzip.compress(b'{"role":"head","completion_text":"public"}\n', mtime=0)
    head_sha = hashlib.sha256(head).hexdigest()
    store.put(layout.public_rollout_gate("r000001"), head)
    file_meta = dict(role="head", key=layout.public_rollout_gate("r000001"), rows=1, bytes=len(head), sha256=head_sha)
    combined = hashlib.sha256(json.dumps({"head": head_sha}, sort_keys=True).encode()).hexdigest()
    gate_meta = dict(schema="sn120-v2-rollouts-gate-1", round_id="r000001", rows=1, bytes=len(head), sha256=combined,
                     params_hash="p", round_seed="s", closed_at="1970-01-01T00:00:00Z", files=[file_meta])
    store.put(layout.public_rollout_gate_meta("r000001"), json.dumps(gate_meta).encode())
    for key in ("public/rollouts/bench/model.per_task.jsonl.gz", "public/rollouts/gate/head_outcomes.r000001-r000068.jsonl.gz",
                "public/rollouts/gate/tasks.jsonl.gz", "public/rollouts/bench/m/math.jsonl"):
        store.put(key, b"must not copy")
    pulled = site_pull.Pull(store, site, 0)
    pulled.rollouts()
    index = json.loads((site / layout.ROLLOUT_INDEX).read_bytes())
    assert index["schema"] == "sn120-v2-rollouts-index-2" and index["backfill"] == {"schema": "backfill", "rounds": [1]}
    assert index["bench"] == [meta] and index["gate"][0]["round_id"] == "r000001"
    assert (site / bench_key).read_bytes() == data and (site / layout.public_rollout_gate("r000001")).read_bytes() == head
    assert not (site / "public/rollouts/bench/m/math.jsonl").exists()
    for key, old in backfill.items():
        if key != layout.ROLLOUT_INDEX:
            assert (site / key).read_bytes() == old
    owned = json.loads((site / site_pull.INDEX).read_bytes())["files"]
    assert layout.ROLLOUT_INDEX in owned and not set(backfill) - {layout.ROLLOUT_INDEX} & set(owned)


def test_rollout_size_and_retention_are_separate(tmp_path):
    store = LocalStore(tmp_path / "store")
    for number in (1, 2):
        rid = f"r{number:06d}"
        key = layout.public_rollout_gate(rid)
        data = gzip.compress(b"{}\n", mtime=0)
        store.put(key, data)
        file = dict(role="head", key=key, rows=1, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        meta = dict(schema="sn120-v2-rollouts-gate-1", round_id=rid, files=[file], rows=1, bytes=len(data))
        store.put(layout.public_rollout_gate_meta(rid), json.dumps(meta).encode())
    site = tmp_path / "site"
    site.mkdir()
    pulled = site_pull.Pull(store, site, 0, max_rollout_bytes=1, keep_rollout_rounds=1)
    pulled.rollouts()
    assert not (site / layout.public_rollout_gate("r000002")).exists()
    assert pulled.problems == []
    pulled.max_rollout_bytes = 1 << 20
    pulled.rollouts()
    assert (site / layout.public_rollout_gate("r000002")).is_file()
    assert not (site / layout.public_rollout_gate("r000001")).exists()
    assert (site / layout.public_rollout_gate_meta("r000002")).is_file()
