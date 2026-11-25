"""Restore public reserve caches without discarding a packaged extension."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.math_pool import load_pool
from sn120_v2.envs.reserve import empty_manifest
from sn120_v2.errors import VerifyError
from sn120_v2.params import Params
from sn120_v2.pool_runtime import PoolRuntime
from sn120_v2.state import State
from sn120_v2.store import LocalStore


def reserve_root(tmp_path: Path) -> Path:
    root = tmp_path / "reserve"
    (root / "shards").mkdir(parents=True)
    (root / "MANIFEST.jsonl").write_bytes(empty_manifest())
    (root / "withdrawn.jsonl").write_bytes(b"")
    return root


def add_shard(root: Path, number: int, prefix: str) -> None:
    name = f"{number:04d}-test.jsonl"
    row = {"id": f"{prefix}:{number}", "env": "affine_math", "problem": f"restore test problem {prefix} {number}",
           "answer": str(number), "source": f"test@rev:{number}", "license": "CC0-1.0", "added_at": "2026-11-18",
           "audit": {"verdict": "keep", "method": "unit-test", "ref": "test_pool_runtime_restore"}}
    (root / "shards" / name).write_bytes(canonical_json(row) + b"\n")
    manifest = [json.loads(line) for line in (root / "MANIFEST.jsonl").read_bytes().splitlines()]
    manifest[-1:-1] = [{"shard": name, "sha256": sha256_hex((root / "shards" / name).read_bytes()),
                        "count": 1, "added_at": "2026-11-18"}]
    (root / "MANIFEST.jsonl").write_bytes(b"".join(canonical_json(row) + b"\n" for row in manifest))


def withdraw(root: Path, task_id: str) -> None:
    with (root / "withdrawn.jsonl").open("ab") as stream:
        stream.write(canonical_json({"id": task_id, "reason": "restore test", "added_at": "2026-11-18"}) + b"\n")
    manifest = [json.loads(line) for line in (root / "MANIFEST.jsonl").read_bytes().splitlines()]
    withdrawn = (root / "withdrawn.jsonl").read_bytes()
    manifest[-1] = {"withdrawn": {"sha256": sha256_hex(withdrawn), "count": len(withdrawn.splitlines())}}
    (root / "MANIFEST.jsonl").write_bytes(b"".join(canonical_json(row) + b"\n" for row in manifest))


def runtime(store: LocalStore, root: Path) -> PoolRuntime:
    params = Params(rotation_enabled=True)
    return PoolRuntime(store, params, State(params=params), root=root, base=load_pool(), initial_ids=("math:0", "math:1"))


def restore(store: LocalStore, root: Path) -> None:
    runtime(store, root)


def test_local_packaged_extension_and_withdrawals_are_kept(tmp_path):
    root = reserve_root(tmp_path)
    add_shard(root, 1, "published")
    add_shard(root, 2, "packaged")
    withdraw(root, "published:1")
    withdraw(root, "packaged:2")
    store = LocalStore(tmp_path / "store")
    manifest = [json.loads(line) for line in (root / "MANIFEST.jsonl").read_bytes().splitlines()]
    store.put(layout.RESERVE_MANIFEST, canonical_json(manifest[0]) + b"\n" + canonical_json(manifest[-1]) + b"\n")
    published_withdrawal = (root / "withdrawn.jsonl").read_bytes().splitlines(keepends=True)[0]
    store.put(layout.RESERVE_WITHDRAWN, published_withdrawal)
    store.put(layout.reserve_shard("0001-test.jsonl", manifest[0]["sha256"]), (root / "shards" / "0001-test.jsonl").read_bytes())
    before = ((root / "MANIFEST.jsonl").read_bytes(), (root / "withdrawn.jsonl").read_bytes())

    pool = runtime(store, root)

    assert ((root / "MANIFEST.jsonl").read_bytes(), (root / "withdrawn.jsonl").read_bytes()) == before
    assert pool.reserve.manifest_sha == sha256_hex(before[0])
    assert not pool.reserve.queue


def test_equal_manifest_is_a_byte_for_byte_no_op(tmp_path):
    root = reserve_root(tmp_path)
    add_shard(root, 1, "published")
    store = LocalStore(tmp_path / "store")
    store.put(layout.RESERVE_MANIFEST, (root / "MANIFEST.jsonl").read_bytes())
    store.put(layout.RESERVE_WITHDRAWN, b"")
    store.put(layout.reserve_shard("0001-test.jsonl", sha256_hex((root / "shards" / "0001-test.jsonl").read_bytes())),
              (root / "shards" / "0001-test.jsonl").read_bytes())
    before = ((root / "MANIFEST.jsonl").read_bytes(), (root / "withdrawn.jsonl").read_bytes())

    assert restore(store, root) is None
    assert ((root / "MANIFEST.jsonl").read_bytes(), (root / "withdrawn.jsonl").read_bytes()) == before


@pytest.mark.parametrize("empty", [True, False])
def test_cache_miss_is_filled_from_public(tmp_path, empty):
    root = reserve_root(tmp_path)
    add_shard(root, 1, "published")
    store = LocalStore(tmp_path / "store")
    store.put(layout.RESERVE_MANIFEST, (root / "MANIFEST.jsonl").read_bytes())
    store.put(layout.RESERVE_WITHDRAWN, b"")
    store.put(layout.reserve_shard("0001-test.jsonl", sha256_hex((root / "shards" / "0001-test.jsonl").read_bytes())),
              (root / "shards" / "0001-test.jsonl").read_bytes())
    if empty:
        (root / "MANIFEST.jsonl").write_bytes(empty_manifest())
    else:
        (root / "withdrawn.jsonl").write_bytes(canonical_json({"id": "published:1", "reason": "old cache", "added_at": "2026-11-18"}) + b"\n")
        raw = (root / "withdrawn.jsonl").read_bytes()
        (root / "MANIFEST.jsonl").write_bytes(canonical_json({"withdrawn": {"sha256": sha256_hex(raw), "count": 1}}) + b"\n")
    for shard in (root / "shards").iterdir():
        shard.unlink()

    pool = runtime(store, root)
    assert (root / "MANIFEST.jsonl").read_bytes() == store.get(layout.RESERVE_MANIFEST)
    assert (root / "withdrawn.jsonl").read_bytes() == b""
    assert pool.reserve.queue == ("published:1",)


def test_older_strict_prefix_is_restored_from_public(tmp_path):
    root = reserve_root(tmp_path)
    add_shard(root, 1, "published")
    add_shard(root, 2, "later")
    withdraw(root, "published:1")
    withdraw(root, "later:2")
    store = LocalStore(tmp_path / "store")
    store.put(layout.RESERVE_MANIFEST, (root / "MANIFEST.jsonl").read_bytes())
    store.put(layout.RESERVE_WITHDRAWN, (root / "withdrawn.jsonl").read_bytes())
    for number in (1, 2):
        shard = root / "shards" / f"{number:04d}-test.jsonl"
        store.put(layout.reserve_shard(shard.name, sha256_hex(shard.read_bytes())), shard.read_bytes())
    manifest = [json.loads(line) for line in (root / "MANIFEST.jsonl").read_bytes().splitlines()]
    (root / "MANIFEST.jsonl").write_bytes(canonical_json(manifest[0]) + b"\n" + canonical_json(manifest[-1]) + b"\n")
    (root / "withdrawn.jsonl").write_bytes((root / "withdrawn.jsonl").read_bytes().splitlines(keepends=True)[0])

    pool = runtime(store, root)
    assert (root / "MANIFEST.jsonl").read_bytes() == store.get(layout.RESERVE_MANIFEST)
    assert (root / "withdrawn.jsonl").read_bytes() == store.get(layout.RESERVE_WITHDRAWN)
    assert not pool.reserve.queue


def test_diverging_local_manifest_is_rejected_without_overwrite(tmp_path):
    root = reserve_root(tmp_path)
    add_shard(root, 1, "divergent")
    store = LocalStore(tmp_path / "store")
    published = reserve_root(tmp_path / "published")
    add_shard(published, 1, "published")
    store.put(layout.RESERVE_MANIFEST, (published / "MANIFEST.jsonl").read_bytes())
    store.put(layout.RESERVE_WITHDRAWN, b"")
    before = (root / "MANIFEST.jsonl").read_bytes()
    with pytest.raises(VerifyError, match="does not extend"):
        runtime(store, root)
    assert (root / "MANIFEST.jsonl").read_bytes() == before


def test_local_withdrawn_shrink_is_rejected_without_overwrite(tmp_path):
    root = reserve_root(tmp_path)
    add_shard(root, 1, "published")
    add_shard(root, 2, "packaged")
    withdraw(root, "published:1")
    store = LocalStore(tmp_path / "store")
    store.put(layout.RESERVE_MANIFEST, (root / "MANIFEST.jsonl").read_bytes())
    store.put(layout.RESERVE_WITHDRAWN, (root / "withdrawn.jsonl").read_bytes())
    store.put(layout.reserve_shard("0001-test.jsonl", sha256_hex((root / "shards" / "0001-test.jsonl").read_bytes())),
              (root / "shards" / "0001-test.jsonl").read_bytes())
    (root / "withdrawn.jsonl").write_bytes(b"")
    manifest = [json.loads(line) for line in (root / "MANIFEST.jsonl").read_bytes().splitlines()]
    manifest[-1] = {"withdrawn": {"sha256": sha256_hex(b""), "count": 0}}
    (root / "MANIFEST.jsonl").write_bytes(b"".join(canonical_json(row) + b"\n" for row in manifest))
    before = ((root / "MANIFEST.jsonl").read_bytes(), (root / "withdrawn.jsonl").read_bytes())
    with pytest.raises(VerifyError, match="withdrawals do not extend"):
        runtime(store, root)
    assert ((root / "MANIFEST.jsonl").read_bytes(), (root / "withdrawn.jsonl").read_bytes()) == before
