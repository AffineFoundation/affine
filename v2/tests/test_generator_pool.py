"""Pinned prior-pool adapter: raw hashes, original IDs, cache and production Task."""

from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import pytest

from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.math_pool import MANDATE, Task
from sn120_v2.errors import ConfigError, ContractError, VerifyError
from sn120_v2.generator import pool


def manifest(**changes):
    return {
        "name": "sn120-math-prior",
        "version": "v1",
        "rows": 811_609,
        "columns": list(pool.POOL_COLUMNS),
        "shards": [
            {"file": f"prior-{number:02d}-of-08.parquet", "rows": 101_452 if number == 0 else 101_451, "sha256": f"{number + 1:064x}"}
            for number in range(8)
        ],
    } | changes


@pytest.fixture
def valid_manifest():
    raw = canonical_json(manifest())
    # Tests validate the release schema while overriding the single content-pin check.
    original = pool.POOL_MANIFEST_SHA256
    pool.POOL_MANIFEST_SHA256 = sha256_hex(raw)
    try:
        yield raw
    finally:
        pool.POOL_MANIFEST_SHA256 = original


def test_release_url_name_version_rows_and_full_pin_are_default():
    assert pool.POOL_MANIFEST_URL == "https://github.com/AffineFoundation/affine/releases/download/sn120-math-prior-v1/MANIFEST.json"
    assert pool.POOL_ID == "sn120-math-prior-v1:45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115"
    assert pool.POOL_ROWS == 811_609
    parsed = pool.parse_manifest(pool.read_url(pool.POOL_MANIFEST_URL))
    assert parsed.pool == pool.POOL_ID
    assert parsed.rows == 811_609
    assert len(parsed.shards) == 8
    assert parsed.to_dict()["manifest_sha256"] == pool.POOL_MANIFEST_SHA256


def test_manifest_location_is_explicit_and_configurable(monkeypatch):
    assert pool.manifest_url() == pool.POOL_MANIFEST_URL
    assert pool.manifest_url("file:///tmp/MANIFEST.json") == "file:///tmp/MANIFEST.json"
    monkeypatch.setenv(pool.POOL_MANIFEST_ENV, "https://mirror.test/MANIFEST.json")
    assert pool.manifest_url() == "https://mirror.test/MANIFEST.json"
    for bad in ("", None, "relative/path", "s3://bucket/key", "file://remote/etc/passwd", "https:///missing"):
        if bad is None:
            monkeypatch.setenv(pool.POOL_MANIFEST_ENV, "")
        else:
            with pytest.raises(ConfigError):
                pool.manifest_url(bad)
    with pytest.raises(ConfigError, match=pool.POOL_MANIFEST_ENV):
        pool.manifest_url()


def test_small_manifest_read_is_bounded(tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_bytes(b"x" * (pool.MAX_POOL_MANIFEST_BYTES + 1))
    with pytest.raises(VerifyError, match="exceeds"):
        pool.read_url(path.as_uri())


def test_wrong_raw_manifest_bytes_fail_even_when_json_is_equivalent(valid_manifest):
    assert pool.parse_manifest(valid_manifest).pool == pool.POOL_ID
    with pytest.raises(VerifyError, match="raw SHA-256"):
        pool.parse_manifest(valid_manifest + b"\n")
    with pytest.raises(VerifyError, match="raw SHA-256"):
        pool.parse_manifest(json.dumps(json.loads(valid_manifest), indent=2).encode())


@pytest.mark.parametrize(
    "change",
    [
        {"name": "other-pool"},
        {"version": "v2"},
        {"rows": 1},
        {"columns": ["id"]},
        {"shards": []},
        {"shards": [{"file": "x", "rows": 1, "sha256": "0" * 64}]},
    ],
)
def test_wrong_manifest_schema_is_rejected(valid_manifest, change):
    data = json.loads(valid_manifest) | change
    raw = canonical_json(data)
    pool.POOL_MANIFEST_SHA256 = sha256_hex(raw)
    with pytest.raises(VerifyError):
        pool.parse_manifest(raw)


def test_shard_rows_are_strict_ints_not_booleans(valid_manifest):
    data = json.loads(valid_manifest)
    data["shards"][0]["rows"] = True
    raw = canonical_json(data)
    pool.POOL_MANIFEST_SHA256 = sha256_hex(raw)
    with pytest.raises(VerifyError):
        pool.parse_manifest(raw)


def _tiny_pool(tmp_path, valid_manifest, *, duplicate=False):
    parsed = pool.parse_manifest(valid_manifest)
    shard = parsed.shards[0]
    raw = b"first-parquet-bytes"
    path = tmp_path / "shards" / shard.file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    parsed = pool.PoolManifest(parsed.pool, parsed.manifest_url, parsed.manifest_sha256, (pool.PoolShard(shard.file, 2, sha256_hex(raw)),))
    (tmp_path / "MANIFEST.json").write_bytes(valid_manifest)
    rows = [
        dict.fromkeys(pool.POOL_COLUMNS, ""),
        dict.fromkeys(pool.POOL_COLUMNS, ""),
    ]
    for number, row in enumerate(rows):
        row.update(id=f"{number + 1:020x}", problem=f"What is {number + 1}?", answer=str(number + 1))
    if duplicate:
        rows[1]["id"] = rows[0]["id"]
    return parsed, rows


def test_verified_registry_preserves_id_and_uses_production_prompt(tmp_path, valid_manifest, monkeypatch):
    parsed, rows = _tiny_pool(tmp_path, valid_manifest)
    monkeypatch.setattr(pool, "POOL_ROWS", 2)
    seen = []

    def reader(path, shard):
        seen.append((path.name, shard.sha256))
        yield from enumerate(rows)

    registry = pool.PriorPool(tmp_path, manifest=parsed, reader=reader, build_index=True)
    task_id = rows[0]["id"]
    assert registry.require_ids([task_id]) == (task_id,)
    assert task_id in registry
    assert registry.row(task_id) == rows[0]
    task = registry.tasks([task_id])[0]
    assert isinstance(task, Task)
    assert task.task_id == task_id
    assert task.task_id == rows[0]["id"] and not task.task_id.startswith("mp:")
    assert registry.messages(task_id) == [
        {"role": "system", "content": MANDATE},
        {"role": "user", "content": "What is 1?"},
    ]
    assert task.messages() == registry.messages(task_id)
    # Cache reload uses only the original IDs and shard/row index.
    reloaded = pool.PriorPool(tmp_path, manifest=parsed, reader=reader)
    assert reloaded.row(task_id) == rows[0]
    assert len(seen) >= 2


def test_id_index_is_canonical_pinned_and_not_a_new_task_dataset(tmp_path, valid_manifest, monkeypatch):
    parsed, rows = _tiny_pool(tmp_path, valid_manifest)
    monkeypatch.setattr(pool, "POOL_ROWS", 2)
    registry = pool.PriorPool(tmp_path, manifest=parsed, reader=lambda path, shard: enumerate(rows), build_index=True)
    lines = registry.index_path.read_bytes().splitlines()
    assert all(canonical_json(json.loads(line)) == line for line in lines)
    assert json.loads(lines[0]) == {"id": rows[0]["id"], "shard": 0, "row": 0}
    assert json.loads(lines[-1]) == {"index_version": "affine-generator-pool-index-v1", "pool": pool.POOL_ID, "rows": 2}
    assert not any("problem" in json.loads(line) for line in lines)
    registry.index_path.write_bytes(b"\n".join(lines[:-1]) + b"\n")
    with pytest.raises(VerifyError):
        pool.PriorPool(tmp_path, manifest=parsed, reader=lambda path, shard: enumerate(rows))


def test_shard_sha_is_verified_before_reader_is_called(tmp_path, valid_manifest):
    parsed, rows = _tiny_pool(tmp_path, valid_manifest)
    next(tmp_path.glob("shards/*")).write_bytes(b"tampered")
    called = False

    def reader(path, shard):
        nonlocal called
        called = True
        return enumerate(rows)

    with pytest.raises(VerifyError, match="SHA-256"):
        pool.PriorPool(tmp_path, manifest=parsed, reader=reader)
    assert not called


def test_missing_shard_does_not_fall_back_to_network(tmp_path, valid_manifest):
    parsed, rows = _tiny_pool(tmp_path, valid_manifest)
    next(tmp_path.glob("shards/*")).unlink()
    with pytest.raises(VerifyError, match="missing"):
        pool.PriorPool(tmp_path, manifest=parsed, reader=lambda path, shard: rows)


def test_duplicate_or_unknown_id_never_gets_minted_or_deduplicated(tmp_path, valid_manifest, monkeypatch):
    parsed, rows = _tiny_pool(tmp_path, valid_manifest, duplicate=True)
    monkeypatch.setattr(pool, "POOL_ROWS", 2)
    with pytest.raises(VerifyError, match="duplicate"):
        pool.PriorPool(tmp_path, manifest=parsed, reader=lambda path, shard: enumerate(rows), build_index=True)
    assert not (tmp_path / "id-index.jsonl").exists()
    (tmp_path / "id-index.jsonl").unlink(missing_ok=True)
    monkeypatch.undo()
    parsed, rows = _tiny_pool(tmp_path, valid_manifest)
    monkeypatch.setattr(pool, "POOL_ROWS", 2)
    registry = pool.PriorPool(tmp_path, manifest=parsed, reader=lambda path, shard: enumerate(rows), build_index=True)
    with pytest.raises(ContractError):
        registry.require_ids([rows[0]["id"], rows[0]["id"]])
    with pytest.raises(ContractError):
        registry.require_ids(["mp:123"])
    with pytest.raises(ContractError):
        registry.row("f" * 20)


def test_index_must_resolve_to_original_row(tmp_path, valid_manifest, monkeypatch):
    parsed, rows = _tiny_pool(tmp_path, valid_manifest)
    monkeypatch.setattr(pool, "POOL_ROWS", 2)
    registry = pool.PriorPool(tmp_path, manifest=parsed, reader=lambda path, shard: enumerate(rows), build_index=True)
    changed = copy.deepcopy(rows)
    changed[0]["id"] = "f" * 20
    registry.reader = lambda path, shard: enumerate(changed)
    with pytest.raises(VerifyError, match="does not resolve"):
        registry.row(rows[0]["id"])


def test_grader_delegates_to_v2_math_grader(tmp_path, valid_manifest, monkeypatch):
    parsed, rows = _tiny_pool(tmp_path, valid_manifest)
    monkeypatch.setattr(pool, "POOL_ROWS", 2)
    registry = pool.PriorPool(tmp_path, manifest=parsed, reader=lambda path, shard: enumerate(rows), build_index=True)
    module = types.ModuleType("sn120_v2.envs.math_grader")
    called = []

    def grade(task, output, capped):
        called.append((task, output, capped))
        return "production-grade"

    module.grade = grade
    monkeypatch.setitem(sys.modules, "sn120_v2.envs.math_grader", module)
    assert registry.grade(rows[0]["id"], "output", False) == "production-grade"
    assert called[0][0].messages() == registry.messages(rows[0]["id"])
    assert called[0][1:] == ("output", False)


def test_pyarrow_reader_checks_schema_and_count(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    shard = pool.PoolShard("prior-00-of-08.parquet", 1, "0" * 64)
    path = tmp_path / shard.file
    row = dict.fromkeys(pool.POOL_COLUMNS, "")
    pq.write_table(pa.Table.from_pylist([row], schema=pa.schema([(name, pa.string()) for name in pool.POOL_COLUMNS])), path)
    assert list(pool._pyarrow_rows(path, shard)) == [(0, row)]
    with pytest.raises(VerifyError, match="schema or row count"):
        list(pool._pyarrow_rows(path, pool.PoolShard(shard.file, 2, shard.sha256)))


def test_cache_path_must_be_explicit(monkeypatch):
    monkeypatch.delenv(pool.POOL_CACHE_ENV, raising=False)
    with pytest.raises(ConfigError, match=pool.POOL_CACHE_ENV):
        pool.cache_directory()
