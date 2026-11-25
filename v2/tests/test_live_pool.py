"""Live official-active pool discovery, verification and catalog resolution. No network in tests."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.math_pool import load_pool
from sn120_v2.errors import ConfigError

SPEC = importlib.util.spec_from_file_location("live_pool_tool", Path(__file__).parents[1] / "miner" / "live_pool.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def public_files(tmp_path: Path) -> dict[str, bytes]:
    base = load_pool()
    chosen = base.task(base.task_ids[0])
    retired = [{"id": "gen:9", "env": "affine_math", "problem": "old", "answer": "0", "reason": "high", "retired_at": 1}]
    tasks = [{"id": "gen:9", "env": "affine_math", "problem": "old", "answer": "0"}]
    shard_body = dict(id="gen:1", env="affine_math", problem="A synthetic public reserve problem", answer="2",
                      source="generator", license="apache-2-0", added_at="2026-11-25", prior_p=0.5,
                      audit={"verdict": "keep", "method": "public", "ref": "test"})
    shard = canonical_json(shard_body) + b"\n"
    withdrawn = b""
    manifest = (canonical_json({"shard": "0001-public.jsonl", "sha256": sha256_hex(shard), "count": 1, "added_at": "2026-11-25"})
                + b"\n" + canonical_json({"withdrawn": {"sha256": sha256_hex(withdrawn), "count": 0}}) + b"\n")
    active = canonical_json(sorted([chosen.task_id, "gen:1"]))
    active_sha = sha256_hex(active)
    rotation = {"round_id": "r000123", "active_sha": active_sha, "reserve_manifest_sha": sha256_hex(manifest),
                "rule_version": "affine-pool-rotation-v2"}
    pool = {**rotation, "active": layout.active_pool(active_sha), "size": 2}
    return {
        layout.MINING_JSON: canonical_json({"round": {"pool": pool}, "params": {"rotation_rule": "info"}}),
        layout.POOL_SUMMARY: canonical_json({"active_sha": "0" * 64}),
        layout.active_pool(active_sha): active,
        layout.RESERVE_MANIFEST: manifest,
        layout.RESERVE_WITHDRAWN: withdrawn,
        layout.reserve_shard("0001-public.jsonl", sha256_hex(shard)): shard,
        layout.RETIRED_POOL: canonical_json(retired),
        layout.RETIRED_TASKS: canonical_json(tasks),
        layout.pool_rotation("r000123"): canonical_json(rotation),
    }


def mocked_public(monkeypatch, documents):
    requests = []

    def read(public_url, key, *, attempts=3):
        requests.append((key, attempts))
        if key not in documents:
            raise urllib.error.URLError("missing")
        return documents[key]

    monkeypatch.setattr(tool, "read_bytes", read)
    return requests


def test_mining_round_is_authoritative_and_active_sha_is_verified(monkeypatch, tmp_path):
    files = public_files(tmp_path)
    requests = mocked_public(monkeypatch, files)
    expected = json.loads(files[next(key for key in files if key.startswith("public/pool/active/"))])
    assert tool.fetch_active_ids("https://files.example") == tuple(expected)
    assert requests[0][0] == layout.MINING_JSON and layout.POOL_SUMMARY not in dict(requests)


def test_summary_is_only_a_fallback_and_sha_failure_is_clear(monkeypatch, tmp_path):
    files = public_files(tmp_path)
    mining = json.loads(files[layout.MINING_JSON])
    mining["round"] = {}
    files[layout.MINING_JSON] = canonical_json(mining)
    active_key = next(key for key in files if key.startswith("public/pool/active/"))
    files[layout.POOL_SUMMARY] = canonical_json({"active_sha": Path(active_key).stem, "round_id": "r000123"})
    requests = mocked_public(monkeypatch, files)
    assert len(tool.fetch_active_ids("https://files.example")) == 2
    assert requests[1][0] == layout.POOL_SUMMARY
    files[active_key] += b"\n"
    with pytest.raises(tool.PoolError, match="SHA256 mismatch"):
        tool.fetch_active_ids("https://files.example")


def test_active_tasks_resolve_base_and_public_reserve_bodies(monkeypatch, tmp_path):
    files = public_files(tmp_path)
    requests = mocked_public(monkeypatch, files)
    rows = tool.active_tasks("https://files.example", tmp_path / "cache")
    assert [row["task_id"] for row in rows] == sorted(row["task_id"] for row in rows)
    reserve = next(row for row in rows if row["task_id"] == "gen:1")
    assert reserve["problem"] == "A synthetic public reserve problem" and reserve["answer"] == "2"
    assert reserve["prompt"].endswith("\n\n" + reserve["problem"])
    assert any(key.startswith("public/pool/reserve/shards/") for key, _ in requests)
    before = len(requests)
    assert tool.active_tasks("https://files.example", tmp_path / "cache") == rows
    assert sum(key.startswith("public/pool/reserve/shards/") for key, _ in requests[before:]) == 0


def test_wrong_reserve_manifest_and_corrupt_cached_shard_are_refused(monkeypatch, tmp_path):
    files = public_files(tmp_path)
    mocked_public(monkeypatch, files)
    manifest = files[layout.RESERVE_MANIFEST]
    files[layout.RESERVE_MANIFEST] = b'{"withdrawn":{"sha256":"0","count":0}}\n'
    with pytest.raises(tool.PoolError, match="reserve_manifest_sha"):
        tool.active_tasks("https://files.example", tmp_path / "cache")
    files[layout.RESERVE_MANIFEST] = manifest
    rotation = json.loads(files[layout.pool_rotation("r000123")])
    rotation["reserve_manifest_sha"] = sha256_hex(manifest)
    files[layout.pool_rotation("r000123")] = canonical_json(rotation)
    mining = json.loads(files[layout.MINING_JSON])
    mining["round"]["pool"]["reserve_manifest_sha"] = rotation["reserve_manifest_sha"]
    files[layout.MINING_JSON] = canonical_json(mining)
    rows = tool.active_tasks("https://files.example", tmp_path / "cache")
    assert len(rows) == 2
    shard = next(path for path in (tmp_path / "cache" / "shards").iterdir())
    shard.write_bytes(b"corrupt")
    assert len(tool.active_tasks("https://files.example", tmp_path / "cache")) == 2


def test_every_http_request_has_a_client_no_cache_and_retries(monkeypatch):
    calls = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b"{}"

    def open_url(request, timeout):
        calls.append(request)
        if len(calls) < 3:
            raise urllib.error.URLError("temporary")
        return Reply()

    monkeypatch.setattr(tool.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(tool.time, "sleep", lambda _: None)
    assert tool.fetch_bytes("https://files.example", layout.MINING_JSON) == b"{}"
    assert len(calls) == 3
    for request in calls:
        assert request.get_header("User-agent") == "sn120-v2-live-pool/1"
        assert request.get_header("Cache-control") == "no-cache" and "_=" in request.full_url


def test_cli_writes_official_jsonl_and_reports_failures(monkeypatch, tmp_path, capsys):
    files = public_files(tmp_path)
    mining = json.loads(files[layout.MINING_JSON])
    rotation = json.loads(files[layout.pool_rotation("r000123")])
    mining["round"]["pool"]["reserve_manifest_sha"] = rotation["reserve_manifest_sha"]
    files[layout.MINING_JSON] = canonical_json(mining)
    mocked_public(monkeypatch, files)
    out = tmp_path / "tasks.jsonl"
    assert tool.main(["--public-url", "https://files.example", "--cache", str(tmp_path / "cache"), "--out", str(out)]) == 0
    assert len(out.read_text().splitlines()) == 2 and "live official-active pool" in capsys.readouterr().out
    monkeypatch.setattr(tool, "active_tasks", lambda *args: (_ for _ in ()).throw(ConfigError("unreadable")))
    assert tool.main(["--out", str(out)]) == 1 and "live pool failed" in capsys.readouterr().err
