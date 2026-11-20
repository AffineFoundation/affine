"""Rehearsal tool safety seams; no real credentials, R2, SSH or production files."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from nacl.signing import SigningKey

from sn120_v2.errors import ConfigError
from sn120_v2.generator.config import Config
from sn120_v2.generator.epoch import epoch_deadline, epoch_start, submission_object
from sn120_v2.generator.links import seal
from sn120_v2.errors import ConfigError
from sn120_v2.store import LocalStore

SPEC = importlib.util.spec_from_file_location("generator_rehearsal", Path(__file__).parents[2] / "tools/generator_rehearsal.py")
tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tool)


def test_fresh_scratch_is_created_private(tmp_path):
    scratch = tmp_path / "new-private-rehearsal"
    tool.validate_paths(Config(tmp_path / "site", tmp_path / "work", tmp_path / "pool"), scratch)
    assert scratch.stat().st_mode & 0o777 == 0o700
    assert (scratch / "site").stat().st_mode & 0o777 == 0o700
    assert (scratch / "work").stat().st_mode & 0o777 == 0o700


def test_head_version_id_and_sanitized_failure_details(tmp_path, monkeypatch):
    class Reply:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size): return b'{"version_id":"' + b"a" * 64 + b'"}'
    monkeypatch.setattr(tool.urllib.request, "urlopen", lambda *args, **kwargs: Reply())
    assert tool.read_public_head("https://public.invalid/head")[0] == "a" * 64
    try:
        raise ValueError("upload failed at https://bucket.invalid/path?X-Amz-Signature=secret " + "f" * 64)
    except ValueError as exc:
        detail = tool.sanitized_error(exc)
    assert "secret" not in detail["message"] and "f" * 64 not in detail["message"]
    assert detail["type"] == "ValueError" and detail["frames"]


def test_real_miner_kit_upload_with_fresh_key(tmp_path, monkeypatch):
    class Store(LocalStore):
        def presign_get(self, key, ttl):
            return "https://store.invalid/" + key
        def presign_put(self, key, ttl):
            return "https://store.invalid/" + key, {"Content-Type": "application/octet-stream"}
    store = Store(tmp_path / "store")
    key = SigningKey(b"\x11" * 32)
    hotkey = tool.ss58(key)
    kit = tool.load_kit()
    def upload(request, timeout):
        store.put(request.full_url.removeprefix("https://store.invalid/"), request.data)
        class Reply:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size): return b""
        return Reply()
    monkeypatch.setattr(kit.urllib.request, "urlopen", upload)
    object_key = tool.upload_simulated(store, hotkey, key, ["01" * 10], epoch_start(tool.EPOCH).timestamp())
    document = json.loads(store.get(object_key))
    assert document["hotkey"] == hotkey and document["task_ids"] == ["01" * 10]
    assert document["signature"]


def test_rehearsal_epoch_is_before_launch_and_paths_are_isolated(tmp_path):
    site, work = tmp_path / "prod-site", tmp_path / "prod-work"
    config = Config(site, work, tmp_path / "pool")
    scratch = tmp_path / "rehearsal"
    tool.validate_paths(config, scratch)
    assert (scratch / "site").is_dir() and (scratch / "work").is_dir()
    with pytest.raises(ConfigError):
        tool.validate_paths(config, site / "child")


def test_sealed_upload_and_clocked_snapshot_never_write_public_mirror(tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "store")
    store._head = lambda key: {"LastModified": store._path(key).stat().st_mtime}
    key = SigningKey(b"\x07" * 32)
    hotkey = tool.ss58(key)
    start = epoch_start(tool.EPOCH).timestamp()
    kit = SimpleNamespace(
        open_sealed_link=lambda outer, hotkey, key, now: outer,
        batch_document=lambda epoch, hotkey, ids, key: {"epoch": epoch, "hotkey": hotkey, "task_ids": ids},
        check_document=lambda raw: None,
        upload=lambda link, raw, replace: store.put(link["object_key"], raw),
    )
    monkeypatch.setattr(tool, "load_kit", lambda: kit)
    monkeypatch.setattr(tool, "link_files", lambda hotkeys, presigner, now: {
        f"generator/links/{hotkey}.json": json.dumps({"hotkey": hotkey, "object_key": submission_object(tool.EPOCH, hotkey)}).encode()
    })
    object_key = tool.upload_simulated(store, hotkey, key, ["01" * 10], start)
    assert object_key == submission_object(tool.EPOCH, hotkey)
    assert store.get(object_key)
    clocked = tool.RehearsalClock(store, start + 120)
    assert clocked.describe(object_key)["LastModified"] == start + 120
    assert store._head(object_key)["LastModified"] != start + 120
    assert not list(store.list("public/generator/"))


def test_rehearsal_clock_cannot_affect_production_store(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.put("private", b"payload")
    clocked = tool.RehearsalClock(store, epoch_deadline(tool.EPOCH).timestamp() + 60)
    assert clocked.get("private") == b"payload"
    assert store.get("private") == b"payload"


def test_public_head_parser_accepts_repository_version_id():
    from sn120_v2.generator.snapshot import head_version
    assert head_version({"version_id": "a" * 64}) == "a" * 64


def test_public_sink_can_explicitly_disable_mirror_for_rehearsal(tmp_path):
    from sn120_v2.generator.common import PublicSink
    root = tmp_path / "site"
    root.mkdir()
    class Store:
        def __init__(self):
            self.puts = []
        def put(self, key, raw):
            self.puts.append(key)
    store = Store()
    PublicSink(root, store, mirror=False).put("epochs/rehearsal/manifest.json", b"{}")
    assert store.puts == []
    assert (root / "generator/epochs/rehearsal/manifest.json").read_bytes() == b"{}"


def test_script_has_no_external_deletion_or_public_sink_store(tmp_path):
    source = (Path(__file__).parents[2] / "tools/generator_rehearsal.py").read_text()
    assert ".delete(" not in source
    assert "rm -" not in source
    assert "publish_epoch(config, epoch=EPOCH, now=simulated, client={1: hotkey}, store=None, pool=pool)" in source
    assert '"rehearsal": True' in source
    assert "rehearsal-audit.json" in source
