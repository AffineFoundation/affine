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


def test_rehearsal_epoch_is_before_launch_and_paths_are_isolated(tmp_path):
    site, work = tmp_path / "prod-site", tmp_path / "prod-work"
    config = Config(site, work, tmp_path / "pool")
    scratch = tmp_path / "rehearsal"
    scratch.mkdir(mode=0o700)
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
