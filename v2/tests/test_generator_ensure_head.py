"""Frozen-head preparation and dispatch guard tests; no network or GPU."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sn120_v2.errors import InfraError, VerifyError
from sn120_v2.generator.ensure_head import (
    ensure_head, installed_version, require_installed, restart_servers, select_base,
)
from test_generator_remote import EPOCH, make_remote

VERSION = "a" * 64
OTHER = "b" * 64
SECRET_URL = "https://downloads.invalid/base?X-Amz-Signature=not-logged"


def documents(version=VERSION, sha=None):
    content = b"verified bf16 fixture"
    sha = sha or hashlib.sha256(content).hexdigest()
    entry = {"files": [{"name": "model.bin", "sha256": sha, "size": len(content), "url": SECRET_URL}]}
    head = {"head": {"version_id": version}, "base_files": {"model.bin": sha}}
    links = {"head_version_id": version, "base": entry}
    return content, head, links


def test_current_head_download_hash_swap_and_restart(tmp_path):
    content, head, links = documents()
    base = tmp_path / "base"
    base.mkdir()
    (base / "old").write_text("old model")
    (base / ".head_version").write_text(OTHER)
    calls = []

    def fetch(entry, target):
        assert entry["model.bin"]["sha256"] == links["base"]["files"][0]["sha256"]
        (target / "model.bin").write_bytes(content)

    report = ensure_head(VERSION, base=base, read_record=lambda url: head if url.endswith("head.json") else links, fetch_files=fetch, restart=lambda: calls.append("restart"))
    assert report["status"] == "prepared" and calls == ["restart"]
    assert installed_version(base) == VERSION
    assert (base / "model.bin").read_bytes() == content and not (base / "old").exists()
    assert not (tmp_path / f"base.{VERSION}").exists()
    require_installed(VERSION, base)


def test_checkpoint_head_can_change_between_epochs():
    content, head, links = documents()
    _, old_head, old_links = documents(OTHER)
    links["head_version_id"] = OTHER
    links["base"] = None
    links["checkpoints"] = [{"version_id": VERSION, "base": old_links["base"]}]
    entry, hashes = select_base(VERSION, old_head, links)
    assert entry["model.bin"]["sha256"] == hashes["model.bin"]


def test_unknown_head_fails_before_download_or_restart(tmp_path):
    content, head, links = documents()
    calls = []
    with pytest.raises(VerifyError, match="neither"):
        ensure_head(OTHER, base=tmp_path / "base", read_record=lambda url: head if url.endswith("head.json") else links, fetch_files=lambda *args: calls.append("download"), restart=lambda: calls.append("restart"))
    assert not calls


def test_wrong_hash_leaves_installed_base_untouched(tmp_path):
    content, head, links = documents()
    base = tmp_path / "base"
    base.mkdir()
    (base / ".head_version").write_text(OTHER)
    (base / "existing.bin").write_text("keep")
    with pytest.raises(VerifyError):
        ensure_head(VERSION, base=base, read_record=lambda url: head if url.endswith("head.json") else links, fetch_files=lambda entry, target: (target / "model.bin").write_bytes(b"wrong"), restart=lambda: pytest.fail("must not restart"))
    assert installed_version(base) == OTHER and (base / "existing.bin").read_text() == "keep"


def test_scorer_refuses_missing_or_different_installed_head(tmp_path):
    with pytest.raises(VerifyError):
        require_installed(VERSION, tmp_path / "missing")
    base = tmp_path / "base"
    base.mkdir()
    (base / ".head_version").write_text(OTHER + "\n")
    with pytest.raises(VerifyError):
        require_installed(VERSION, base)


def test_restart_is_stop_then_start_with_long_timeout():
    calls = []
    def run(argv, **options):
        calls.append((argv, options))
        return SimpleNamespace(returncode=0)
    restart_servers(command=run)
    assert [call[0][-1] for call in calls] == ["stop", "start"]
    assert all(call[1]["timeout"] >= 900 for call in calls)
    assert all(call[1]["cwd"] == "/" for call in calls)


def test_health_python_is_isolated_from_package_directory():
    script = Path(__file__).parents[1] / "sn120_v2/generator/serve.sh"
    assert '"$VENV/bin/python" -I - "$ROOT" <<\'PY\'' in script.read_text()


def test_server_failure_is_loud():
    with pytest.raises(InfraError, match="eight-server"):
        restart_servers(command=lambda *args, **kwargs: SimpleNamespace(returncode=1))


def test_dispatch_prepares_exact_job_head_before_scorer(tmp_path):
    remote, fake, _ = make_remote(tmp_path)
    job = remote.work_dir / "jobs" / EPOCH / "job.json"
    job.write_text(json.dumps({"epoch": EPOCH, "head_version": VERSION, "tasks": []}))
    timeouts = []
    original = fake.ssh
    def ssh(command, *, timeout=None):
        timeouts.append(timeout)
        return original(command, timeout=timeout)
    fake.ssh = ssh
    remote.dispatch(EPOCH)
    commands = [command for kind, command in fake.calls if kind == "ssh"]
    assert len(commands) == 3
    assert "sn120_v2.generator.ensure_head " + VERSION in commands[1]
    assert timeouts[1] == 900
    assert "nohup" in commands[2]


def test_dispatch_head_failure_never_starts_scorer(tmp_path):
    remote, fake, _ = make_remote(tmp_path)
    (remote.work_dir / "jobs" / EPOCH / "job.json").write_text(json.dumps({"epoch": EPOCH, "head_version": VERSION, "tasks": []}))
    original = fake.ssh
    def ssh(command, *, timeout=None):
        if "ensure_head" in command:
            fake.calls.append(("ssh", command))
            raise InfraError("unknown head")
        return original(command, timeout=timeout)
    fake.ssh = ssh
    with pytest.raises(InfraError, match="unknown head"):
        remote.dispatch(EPOCH)
    assert not any("nohup" in command for kind, command in fake.calls if kind == "ssh")
    fake.ssh = original
    remote.dispatch(EPOCH)
    assert sum("ensure_head" in command for kind, command in fake.calls if kind == "ssh") == 2
    assert any("nohup" in command for kind, command in fake.calls if kind == "ssh")
