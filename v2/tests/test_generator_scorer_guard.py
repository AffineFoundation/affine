"""Scorer command guard, tested against the sibling scorer runner module."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sn120_v2.errors import VerifyError
from sn120_v2.generator import scorer


def test_scorer_command_refuses_wrong_head_before_delegating(tmp_path, monkeypatch):
    source = Path("/home/bb/.bb/plugins/environment-git-worktree/host-data/worktrees/thr_gk2fjxb7ny-1/affine/v2/sn120_v2/generator/scoring.py")
    if not source.is_file():
        pytest.skip("sibling scorer source not present")
    spec = importlib.util.spec_from_file_location("sn120_v2.generator.runner", source)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    runner.main = lambda args: called.append(args) or 0
    monkeypatch.setattr("sn120_v2.generator.ensure_head.BASE", tmp_path / "base")
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "version": runner.JOB_VERSION, "epoch": "20261120T0000Z", "tasks": [{"id": "01" * 10, "hotkey": "hk"}],
        "head_version": "a" * 64, "seed_secret": "test-only", "budget_deadline": 2_000_000_000, "pool_pin": runner.POOL_ID,
    }))
    called = []
    with pytest.raises(VerifyError):
        scorer.main(["run", str(job)])
    assert not called
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / ".head_version").write_text("a" * 64 + "\n")
    assert scorer.main(["run", str(job)]) == 0
    assert called == [["run", str(job)]]
