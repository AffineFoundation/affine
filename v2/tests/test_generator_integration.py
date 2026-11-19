"""Control driver and sibling-scorer contract tests, entirely with fakes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sn120_v2.canonical import sha256_hex
from sn120_v2.errors import ConfigError
from sn120_v2.generator.config import Config
from sn120_v2.generator.control import tick, write_seed
from sn120_v2.generator.decisions import decision
from sn120_v2.generator.epoch import epoch_deadline, epoch_start
from sn120_v2.generator.publish import publication
from sn120_v2.generator.snapshot import scorer_job

EPOCH = "20261120T0000Z"
DEADLINE = epoch_deadline(EPOCH).timestamp()
SECRET = "private test seed"
COMMITMENT = sha256_hex(SECRET.encode())
POOL = "sn120-math-prior-v1:45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115"


class Pool:
    def row(self, task_id):
        return {"problem": "2 + 2?", "answer": "4", "source_id": "prior-1"}


def test_seed_is_created_once_and_commitment_is_stable(tmp_path):
    first = write_seed(tmp_path, EPOCH)
    second = write_seed(tmp_path, EPOCH)
    assert first == second and first["seed_commitment"] == sha256_hex(first["secret"].encode())
    assert first["window_open"].endswith("Z")
    assert first["secret"] not in str(list(tmp_path.rglob("*.json")))


def test_scorer_job_matches_sibling_parser_if_scorer_branch_is_available():
    source = Path("/home/bb/.bb/plugins/environment-git-worktree/host-data/worktrees/thr_gk2fjxb7ny-1/affine/v2/sn120_v2/generator/scoring.py")
    if not source.is_file():
        pytest.skip("sibling scorer source not present")
    evidence = {
        "epoch": EPOCH, "pool": POOL, "king": "head-v1", "seed_commitment": COMMITMENT,
        "tasks": [{"task_id": "01" * 10, "hotkey": "hk", "uid": 1, "uploaded_at": DEADLINE - 4, "file_sha256": "a" * 64}],
    }
    job = scorer_job(evidence, SECRET)
    assert set(job) == {"version", "epoch", "tasks", "head_version", "seed_secret", "budget_deadline", "pool_pin"}
    assert job["tasks"] == [{"id": "01" * 10, "hotkey": "hk"}]
    assert job["budget_deadline"] == DEADLINE + 3 * 3600 + 45 * 60
    assert "uid" not in str(job) and "file_sha256" not in str(job)
    assert scorer_job(evidence, SECRET) == job


def test_sibling_shaped_output_and_rollout_publish(tmp_path):
    evidence = {
        "epoch": EPOCH, "pool": POOL, "king": "head-v1", "seed_commitment": COMMITMENT,
        "tasks": [{"task_id": "01" * 10, "hotkey": "hk", "uid": 1, "uploaded_at": DEADLINE - 4, "file_sha256": "a" * 64}],
        "counts": {"hk": {"submitted": 3, "eligible": 1}},
    }
    output = {
        "version": "affine-generator-results-v1", "epoch": EPOCH, "head_version": "head-v1", "pool": POOL, "pool_pin": POOL,
        "seed_commitment": COMMITMENT, "seed_secret": SECRET,
        "tasks": [{"id": "01" * 10, "hotkey": "hk", "s": 4, "n": 8, "q": decision(4, 8).q, "decision": "accept", "reason": "q >= 0.8", "rollouts": "public/generator/epochs/epoch/rollouts/01.jsonl"}],
    }
    path = tmp_path / "rollouts" / ("01" * 10 + ".jsonl")
    path.parent.mkdir()
    row = {
        "version": "affine-generator-rollout-v1", "epoch": EPOCH, "task_id": "01" * 10, "hotkey": "hk", "j": 0,
        "seed": 1, "head_version": "head-v1", "messages": [{"role": "user", "content": "2+2"}],
        "response": "The answer is \\boxed{4}", "grade": {"passed": True, "code": "P"}, "capped": False,
    }
    path.write_text(json.dumps(row) + "\n")
    plan = publication(evidence, output, out_dir=tmp_path, secret=SECRET, history=[], registered={1: "hk"}, now=DEADLINE + 20, pool_rows=Pool(), frozen_at=DEADLINE)
    assert plan["intake"] == [{"problem": "2 + 2?", "answer": "4", "source_id": "prior-1", "prior_p": 0.5}]
    assert b"response" in plan["rollouts"]["01" * 10]
    assert b"grade" in plan["rollouts"]["01" * 10]


def test_tick_runs_all_due_phases_and_does_not_republish(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    config = Config(site, tmp_path / "work", tmp_path / "pool", first_epoch=EPOCH)
    now = DEADLINE + 60
    calls = []
    evidence = config.work_dir / "jobs" / EPOCH / "evidence.json"
    def fake_snapshot(**kwargs):
        calls.append("snapshot")
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("{}")
        (evidence.parent / "job.json").write_text('{"epoch":"' + EPOCH + '"}')
        return {"status": "snapshot", "epoch": EPOCH}
    steps = {
        "links": lambda **kwargs: calls.append("links") or {"status": "issued"},
        "window_state": lambda **kwargs: calls.append("state"),
        "snapshot": fake_snapshot,
        "dispatch": lambda **kwargs: calls.append("dispatch") or SimpleNamespace(to_dict=lambda: {"state": "dispatched", "epoch": EPOCH}),
        "collect": lambda **kwargs: calls.append("collect") or SimpleNamespace(to_dict=lambda: {"state": "running", "epoch": EPOCH}),
    }
    remote = SimpleNamespace(dispatch=lambda epoch: steps["dispatch"](epoch=epoch), collect=lambda epoch: steps["collect"](epoch=epoch))
    write_seed(config.work_dir, EPOCH)
    # Avoid network through the same step injection used by the pure driver.
    with pytest.MonkeyPatch.context() as patch:
        from sn120_v2.generator import control
        patch.setattr(control, "update_window_state", lambda *args, **kwargs: steps["window_state"]())
        patch.setattr(control, "fetch_head", lambda url, path, now: path.parent.mkdir(parents=True, exist_ok=True) or path.write_text('{"version":"test-king"}'))
        patch.setattr(control, "read_registrations", lambda client, clock: ({1: "hk"}, {"hk"}))
        tick(config, now=now, client=None, store=None, pool=None, remote=remote, steps=steps)
        assert calls == ["snapshot", "dispatch", "collect", "links", "state"]
        calls.clear()
        tick(config, now=now + 1, client=None, store=None, pool=None, remote=remote, steps=steps)
        assert calls == ["dispatch", "collect", "links", "state"]
        # A completed intake is the one authoritative do-not-publish marker.
        intake = site / "generator" / "epochs" / EPOCH / "intake.sha256"
        intake.parent.mkdir(parents=True)
        intake.write_text(sha256_hex(b"") + "  intake.jsonl\n")
        calls.clear()
        tick(config, now=now + 2, client=None, store=None, pool=None, remote=remote, steps=steps)
        assert calls == ["links", "state"], "a complete epoch is never snapshotted, dispatched or republished"


@pytest.mark.parametrize("offset", [-1, 0, 60, 180])
def test_tick_captures_only_just_closed_epoch_within_grace(tmp_path, monkeypatch, offset):
    site = tmp_path / "site"
    site.mkdir()
    config = Config(site, tmp_path / "work", tmp_path / "pool", first_epoch=EPOCH)
    write_seed(config.work_dir, EPOCH)
    calls = []
    monkeypatch.setattr("sn120_v2.generator.control.fetch_head", lambda url, path, now: calls.append(now) or path.parent.mkdir(parents=True, exist_ok=True) or path.write_text('{"version":"test-king"}'))
    monkeypatch.setattr("sn120_v2.generator.control.issue_links", lambda **kwargs: {"status": "not_due"})
    monkeypatch.setattr("sn120_v2.generator.control.update_window_state", lambda **kwargs: None)
    phases = {
        "snapshot": lambda **kwargs: {"status": "snapshot", "epoch": kwargs["epoch"]},
        "dispatch": lambda **kwargs: SimpleNamespace(to_dict=lambda: {"state": "running"}),
        "collect": lambda **kwargs: SimpleNamespace(to_dict=lambda: {"state": "running"}),
    }
    now = DEADLINE + offset
    if offset > 120:
        with pytest.raises(ConfigError, match="grace"):
            tick(config, now=now, client=None, store=None, pool=None, steps=phases)
    else:
        tick(config, now=now, client=None, store=None, pool=None, steps=phases)
        if offset >= 0:
            king = json.loads((config.work_dir / "king" / f"{EPOCH}.json").read_text())
            assert king["captured_at_epoch"] == now
            assert king["epoch"] == EPOCH
        else:
            assert not (config.work_dir / "king" / f"{EPOCH}.json").exists()
    assert calls == ([now] if 0 <= offset <= 120 else [])


def test_first_boot_does_not_process_epochs_before_launch(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    config = Config(site, tmp_path / "work", tmp_path / "pool", first_epoch="20261120T0400Z")
    before = epoch_deadline("20261120T0000Z").timestamp() - 3600
    monkeypatch.setattr("sn120_v2.generator.control.fetch_head", lambda *args, **kwargs: pytest.fail("no earlier king"))
    monkeypatch.setattr("sn120_v2.generator.control.issue_links", lambda **kwargs: {"status": "before_first_epoch", "count": 0})
    monkeypatch.setattr("sn120_v2.generator.control.snapshot_epoch", lambda **kwargs: pytest.fail("no earlier snapshot"))
    events = tick(config, now=before, client=None, store=None, pool=None)
    assert [event["phase"] for event in events] == ["links"]
    state = json.loads((site / "generator" / "state.json").read_text())
    assert state["current_epoch"] == "20261120T0400Z"
    assert (config.work_dir / "seeds" / "20261120T0400Z.json").is_file()
    assert not (config.work_dir / "seeds" / "20261120T0000Z.json").exists()
    assert not (config.work_dir / "king").exists()


def test_tick_rejects_publication_after_results_deadline(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    config = Config(site, tmp_path / "work", tmp_path / "pool", first_epoch=EPOCH)
    results = config.work_dir / "epochs" / EPOCH / "out" / "results.json"
    results.parent.mkdir(parents=True)
    results.write_text("{}")
    write_seed(config.work_dir, EPOCH)
    with pytest.MonkeyPatch.context() as patch:
        from sn120_v2.generator import control
        patch.setattr(control, "update_window_state", lambda *args, **kwargs: None)
        patch.setattr(control, "issue_links", lambda **kwargs: {"status": "not_due"})
        with pytest.raises(ConfigError, match="missed"):
            tick(config, now=DEADLINE + 3 * 3600 + 46 * 60, client=None, store=None, pool=None)
