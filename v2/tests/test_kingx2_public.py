"""Public-only verification for the future king-x2 deployment kit."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import urllib.error

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json
from sn120_v2.irt import IRTSampler
from sn120_v2.params import Params
from sn120_v2.rotation import Rotation

KIT = Path(__file__).parents[1] / "tools/kingx2-deploy"


@pytest.fixture(autouse=True)
def isolated_public_root(tmp_path, monkeypatch):
    # pub_get resolves its root at call time, not while the module is loaded.
    # Keep every test on its synthetic LocalStore and away from affine.io.
    monkeypatch.setenv("KINGX2_PUBLIC_ROOT", str(tmp_path / "public"))
    monkeypatch.setenv("KINGX2_WORK_DIR", str(tmp_path / "work"))

def operator(tmp_path: Path):
    # The real entry point is invoked with -I. This fixture also loads its
    # selected v2 package explicitly, never through the kit's directory.
    import sn120_v2
    import os
    old = dict(os.environ)
    module_environment = old | {name: value for name, value in {
        "KINGX2_V2": str(KIT.parents[1]), "KINGX2_HERE": str(KIT),
        "KINGX2_STATE": str(tmp_path / "state"), "KINGX2_DRY": "1",
        "KINGX2_PUBLIC_ROOT": str(tmp_path / "public"),
    }.items()}
    os.environ.update(module_environment)
    sys.modules["sn120_v2"] = sn120_v2
    spec = importlib.util.spec_from_file_location("kingx2_public_test_operator", KIT / "kx2_operator.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        os.environ.clear()
    os.environ.update(old)
    return module


def fixture(tmp_path: Path):
    module = operator(tmp_path)
    rid = "r000087"
    ids = tuple(f"task:{index}" for index in range(10000))
    snapshot = IRTSampler(ids, Params(rotation_active_size=10000, irt_draw_rule="sqrt_fisher")).public_snapshot()
    active = tuple(snapshot["tasks"])
    from sn120_v2.rotation import active_sha
    previous = Rotation(round_id="r000086", active_ids=active, active_sha=active_sha(active),
                        prev_active_sha="0" * 64, admitted=(), retired=(), reserve_manifest_sha="1" * 64,
                        queue_remaining=0, next_up=(), reserve_history={"shards": [], "queue": [], "withdrawn": []},
                        ledger={task: (0, 0) for task in active}, source_snapshot_id=snapshot["snapshot_id"],
                        rule_version="affine-pool-rotation-v1", rule_params={})
    rotation = Rotation.from_dict(previous.to_dict() | {"round_id": rid})
    journal_rotation = rotation.journal_block() | {"state_sha": rotation.digest(), "size": len(active)}
    record = {"round_id": rid, "params_hash": module.EXPECT["new_params_hash"], "n_tasks": 2000,
              "snapshot_id": snapshot["snapshot_id"], "rotation": journal_rotation}
    opened = SimpleNamespace(kind="round_opened", data={"round": record})
    module.read_journal = lambda path: [opened]
    root = tmp_path / "public"
    root.mkdir()
    (root / "state").mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "state/journal.jsonl").write_text("local journal is read through the fixture above\n")
    (root / layout.pool_rotation(rid)).parent.mkdir(parents=True)
    (root / layout.pool_rotation(rid)).write_bytes(rotation.to_json())
    (root / layout.active_pool(rotation.active_sha)).parent.mkdir(parents=True)
    (root / layout.active_pool(rotation.active_sha)).write_bytes(canonical_json(list(active)))
    (root / layout.irt_snapshot(snapshot["snapshot_id"])).parent.mkdir(parents=True)
    (root / layout.irt_snapshot(snapshot["snapshot_id"])).write_bytes(canonical_json(snapshot))
    (root / layout.round_file(rid)).parent.mkdir(parents=True)
    (root / layout.round_file(rid)).write_bytes(canonical_json({"round_id": rid, "params_hash": module.EXPECT["new_params_hash"]}))
    return module, rid, rotation


def test_open_and_public_checks_do_not_require_public_pool_state(tmp_path):
    module, rid, rotation = fixture(tmp_path)
    assert not (tmp_path / "public" / "public/pool/state").exists()
    assert module.open_check(rid)["ok"], module.open_check(rid)
    assert module.public_check(rid)["ok"], module.public_check(rid)


@pytest.mark.parametrize("check", ["open_check", "public_check"])
def test_public_checks_retry_each_artifact_within_one_bounded_window(tmp_path, monkeypatch, check):
    module, rid, rotation = fixture(tmp_path)
    module.EXPECT["public_retry_seconds"] = 10
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    original = module.pub_get
    attempts = []
    transient = {
        layout.pool_rotation(rid): urllib.error.HTTPError("https://example.invalid/rotation", 404, "late", {}, None),
        layout.active_pool(rotation.active_sha): urllib.error.HTTPError("https://example.invalid/active", 503, "busy", {}, None),
        layout.irt_snapshot(rotation.source_snapshot_id): socket.timeout("late mirror"),
    }
    if check == "public_check":
        transient[layout.round_file(rid)] = urllib.error.URLError(socket.timeout("late report"))
    def intermittently_unavailable(key, **kwargs):
        attempts.append(key)
        if attempts.count(key) == 1:
            raise transient[key]
        return original(key, **kwargs)
    module.pub_get = intermittently_unavailable
    result = getattr(module, check)(rid)
    assert result["ok"], result
    assert set(attempts) == set(transient)
    assert all(attempts.count(key) == 2 for key in transient)


def test_public_rotation_still_checks_digest_and_journal_block(tmp_path):
    module, rid, rotation = fixture(tmp_path)
    altered = Rotation.from_dict(rotation.to_dict() | {"queue_remaining": 1})
    (tmp_path / "public" / layout.pool_rotation(rid)).write_bytes(altered.to_json())
    assert not module.open_check(rid)["ok"]
    assert any("state_sha/size/block" in why for why in module.open_check(rid)["why"])


def test_public_poll_retries_404_5xx_and_timeouts_within_one_deadline(tmp_path, monkeypatch):
    module = operator(tmp_path)
    module.EXPECT["public_retry_seconds"] = 10
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    attempts = []
    errors = (urllib.error.HTTPError("https://example.invalid/key", 404, "not mirrored", {}, None),
              urllib.error.HTTPError("https://example.invalid/key", 503, "busy", {}, None),
              socket.timeout("late mirror"))
    def fetch(deadline):
        attempts.append(deadline)
        if len(attempts) <= len(errors):
            raise errors[len(attempts) - 1]
        return b"public bytes"
    assert module.public_poll(fetch) == b"public bytes"
    assert len(attempts) == 4 and len(set(attempts)) == 1


def test_public_poll_does_not_retry_forbidden_or_corrupt_content(tmp_path, monkeypatch):
    module = operator(tmp_path)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    calls = []
    def forbidden(deadline):
        calls.append(deadline)
        raise urllib.error.HTTPError("https://example.invalid/key", 403, "forbidden", {}, None)
    with pytest.raises(urllib.error.HTTPError):
        module.public_poll(forbidden)
    assert len(calls) == 1
    with pytest.raises(ValueError):
        module.public_poll(lambda deadline: json.loads(b"not json"))


def test_run_log_eval_wall_reads_only_this_rounds_production_summary(tmp_path):
    module = operator(tmp_path)
    log = tmp_path / "run.log"
    log.write_text(
        "SN120 v2 round r000086: NULL\nwall time: train 1m00s | eval 1m00s | total 2m00s\n"
        "2026-11-21T12:00:00Z SN120 v2 round r000087: NULL\n"
        "wall time: train 8m22s | eval 116m24s | promote 0m01s | fold 0m02s | total 124m49s\n"
        "text containing SN120 v2 round r000088: NULL is not a report header\n"
        "wall time: eval 9m99s\n"
    )
    assert module.run_log_eval_wall("r000087", log) == 6984
    assert module.run_log_eval_wall("r000086", log) == 60
    assert module.run_log_eval_wall("r000088", log) is None
    assert module.run_log_eval_wall("r000087", tmp_path / "missing.log") is None


def health_fixture(tmp_path, monkeypatch, *, phases=False, eval_wall=6984):
    module, rid, rotation = fixture(tmp_path)
    module.EXPECT["n_tasks"] = 4
    module.EXPECT["null_samples"] = 2
    module.read_journal = lambda path: []
    module.journal_close_summary = lambda rid, next_open_ok: {
        "clean": True, "why": [], "outcome": "null", "n_tasks": 4,
        "round_n_tasks": 4, "round_params_hash": module.EXPECT["new_params_hash"],
        "boxes": ["box-a"],
    }
    minutes, seconds = divmod(eval_wall, 60)
    report = {
        "round_id": rid, "params_hash": module.EXPECT["new_params_hash"],
        "outcome": "null", "n_tasks": 4, "wall_s": {"training": 1, "evaluation": eval_wall},
        "null_outcomes": {"box-a": ["P" * 4, "P" * 4]},
    }
    (tmp_path / "public" / layout.round_file(rid)).write_bytes(canonical_json(report))
    run_log = tmp_path / "work/logs/run.log"
    run_log.parent.mkdir(parents=True)
    run_log.write_text(f"SN120 v2 round {rid}: NULL\nwall time: train 0m01s | eval {minutes}m{seconds:02d}s\n")
    monkeypatch.setenv("KINGX2_WORK_DIR", str(tmp_path / "work"))
    if phases:
        phase_log = tmp_path / "work/logs/phases.jsonl"
        phase_log.write_text(json.dumps({"phase": "eval", "round_id": rid, "wall_s": eval_wall}) + "\n")
    return module, rid


@pytest.mark.parametrize("phases", [False, True])
def test_health_accepts_public_eval_wall_without_production_phase_records(tmp_path, monkeypatch, phases):
    module, rid = health_fixture(tmp_path, monkeypatch, phases=phases)
    result = module.health(rid)
    assert result["ok"], result
    assert result["eval_wall_s"] == 6984 and result["run_log_eval_wall_s"] == 6984
    assert result["eval_phase_wall_s"] == ([{"at": None, "phase": "eval", "look": None, "wall_s": 6984}] if phases else [])
    assert result["eval_timing_evidence"] == ("phases" if phases else "run_log")


def test_health_accepts_missing_run_log_but_rejects_conflicting_or_missing_wall_time(tmp_path, monkeypatch):
    module, rid = health_fixture(tmp_path, monkeypatch)
    (tmp_path / "work/logs/run.log").unlink()
    result = module.health(rid)
    assert result["ok"] and result["eval_timing_evidence"] == "public_eval_wall_s", result
    (tmp_path / "work/logs/run.log").write_text(f"SN120 v2 round {rid}: NULL\nwall time: train 0m01s | eval 116m24s\n")
    result = module.health(rid)
    report_path = tmp_path / "public" / layout.round_file(rid)
    report = json.loads(report_path.read_bytes())
    report["wall_s"]["evaluation"] = 6984.4
    report_path.write_bytes(canonical_json(report))
    assert module.health(rid)["ok"]
    report["wall_s"]["evaluation"] = 6987
    report_path.write_bytes(canonical_json(report))
    result = module.health(rid)
    assert not result["ok"] and "run-log evaluation wall time differs" in result["why"][-1]
    report["wall_s"]["evaluation"] = 0
    report_path.write_bytes(canonical_json(report))
    result = module.health(rid)
    assert not result["ok"] and "public round evaluation wall time is missing or not positive" in result["why"]
