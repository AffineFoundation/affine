"""Offline unit proofs for the dedup deploy's public and cut evidence checks."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import shutil
import urllib.error

import pytest

ROOT = Path(__file__).parents[1]
KIT = ROOT / "tools" / "dedup-deploy"


def operator(tmp_path, monkeypatch):
    monkeypatch.setenv("DEDUP_V2", str(ROOT))
    monkeypatch.setenv("DEDUP_HERE", str(KIT))
    monkeypatch.setenv("DEDUP_STATE", str(tmp_path))
    monkeypatch.setenv("DEDUP_WORK_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("dedup_deploy_operator_test", KIT / "dedup_operator.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_line(*, invalid=0, rules="", no_file=0):
    detail = f" ({rules})" if rules else ""
    return (f"files: check at 2026-11-21 00:00:00 UTC: 8 waiting seats, {8-invalid-no_file} valid, "
            f"{invalid} not valid{detail}, {no_file} with no file, 0 changed")


def test_run_log_cut_counts_bind_the_round_and_distinguish_zero_from_missing(tmp_path, monkeypatch):
    op = operator(tmp_path, monkeypatch)
    log = tmp_path / "run.log"
    log.write_text("\n".join([
        check_line(rules="min_rollouts 3, null_update 2"),
        "open: round r000089 is open, 3 seats. From the file check to here",
        check_line(rules="min_rollouts 4, null_update 1"),
        "open: round r000090 is open, 3 seats. From the file check to here",
    ]) + "\n")
    assert op.run_log_cut_counts("r000090", log) == {"min_rollouts": 4, "null_update": 1}
    assert op.run_log_cut_counts("r000089", log) == {"min_rollouts": 3, "null_update": 2}
    log.write_text(check_line() + "\nopen: round r000090 is open, 3 seats. From the file check to here\n")
    assert op.run_log_cut_counts("r000090", log) == {"min_rollouts": 0, "null_update": 0}
    log.write_text("open: round r000090 is open\n")
    assert op.run_log_cut_counts("r000090", log) == {}


def test_run_log_eval_wall_and_optional_phase_evidence(tmp_path, monkeypatch):
    op = operator(tmp_path, monkeypatch)
    log = tmp_path / "run.log"
    log.write_text("2026-11-21 SN120 v2 round r000090: null\n"
                   "  wall time: train 1m00s | eval 20m34s | apply 0m01s\n")
    assert op.run_log_eval_wall("r000090", log) == 1234
    assert op.run_log_eval_wall("r000091", log) is None


def test_public_retry_is_only_for_publication_transport(tmp_path, monkeypatch):
    op = operator(tmp_path, monkeypatch)
    calls = 0

    def eventually(left):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise urllib.error.HTTPError("https://example.invalid", 404, "not mirrored", {}, None)
        return b"ok"

    import time
    monkeypatch.setattr(time, "sleep", lambda _: None)
    assert op.public_poll(eventually, deadline=time.monotonic() + 30) == b"ok"
    assert calls == 3
    for error in (ValueError("bad schema"), urllib.error.HTTPError("https://example.invalid", 400, "bad", {}, None)):
        with pytest.raises(type(error)):
            op.public_poll(lambda _: (_ for _ in ()).throw(error), deadline=time.monotonic() + 30)


def journal_round(op, tmp_path, *, duplicate_only=False, all_cut=False):
    from sn120_v2.canonical import canonical_json
    from sn120_v2.journal import JournalRecord
    from sn120_v2.types import EntryDecision, RoundDecision
    params = op.EXPECT["new_params_hash"]
    statuses = (["duplicate", "duplicate"] if all_cut else
                ["duplicate", "null_update", "lost"] if not duplicate_only else ["duplicate"])
    entries = [EntryDecision(i, i + 1, f"hk{i}", i + 100, "a" * 64, status)
               for i, status in enumerate(statuses)]
    decision = RoundDecision("r000090", "head", None, entries, 2000, 1.0, 0.0, "synthetic")
    rotation = {"state_sha": "b" * 64, "size": 10000}
    record = {"round_id": "r000090", "params_hash": params, "n_tasks": 2000,
              "leases": ([] if duplicate_only else [{"box_id": "box0", "lease_id": "r000090-l0", "entries": []}]),
              "decision": decision.to_dict() if duplicate_only else None}
    rows = [JournalRecord(0, 0, 1, "round_opened", {"round": record, "rotation": rotation}, "")]
    if not duplicate_only:
        rows.append(JournalRecord(1, 0, 2, "round_settled", {"round_id": "r000090", "decision": decision.to_dict(), "results": {}}, rows[0].checksum))
    rows.append(JournalRecord(len(rows), 0, 3, "round_closed", {"round_id": "r000090", "outcome": "null", "served": [],
                                               "front": [], "attempted": [], "held": [], "released": []}, rows[-1].checksum))
    (tmp_path / "journal.jsonl").write_bytes(b"".join(row.line() + b"\n" for row in rows))


def test_health_cut_semantics_and_duplicate_only_open_decision(tmp_path, monkeypatch):
    op = operator(tmp_path, monkeypatch)
    journal_round(op, tmp_path)
    runlog = tmp_path / "run.log"
    runlog.write_text(check_line(rules="min_rollouts 0, null_update 0") +
                      "\nopen: round r000090 is open, 3 seats. From the file check to here\n")
    monkeypatch.setenv("DEDUP_RUN_LOG", str(runlog))
    report = {"round_id": "r000090", "params_hash": op.EXPECT["new_params_hash"], "outcome": "null",
              "n_tasks": 2000, "null_outcomes": {"box0": ["P" * 2000, "E" * 2000]}, "wall_s": {"evaluation": 1234}}
    monkeypatch.setattr(op, "public_round", lambda rid: {"report": report, "raw": op.layout and b"{}"})
    out = op.health("r000090")
    assert out["ok"], out["why"]
    assert out["cut_counts"] == {"duplicate": 1, "null_update": 1}
    assert out["admission_cut_counts"] == {"min_rollouts": 0, "null_update": 0}
    assert out["surviving_batches"] == 1
    runlog.write_text("open: round r000090 is open\n")
    out = op.health("r000090")
    assert not out["ok"] and any("not recorded" in why for why in out["why"])

    journal_round(op, tmp_path, all_cut=True)
    runlog.write_text(check_line(rules="min_rollouts 0, null_update 0") + "\nopen: round r000090 is open\n")
    out = op.health("r000090")
    assert not out["ok"] and "dedup/null-update cuts removed every selected batch" in out["why"]

    journal_round(op, tmp_path, duplicate_only=True)
    out = op.journal_close_summary("r000090", next_open_ok=True)
    assert out["cut_counts"] == {"duplicate": 1, "null_update": 0}
    assert out["selected_batches"] == 1 and out["surviving_batches"] == 0
    assert out["round_params_hash"] == op.EXPECT["new_params_hash"]
    assert out["outcome"] == "null"


def test_apply_requires_reviewed_old_worker_recipe(tmp_path, monkeypatch):
    op = operator(tmp_path, monkeypatch)
    # This assertion is pinned explicitly so a release-source digest cannot
    # accidentally replace the retained king-x2 worker recipe in apply checks.
    from sn120_v2.params import load_params
    params = load_params(KIT / "new-params.json")
    assert params.recipe.code_digest == op.EXPECT["old_digest"]
    assert params.recipe.code_digest != op.EXPECT["new_digest"]


def test_legacy_params_require_old_head(tmp_path, monkeypatch):
    # Never stage command output in the reviewed source kit, even in tests.
    kit = tmp_path / "kit"
    shutil.copytree(KIT, kit, ignore=shutil.ignore_patterns("__pycache__", "set-params-*.json"))
    monkeypatch.setenv("DEDUP_HERE", str(kit))
    op = operator(tmp_path, monkeypatch)
    op.HERE = kit
    monkeypatch.setattr(op, "head", lambda: op.EXPECT["new_commit"])
    with pytest.raises(RuntimeError, match="legacy king-x2 params require"):
        op.set_params("old-params.json")
    monkeypatch.setattr(op, "head", lambda: op.EXPECT["old_commit"])
    monkeypatch.setattr(op, "alive", lambda: [])
    monkeypatch.setattr(op, "sh", lambda *args, **kwargs: type("R", (), {"stdout": "{}", "returncode": 0})())
    monkeypatch.setenv("DEDUP_DRY_SET_PARAMS", "true")
    monkeypatch.setattr(op, "DRY", True)
    op.set_params("old-params.json")
    assert (kit / "set-params-old-params.json").read_text() == "{}"
    assert not (KIT / "set-params-old-params.json").exists()


def test_bounded_recovery_passes_stop_path_and_requires_real_confirmation(tmp_path, monkeypatch, capsys):
    op = operator(tmp_path, monkeypatch)
    monkeypatch.setattr(op, "DRY", True)
    monkeypatch.setattr(op, "alive", lambda: [])
    monkeypatch.setenv("DEDUP_DRY_RUN", "isolated real run")
    calls = []

    def run_once(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs["check"] is False
        assert kwargs["env"]["SN120_WORK_DIR"] == str(tmp_path)
        (tmp_path / "logs/run-loop.stop-after-round").unlink()
        return type("R", (), {"stdout": "", "stderr": "stop after the round: obeyed. fixture\n", "returncode": 0})()

    monkeypatch.setattr(op, "sh", run_once)
    result = op.cli("run", stop_after_round=True)
    assert result.returncode == 0
    assert len(calls) == 1 and not (tmp_path / "logs/run-loop.stop-after-round").exists()

    journal_round(op, tmp_path)
    # Stop confirmation must be checked before replaying the post-run state.
    # This test deliberately does not reach that later journal assertion.
    from types import SimpleNamespace
    state = SimpleNamespace(round=SimpleNamespace(
        round_id="r000090", params_hash=op.EXPECT["new_params_hash"], decision=None))
    monkeypatch.setattr(op, "load", lambda _: state)
    monkeypatch.setattr(op, "head", lambda: op.EXPECT["new_commit"])
    monkeypatch.setattr(op, "sealed_result_leases", lambda: [])
    monkeypatch.setattr(op, "cli", lambda *args, **kwargs: type("R", (), {
        "stdout": "", "stderr": "real bounded-run diagnostic\n", "returncode": 0})())
    with pytest.raises(RuntimeError, match="did not confirm its stop-after-round request"):
        op.finish_open_for_rollback()
    assert "real bounded-run diagnostic" in capsys.readouterr().err
