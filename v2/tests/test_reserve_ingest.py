"""Reserve ingest: live grader, complete dedupe evidence and immutable bytes."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.math_pool import MANDATE, Task, load_pool
from sn120_v2.envs.reserve import empty_manifest, load_reserve
from sn120_v2.envs.reserve.__main__ import ingest, main, utc_now, withdraw
from sn120_v2.errors import V2Error, VerifyError, ConfigError


def setup_reserve(tmp_path, monkeypatch):
    root = tmp_path / "reserve"
    (root / "shards").mkdir(parents=True)
    (root / "MANIFEST.jsonl").write_bytes(empty_manifest())
    (root / "withdrawn.jsonl").write_bytes(b"")
    monkeypatch.setenv("SN120_RESERVE_DIR", str(root))
    monkeypatch.setattr("sn120_v2.envs.reserve.__main__.utc_now", lambda: "2026-11-18")
    return root


def input_file(tmp_path, rows):
    path = tmp_path / "input.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def args(path, **changes):
    return Namespace(**dict(input=str(path), source="testset@deadbeef", license="CC0-1.0", prefix="omni",
                     audit_method="unit-test", audit_ref="v2/tests/test_reserve_ingest.py", reject_mc=False,
                     env="affine_math", readmit=None) | changes)


def test_ingest_grades_assigns_continuous_ids_and_rerun_has_no_effect(tmp_path, monkeypatch):
    root = setup_reserve(tmp_path, monkeypatch)
    path = input_file(tmp_path, [{"problem": "What is 2 + 2?", "answer": "4", "source_id": "row0"},
                                  {"problem": "What is 3 + 3?", "answer": "6", "source_id": "row1"}])
    report = ingest(args(path))
    assert (report["ingested"], report["rejected"], report["next_ids"]) == (2, 0, ["omni:0", "omni:1"])
    assert report["shard"] == "0001-testset-deadbeef.jsonl"
    first = {p.name: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert (root / "shards" / (report["shard"] + ".rejected.json")).exists()
    assert ingest(args(path))["ingested"] == 0
    assert {p.name: p.read_bytes() for p in root.rglob("*") if p.is_file()} == first
    # Different data in a new shard continues, never restarts, this prefix.
    other = input_file(tmp_path / "other", [{"problem": "UNIQUE-RESERVE-CONTINUATION: compute 101 + 204", "answer": "305"}])
    second = ingest(args(other))
    assert second["shard"].startswith("0002-") and second["next_ids"] == ["omni:2"]
    assert main(["verify"]) == 0
    assert main(["status"]) == 0


def test_dedupe_base_archive_reserve_withdrawn_retired_and_normalised_input(tmp_path, monkeypatch):
    root = setup_reserve(tmp_path, monkeypatch)
    base = load_pool().task("math:2")
    archive = load_pool(__import__('sn120_v2.envs.math_pool', fromlist=['POOL_V2_0_FILE']).POOL_V2_0_FILE,
                        __import__('sn120_v2.envs.math_pool', fromlist=['POOL_V2_0_SHA256']).POOL_V2_0_SHA256)
    old = archive.task("dapo:3771")  # short archive-only row; no full archive scan
    # The base/archive rows are rejected by dedupe before grading. Keep the
    # grader's live self-check in the other tests, but do not evaluate their
    # potentially expensive symbolic expressions here.
    monkeypatch.setattr("sn120_v2.envs.math_grader.grade_answer", lambda answer, problem, output, capped: type("G", (), {"code": "P"})())
    rows = [{"problem": base.problem, "answer": base.answer}, {"problem": old.problem, "answer": old.answer},
            {"problem": "  UNIQUE-RESERVE-ALPHA: compute 101 + 202 ", "answer": "303"},
            {"problem": "unique-reserve-alpha: compute 101 + 202", "answer": "303"},
            {"problem": "UNIQUE-RESERVE-BETA: compute 101 + 203", "answer": "304"},
            {"problem": "UNIQUE-RESERVE-GAMMA: compute 101 + 204", "answer": "305"}]
    first = ingest(args(input_file(tmp_path, rows)))
    assert first["ingested"] == 3 and first["counts"]["duplicate_problem"] == 2 and first["counts"]["duplicate_input"] == 1
    # A withdrawn reserve task remains in the dedupe index; reusing its text
    # is refused rather than quietly re-entering the queue.
    removed = withdraw(Namespace(id="omni:2", reason="wrong key"))
    assert removed["id"] == "omni:2"
    repeat = ingest(args(input_file(tmp_path / "repeat", [{"problem": "UNIQUE-RESERVE-BETA: compute 101 + 203", "answer": "304"}])))
    assert repeat["ingested"] == 0 and repeat["counts"]["duplicate_problem"] == 1
    # Archive the original bytes for an explicit high-retired readmission.
    retired_task = Task("omni:4", "affine_math", "UNIQUE-RESERVE-DELTA: compute 101 + 205", "306")
    (root / "retired_tasks.json").write_text(json.dumps([{"id": retired_task.task_id, "env": retired_task.slice,
                                                        "problem": retired_task.problem, "answer": retired_task.answer}]))
    (root / "retired.json").write_text(json.dumps([{"id": "omni:4", "round_id": "r000008", "reason": "high", "p_or_rate": .99, "rounds_seen": 4}]))
    ordinary = args(input_file(tmp_path / "ordinary", [{"problem": retired_task.problem, "answer": retired_task.answer}]))
    assert ingest(ordinary)["counts"]["duplicate_problem"] == 1
    readmit = args(input_file(tmp_path / "readmit", [{"problem": retired_task.problem, "answer": retired_task.answer, "source_id": "omni:4"}]), readmit=["omni:4"])
    result = ingest(readmit)
    assert result["ingested"] == 1
    shard = root / "shards" / result["shard"]
    row = json.loads(shard.read_text())
    assert row["id"] == "omni:4" and row["audit"]["readmit_of"] == "r000008"
    reserve, _ = load_reserve(root)
    assert "omni:4" in reserve.queue and "omni:2" not in reserve.queue
    second_readmit = args(input_file(tmp_path / "second-readmit", [{"problem": retired_task.problem, "answer": retired_task.answer, "source_id": "omni:4"}]), readmit=["omni:4"])
    assert ingest(second_readmit)["ingested"] == 0
    assert ingest(second_readmit)["counts"] == {"already_readmitted": 1}
    assert load_reserve(root)[0].queue.count("omni:4") == 1
    archive = json.loads((root / "retired.json").read_text())
    archive[0]["reason"] = "low"
    (root / "retired.json").write_text(json.dumps(archive))
    with pytest.raises(ConfigError):
        load_reserve(root)


def test_format_grader_self_check_and_optional_mc_reject_are_sidecar_only(tmp_path, monkeypatch):
    root = setup_reserve(tmp_path, monkeypatch)
    rows = [{"problem": "UNIQUE-RESERVE-BAD: compute 101 + 202", "answer": "{"}, {"problem": "", "answer": "1"},
            {"problem": MANDATE, "answer": "1"}, {"problem": "UNIQUE-RESERVE-EMPTY: compute 101 + 204", "answer": ""},
            {"problem": "Choose: \\textbf{(A)} 1 \\qquad \\textbf{(B)} 2", "answer": "1"},
            {"problem": "UNIQUE-RESERVE-GOOD: compute 101 + 203", "answer": "304"}]
    report = ingest(args(input_file(tmp_path, rows), reject_mc=True))
    assert report["ingested"] == 1
    assert {"grader_self_check", "problem_length", "mandate", "answer_format", "multiple_choice"} <= set(report["counts"])
    rejected = json.loads((root / "shards" / (report["shard"] + ".rejected.json")).read_bytes())["rejected"]
    assert len(rejected) == 5 and all("problem" not in row and "answer" not in row for row in rejected)
    for line in (root / "shards" / report["shard"]).read_bytes().splitlines():
        obj = json.loads(line)
        assert canonical_json(obj) == line and obj["audit"]["verdict"] == "keep"


def test_reserve_verify_fails_on_a_tampered_row_and_cli_returns_nonzero(tmp_path, monkeypatch, capsys):
    root = setup_reserve(tmp_path, monkeypatch)
    result = ingest(args(input_file(tmp_path, [{"problem": "What is 2+2?", "answer": "4"}])))
    path = root / "shards" / result["shard"]
    path.write_bytes(path.read_bytes().replace(b'"answer":"4"', b'"answer":"5"'))
    with pytest.raises(ConfigError):
        load_reserve(root)
    assert main(["verify"]) == 2
    assert "error" in capsys.readouterr().err
