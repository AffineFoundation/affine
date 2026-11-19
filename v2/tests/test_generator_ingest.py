"""Accepted generator drops: prior-id dedupe, screened rows and provenance."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.reserve import load_reserve
from sn120_v2.envs.reserve.generator import GENERATOR_METHOD, prior_v1_id


def _row(index: int, *, problem: str | None = None) -> dict:
    problem = problem or f"UNIQUE-GENERATOR-TEST-{index}: compute {index + 10} + 5"
    return {"problem": problem, "answer": str(index + 15), "source_id": prior_v1_id(problem),
            "prior_p": 0.375, "submitter": "miner-test", "n": 8, "k": 3}


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _prior_row(root: Path, row: dict) -> None:
    task = {"id": "bigmath:0", "env": "affine_math", "problem": row["problem"], "answer": row["answer"],
            "source": "SynthLabsAI/Big-Math-RL-Verified@c75d2f11:old",
            "license": "apache-2.0", "added_at": "2026-11-19", "prior_p": .5,
            "audit": {"method": "prior-v1 unscreened, prior_p from difficulty_meta", "verdict": "unscreened"}}
    raw = canonical_json(task) + b"\n"
    (root / "shards" / "0001-bigmath.jsonl").write_bytes(raw)
    (root / "MANIFEST.jsonl").write_bytes(canonical_json({"shard": "0001-bigmath.jsonl", "sha256": sha256_hex(raw), "count": 1, "added_at": "2026-11-19"}) + b"\n" + (root / "MANIFEST.jsonl").read_bytes())


def _release(tmp_path: Path, rows: list[dict]) -> Path:
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    root = tmp_path / "published"
    root.mkdir(parents=True)
    columns = ["id", "problem", "answer", "answer_type", "source", "source_id", "license",
               "subsource", "topic", "difficulty_meta", "also_in", "in_sn120_pool"]
    full = [dict(id=row["source_id"], problem=row["problem"], answer=row["answer"], answer_type="integer",
                 source="SynthLabsAI/Big-Math-RL-Verified@c75d2f11", source_id=str(index), license="apache-2.0",
                 subsource="test", topic="", difficulty_meta="{}", also_in="[]", in_sn120_pool="")
            for index, row in enumerate(rows)]
    path = root / "prior-00-of-08.parquet"
    pq.write_table(pa.Table.from_pylist(full, schema=pa.schema([(name, pa.string()) for name in columns])), path, compression="zstd")
    (root / "MANIFEST.json").write_text(json.dumps({
        "name": "sn120-math-prior", "version": "v1", "rows": len(full), "columns": columns,
        "shards": [{"file": path.name, "rows": len(full), "sha256": sha256_hex(path.read_bytes())}],
    }))
    return root


def _command(accepted: Path, release: Path) -> list[str]:
    return ["ingest-generator", str(accepted), "--prior-dir", str(release), "--added-at", "2026-11-20"]


def test_generator_dedupes_prior_batch_and_is_idempotent(tmp_path, monkeypatch, capsys):
    from test_reserve_ingest import setup_reserve
    from sn120_v2.envs.reserve.__main__ import main
    root = setup_reserve(tmp_path, monkeypatch)
    duplicate, fresh = _row(0), _row(1)
    duplicate["prior_p"], fresh["prior_p"] = 0.0, 0.5
    _prior_row(root, duplicate)
    accepted = _write(tmp_path / "accepted-20261120T080000Z.jsonl", [duplicate, fresh])
    release = _release(tmp_path, [duplicate, fresh])
    before = (root / "shards" / "0001-bigmath.jsonl").read_bytes()
    assert main(_command(accepted, release)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ingested"] == 1 and report["counts"]["already_ingested"] == 1
    reserve, _ = load_reserve(root)
    assert len(reserve.queue) == 2
    public = [json.loads(line) for shard in reserve.shards for line in (root / "shards" / shard.shard).read_bytes().splitlines()]
    new = next(row for row in public if row["id"] != "bigmath:0")
    assert new["id"] == "bigmath:1"  # the same source-prefix scheme as ingest-prior
    assert new["prior_p"] == fresh["prior_p"] == 0.5 and new["audit"] == {"method": GENERATOR_METHOD, "verdict": "screened"}
    assert set(new).isdisjoint({"submitter", "n", "k", "epoch"})
    assert new["source"] == f"SynthLabsAI/Big-Math-RL-Verified@c75d2f11:1"
    assert new["license"] == "apache-2.0"
    assert "miner-test" not in (root / "shards" / report["shards"][0]["shard"]).read_text()
    provenance = json.loads((root / "provenance" / "generator-20261120T080000Z.jsonl").read_text())
    assert provenance == {"id": new["id"], "source_id": fresh["source_id"], "submitter": "miner-test", "n": 8, "k": 3, "epoch": "20261120T080000Z"}
    assert reserve.audit_verdicts == {"bigmath:0": "unscreened"}
    from sn120_v2.rotation import Reserve, rotate_info
    from test_pool_rotation_v2 import info_params, snapshot
    active = {"a:0": [0.0, 0.25, 4]}
    reserve = load_reserve(root)[0]
    ranked = Reserve.from_dict(reserve.to_dict() | {"candidate_meta": {
        "bigmath:0": {"prior_p": 0.0, "unscreened": True},
        new["id"]: {"prior_p": 0.5},
    }})
    decision = rotate_info("r000070", active, snapshot(active, 2.0), {},
                           ranked, info_params(rotation_active_size=2, rotation_churn_frac=0.0, rotation_unscreened_max=1))
    assert decision.admitted == (new["id"],)  # the raw row's slot remains unused
    assert "bigmath:0" in decision.next_up
    first = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert main(_command(accepted, release)) == 0
    assert json.loads(capsys.readouterr().out)["ingested"] == 0
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == first
    assert main(["verify"]) == 0
    assert before == (root / "shards" / "0001-bigmath.jsonl").read_bytes()


def test_generator_dedupes_duplicate_input_and_rejects_bad_prior_id(tmp_path, monkeypatch, capsys):
    from test_reserve_ingest import setup_reserve
    from sn120_v2.envs.reserve.__main__ import main
    setup_reserve(tmp_path, monkeypatch)
    row = _row(2)
    accepted = _write(tmp_path / "accepted-epoch2.jsonl", [row, row | {"prior_p": .5}])
    release = _release(tmp_path, [row])
    assert main(_command(accepted, release)) == 0
    assert json.loads(capsys.readouterr().out)["ingested"] == 1
    bad = _write(tmp_path / "accepted-bad.jsonl", [_row(3) | {"source_id": "not-the-problem-hash"}])
    assert main(_command(bad, release)) == 2
    assert "source_id" in capsys.readouterr().err


def test_generator_epoch_name_is_required(tmp_path, monkeypatch, capsys):
    from test_reserve_ingest import setup_reserve
    from sn120_v2.envs.reserve.__main__ import main
    setup_reserve(tmp_path, monkeypatch)
    accepted = _write(tmp_path / "not-accepted.jsonl", [_row(4)])
    assert main(["ingest-generator", str(accepted), "--prior-dir", str(tmp_path)]) == 2
    assert "accepted-<epoch>" in capsys.readouterr().err


def test_generator_grades_and_requires_prior_metadata_before_writing(tmp_path, monkeypatch, capsys):
    from test_reserve_ingest import setup_reserve
    from sn120_v2.envs.reserve.__main__ import main
    root = setup_reserve(tmp_path, monkeypatch)
    row = _row(5)
    accepted = _write(tmp_path / "accepted-epoch5.jsonl", [row])
    release = _release(tmp_path, [row])
    calls = []
    from sn120_v2.envs import math_grader
    original = math_grader.grade_answer

    def check(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(math_grader, "grade_answer", check)
    assert main(_command(accepted, release)) == 0
    capsys.readouterr()
    assert calls
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    # A well-formed but unknown prior id is rejected without altering reserve bytes.
    unknown = _write(tmp_path / "accepted-unknown.jsonl", [_row(55)])
    assert main(_command(unknown, release)) == 2
    assert "not in the verified prior-v1 release" in capsys.readouterr().err
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    fresh = _row(6)
    ungradeable = _write(tmp_path / "accepted-ungradeable.jsonl", [fresh])
    second_release = _release(tmp_path / "second", [fresh])
    monkeypatch.setattr(math_grader, "grade_answer", lambda *args: type("G", (), {"code": "F"})())
    assert main(_command(ungradeable, second_release)) == 2
    assert "grader answer-format self-check" in capsys.readouterr().err
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def _ops_module():
    import importlib.util
    path = Path(__file__).parents[1] / "ops" / "generator_ingest.py"
    spec = importlib.util.spec_from_file_location("sn120_generator_ops_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ops_dry_run_makes_no_network_commit_merge_or_state_change(tmp_path, monkeypatch, capsys):
    ops = _ops_module()
    monkeypatch.setattr(ops.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry-run executed a subprocess"))
    _write(tmp_path / "accepted-20261120T080000Z.jsonl", [_row(7)])
    state = tmp_path / "state" / "processed.txt"
    assert ops.main([str(tmp_path), "--prior-dir", str(tmp_path / "prior"), "--state", str(state), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["processed"] == 1 and report["dry_run"]
    assert not state.exists()
    # The explicit non-merge dry-run also promises to leave its future PR open.
    assert ops.main([str(tmp_path), "--prior-dir", str(tmp_path / "prior"), "--state", str(state), "--dry-run", "--no-merge"]) == 0
    assert not state.exists()


def test_ops_real_path_checks_reserve_files_and_reports_confirmed_merge_sha(tmp_path):
    ops = _ops_module()
    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        text = ""
        if args[:3] == ["git", "status", "--porcelain"]:
            text = "A  v2/sn120_v2/envs/reserve/shards/0001-test.jsonl\n"
        elif args[:3] == ["git", "diff", "--name-only"]:
            text = "v2/sn120_v2/envs/reserve/shards/0001-test.jsonl\n"
        elif "ingest-generator" in args:
            text = '{"ingested": 1}'
        elif args[:3] == ["gh", "pr", "create"]:
            text = "https://example.test/pull/42\n"
        elif args[:3] == ["gh", "pr", "view"] and "files" in args:
            text = "v2/sn120_v2/envs/reserve/shards/0001-test.jsonl\n"
        elif args[:3] == ["gh", "pr", "view"] and "state,mergeCommit" in args:
            text = '{"state":"MERGED","mergeCommit":{"oid":"abc123"}}\n'
        return type("Result", (), {"stdout": text})()

    report = ops.process_epoch("epoch-test", tmp_path / "accepted-epoch-test.jsonl", work=tmp_path / "work",
                               remote="origin", base="v2", prior_dir=tmp_path / "prior", merge=True, dry_run=False, runner=runner)
    assert report["merge_sha"] == "abc123" and report["merged"]
    assert any(args[:3] == ["gh", "pr", "merge"] for args in calls)
    assert "--prior-dir" in next(args for args in calls if "ingest-generator" in args)
    merge = next(index for index, args in enumerate(calls) if args[:3] == ["gh", "pr", "merge"])
    files = next(index for index, args in enumerate(calls) if args[:3] == ["gh", "pr", "view"] and "files" in args)
    assert files < merge

    calls.clear()
    with pytest.raises(RuntimeError, match="non-reserve"):
        ops.process_epoch("epoch-outside", tmp_path / "accepted-epoch-outside.jsonl", work=tmp_path / "outside",
                          remote="origin", base="v2", prior_dir=tmp_path / "prior", merge=True, dry_run=False,
                          runner=lambda args, **kwargs: type("Result", (), {"stdout": (
                              "A  v2/POOL_ROTATION.md\n" if args[:3] == ["git", "status", "--porcelain"] else
                              '{"ingested": 1}' if "ingest-generator" in args else "")})())
    assert not any(args[:2] in (["git", "push"], ["gh", "pr"]) for args in calls)
