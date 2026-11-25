"""Offline old-result projection and benchmark registry for GSM8K-Platinum."""
import json

import pytest

from affine_bench import backfill_gsm8k_platinum as backfill
from affine_bench import registry as R


ORIGINAL = [{"question": "keep right", "answer": "work #### 1"},
            {"question": "keep wrong", "answer": "work #### 2"},
            {"question": "revised", "answer": "work #### 3"},
            {"question": "removed", "answer": "work #### 4"}]
PLATINUM = [{"question": "keep right", "answer": "work #### 1", "cleaning_status": "verified"},
            {"question": "keep wrong", "answer": "work #### 2", "cleaning_status": "consensus"},
            {"question": "revised", "answer": "work #### 6", "cleaning_status": "revised"}]


def test_backfill_cli_with_offline_dataset_files(tmp_path, capsys):
    directory = tmp_path / "results" / "king-fixture"
    directory.mkdir(parents=True)
    (directory / "gsm8k.per_task.jsonl").write_text("".join(json.dumps(dict(idx=i, n=1, correct=c)) + "\n"
                                                            for i, c in enumerate([1, 0, 1, 1])))
    original_file = tmp_path / "original.jsonl"
    platinum_file = tmp_path / "platinum.jsonl"
    for path, rows in ((original_file, ORIGINAL), (platinum_file, PLATINUM)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "estimate.json"
    backfill.main([str(tmp_path / "results"), "--original-jsonl", str(original_file),
                   "--platinum-jsonl", str(platinum_file), "--out", str(out)],
                  load_datasets=lambda: (ORIGINAL, PLATINUM))
    report = json.loads(out.read_text())
    assert report["rows"][0]["original"]["score"] == 0.75
    assert report["rows"][0]["platinum_unchanged"]["score"] == 0.5
    assert report["rows"][0]["platinum_bounds"] == {
        "tasks": 3, "lower": 1 / 3, "upper": 2 / 3, "revised_tasks": 1, "old_revised_score": 1.0,
    }
    assert json.loads(capsys.readouterr().out) == report


def test_backfill_row_and_incomplete_input(tmp_path):
    path = tmp_path / "king" / "gsm8k.per_task.jsonl"
    path.parent.mkdir()
    path.write_text("".join(json.dumps(dict(idx=i, n=2, correct=2 * c)) + "\n" for i, c in enumerate([1, 0, 0.5, 1])))
    result = backfill.backfill_row(path, ORIGINAL, PLATINUM, require_complete=False)
    assert result == {
        "row": "king",
        "original": {"tasks": 4, "score": 0.625},
        "platinum_unchanged": {"tasks": 2, "score": 0.5},
        "platinum_bounds": {"tasks": 3, "lower": 1 / 3, "upper": 2 / 3,
                            "revised_tasks": 1, "old_revised_score": 0.5},
    }
    with pytest.raises(ValueError, match="expected every original"):
        backfill.backfill_row(path, ORIGINAL[:3], PLATINUM, require_complete=False)
    with pytest.raises(ValueError, match="expected 1,319"):
        backfill.map_questions(ORIGINAL, PLATINUM)


def test_backfill_counts_an_errored_rollout_as_zero(tmp_path):
    path = tmp_path / "king" / "gsm8k.per_task.jsonl"
    path.parent.mkdir()
    path.write_text("".join(json.dumps(dict(idx=i, n=0 if i == 0 else 1, correct=0 if i == 0 else 1)) + "\n"
                            for i in range(4)))
    result = backfill.backfill_row(path, ORIGINAL, PLATINUM, require_complete=False)
    assert result["original"]["score"] == 0.75
    assert result["platinum_unchanged"]["score"] == 0.5
    assert result["platinum_bounds"]["lower"] == 1 / 3


def test_registry_switches_to_platinum_without_losing_old_gsm8k():
    old, new = R.benchmark("gsm8k"), R.benchmark("gsm8k_platinum")
    assert old["enabled"] is False and old["in_avg"] is False
    assert new["label"] == "GSM8K-Platinum" and new["suite"] == "math"
    assert new["env"] == "local-gsm8k-platinum"
    assert new["source"] == "repo:bench/environments/local-gsm8k-platinum"
    assert new["dataset"] == "madrylab/gsm8k-platinum@e762492455a1cf7967de89f05b6bef72fc713b66 (main/test, 1,209)"
    assert (new["rollouts"], new["enabled"], new["in_avg"], new["hide_column"], new["order"]) == (1, False, False, True, 151)
    runnable = [b["id"] for b in R.benchmarks(enabled_only=True, runnable_only=True)]
    assert "gsm8k_platinum" not in runnable and "gsm8k" not in runnable  # both are retired, hidden-but-stored columns
