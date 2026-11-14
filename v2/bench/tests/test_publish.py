"""publish: an errored rollout counts as a wrong sample in the pooled score, and rows that lost samples are flagged incomplete."""
import json, os
from affine_bench import config as C, publish


def test_pooled_counts_errored_rollouts_as_wrong(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "STATE", str(tmp_path))
    os.makedirs(tmp_path / "private" / "m")
    # part a: 2 tasks x 4 rollouts, task 1 lost 2 rollouts (n=2) but both of its valid ones were right
    (tmp_path / "private" / "m" / "a.per_task.jsonl").write_text(
        json.dumps(dict(idx=0, n=4, correct=4.0, errors=0)) + "\n" + json.dumps(dict(idx=1, n=2, correct=2.0, errors=2)) + "\n")
    (tmp_path / "private" / "m" / "b.per_task.jsonl").write_text(json.dumps(dict(idx=0, n=4, correct=0.0, errors=0)) + "\n")
    results = {("m", "a"): dict(rollouts=4, tasks=2, tasks_total=2, samples=6, cap_rate=0.0, finished_at=1),
               ("m", "b"): dict(rollouts=4, tasks=1, tasks_total=1, samples=4, cap_rate=0.0, finished_at=1)}
    full = {"a": 2, "b": 1}
    p = publish.pooled("m", dict(pooled_from=["a", "b"]), results, full)
    assert abs(p["score"] - (1.0 + 0.5 + 0.0) / 3) < 1e-9  # dropping the errors would give (1 + 1 + 0) / 3
    assert p["samples_expected"] == 12 and p["errors"] == 2 and p["complete"] is False


def test_expected_uses_the_full_task_count():
    assert publish.expected(dict(tasks=393, tasks_total=393, rollouts=1), "math500", {"math500": 500}) == 500
    assert publish.expected(dict(tasks=30, tasks_total=30, rollouts=16), "aime22", {"aime22": 30}) == 480
