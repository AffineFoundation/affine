"""publish: an errored rollout counts as a wrong sample in the pooled score, and rows that lost samples are flagged incomplete."""
import json, os
from affine_bench import config as C, publish, source


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


def test_source_reads_reverted_head_ids_and_falls_back_to_mining(monkeypatch):
    docs = {
        source.layout.HEAD_CHAIN: b'{"reverts":[{"reverted_version_ids":["h163"],"reverted_heads":["h164"]}]}',
        source.layout.MINING_JSON: b'{"head_chain":{"reverted_heads":["h165"]}}',
    }
    monkeypatch.setattr(source, "store", lambda: type("Store", (), {"get": lambda self, key: docs[key]})())
    assert source.reverted_version_ids() == {"h163", "h164"}
    docs[source.layout.HEAD_CHAIN] = b'{"reverts":[]}'
    assert source.reverted_version_ids() == {"h165"}


def test_publish_removes_reverted_current_and_published_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "SITE", str(tmp_path))
    monkeypatch.setattr(C, "STATE", str(tmp_path / "state"))
    monkeypatch.setattr(publish, "OUT", str(tmp_path / "benchmarks.json"))
    monkeypatch.setattr(source, "reverted_version_ids", lambda: {"h163"})
    monkeypatch.setattr(publish.R, "benchmarks", lambda **kwargs: [{"id": "a", "suite": "math"}])
    monkeypatch.setattr(publish.R, "serving", lambda: {"version": 1})
    previous = {
        "models": [
            {"model_id": "king-old", "kind": "king", "head_version_id": "h163"},
            {"model_id": "king-kept", "kind": "king", "head_version_id": "h162"},
            {"model_id": "reference", "kind": "reference", "head_version_id": "h163"},
        ],
        "results": [],
    }
    result = tmp_path / "state/results/king-kept/a.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(dict(
        model_id="king-kept", benchmark="a", score=.5, se=.1, tasks=1, tasks_total=1,
        samples=1, rollouts=1, cap_rate=0, mean_out_tokens=1, bench_version="bench-v2",
    )))
    (tmp_path / "benchmarks.json").write_text(json.dumps(previous))
    rows = [
        {"model_id": "king-current", "kind": "king", "head_version_id": "h163"},
        {"model_id": "king-ok", "kind": "king", "head_version_id": "h161"},
    ]
    publish.publish(rows, {})
    table = json.loads((tmp_path / "benchmarks.json").read_text())
    models = {m["model_id"]: m for m in table["models"]}
    assert set(models) == {"king-ok", "king-kept"}
    assert "king-old" not in models
    assert "king-current" not in models
