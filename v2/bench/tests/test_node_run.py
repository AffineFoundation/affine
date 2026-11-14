"""node_run: what counts as done, and which run may go on with --resume (2026-11-14: ArXivMath went from 16 samples to 2)."""
import json, os
from affine_bench import registry as R
from affine_bench.node_run import finished, resumable, summarize


def saved_run(vf, rollouts, name="run1", rows=()):
    d = vf / name / "configs" / "resolved"; d.mkdir(parents=True)
    (d / "eval.json").write_text(json.dumps(dict(env="arxivmath", num_rollouts=rollouts)))
    (vf / name / "traces.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(d / "eval.json")


def row(idx, score):
    tr = [dict(rewards={"r": {"score": score}}, calls=[{"finish_reason": "stop"}], num_output_tokens=10)]
    return dict(task={"data": {"idx": idx}, "key": str(idx)}, ok=True, traces=tr)


def test_arxivmath_has_two_samples_and_a_new_version():
    b = R.benchmark("arxivmath")
    assert b["rollouts"] == 2 and R.bench_version(b).startswith("arxivmath/v2/")
    assert all(x.get("version", 1) == 1 for x in R.benchmarks() if x["id"] != "arxivmath")


def test_a_run_with_the_same_count_goes_on(tmp_path):
    vf = tmp_path / "vf"; saved = saved_run(vf, 2)
    assert resumable(str(vf), 2) == saved and os.listdir(tmp_path) == ["vf"]


def test_a_run_with_another_count_is_moved_aside_and_not_scored(tmp_path):
    # the run at 16 was stopped: problem 0 has 3 rows, the two that ended first are right; problem 1 has none
    vf = tmp_path / "vf"; saved_run(vf, 16, rows=[row(0, 1.0), row(0, 1.0), row(0, 0.0)])
    b = dict(id="arxivmath", env="arxivmath", rollouts=2)
    assert summarize(b, str(vf))[0]["score"] == 1.0  # what a cut of that file to 2 would say: the shortest answers
    assert resumable(str(vf), 2) is None
    assert sorted(os.listdir(tmp_path)) == ["vf", "vf.r16"] and os.listdir(vf) == []
    assert os.path.exists(tmp_path / "vf.r16" / "run1" / "traces.jsonl")
    assert summarize(b, str(vf)) is None
    saved_run(vf, 16, name="run2")
    assert resumable(str(vf), 2) is None and sorted(os.listdir(tmp_path)) == ["vf", "vf.r16", "vf.r16.2"]


def test_a_saved_config_that_cannot_be_read_is_not_resumed(tmp_path):
    vf = tmp_path / "vf"; saved = saved_run(vf, 2); open(saved, "w").write("{")
    assert resumable(str(vf), 2) is None and sorted(os.listdir(tmp_path)) == ["vf", "vf.rx"]


def test_no_saved_run(tmp_path):
    (tmp_path / "vf").mkdir()
    assert resumable(str(tmp_path / "vf"), 2) is None and os.listdir(tmp_path) == ["vf"]


def test_a_summary_of_another_version_is_not_done(tmp_path):
    b = R.benchmark("arxivmath")
    assert not finished(b, str(tmp_path))
    (tmp_path / "summary.json").write_text(json.dumps(dict(bench_version=R.bench_version(dict(b, version=1)))))
    assert not finished(b, str(tmp_path))
    (tmp_path / "summary.json").write_text(json.dumps(dict(bench_version=R.bench_version(b))))
    assert finished(b, str(tmp_path))
    (tmp_path / "summary.json").write_text("{")
    assert not finished(b, str(tmp_path))
