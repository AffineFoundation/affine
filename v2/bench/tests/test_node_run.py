"""node_run: what counts as done, and which run may go on with --resume (2026-11-14: ArXivMath went from 16 samples to 2)."""
import json, os
from affine_bench import registry as R
from affine_bench.node_run import finished, resumable, summarize

ARXIV = R.benchmark("arxivmath")
AIME = R.benchmark("aime25")


def saved_run(vf, rollouts, name="run1", rows=(), version=None):
    d = vf / name / "configs" / "resolved"; d.mkdir(parents=True)
    (d / "eval.json").write_text(json.dumps(dict(env="arxivmath", num_rollouts=rollouts)))
    (vf / name / "traces.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if version:
        (vf / "bench_version").write_text(version)
    return str(d / "eval.json")


def row(idx, score):
    tr = [dict(rewards={"r": {"score": score}}, calls=[{"finish_reason": "stop"}], num_output_tokens=10)]
    return dict(task={"data": {"idx": idx}, "key": str(idx)}, ok=True, traces=tr)


def test_arxivmath_has_two_samples_and_a_new_version():
    assert ARXIV["rollouts"] == 2 and R.bench_version(ARXIV).startswith("arxivmath/v2/")
    assert all(x.get("version", 1) == 1 for x in R.benchmarks() if x["id"] != "arxivmath")


def test_a_run_of_this_count_and_version_goes_on(tmp_path):
    vf = tmp_path / "vf"; saved = saved_run(vf, 2, version=R.bench_version(ARXIV))
    assert resumable(str(vf), ARXIV) == saved and os.listdir(tmp_path) == ["vf"]


def test_a_run_from_before_the_version_file_goes_on_at_version_1(tmp_path):
    # the runs that were on the node on 2026-11-14 have no vf/bench_version: every benchmark had version 1 then
    vf = tmp_path / "vf"; saved = saved_run(vf, 16)
    assert AIME.get("version", 1) == 1 and resumable(str(vf), AIME) == saved and os.listdir(tmp_path) == ["vf"]


def test_a_run_with_another_count_is_moved_aside_and_not_scored(tmp_path):
    # the run at 16 was stopped: problem 0 has 3 rows, the two that ended first are right; problem 1 has none
    vf = tmp_path / "vf"; saved_run(vf, 16, rows=[row(0, 1.0), row(0, 1.0), row(0, 0.0)])
    assert summarize(ARXIV, str(vf))[0]["score"] == 1.0  # what a cut of that file to 2 would say: the shortest answers
    assert resumable(str(vf), ARXIV) is None
    assert sorted(os.listdir(tmp_path)) == ["vf", "vf.old"] and os.listdir(vf) == []
    assert os.path.exists(tmp_path / "vf.old" / "run1" / "traces.jsonl")
    assert summarize(ARXIV, str(vf)) is None
    saved_run(vf, 16, name="run2")
    assert resumable(str(vf), ARXIV) is None and sorted(os.listdir(tmp_path)) == ["vf", "vf.old", "vf.old.2"]


def test_a_run_of_another_version_with_the_same_count_is_moved_aside(tmp_path):
    vf = tmp_path / "vf"; saved_run(vf, 2, version=R.bench_version(dict(ARXIV, version=1)))
    assert resumable(str(vf), ARXIV) is None and sorted(os.listdir(tmp_path)) == ["vf", "vf.old"]
    assert os.path.exists(tmp_path / "vf.old" / "bench_version")
    saved_run(vf, 2)  # no version file: a run of version 1
    assert resumable(str(vf), ARXIV) is None and sorted(os.listdir(tmp_path)) == ["vf", "vf.old", "vf.old.2"]


def test_a_saved_config_that_cannot_be_read_is_not_resumed(tmp_path):
    vf = tmp_path / "vf"; saved = saved_run(vf, 2, version=R.bench_version(ARXIV)); open(saved, "w").write("{")
    assert resumable(str(vf), ARXIV) is None and sorted(os.listdir(tmp_path)) == ["vf", "vf.old"]


def test_no_saved_run(tmp_path):
    (tmp_path / "vf").mkdir()
    assert resumable(str(tmp_path / "vf"), ARXIV) is None and os.listdir(tmp_path) == ["vf"]


def test_a_summary_of_another_version_is_not_done(tmp_path):
    assert not finished(ARXIV, str(tmp_path))
    (tmp_path / "summary.json").write_text(json.dumps(dict(bench_version=R.bench_version(dict(ARXIV, version=1)))))
    assert not finished(ARXIV, str(tmp_path))
    (tmp_path / "summary.json").write_text(json.dumps(dict(bench_version=R.bench_version(ARXIV))))
    assert finished(ARXIV, str(tmp_path))
    (tmp_path / "summary.json").write_text("{")
    assert not finished(ARXIV, str(tmp_path))
