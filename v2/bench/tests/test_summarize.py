"""node_run.summarize: an errored rollout is a failed sample, never dropped (2026-11-04 MiMo-7B-RL / Phi-4-reasoning)."""
import json
from affine_bench.node_run import summarize

B = dict(id="t", env="t", rollouts=4)


def row(idx, ok=True, score=1.0, fin="stop", tok=10):
    tr = [dict(rewards={"r": {"score": score}}, calls=[{"finish_reason": fin}], num_output_tokens=tok)] if ok else []
    return dict(task={"data": {"idx": idx}, "key": str(idx)}, ok=ok, traces=tr)


def run(tmp_path, rows):
    d = tmp_path / "vf" / "run1"; d.mkdir(parents=True)
    (d / "traces.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return summarize(B, str(tmp_path / "vf"))


def test_errors_count_as_failures(tmp_path):
    # task 0: 2 correct + 2 errored -> 0.5 (old code: 1.0); task 1: 4 wrong -> 0
    s, per = run(tmp_path, [row(0), row(0), row(0, ok=False), row(0, ok=False)] + [row(1, score=0.0)] * 4)
    assert s["score"] == 0.25 and s["errors"] == 2 and s["errored_tasks"] == 1 and not s["complete"]
    assert s["samples"] == 6 and s["samples_expected"] == 8 and per[0]["errors"] == 2


def test_all_errored_task_is_zero_not_dropped(tmp_path):
    s, _ = run(tmp_path, [row(0)] * 4 + [row(1, ok=False)] * 4)
    assert s["score"] == 0.5 and s["tasks"] == 2 and s["errors"] == 4


def test_resumed_duplicates_capped_and_clean(tmp_path):
    # a resumed run re-ran the failed rollout: the failed row stays in the file, the retry succeeds -> complete
    s, _ = run(tmp_path, [row(0)] * 3 + [row(0, ok=False), row(0, fin="length", score=0.0)] + [row(0)] * 2)
    assert s["complete"] and s["errors"] == 0 and s["score"] == 0.75 and s["samples"] == 4 and s["cap_rate"] == 0.25
