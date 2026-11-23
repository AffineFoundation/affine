"""Public full rollouts from verifiers traces, including explicit lost samples."""
import gzip
import hashlib
import json

from affine_bench import registry as R
from affine_bench.node_run import SAMPLING_FIELDS, last_boxed, rollout_rows, write_rollouts

B = dict(id="t", env="t", dataset="public/test", rollouts=4)
SP = dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0,
          max_tokens=1024, enable_thinking=True, profile="test", max_model_len=2048, source="not-public", budget=1024)


def trace(score, text, finish="stop", tokens=20):
    return dict(rewards={"correct": {"score": score}}, calls=[dict(message={"content": text}, finish_reason=finish)], num_output_tokens=tokens)


def test_extracts_all_grades_prompt_reasoning_and_last_balanced_box(tmp_path):
    rows = [
        dict(task={"data": {"idx": 0, "prompt": [{"role": "user", "content": "2+2"}], "answer": "4"}, "key": "t0"}, ok=True,
             traces=[trace(1, "<think>work</think>First \\boxed{3}; final \\boxed{4}")]),
        dict(task={"data": {"idx": 0, "prompt": [{"role": "user", "content": "2+2"}], "answer": "4"}, "key": "t0"}, ok=True,
             traces=[trace(0, "wrong \\boxed{5}")]),
        dict(task={"data": {"idx": 0, "prompt": [{"role": "user", "content": "2+2"}], "answer": "4"}, "key": "t0"}, ok=True,
             traces=[trace(0, "unfinished \\boxed{4}", finish="length")]),
        dict(task={"data": {"idx": 0, "prompt": [{"role": "user", "content": "2+2"}], "answer": "4"}, "key": "t0"}, ok=False, traces=[]),
    ]
    path = tmp_path / "vf" / "run"
    path.mkdir(parents=True)
    (path / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    found = list(rollout_rows(B, str(path / "traces.jsonl"), "model", SP))
    assert [row["grade"] for row in found] == ["pass", "fail", "cap", "error"]
    assert [row["sample_index"] for row in found] == [0, 1, 2, 3]
    assert found[0]["prompt"] == [{"role": "user", "content": "2+2"}] and found[0]["gold_answer"] == "4"
    assert found[0]["reasoning_text"] == "work" and found[0]["completion_text"] == "First \\boxed{3}; final \\boxed{4}"
    assert found[0]["extracted_answer"] == "\\boxed{4}" and found[0]["reward"] == 1
    assert found[2]["capped"] and not found[2]["errored"] and found[2]["output_tokens"] == 20
    assert found[3]["errored"] and found[3]["reward"] == 0 and found[3]["completion_text"] == ""
    assert set(found[0]["sampling"]) == set(SAMPLING_FIELDS) and "source" not in found[0]["sampling"]
    assert last_boxed("no box") is None and last_boxed("\\boxed{1}") == "\\boxed{1}"
    assert last_boxed("\\boxed{1} </think> \\boxed{x+{y}} trailing \\boxed{3}") == "\\boxed{3}"
    assert last_boxed("\\boxed{unclosed") is None


def test_resumed_run_takes_latest_k_attempts_and_fills_a_missing_expected_row(tmp_path):
    path = tmp_path / "vf" / "run"
    path.mkdir(parents=True)
    rows = [dict(task={"data": {"idx": 0, "answer": "4"}, "key": "0"}, ok=False, traces=[]),
            dict(task={"data": {"idx": 0, "answer": "4"}, "key": "0"}, ok=True, traces=[trace(1, "\\boxed{4}")])]
    (path / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    found = list(rollout_rows(B, str(path / "traces.jsonl"), "model", SP))
    assert len(found) == 4 and found[0]["grade"] == "pass"
    assert [row["errored"] for row in found[1:]] == [True, True, True]


def test_gzip_is_deterministic_and_has_no_run_path(tmp_path):
    source = tmp_path / "vf" / "run"
    source.mkdir(parents=True)
    (source / "traces.jsonl").write_text(json.dumps(dict(task={"data": {"idx": 0, "answer": "4"}, "key": "0"}, ok=True,
                                                           traces=[trace(1, "<think>x</think>\\boxed{4}")])) + "\n")
    b = B | {"rollouts": 1}
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir(); second.mkdir()
    write_rollouts(b, str(tmp_path / "vf"), str(first), "model", SP)
    write_rollouts(b, str(tmp_path / "vf"), str(second), "model", SP)
    data = (first / "rollouts.jsonl.gz").read_bytes()
    assert data == (second / "rollouts.jsonl.gz").read_bytes()
    assert data[4:8] == b"\0\0\0\0" and str(tmp_path).encode() not in data
    assert hashlib.sha256(data).hexdigest() == hashlib.sha256(data).hexdigest()
    assert json.loads(gzip.decompress(data))["extracted_answer"] == "\\boxed{4}"
