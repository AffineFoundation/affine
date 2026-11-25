"""Option A public math environments: offline pinned-loader and grader tests."""
import asyncio
import importlib
import sys
import types
from pathlib import Path

import pytest

from affine_math_grader import MathContestConfig, MathContestTaskset, check, implicit_mult
from affine_math_grader.taskset import INSTRUCTION, MathContestTask

ROOT = Path(__file__).parents[1] / "environments"
CASES = [
    ("hmmt_feb_2025", "HMMTFeb2025Taskset", "MathArena/hmmt_feb_2025", "train", "6fdc4277120810ff75aa22d2d5489b91f7a262a1", 30, True),
    ("hmmt_nov_2025", "HMMTNov2025Taskset", "MathArena/hmmt_nov_2025", "train", "118dbfb45c4c9467c672268ed55166642897aa46", 30, True),
    ("hmmt_feb_2026", "HMMTFeb2026Taskset", "MathArena/hmmt_feb_2026", "train", "589a53fe152febc0f4f99f3e561007bf8da8e664", 33, True),
    ("brumo_2025", "BRUMO2025Taskset", "MathArena/brumo_2025", "train", "6de9bc0987f5710c6376ee980e1483dbeb55cfda", 30, True),
    ("cmimc_2025", "CMIMC2025Taskset", "MathArena/cmimc_2025", "train", "7b3ca7c4b417d31b1f97f781f39fc4f97f1b9185", 40, True),
    ("beyondaime", "BeyondAIMETaskset", "ByteDance-Seed/BeyondAIME", "test", "c705198ae1043810b1e1693bd879250b51a7a523", 100, False),
]


@pytest.fixture(autouse=True)
def environments(monkeypatch):
    """Load repo-local packages without downloading datasets or installing each env."""
    for path in (ROOT / "affine_math_grader", *(ROOT / case[0] for case in CASES)):
        monkeypatch.syspath_prepend(str(path))


@pytest.mark.parametrize("name,class_name,repo,split,revision,count,sorted_by_index", CASES)
def test_taskset_loads_pinned_public_dataset(monkeypatch, name, class_name, repo, split, revision, count, sorted_by_index):
    rows = [{"problem": f"problem {i}", "answer": i if name == "beyondaime" else f"answer {i}",
             "problem_idx": count - i} for i in range(count)]
    calls = []
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: (calls.append((args, kwargs)), rows)[1]
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    module = importlib.import_module(name)
    taskset = getattr(module, class_name)()
    tasks = list(taskset.load())

    assert calls == [((repo,), {"split": split, "revision": revision})]
    assert len(tasks) == count
    assert taskset.system_prompt is None and taskset.config.task.math_verify_timeout == 5
    assert [task.data.idx for task in tasks] == list(range(count))
    assert all(task.data.system_prompt is None for task in tasks)
    expected = sorted(rows, key=lambda row: row["problem_idx"]) if sorted_by_index else rows
    assert [task.data.prompt for task in tasks] == [INSTRUCTION + row["problem"] for row in expected]
    assert [task.data.answer for task in tasks] == [str(row["answer"]) for row in expected]
    if sorted_by_index:
        loaded = {row["problem"]: row["problem_idx"] for row in rows}
        assert [loaded[task.data.prompt.removeprefix(INSTRUCTION)] for task in tasks] == list(range(1, count + 1))
    else:
        assert [task.data.prompt.removeprefix(INSTRUCTION) for task in tasks] == [row["problem"] for row in rows]
        assert [row["problem_idx"] for row in rows] == list(range(count, 0, -1))  # parquet order, no index sort


@pytest.mark.parametrize("name,class_name,repo,split,revision,count,sorted_by_index", CASES)
def test_taskset_refuses_changed_pinned_count(monkeypatch, name, class_name, repo, split, revision, count, sorted_by_index):
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: [{"problem": "1+1", "answer": 2, "problem_idx": 1}] * (count - 1)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    taskset = getattr(importlib.import_module(name), class_name)()
    with pytest.raises(ValueError, match=f"has {count - 1} rows, expected {count}"):
        list(taskset.load())


def test_math_grader_direct_and_symbolic_retry(monkeypatch):
    calls = []

    def verify(reply, answer, *, timeout_seconds):
        calls.append((reply, answer, timeout_seconds))
        return 1.0 if answer == "2x y" and reply == "\\boxed{2x y}" else 0.0

    monkeypatch.setattr("affine_math_grader.taskset.vf.verify_boxed_math_answer", verify)
    reply = "<think>work</think>\nThe answer is \\boxed{2xy}"
    assert check(reply, "2xy", timeout=5) == 1.0
    assert calls == [(reply, "2xy", 5), ("\\boxed{2x y}", "2x y", 5)]
    assert implicit_mult(r"\ln(2dr)") == r"\ln(2\vartheta  r)"
    assert implicit_mult("d r") == r"\vartheta  r"


def test_math_grader_calls_verifier_with_five_second_timeout_and_no_retry_when_direct(monkeypatch):
    calls = []
    monkeypatch.setattr("affine_math_grader.taskset.vf.verify_boxed_math_answer",
                        lambda reply, answer, *, timeout_seconds: (calls.append((reply, answer, timeout_seconds)), 1.0)[1])
    reply = "<think>work</think>\\boxed{\\frac{13}{32}}"
    assert check(reply, r"\frac{13}{32}") == 1.0
    assert calls == [(reply, r"\frac{13}{32}", 5)]
    task = MathContestTask(types.SimpleNamespace(answer="13/32"), MathContestConfig().task)
    assert asyncio.run(task.correct(types.SimpleNamespace(last_reply=reply))) == 1.0
    assert calls[-1][-1] == 5


def test_math_grader_base_class_requires_dataset_metadata():
    with pytest.raises(AttributeError):
        list(MathContestTaskset().load())
