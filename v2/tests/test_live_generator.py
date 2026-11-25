"""Reference rollout generation: live pool selection without network or GPU imports."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("reference_live_generator", ROOT / "miner" / "kit" / "generate.py")


@pytest.fixture
def generator(monkeypatch):
    """Install only the kit/helper surfaces used by generate.py."""
    prompts = SimpleNamespace(
        batch_prompt=lambda problem: "MANDATE\n\n" + problem,
        batch_ids=lambda tok, prompt: ["ids", prompt],
        pool=lambda: pytest.fail("the default live pool must not load the legacy local pool"),
    )
    sampler = SimpleNamespace(
        EOS=(99, 98),
        request_seed=lambda plan, env, position, sample: f"{plan}:{env}:{position}:{sample}",
        sample_all=lambda *args, **kwargs: pytest.fail("async sample_all must be replaced by run() tests"),
    )
    grader = SimpleNamespace(
        init=lambda head: None,
        _G={"tok": None},
        grade=lambda out, stop, answer, problem: ("positive", "mocked"),
    )
    live = SimpleNamespace(active_tasks=lambda **kwargs: pytest.fail("active_tasks must be replaced by a fixture"))
    for name, module in {"prompts": prompts, "sampler": sampler, "grade": grader, "live_pool": live}.items():
        monkeypatch.setitem(sys.modules, name, module)
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def args(generator, *extra):
    return generator.parse_args(["--head", "unused", "--url", "unused", "--out", "unused", *extra])


def test_live_is_default_and_selects_every_active_id(generator, monkeypatch):
    tasks = [
        {"task_id": "math:8", "problem": "base problem", "answer": "8", "prompt": "base batch prompt"},
        {"task_id": "gen:alpha", "problem": "new problem", "answer": "42", "prompt": "new batch prompt"},
        {"task_id": "gen:omega", "problem": "another problem", "answer": "7", "prompt": "another batch prompt"},
    ]
    calls = []

    def active_tasks(**kwargs):
        calls.append(kwargs)
        yield from tasks

    monkeypatch.setattr(generator.live_pool, "active_tasks", active_tasks)
    parsed = args(generator)
    assert (parsed.pool, parsed.public_url, parsed.cache, parsed.tasks) == ("live", "https://affine.io", None, None)
    selected = generator.select_pool(parsed)
    assert [(position, row["task_id"]) for position, row, _ in selected] == list(enumerate(task["task_id"] for task in tasks))
    assert calls == [{"public_url": "https://affine.io", "cache": None}]
    assert [row[2] for row in selected] == [task["prompt"] for task in tasks]


def test_live_selection_accepts_exact_string_ids_and_positional_ranges(generator):
    tasks = [{"task_id": task_id, "problem": task_id, "answer": "a", "prompt": task_id} for task_id in ("math:8", "gen:alpha", "gen:omega", "math:40")]
    selected = generator.select_tasks(tasks, "gen:omega,0,2-3")
    assert [(position, task["task_id"]) for position, task in selected] == [(2, "gen:omega"), (0, "math:8"), (2, "gen:omega"), (3, "math:40")]
    with pytest.raises(ValueError, match="outside the active list"):
        generator.select_tasks(tasks, "9")


def test_run_emits_live_ids_but_seeds_by_stable_position_and_preserves_prompt_and_grade(generator, monkeypatch, tmp_path):
    tasks = [
        {"task_id": "math:9000", "problem": "base problem", "answer": "base answer", "prompt": "EXACT BASE BATCH PROMPT"},
        {"task_id": "gen:string-id", "problem": "reserve problem", "answer": "reserve answer", "prompt": "EXACT RESERVE BATCH PROMPT"},
    ]
    rendered, graded, requests = [], [], []

    class Tokenizer:
        def decode(self, ids, skip_special_tokens):
            return "decoded answer"

        def __call__(self, text, add_special_tokens):
            return {"input_ids": [10]}

    monkeypatch.setattr(generator.live_pool, "active_tasks", lambda **kwargs: iter(tasks))
    monkeypatch.setattr(generator.prompts, "batch_ids", lambda tok, prompt: rendered.append(prompt) or [10])
    monkeypatch.setattr(generator.grade, "init", lambda head: generator.grade._G.update(tok=Tokenizer()))

    def request_seed(seed, env, position, sample):
        requests.append((seed, env, position, sample))
        assert type(position) is int
        return position * 10 + sample

    async def sample_all(urls, jobs, **kwargs):
        assert [job[2] for job in jobs] == [0, 10, 0]
        return {(key, 0): ([10, 99], "eos") for key in range(3)}

    monkeypatch.setattr(generator.sampler, "request_seed", request_seed)
    monkeypatch.setattr(generator.sampler, "sample_all", sample_all)
    monkeypatch.setattr(generator.grade, "grade", lambda out, stop, answer, problem: graded.append((answer, problem)) or ("positive", "mocked"))
    output = tmp_path / "rollouts.jsonl"
    parsed = args(generator, "--seed", "plan", "--tasks", "math:9000,1,0", "--k", "1", "--out", str(output))
    generator.run(parsed)

    assert rendered == ["EXACT BASE BATCH PROMPT", "EXACT RESERVE BATCH PROMPT", "EXACT BASE BATCH PROMPT"]
    assert requests == [("plan", "miner", 0, 0), ("plan", "miner", 1, 0), ("plan", "miner", 0, 0)]
    assert graded == [("base answer", "base problem"), ("reserve answer", "reserve problem"), ("base answer", "base problem")]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["task"] for row in rows] == ["math:9000", "gen:string-id", "math:9000"]
    assert [row["prompt"] for row in rows] == ["EXACT BASE BATCH PROMPT", "EXACT RESERVE BATCH PROMPT", "EXACT BASE BATCH PROMPT"]
    assert [row["answer"] for row in rows] == ["base answer", "reserve answer", "base answer"]
    assert rows[1]["rollouts"] == [{"output": "decoded answer", "tokens": 2, "stop": "eos", "kind": "positive", "how": "mocked", "reencode_ok": True}]


def test_local_is_explicit_with_legacy_defaults_and_optional_jsonl_path(generator, monkeypatch, tmp_path):
    monkeypatch.setattr(generator.live_pool, "active_tasks", lambda **kwargs: pytest.fail("explicit local mode must not access the live pool"))
    legacy = [{"prompt": "first local problem", "answer": "1"}, {"prompt": "second local problem", "answer": "2"}]
    monkeypatch.setattr(generator.prompts, "pool", lambda: legacy)
    selected = generator.select_pool(args(generator, "--pool", "local"))
    assert [(position, row["task_id"], prompt) for position, row, prompt in selected] == [
        (0, 0, "MANDATE\n\nfirst local problem"),
        (1, 1, "MANDATE\n\nsecond local problem"),
    ]

    path = tmp_path / "pool.jsonl"
    path.write_text('{"task_id":"custom:string","problem":"custom problem","answer":"3"}\n')
    selected = generator.select_pool(args(generator, "--pool", "local", "--local-pool", str(path)))
    assert selected == [(0, {"task_id": "custom:string", "problem": "custom problem", "answer": "3", "prompt": None}, "MANDATE\n\ncustom problem")]
    with pytest.raises(SystemExit):
        args(generator, "--local-pool", str(path))
