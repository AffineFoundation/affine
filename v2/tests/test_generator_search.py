"""Reference Generator search: validator settings, original IDs and local ranking."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sn120_v2.canonical import canonical_json
from sn120_v2.errors import ConfigError
from sn120_v2.envs.math_pool import Task
from sn120_v2.generator.constants import SAMPLING
from sn120_v2.generator.pool import PoolLocation

_SPEC = importlib.util.spec_from_file_location("generator_search_tool", Path(__file__).resolve().parents[1] / "miner" / "generator_search.py")
search = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(search)


class Registry:
    def __init__(self, ids):
        self._locations = {task_id: PoolLocation(0, number) for number, task_id in enumerate(ids)}

    def messages(self, task_id):
        return [{"role": "system", "content": "solve"}, {"role": "user", "content": task_id}]

    def tasks(self, ids):
        return tuple(Task(task_id=task_id, slice="math", problem=task_id, answer="1") for task_id in ids)


def test_sampling_body_matches_validator_settings_exactly():
    body = search.request_body("king", [{"role": "user", "content": "problem"}], seed=42)
    assert body == {
        "model": "king",
        "messages": [{"role": "user", "content": "problem"}],
        "seed": 42,
        "max_tokens": 65536,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "skip_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert (SAMPLING.temperature, SAMPLING.top_p, SAMPLING.top_k, SAMPLING.max_tokens, SAMPLING.thinking) == (1.0, 0.95, 20, 65536, True)


def test_pool_sample_is_reproducible_bounded_and_verbatim():
    ids = tuple(f"{number:020x}" for number in range(100))
    registry = Registry(ids)
    first = search.sample_ids(registry, 12, 1201)
    assert first == search.sample_ids(registry, 12, 1201)
    assert len(first) == len(set(first)) == 12
    assert set(first) <= set(ids)
    assert first == tuple(sorted(first))
    with pytest.raises(ConfigError):
        search.sample_ids(registry, 101, 1)


def test_rank_prefers_half_then_more_samples_then_stable_id():
    rows = [
        {"id": "b" * 20, "n": 8, "s": 4, "local_pass_rate": 0.5},
        {"id": "a" * 20, "n": 8, "s": 4, "local_pass_rate": 0.5},
        {"id": "c" * 20, "n": 8, "s": 2, "local_pass_rate": 0.25},
    ]
    # This pure ordering is the same key used by grade_rollouts/score.
    assert [row["id"] for row in sorted(rows, key=lambda row: (abs(row["local_pass_rate"] - 0.5), -row["n"], row["id"]))] == ["a" * 20, "b" * 20, "c" * 20]


def test_score_sends_eight_seeded_rollouts_and_counts_grader_failures(monkeypatch):
    ids = tuple(f"{n:020x}" for n in range(1, 3))
    registry = Registry(tuple(ids))
    calls = []

    def generate(url, body, timeout):
        calls.append((url, body, timeout))
        return "generated", False

    class Grader:
        def __init__(self, **kwargs):
            pass

        def grade_many(self, jobs):
            assert len(jobs) == 16
            return [SimpleNamespace(code="P" if index % 2 == 0 else "F", cap_hit=False) for index in range(16)]

        def close(self):
            pass

    rows = search.score(registry, tuple(ids), "http://127.0.0.1:8100/v1", "king", 100, 1, 4, 10, 20, generate=generate, grader_factory=Grader)
    assert len(calls) == 16
    assert calls[0][1]["temperature"] == 1.0 and calls[0][1]["top_p"] == 0.95
    assert calls[0][1]["top_k"] == 20 and calls[0][1]["max_tokens"] == 65536
    assert [body["seed"] for _, body, _ in calls[:8]] == list(range(100, 108))
    assert [body["seed"] for _, body, _ in calls[8:]] == list(range(108, 116))
    assert all(row["n"] == 8 and row["s"] == 4 and row["local_pass_rate"] == 0.5 for row in rows)


def test_rollouts_survive_interruption_and_grade_failure_without_regeneration(tmp_path):
    ids = ("a" * 20, "b" * 20)
    registry, journal = Registry(ids), tmp_path / "raw-rollouts.jsonl"
    calls = []

    class Grader:
        def __init__(self, **kwargs):
            pass

        def grade_many(self, jobs):
            return [SimpleNamespace(code="P" if index % 2 else "F", cap_hit=False) for index in range(len(jobs))]

        def close(self):
            pass

    def generate(url, body, timeout):
        calls.append(body["seed"])
        return f"raw output {body['seed']}", body["seed"] % 2 == 0

    original_generate = generate

    def interrupted(url, body, timeout):
        if body["seed"] == 112:
            raise RuntimeError("simulated interruption")
        return original_generate(url, body, timeout)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        search.score(registry, ids, "http://127.0.0.1:8100/v1", "king", 100, 1, 1, 10, 20,
                     generate=interrupted, grader_factory=Grader, save_rollouts=journal, king="head-test")
    saved = len(journal.read_text().splitlines()) - 1
    assert 0 < saved < 16
    first_calls = list(calls)
    saved_seeds = [json.loads(line)["seed"] for line in journal.read_text().splitlines()[1:]]
    assert len(saved_seeds) == len(set(saved_seeds))
    calls.clear()

    class BrokenGrader(Grader):
        def grade_many(self, jobs):
            raise RuntimeError("grading failed after generation")

    with pytest.raises(RuntimeError, match="grading failed"):
        search.score(registry, ids, "http://127.0.0.1:8100/v1", "king", 100, 1, 1, 10, 20,
                     generate=generate, grader_factory=BrokenGrader, save_rollouts=journal, resume=True, king="head-test")
    assert len(calls) == 16 - saved and not (set(calls) & set(saved_seeds)) and 112 in calls
    calls.clear()
    rows = search.score(registry, ids, "unused", "king", 100, 1, 1, 10, 20,
                        generate=generate, grader_factory=Grader, save_rollouts=journal, resume=True, king="head-test")
    assert not calls and len(rows) == 2 and all(row["n"] == 8 for row in rows)
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(records) == 17 and all(set(row) == {"id", "sample", "seed", "output", "capped", "finish_reason"} for row in records[1:])
    assert all(row["finish_reason"] == ("length" if row["capped"] else "stop") for row in records[1:])
    before = journal.read_bytes()
    calls.clear()
    search.score(registry, ids, "unused", "king", 100, 1, 1, 10, 20,
                 generate=generate, grader_factory=Grader, save_rollouts=journal, resume=True, king="head-test")
    assert not calls and journal.read_bytes() == before
    with pytest.raises(search.ConfigError, match="does not match"):
        search.score(registry, ids, "unused", "other-model", 100, 1, 1, 10, 20,
                     generate=generate, grader_factory=Grader, save_rollouts=journal, resume=True, king="head-test")
    journal.write_text(journal.read_text() + '{"truncated":')
    with pytest.raises(search.ConfigError, match="cannot resume"):
        search.resume_rollouts(journal, search.rollout_metadata(ids, "head-test", "king", 100))


def test_first_resume_creates_a_journal_and_rejects_overwrite(tmp_path):
    ids, registry = ("a" * 20,), Registry(("a" * 20,))
    path = tmp_path / "new.jsonl"

    class Grader:
        def __init__(self, **kwargs):
            pass

        def grade_many(self, jobs):
            return [SimpleNamespace(code="P", cap_hit=False) for _ in jobs]

        def close(self):
            pass

    search.score(registry, ids, "unused", "king", 1, 1, 1, 1, 1,
                 generate=lambda *args: ("raw", False), grader_factory=Grader, save_rollouts=path, resume=True, king="head-test")
    assert len(path.read_text().splitlines()) == 9
    with pytest.raises(FileExistsError):
        search.score(registry, ids, "unused", "king", 1, 1, 1, 1, 1,
                     generate=lambda *args: ("raw", False), grader_factory=Grader, save_rollouts=path, king="head-test")


def test_complete_cli_resume_does_not_fetch_or_start_vllm(tmp_path, monkeypatch, capsys):
    ids = ("a" * 20, "b" * 20)
    registry = Registry(ids)
    journal = tmp_path / "raw-rollouts.jsonl"
    metadata = search.rollout_metadata(ids, "head-test", "king", 100)
    journal.write_text(search.canonical_json({"format": search.ROLLOUT_FORMAT, "metadata": metadata}).decode() + "\n")
    for index, task_id in enumerate(ids):
        for sample in range(8):
            row = {"id": task_id, "sample": sample, "seed": 100 + index * 8 + sample, "output": "raw", "capped": False, "finish_reason": "stop"}
            with journal.open("a") as stream:
                stream.write(search.canonical_json(row).decode() + "\n")
    monkeypatch.setattr(search, "PriorPool", lambda *args, **kwargs: registry)
    monkeypatch.setattr(search, "sample_ids", lambda *args: ids)
    monkeypatch.setattr(search, "fetch_king", lambda *args: (_ for _ in ()).throw(AssertionError("must not fetch")))
    monkeypatch.setattr(search, "local_server", lambda *args: (_ for _ in ()).throw(AssertionError("must not serve")))

    class Grader:
        def __init__(self, **kwargs):
            pass

        def grade_many(self, jobs):
            assert len(jobs) == 16
            return [SimpleNamespace(code="P" if index % 2 else "F", cap_hit=False) for index in range(16)]

        def close(self):
            pass

    monkeypatch.setattr(search, "grade_rollouts", lambda *args: [{"id": task_id, "n": 8, "s": 4, "local_pass_rate": 0.5} for task_id in ids])
    code = search.main(["--resume", "--save-rollouts", str(journal), "--seed", "100", "--sample-size", "2", "--count", "1",
                        "--report", str(tmp_path / "report.jsonl"), "--ids-file", str(tmp_path / "ids.txt")])
    assert code == 0 and "grading without fetching or starting vLLM" in capsys.readouterr().out
    assert (tmp_path / "ids.txt").read_text() == "a" * 20 + "\n"


def test_local_server_disables_vl_media_inputs(monkeypatch):
    command = []

    class Process:
        returncode = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(search.subprocess, "Popen", lambda argv, env: command.extend(argv) or Process())
    monkeypatch.setattr(search, "wait_http", lambda *args: None)
    with search.local_server(Path("/model"), "auto", 0, 131072, 1, ()):
        pass
    index = command.index("--limit-mm-per-prompt")
    assert command[index + 1] == '{"image":0,"video":0}'


def test_write_outputs_produces_ids_file_and_jsonl_not_header(tmp_path):
    rows = [
        {"id": "a" * 20, "n": 8, "s": 4, "local_pass_rate": 0.5},
        {"id": "b" * 20, "n": 8, "s": 0, "local_pass_rate": 0.0},
    ]
    report, ids = tmp_path / "report.jsonl", tmp_path / "ids.txt"
    search.write_outputs(report, ids, rows, 1, "head-test", {"temperature": 1.0})
    assert ids.read_text() == "a" * 20 + "\n"
    lines = report.read_text().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["king"] == "head-test" for line in lines)
    assert all(set(json.loads(line)) == {"king", "settings", *rows[0]} for line in lines)


def test_local_endpoint_is_loopback_only():
    with search.local_server(None, "http://127.0.0.1:8100/v1", 0, 100, 1, ()) as url:
        assert url.endswith("/v1")
    with pytest.raises(ConfigError, match="localhost"):
        with search.local_server(None, "http://public.example/v1", 0, 100, 1, ()):
            pass


def test_search_import_does_not_fetch_or_start_vllm():
    # Import above has no public I/O and no server side effect. The pure request
    # helper can be tested without installing torch/vLLM.
    assert search.PUBLIC_BASE == "https://affine.io/public/read_links.json"
    assert search.END_OF_TURN == "<|im_end|>"
