"""Miner check/reproduce support for staged affine-batch-v4 files."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sn120_v2.worker import LocalRunner
from test_miner_tools import Box, draw

_SPEC = importlib.util.spec_from_file_location("affine_miner_v4_tool", Path(__file__).resolve().parents[1] / "miner" / "affine_miner.py")
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)

HOTKEY = "hk1"
THINK = "step " * 40


def batch() -> bytes:
    return json.dumps({
        "version": "affine-batch-v4", "hotkey": HOTKEY, "train": {"steps": 1},
        "stages": [
            {"train": {"lora_lr": 0.00002}, "groups": [{"prompt": "first prompt", "rollouts": [
                {"output": THINK + "\\boxed{1}", "reward": 1.0},
                {"output": THINK + "\\boxed{0}", "reward": 0.0},
            ]}]},
            {"sft": [{"prompt": "second prompt", "output": THINK + "\\boxed{1}"}]},
        ],
    }).encode()


def test_check_and_reproduce_keep_the_admitted_stage_plan(build):
    report = tool.check(batch(), HOTKEY, build.params())
    assert report["format"] == "affine-batch-v4"
    assert (report["n_prompts"], report["n_rollouts"]) == (2, 3)
    box = Box(build)
    runner = LocalRunner(box, box, lambda: 7, box.image)
    reproduced = tool.reproduce(batch(), HOTKEY, build.params(), build.head(), draw(), box, runner)
    assert reproduced["status"] == "winner"
    request = box.requests[0]
    stages = request.entry.batch.stages
    assert len(stages) == 2 and stages[0]["train"] == {"lora_lr": 0.00002}
    assert all(set(stage) == {"train"} for stage in stages)
    assert "sft" not in request.entry.batch.to_json().decode()
    assert box[request.entry.batch.uri] == batch()


def test_null_cache_disabled_is_independent_of_v4_contents(build, tmp_path):
    box = Box(build)
    runner = LocalRunner(box, box, lambda: 7, box.image)
    assert tool.cached_null_runner(runner, None, build.head(), build.params(), draw()) is runner
    assert list(tmp_path.iterdir()) == []


def test_v4_example_matches_the_miner_grammar():
    example = json.loads((Path(__file__).parents[1] / "examples" / "affine-batch-v4.json").read_text())
    body = json.loads(batch())
    assert example["version"] == body["version"]
    assert set(example) == set(body)
    assert set(example["stages"][0]["train"]) <= {"lora_lr", "adam_eps", "steps", "minibatches", "group_size", "sample_temperature", "sft_weight"}
    assert set(example["stages"][1]["sft"][0]) == {"prompt", "output"}
