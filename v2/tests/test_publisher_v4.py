"""Public affine-batch-v4 documentation and the crowned-only batch privacy boundary."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sn120_v2 import publisher, public, sim
from sn120_v2.admission import FORMAT_V4, STAGE_TRAIN_KEYS
from sn120_v2.errors import ContractError
from sn120_v2.params import DEFAULT_MINER_BOUNDS, SFT_MINER_BOUND, Recipe


def snapshot(tmp_path: Path, **changes):
    params = sim.sim_params(**changes)
    return public.snapshot(sim.SimWorld(tmp_path / "world", params).controller.state, params, 1_800_000_000)


def render(tmp_path: Path, **changes):
    mining, llms = publisher.render(snapshot(tmp_path, **changes), "https://files.example")
    return json.loads(mining), llms.decode()


def flat(text: str) -> str:
    return " ".join(text.split())


def test_v4_documentation_matches_admission_and_resolved_stage_cost(tmp_path):
    doc, text = render(tmp_path)
    one_line = flat(text)
    assert doc["contract"]["batch_file"]["v4"] == {
        "format": FORMAT_V4,
        "max_stages": 5,
        "top_level_keys": ["version", "hotkey", "stages", "train"],
        "stage_keys": ["train", "groups", "sft"],
        "stage_train_keys": sorted(STAGE_TRAIN_KEYS),
        "steps": "epochs (full passes) over the stage",
        "cost": "sum over stages of steps * trained_tokens, including GRPO and SFT",
        "sft": "official verification requires a known prompt with official reward exactly 1",
    }
    assert doc["contract"]["training_time_floor"] is None
    for phrase in (
        "an ordered list of 1 to 5 stages (`max_stages`)",
        "`steps` means epochs: full passes over that stage's trainable rows, not gradient-update counts",
        "Per-stage cost is `steps * trained_tokens`",
        "file cost is the sum of this value over all stages",
        "SFT trained tokens count in the same stage total",
        "Official verification requires a prompt known to the official grader and an output whose official reward is exactly 1",
        "This recipe publishes no `sft_weight` MinerBound",
        "a miner may not set that key at either level until the bound is explicitly enabled",
        "Waiting and losing seats reveal no batch content, stage counts, SFT counts, stage train values or SFT verification details",
    ):
        assert phrase in one_line
    assert publisher.check_pair(json.dumps(doc).encode(), text.encode()) == []


def test_v4_example_has_stages_and_sft_and_preserves_flat_example(tmp_path):
    doc, text = render(tmp_path)
    examples = [json.loads(block) for block in text.split("```json\n")[1:] for block in [block.split("```", 1)[0]]]
    legacy, v4 = examples[:2]
    assert legacy["version"] == "affine-batch-v2" and set(legacy) == {"version", "hotkey", "train", "groups"}
    assert v4["version"] == FORMAT_V4 and set(v4) == {"version", "hotkey", "stages"}
    assert set(v4["stages"][0]) == {"train", "groups", "sft"}
    assert v4["stages"][0]["train"] == {"steps": doc["params"]["recipe"].get("steps", 1)}
    assert set(v4["stages"][0]["sft"][0]) == {"prompt", "output"}
    assert "reward" not in v4["stages"][0]["sft"][0]


def test_sft_weight_is_documented_only_when_its_bound_is_opted_in(tmp_path):
    recipe = Recipe(image_digests=(sim.SIM_IMAGE,), miner_bounds=DEFAULT_MINER_BOUNDS + (SFT_MINER_BOUND,))
    doc, text = render(tmp_path, recipe=recipe)
    rows = {row["key"]: row for row in doc["contract"]["miner_settable"]}
    assert rows["sft_weight"]["default"] == 1
    assert rows["sft_weight"]["low"] == 0 and rows["sft_weight"]["high"] == 1
    assert "This recipe publishes no `sft_weight` MinerBound" not in text
    assert "Set it in top-level `train`, or override it in a stage's `train`" in flat(text)
    assert json.loads(text.split("```json\n")[2].split("```", 1)[0])["stages"][0]["train"]["sft_weight"] == 1


def test_enabled_training_time_floor_is_published_in_both_files(tmp_path):
    params = dict(max_train_token_steps=16_000_000, train_tokens_per_s_floor=100_000, train_time_budget_s=760)
    doc, text = render(tmp_path, **params)
    floor = doc["contract"]["training_time_floor"]
    assert floor == {
        "enabled": True,
        "train_tokens_per_s_floor": 100_000,
        "minimum_seconds": 160,
        "post_training_reserve_s": 600,
        "train_time_budget_s": 760,
    }
    one_line = flat(text)
    assert "Training-time floor enabled: `train_tokens_per_s_floor=100000`" in one_line
    assert "ceil(max_train_token_steps / train_tokens_per_s_floor) = 160 seconds plus 600 seconds" in one_line
    assert "scheduling/reservation floor, not a throughput requirement for miners" in one_line
    assert publisher.check_pair(json.dumps(doc).encode(), text.encode()) == []


def test_changelog_is_date_only_utc_and_v4(tmp_path):
    _, text = render(tmp_path)
    entries = [line for line in publisher.RULE_CHANGELOG if line.startswith("- ")]
    assert all(line.startswith("- 2026-") for line in entries)
    assert all("UTC-" not in line and "T" not in line.split(" / ", 1)[0] for line in entries)
    assert "- 2026-11-25 / `affine-batch-v4`" in text


@pytest.mark.parametrize("private", [
    {"stages": [{"train": {"steps": 2}, "groups": [], "sft": []}]},
    {"stage_count": 2},
    {"sft_count": 1},
    {"n_stages": 2},
    {"n_sft": 1},
    {"sft": [{"prompt": "private prompt", "output": "private output"}]},
    {"groups": [{"prompt": "private prompt", "rollouts": []}]},
])
def test_v4_stage_plans_and_content_are_refused_not_published(tmp_path, private):
    original = snapshot(tmp_path)
    original["round"] = {"entries": [{"uid": 7, "hotkey": "waiting-or-losing", **private}]}
    with pytest.raises(ContractError):
        publisher.render(original)
    doc, _ = render(tmp_path / "clean")
    # Public documentation can describe ``stages`` without opening the private state path.
    assert "stages" in doc["contract"]["batch_file"]["v4"]["top_level_keys"]
    assert "official verification" in doc["contract"]["batch_file"]["v4"]["sft"]


def test_crowned_winner_public_batch_may_have_stages_but_other_seats_may_not(tmp_path):
    made = sim.SimWorld(tmp_path / "world", sim.sim_params())
    made.register(11, 0.4)
    made.register(12, 0.0)
    made.register(13, None)
    made.round()
    snap = copy.deepcopy(made.publisher.snapshots[-1])
    # The exact batch file has its own crowned-winner object key, separate from mining.json.
    public_batch = {"version": FORMAT_V4, "stages": [{"groups": [], "sft": []}]}
    assert public_batch["stages"]

    losing = next(entry for entry in snap["history"][0]["entries"] if entry["status"] != "winner")
    losing["stages"] = [{"sft": [{"prompt": "losing prompt", "output": "losing output"}]}]
    with pytest.raises(ContractError):
        publisher.mining_document({key: value for key, value in snap.items() if key != "winner_stats"})
    del losing["stages"]

    # Existing public views are also checked directly: they do not project stage/SFT counts.
    made.controller.state.round = None
    current = public.snapshot(made.controller.state, made.params, 1_800_000_100)
    doc = publisher.mining_document(current)
    rows = [*doc["queue"]["next"], *doc["queue"]["line"], *doc["queue"]["files"]["verdicts"]]
    rows += [entry for record in doc["history"] for entry in record["entries"] if entry.get("status") != "winner"]
    forbidden = {"stages", "sft", "groups", "rollouts", "stage_count", "sft_count", "n_stages", "n_sft", "train"}
    assert rows and all(not (set(row) & forbidden) for row in rows)
