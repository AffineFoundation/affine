from __future__ import annotations

import dataclasses
import json

import pytest

from sn120_v2.errors import ConfigError, ContractError
from sn120_v2.params import (
    GENESIS_BASE_HASH,
    GENESIS_MODEL_REPO,
    GENESIS_MODEL_REVISION,
    GenesisSpec,
    Params,
    Recipe,
    load_params,
    require_env,
)

SPEC_DEFAULTS = {
    "netuid": 120,
    "env": "math",
    "batch_cap": 32,
    "max_loras_per_box": 32,
    "n_min_start": 1,
    "n_gpus": 8,
    "samples_per_task": 1,
    "looks": (0.25, 0.5, 1.0),
    "max_output_tokens": 65536,
    "max_context_tokens": 69632,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "dtype": "bfloat16",
    "z_bar": 1.0,
    "cap_hit_floor": 0.33,
    "cap_hit_margin": 0.02,
    "fold_every_k": 1,
    "max_batch_bytes": 64 * 1024 * 1024,
    "max_prompts": 250,
    "max_rollouts": 2000,
    "min_reasoning_tokens": 32,
    "reasoning_token_share": 0.5,
    "max_answer_only_frac": 0.05,
    "max_answer_repeat_frac": 0.10,
    "max_answer_repeat_count": 32,
    "reward_tolerance": 0.0,
    "calibration_tol": 0.003,
    "kings_window": 10,
    "burn_uid": 0,
    "weights_cadence_s": 3600,
    "budget_usd_per_hour": 200.0,
}

SPEC_RECIPE_DEFAULTS = {
    "steps": 1,
    "lora_lr": 1e-5,
    "max_grad_norm": 1.0,
    "rank": 16,
    "alpha": 32.0,
    "dropout": 0.0,
    "ppo_clip": 0.2,
    "dual_clip": 3.0,
    "kl": 0.0,
}


@pytest.mark.parametrize("name,expected", sorted(SPEC_DEFAULTS.items()))
def test_default_matches_the_spec_parameter_table(name, expected):
    assert getattr(Params(), name) == expected


@pytest.mark.parametrize("name,expected", sorted(SPEC_RECIPE_DEFAULTS.items()))
def test_recipe_default_matches_the_spec(name, expected):
    assert getattr(Recipe(), name) == expected


def test_training_cap_is_independent_of_the_scoring_cap():
    params = Params()
    assert params.recipe.train_max_output_tokens == 32768
    assert params.max_output_tokens == 65536


def test_params_hash_is_stable_and_changes_with_any_field():
    assert Params().params_hash == Params().params_hash
    assert dataclasses.replace(Params(), n_tasks=500).params_hash != Params().params_hash
    changed_recipe = dataclasses.replace(Params(), recipe=Recipe(rank=8))
    assert changed_recipe.params_hash != Params().params_hash


def test_recipe_id_changes_when_the_recipe_changes():
    assert Recipe().recipe_id == Params().recipe_id
    assert Recipe(lora_lr=2e-5).recipe_id != Recipe().recipe_id
    assert Recipe(image_digests=("sha256:abc",)).recipe_id != Recipe().recipe_id


def test_params_round_trip_through_json():
    params = dataclasses.replace(Params(), n_tasks=64, recipe=Recipe(rank=8, global_norm_cap=0.5))
    assert Params.from_json(params.to_json()) == params


@pytest.mark.parametrize("batches,expected", [(1, 8), (2, 4), (3, 2), (4, 2), (5, 1), (8, 1), (9, 1), (32, 1)])
def test_gpus_per_lora_is_floor_of_eight_over_batches(batches, expected):
    assert Params().gpus_per_lora(batches) == expected


def test_gpus_per_lora_rejects_an_empty_round():
    with pytest.raises(ConfigError):
        Params().gpus_per_lora(0)


def test_look_sizes_end_with_every_task():
    assert Params().look_sizes() == (500, 1000, 2000)
    assert Params().look_sizes(10) == (3, 5, 10)
    assert Params().look_sizes(2) == (1, 2)
    assert Params().look_sizes(1) == (1,)
    assert dataclasses.replace(Params(), looks=(1.0,)).look_sizes(7) == (7,)


@pytest.mark.parametrize(
    "change",
    [
        {"batch_cap": 0},
        {"batch_cap": 33},
        {"n_min_start": 0},
        {"n_tasks": 0},
        {"samples_per_task": 0},
        {"looks": ()},
        {"looks": (0.5, 0.9)},
        {"looks": (0.5, 0.25, 1.0)},
        {"looks": (0.0, 1.0)},
        {"z_bar": float("nan")},
        {"cap_hit_floor": 1.5},
        {"min_cover": 0.0},
        {"min_scored_tasks": 1},
        {"irt_middle_band": (0.8, 0.2)},
        {"fold_every_k": 0},
        {"max_output_tokens": 70000},
        {"lease_timeout_s": 0},
        {"kings_window": 0},
        {"budget_usd_per_hour": 0.0},
    ],
)
def test_an_invalid_setting_fails_closed(change):
    with pytest.raises(ConfigError):
        dataclasses.replace(Params(), **change)


@pytest.mark.parametrize("change", [{"steps": 0}, {"rank": 0}, {"lora_lr": 0.0}, {"target_modules": ()}])
def test_an_invalid_recipe_fails_closed(change):
    with pytest.raises(ConfigError):
        Recipe(**change)


def test_miner_bounds_are_inclusive_ranges_and_unknown_keys_have_none():
    recipe = Recipe(miner_bounds=(("lr", 1e-6, 1e-4),))
    assert recipe.miner_range("lr") == (1e-6, 1e-4)
    assert recipe.miner_range("rank") is None
    with pytest.raises(ConfigError):
        Recipe(miner_bounds=(("lr", 1e-4, 1e-6),))


def test_load_params_applies_overrides_and_keeps_other_defaults(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"n_tasks": 16, "min_scored_tasks": 8, "recipe": {"rank": 8}}))
    params = load_params(path)
    assert (params.n_tasks, params.min_scored_tasks, params.recipe.rank) == (16, 8, 8)
    assert params.recipe.alpha == 32.0
    assert params.batch_cap == 32


@pytest.mark.parametrize(
    "content",
    ['{"n_task": 16}', '{"n_tasks": "16"}', '{"recipe": {"ranks": 8}}', '{"n_tasks": true}'],
)
def test_load_params_rejects_unknown_keys_and_wrong_types(tmp_path, content):
    path = tmp_path / "params.json"
    path.write_text(content)
    with pytest.raises(ContractError):
        load_params(path)


def test_load_params_fails_closed_on_a_missing_or_broken_file(tmp_path):
    with pytest.raises(ConfigError):
        load_params(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    with pytest.raises(ConfigError):
        load_params(broken)


def test_require_env_never_invents_a_default():
    assert require_env("SN120_STATE_DIR", {"SN120_STATE_DIR": "/data/state"}) == "/data/state"
    for environ in ({}, {"SN120_STATE_DIR": ""}, {"SN120_STATE_DIR": "   "}):
        with pytest.raises(ConfigError):
            require_env("SN120_STATE_DIR", environ)


def test_genesis_constants_name_the_launch_model():
    assert GENESIS_MODEL_REPO == "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
    assert GENESIS_BASE_HASH.startswith("87de8304") and len(GENESIS_BASE_HASH) == 64
    assert len(GENESIS_MODEL_REVISION) == 40


def test_genesis_needs_real_digests_and_a_validator_hotkey():
    good = dict(
        model_repo=GENESIS_MODEL_REPO,
        model_revision=GENESIS_MODEL_REVISION,
        base_hash=GENESIS_BASE_HASH,
        base_fp32_master_digest="c" * 64,
        validator_hotkey="5ValidatorHotkey",
    )
    assert GenesisSpec(**good).base_hash == GENESIS_BASE_HASH
    for bad in ({"base_hash": "87de8304"}, {"base_fp32_master_digest": ""}, {"validator_hotkey": ""}, {"model_repo": ""}):
        with pytest.raises(ConfigError):
            GenesisSpec(**(good | bad))
