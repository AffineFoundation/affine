from __future__ import annotations

import dataclasses
import json

import pytest

from sn120_v2.errors import BatchRejected, ConfigError, ContractError
from sn120_v2.params import GENESIS_BASE_HASH, GENESIS_MODEL_REPO, GenesisSpec, MinerBound, Params, Recipe, load_params, require_env

SPEC_DEFAULTS = dict(
    netuid=120, env="math", batch_cap=32, max_loras_per_box=32, n_min_start=1, n_gpus=8, samples_per_task=1,
    looks=(0.25, 0.5, 1.0), max_output_tokens=65536, max_context_tokens=69632, temperature=1.0, top_p=0.95, top_k=20,
    dtype="bfloat16", z_bar=1.0, cap_hit_floor=0.33, cap_hit_margin=0.02, fold_every_k=1, max_batch_bytes=64 * 1024 * 1024,
    max_prompts=250, max_rollouts=2000, min_reasoning_tokens=32, reasoning_token_share=0.5, max_answer_only_frac=0.05,
    max_answer_repeat_frac=0.10, max_answer_repeat_count=32, reward_tolerance=0.0, calibration_tol=0.003, kings_window=10,
    burn_uid=0, weights_cadence_s=3600, budget_usd_per_hour=200.0,
)
SPEC_RECIPE_DEFAULTS = dict(steps=1, lora_lr=1e-5, max_grad_norm=1.0, rank=16, alpha=32.0, dropout=0.0, ppo_clip=0.2, dual_clip=3.0, kl=0.0)


def test_defaults_match_the_spec_parameter_table():
    params = Params()
    assert {name: getattr(params, name) for name in SPEC_DEFAULTS} == SPEC_DEFAULTS
    assert {name: getattr(params.recipe, name) for name in SPEC_RECIPE_DEFAULTS} == SPEC_RECIPE_DEFAULTS
    assert params.recipe.train_max_output_tokens == 32768, "the training cap is independent of the scoring cap"


def test_params_hash_and_recipe_id_pin_every_field_and_survive_a_json_round_trip():
    base = Params()
    assert base.params_hash == Params().params_hash and base.recipe_id == Recipe().recipe_id
    assert dataclasses.replace(base, n_tasks=500).params_hash != base.params_hash
    assert dataclasses.replace(base, recipe=Recipe(rank=8)).params_hash != base.params_hash
    assert Recipe(image_digests=("sha256:abc",)).recipe_id != Recipe().recipe_id
    changed = dataclasses.replace(base, n_tasks=64, recipe=Recipe(rank=8, global_norm_cap=0.5))
    assert Params.from_json(changed.to_json()) == changed and Params.from_json(changed.to_json()).params_hash == changed.params_hash


@pytest.mark.parametrize("batches,expected", [(1, 8), (2, 4), (3, 2), (4, 2), (5, 1), (8, 1), (9, 1), (32, 1)])
def test_gpus_per_lora_is_floor_of_eight_over_batches(batches, expected):
    assert Params().gpus_per_lora(batches) == expected


def test_look_sizes_end_with_every_task():
    params = Params()
    assert (params.look_sizes(), params.look_sizes(10), params.look_sizes(2), params.look_sizes(1)) == ((500, 1000, 2000), (3, 5, 10), (1, 2), (1,))
    assert dataclasses.replace(params, looks=(1.0,)).look_sizes(7) == (7,)


@pytest.mark.parametrize(
    "change",
    [
        {"batch_cap": 0}, {"batch_cap": 33}, {"n_min_start": 0}, {"n_tasks": 0}, {"looks": ()}, {"looks": (0.5, 0.9)},
        {"looks": (0.5, 0.25, 1.0)}, {"z_bar": float("nan")}, {"cap_hit_floor": 1.5}, {"min_cover": 0.0}, {"min_scored_tasks": 1},
        {"drop_alpha": 0.0}, {"fold_every_k": 0}, {"max_output_tokens": 70000}, {"lease_timeout_s": 0}, {"kings_window": 0},
    ],
)
def test_an_invalid_setting_fails_closed(change):
    with pytest.raises(ConfigError):
        dataclasses.replace(Params(), **change)


def test_all_five_miner_settable_params_are_exposed_in_one_place():
    recipe = Recipe()
    assert {bound.key for bound in recipe.miner_bounds} == {"lora_lr", "steps", "group_size", "sample_temperature", "rank", "alpha", "target_modules"}
    assert {b.key for b in recipe.miner_bounds if b.decided} == {"rank", "target_modules"}, "the spec gives only these two bounds"
    assert (recipe.bound("rank").low, recipe.bound("rank").high) == (8, 64)
    assert recipe.bound("target_modules").choices == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    assert recipe.bound("dropout") is None


def test_miner_values_inside_their_bounds_are_applied_with_both_ends_included():
    recipe = Recipe()
    train = {"lora_lr": 1e-4, "steps": 4, "group_size": 2, "sample_temperature": 0.5, "rank": 64, "alpha": 8, "target_modules": ["v_proj", "q_proj"]}
    applied = recipe.for_miner(train)
    assert (applied.lora_lr, applied.steps, applied.group_size, applied.sample_temperature, applied.rank, applied.alpha) == (1e-4, 4, 2, 0.5, 64, 8.0)
    assert applied.target_modules == ("q_proj", "v_proj"), "a subset is stored in one fixed order"
    assert applied.ppo_clip == recipe.ppo_clip and recipe.for_miner({}) is recipe
    assert Recipe.from_json(applied.to_json()) == applied


@pytest.mark.parametrize(
    "train",
    [
        {"lora_lr": 2e-4}, {"lora_lr": 0.0}, {"lora_lr": float("inf")}, {"steps": 5}, {"steps": 2.0}, {"steps": True},
        {"rank": 7}, {"rank": 65}, {"rank": "16"}, {"rank": 10**400}, {"alpha": float("nan")}, {"alpha": 10**400},
        {"group_size": 1}, {"sample_temperature": 1.6}, {"target_modules": []}, {"target_modules": ["q_proj", "lm_head"]},
        {"target_modules": ["q_proj", "q_proj"]}, {"target_modules": "q_proj"}, {"target_modules": [["q_proj"]]},
        {"dropout": 0.1}, {"kl": 0.02}, {"image_digests": ["x"]},
    ],
)
def test_a_miner_value_outside_its_bound_rejects_the_file_and_is_never_clamped(train):
    with pytest.raises(BatchRejected):
        Recipe().for_miner(train)


def test_a_rejection_never_repeats_the_miners_own_key_text_and_bounds_must_name_recipe_fields():
    with pytest.raises(BatchRejected) as caught:
        Recipe().for_miner({"secret-batch-text": 1})
    assert "secret-batch-text" not in str(caught.value)
    for bounds in ((MinerBound("no_such_field", 0, 1),), (MinerBound("rank", 9, 8, integer=True),), (MinerBound("image_digests", choices=("x",)),)):
        with pytest.raises(ConfigError):
            Recipe(miner_bounds=bounds)
    for change in ({"steps": 0}, {"rank": 0}, {"lora_lr": 0.0}, {"target_modules": ()}):
        with pytest.raises(ConfigError):
            Recipe(**change)


def test_load_params_applies_overrides_and_fails_closed(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"n_tasks": 16, "min_scored_tasks": 8, "recipe": {"rank": 8}}))
    params = load_params(path)
    assert (params.n_tasks, params.min_scored_tasks, params.recipe.rank, params.recipe.alpha, params.batch_cap) == (16, 8, 8, 32.0, 32)
    for content in ('{"n_task": 16}', '{"n_tasks": "16"}', '{"recipe": {"ranks": 8}}', '{"n_tasks": true}'):
        path.write_text(content)
        with pytest.raises(ContractError):
            load_params(path)
    path.write_text("{")
    for broken in (path, tmp_path / "absent.json"):
        with pytest.raises(ConfigError):
            load_params(broken)


def test_a_missing_environment_variable_or_genesis_digest_is_an_error_never_a_default():
    assert require_env("SN120_STATE_DIR", {"SN120_STATE_DIR": "/data/state"}) == "/data/state"
    for environ in ({}, {"SN120_STATE_DIR": ""}, {"SN120_STATE_DIR": "   "}):
        with pytest.raises(ConfigError):
            require_env("SN120_STATE_DIR", environ)
    assert GENESIS_MODEL_REPO == "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B" and GENESIS_BASE_HASH.startswith("87de8304")
    good = dict(model_repo=GENESIS_MODEL_REPO, model_revision="rev", base_hash=GENESIS_BASE_HASH, base_fp32_master_digest="c" * 64, validator_hotkey="5Validator")
    assert GenesisSpec(**good).base_hash == GENESIS_BASE_HASH
    for bad in ({"base_hash": "87de8304"}, {"base_fp32_master_digest": ""}, {"validator_hotkey": ""}, {"model_repo": ""}):
        with pytest.raises(ConfigError):
            GenesisSpec(**(good | bad))
