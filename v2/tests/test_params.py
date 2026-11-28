from __future__ import annotations

import dataclasses
import json

import pytest

from sn120_v2.canonical import digest, to_jsonable
from sn120_v2.errors import BatchRejected, ConfigError, ContractError
from conftest import Build
from sn120_v2.params import GENESIS_BASE_HASH, GENESIS_MODEL_REPO, GenesisSpec, MinerBound, Params, Recipe, load_params, require_env

SPEC_DEFAULTS = dict(
    netuid=120, env="math", batch_cap=32, max_loras_per_box=32, n_min_start=1, n_gpus=8, samples_per_task=1,
    looks=(0.25, 0.5, 1.0), max_output_tokens=65536, max_context_tokens=69632, temperature=1.0, top_p=0.95, top_k=20,
    dtype="bfloat16", kv_cache_dtype="auto", max_num_seqs=256, z_bar=1.0, cap_hit_floor=0.33, cap_hit_margin=0.02, fold_every_k=1, max_batch_bytes=128 * 1024 * 1024,
    max_prompts=2000, max_rollouts=8000, min_reasoning_tokens=32, reasoning_token_share=0.5, max_answer_only_frac=0.05,
    max_answer_repeat_frac=0.10, max_answer_repeat_count=32, reward_tolerance=0.0, calibration_tol=0.003, kings_window=10,
    burn_uid=0, weights_cadence_s=3600, budget_usd_per_hour=200.0,
)
SPEC_RECIPE_DEFAULTS = dict(steps=1, minibatches=1, global_norm_cap=6.0, lora_lr=1e-5, adam_eps=1e-8, max_grad_norm=1.0, rank=16, alpha=32.0, dropout=0.0, ppo_clip=0.2, dual_clip=3.0, kl=0.0)


def test_defaults_match_the_spec_parameter_table():
    params = Params()
    assert {name: getattr(params, name) for name in SPEC_DEFAULTS} == SPEC_DEFAULTS
    assert {name: getattr(params.recipe, name) for name in SPEC_RECIPE_DEFAULTS} == SPEC_RECIPE_DEFAULTS
    assert params.recipe.train_max_output_tokens == 32768, "the training cap is independent of the scoring cap"


def test_serving_defaults_preserve_old_json_params_and_ticket_digests():
    params = Params()
    assert params.kv_cache_dtype == "auto" and params.max_num_seqs == 256
    assert params.params_hash == "7eb09fa89387d1d2c5db2d8d0ffd627ad87ece8ef31bdfe93f3374f0c7bb1e50"
    assert "kv_cache_dtype" not in params.to_dict() and "max_num_seqs" not in params.to_dict()
    assert params.null_samples_per_task is None and params.null_samples == params.samples_per_task
    assert "null_samples_per_task" not in params.to_dict()
    two = dataclasses.replace(params, null_samples_per_task=2)
    assert two.to_dict()["null_samples_per_task"] == 2 and Params.from_dict(two.to_dict()).null_samples == 2
    assert two.params_hash != params.params_hash
    raw = json.dumps(params.to_dict(), sort_keys=True)
    assert Params.from_dict(json.loads(raw)).params_hash == params.params_hash
    assert dataclasses.replace(params, kv_cache_dtype="fp8", max_num_seqs=512).params_hash == "5e2169191be5163443b2fb30821234b7a2be058b08589d1eeb95a1488a309571"
    ticket = Build.ticket(params=params)
    assert ticket.digest() == ticket.digest()
    from sn120_v2.types import RoundTicket
    restored = RoundTicket.from_dict(ticket.to_dict())
    assert restored.digest() == ticket.digest() and restored.params == params

    old = Params.from_dict({"n_tasks": 16, "min_scored_tasks": 8})
    assert old.train_seed_scope == "batch" and old.dedup_enabled is False
    ticket = Build.ticket(params=old)
    restored = RoundTicket.from_dict(ticket.to_dict())
    # Nested Params are legacy records too: a reopened old ticket trains and verifies with its original seed scope.
    assert restored.params.to_dict() == old.to_dict() and restored.digest() == ticket.digest()
    assert restored.params.train_seed_scope == "batch" and restored.params.dedup_enabled is False
    assert "train_seed_scope" not in restored.to_dict()["params"]


def test_new_empty_round_fields_keep_the_historical_canonical_form():
    from sn120_v2.state import RoundRecord
    from sn120_v2.types import RoundDecision

    decision = RoundDecision("r000001", "head", None, (), 0, 0.0, 0.0)
    data = decision.to_dict()
    assert "duplicate_clusters" not in data
    assert RoundDecision.from_dict(data).to_dict() == data
    record = RoundRecord("r000001", "head", "params", "recipe", "seed", "snapshot", 1, 1, ())
    data = record.to_dict()
    assert "first_seen" not in data and "duplicate_decisions" not in data and "duplicate_clusters" not in data
    assert RoundRecord.from_dict(data).to_dict() == data


def test_nested_reserve_and_existing_factory_defaults_keep_their_identity():
    from sn120_v2.rotation import Reserve, ReserveShard, Rotation

    reserve = Reserve("a" * 64, ("task",), (ReserveShard("s", "b" * 64, 1, "2026-11-20"),))
    assert Reserve.from_dict(reserve.to_dict()) == reserve
    empty = Reserve("a" * 64)
    assert empty.to_dict()["candidate_meta"] == {}
    for record in (empty,):
        assert type(record).from_dict(record.to_dict()).digest() == record.digest()


def test_changed_params_shows_left_out_at_default_fields_in_both_directions():
    from pathlib import Path
    from sn120_v2.cli import changed_params

    off = Params.from_json((Path(__file__).parent / "data" / "params_prod_shaped.json").read_bytes())
    on = dataclasses.replace(off, rotation_enabled=True)
    assert off.params_hash == "5359a448461c3f3a3e09bd9a2d8df9f387d916fb02b8d02eecea3582649c74a1"
    assert Params().params_hash == "7eb09fa89387d1d2c5db2d8d0ffd627ad87ece8ef31bdfe93f3374f0c7bb1e50"
    assert changed_params(on, off) == {"rotation_enabled": {"before": True, "after": False}}
    assert changed_params(off, on) == {"rotation_enabled": {"before": False, "after": True}}

    default = Params()
    changes = {"irt_draw_rule": "band", "kv_cache_dtype": "fp8", "max_num_seqs": 512, "null_samples_per_task": 2,
               "rotation_k": 31, "rotation_retire_p": 0.9, "rotation_n_min": 2,
               "rotation_retire_low_p": 0.1, "rotation_n_min_low": 7, "rotation_active_size": 20000}
    for name, value in changes.items():
        changed = dataclasses.replace(default, **{name: value})
        old = getattr(default, name)
        assert changed_params(changed, default)[name] == {"before": value, "after": old}
        assert changed_params(default, changed)[name] == {"before": old, "after": value}


def test_public_params_load_the_current_gate_and_paired_parent_fields():
    public = Params().to_dict() | {
        "confirm_next_round": True,
        "min_kl_seq": 10.0,
        "kl_rollouts": 128,
        "kl_max_tokens": 8192,
        "samples_per_task": 2,
        "null_samples_per_task": 4,
        "future_validator_field": {"ignored": True},
    }
    params = Params.from_public_dict(public)
    assert params.confirm_next_round is True
    assert (params.min_kl_seq, params.kl_rollouts, params.kl_max_tokens) == (10.0, 128, 8192)
    assert (params.samples_per_task, params.null_samples_per_task, params.null_samples) == (2, 4, 4)
    with pytest.raises(ContractError, match="unknown field"):
        Params.from_dict(public)


def test_public_params_keep_live_crown_recitation_and_kl_ceiling_fields():
    recipe = Recipe().to_dict()
    recipe["miner_bounds"] = [*recipe["miner_bounds"], {"key": "future_field", "low": 0, "high": 1}]
    public = Params().to_dict() | {
        "recipe": recipe,
        "crown_gate": "argmax_symmetric",
        "z_crown": 1.5,
        "recite_max_share": 0.05,
        "max_kl_tok": 0.25,
    }
    params = Params.from_public_dict(public)
    assert (params.crown_bar, params.recite_max_share, params.max_kl_tok) == (1.5, 0.05, 0.25)
    assert params.kl_floor_on
    # The 2026-11-27 live values hashed bd7b9aee... here; the old drop-unknown
    # behavior on the king branch at 08a936d5 hashed the reduced dict 54c5656c...
    old = {key: value for key, value in public.items() if key not in ("z_crown", "recite_max_share", "max_kl_tok")}
    assert params.params_hash != Params.from_public_dict(old).params_hash
    for key in ("z_crown", "recite_max_share", "max_kl_tok"):
        assert public[key] == params.to_dict()[key]
        assert key in Params.LEFT_OUT_AT_DEFAULT
    assert not {"z_crown", "recite_max_share", "max_kl_tok"} & Params().to_dict().keys()
    assert dataclasses.replace(Params(), z_crown=None).params_hash == Params().params_hash
    assert {bound.key for bound in params.recipe.miner_bounds} == {bound.key for bound in Recipe().miner_bounds}
    with pytest.raises(ConfigError):
        Params.from_dict(public)


def test_public_params_keep_anchor_fields_and_hash_them():
    anchor_values = {
        "anchor_n": 300,
        "anchor_min_p": 0.85,
        "anchor_min_rollouts": 8,
        "anchor_hi_p": 0.9,
        "anchor_hi_share": 0.2,
        "anchor_z_veto": 2.0,
        "anchor_min_scored": 80,
    }
    public = Params().to_dict() | {
        "crown_gate": "argmax_symmetric",
        "eval_merged": True,
        "irt_draw_rule": "sqrt_fisher",
        "z_crown": 1.5,
        "recite_max_share": 0.05,
        "max_kl_tok": 0.25,
        **anchor_values,
    }
    params = Params.from_public_dict(public)
    assert {key: getattr(params, key) for key in anchor_values} == anchor_values
    assert params.to_dict().items() >= anchor_values.items()
    without_anchor_n = Params.from_public_dict({key: value for key, value in public.items() if key != "anchor_n"})
    assert without_anchor_n.anchor_n == 0
    assert params.params_hash != without_anchor_n.params_hash
    # The 2026-11-28 live contract was 3a3f6252...; trees without anchor fields
    # computed bd7b9aee.... Other live fields may change, so compare dynamically.
    for key in anchor_values:
        assert key in Params.LEFT_OUT_AT_DEFAULT
        assert key not in Params().to_dict()
    assert Params().params_hash == "7eb09fa89387d1d2c5db2d8d0ffd627ad87ece8ef31bdfe93f3374f0c7bb1e50"
    assert not {"max_stages", "train_tokens_per_s_floor"} & Params().to_dict().keys()
    assert Params.from_public_dict(public | {"train_tokens_per_s_floor": 100_000}).train_tokens_per_s_floor == 100_000


@pytest.mark.parametrize(
    "change",
    [
        {"anchor_n": -1}, {"anchor_n": True}, {"anchor_n": 1.5},
        {"anchor_min_rollouts": -1}, {"anchor_min_rollouts": True},
        {"anchor_min_scored": -1}, {"anchor_min_scored": 1.5},
        {"anchor_min_p": -0.1}, {"anchor_hi_p": 1.1}, {"anchor_hi_share": 1.1},
        {"anchor_z_veto": -1}, {"anchor_z_veto": float("nan")}, {"anchor_z_veto": float("inf")},
        {"anchor_n": 300},
        {"max_stages": 0}, {"max_stages": 6}, {"max_stages": True},
        {"train_tokens_per_s_floor": -1}, {"train_tokens_per_s_floor": True},
        {"train_tokens_per_s_floor": 5_000},
    ],
)
def test_anchor_and_training_fields_fail_closed(change):
    with pytest.raises(ConfigError):
        Params(**change)


def test_public_params_accept_known_rotation_rules_and_reject_unknown_rules():
    for rule in ("v1", "info", "info-guard"):
        assert Params.from_public_dict(Params().to_dict() | {"rotation_rule": rule}).rotation_rule == rule
    with pytest.raises(ConfigError, match="rotation_rule must be 'v1', 'info' or 'info-guard'"):
        Params.from_public_dict(Params().to_dict() | {"rotation_rule": "unknown"})


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


def test_adam_eps_is_optional_and_a_chosen_value_is_hashed():
    assert Recipe().adam_eps == 1e-8 and "adam_eps" not in Recipe().to_dict()
    assert Recipe().recipe_id == digest(to_jsonable(Recipe())) == digest(Params().to_dict()["recipe"])
    chosen = Recipe().for_miner({"adam_eps": 1e-6})
    assert chosen.adam_eps == 1e-6 and chosen.to_dict()["adam_eps"] == 1e-6
    assert chosen.recipe_id != Recipe().recipe_id
    assert Recipe.from_dict(chosen.to_dict()) == chosen
    assert Recipe.from_dict(Recipe().to_dict()).to_dict() == Recipe().to_dict()
    assert Recipe.from_dict({}).adam_eps == 1e-8  # a stored recipe without the field uses the current default


@pytest.mark.parametrize("eps", [0, -1e-8, 1e-11, 1e-3, float("nan"), float("inf"), -float("inf"), None, True, "1e-8", [1e-8]])
def test_adam_eps_must_be_a_finite_positive_float_inside_its_bound(eps):
    with pytest.raises(BatchRejected):
        Recipe().for_miner({"adam_eps": eps})
    for value in (0, -1e-8, float("nan"), float("inf")):
        with pytest.raises(ConfigError):
            Recipe(adam_eps=value)
    for value in (1e-10, 1e-4):
        assert Recipe().for_miner({"adam_eps": value}).adam_eps == value


def test_look_sizes_end_with_every_task():
    params = Params()
    assert (params.look_sizes(), params.look_sizes(10), params.look_sizes(2), params.look_sizes(1)) == ((500, 1000, 2000), (3, 5, 10), (1, 2), (1,))
    assert dataclasses.replace(params, looks=(1.0,)).look_sizes(7) == (7,)


@pytest.mark.parametrize(
    "change",
    [
        {"batch_cap": 0}, {"batch_cap": 33}, {"n_min_start": 0}, {"n_tasks": 0}, {"looks": ()}, {"looks": (0.5, 0.9)},
        {"looks": (0.5, 0.25, 1.0)}, {"z_bar": float("nan")}, {"cap_hit_floor": 1.5}, {"min_cover": 0.0}, {"min_scored_tasks": 1},
        {"drop_alpha": 0.0}, {"fold_every_k": 0}, {"max_output_tokens": 70000}, {"max_num_seqs": 0}, {"lease_timeout_s": 0}, {"kings_window": 0},
        {"irt_draw_rule": "middle"}, {"irt_draw_rule": ""}, {"irt_draw_rule": "Fisher"}, {"irt_draw_rule": None},
        {"rotation_rule": "unknown"},
        {"null_samples_per_task": 0}, {"null_samples_per_task": -1}, {"null_samples_per_task": 1.5}, {"null_samples_per_task": True},
    ],
)
def test_an_invalid_setting_fails_closed(change):
    with pytest.raises(ConfigError):
        dataclasses.replace(Params(), **change)


def test_the_default_irt_rule_is_fisher_and_band_remains_selectable():
    assert Params().irt_draw_rule == "fisher"
    assert Params(irt_draw_rule="band").irt_draw_rule == "band"
    assert Params.from_dict({"irt_draw_rule": "fisher"}).irt_draw_rule == "fisher"
    assert "irt_draw_rule" not in Params().to_dict(), "old stored params keep their JSON and hash with the new default"
    assert Params(irt_draw_rule="band").to_dict()["irt_draw_rule"] == "band"
    assert Params(irt_draw_rule="band").params_hash != Params().params_hash


def test_all_miner_settable_params_are_exposed_in_one_place():
    recipe = Recipe()
    assert {bound.key for bound in recipe.miner_bounds} == {"lora_lr", "adam_eps", "steps", "minibatches", "group_size", "sample_temperature", "rank", "alpha", "target_modules", "global_norm_cap"}
    assert {b.key for b in recipe.miner_bounds if b.decided} == {"adam_eps", "rank", "target_modules", "minibatches", "global_norm_cap"}, "the spec gives only these bounds"
    assert (recipe.bound("rank").low, recipe.bound("rank").high) == (8, 64)
    assert recipe.bound("target_modules").choices == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj")
    assert recipe.target_modules == recipe.bound("target_modules").choices[:7], "the default list is the 7 names of before"
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
    for change in ({"steps": 0}, {"rank": 0}, {"lora_lr": 0.0}, {"adam_eps": 0.0}, {"target_modules": ()}):
        with pytest.raises(ConfigError):
            Recipe(**change)


def test_kingx2_and_dedup_params_keep_their_combined_and_legacy_identities():
    base = Params()
    assert "null_samples_per_task" not in base.to_dict()
    assert base.params_hash == "7eb09fa89387d1d2c5db2d8d0ffd627ad87ece8ef31bdfe93f3374f0c7bb1e50"
    both = dataclasses.replace(base, null_samples_per_task=2, irt_draw_rule="sqrt_fisher")
    assert both.null_samples == 2 and both.dedup_enabled and both.train_seed_scope == "round"
    restored = Params.from_dict(both.to_dict())
    assert restored.to_dict() == both.to_dict() and restored.params_hash == both.params_hash
    assert restored._legacy_fields is None

    old_data = {name: value for name, value in both.to_dict().items() if name not in Params.LEGACY_FIELDS}
    old = Params.from_dict(old_data)
    assert (old.null_samples_per_task, old.irt_draw_rule, old.null_samples) == (2, "sqrt_fisher", 2)
    assert (old.train_seed_scope, old.dedup_enabled, old.min_signal_rollouts, old.min_step_norm) == ("batch", False, 1, 0.0)
    assert old._legacy_fields == frozenset(Params.LEGACY_FIELDS)
    assert old.to_dict() == old_data and Params.from_dict(old.to_dict()).params_hash == old.params_hash

    # Enabling the dedup rules on a king-x2 historical record writes only the
    # explicitly selected legacy field; the king samples and sampler remain.
    changed = dataclasses.replace(old, dedup_enabled=True)
    assert changed.to_dict()["dedup_enabled"] is True and "train_seed_scope" not in changed.to_dict()
    assert changed.null_samples == 2 and changed.irt_draw_rule == "sqrt_fisher"
    assert Params.from_dict(changed.to_dict()).to_dict() == changed.to_dict()


def test_legacy_params_replacement_is_explicit_and_never_keeps_stale_omissions():
    from pathlib import Path
    old = Params.from_json((Path(__file__).parent / "data" / "params_prod_shaped.json").read_bytes())
    unchanged = dataclasses.replace(old, n_tasks=old.n_tasks)
    assert unchanged.to_dict() == old.to_dict() and unchanged.params_hash == old.params_hash
    changed = dataclasses.replace(old, dedup_enabled=True)
    assert changed.train_seed_scope == "batch"
    # An old round keeps batch-scoped seeds, but an explicitly changed behavior is written and changes the hash.
    assert "train_seed_scope" not in changed.to_dict()
    assert changed.to_dict()["dedup_enabled"] is True and changed.params_hash != old.params_hash
    restored = Params.from_json(changed.to_json())
    assert restored.params_hash == changed.params_hash and restored.to_dict() == changed.to_dict()


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
