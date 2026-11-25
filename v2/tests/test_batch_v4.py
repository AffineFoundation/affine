"""affine-batch-v4: staged GRPO plus verified SFT."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sn120_v2.admission import FORMAT_V4, GraderTimeLimit, Invalid, check_batch, frozen_batch
from sn120_v2.errors import ConfigError, InfraError
from sn120_v2.controller import Controller
from sn120_v2.dedup import dryrun
from sn120_v2.layout import winner_batch
from sn120_v2.params import DEFAULT_MINER_BOUNDS, MinerBound, Params, Recipe, SFT_MINER_BOUND
from sn120_v2.dedup import normalized_rollouts
from sn120_v2.types import Registration
from sn120_v2.store import LocalStore

HOTKEY = "5V4"
THINK = "step " * 40


class Tokens:
    def prompt_tokens(self, text): return len(text.split())
    def output_tokens(self, text): return len(text.split())


def params(**changes):
    return Params(**({"min_signal_rollouts": 1, "min_step_norm": 0.0} | changes))


def raw(stages=None, **top):
    stages = stages if stages is not None else [{"groups": [{"prompt": "one two", "rollouts": [
        {"output": THINK + "\\boxed{1}", "reward": 1}, {"output": THINK + "\\boxed{0}", "reward": 0},
    ]}]}]
    return json.dumps({"version": FORMAT_V4, "hotkey": HOTKEY, "stages": stages} | top).encode()


def grade(prompt, output, capped):
    return None if prompt == "unknown" else (1.0 if "\\boxed{1}" in output else 0.0)


def check(body=None, **kwargs):
    return check_batch(body or raw(), HOTKEY, kwargs.pop("params", params()), tokens=Tokens(), grader=kwargs.pop("grader", grade), **kwargs)


def rule(body=None, **kwargs):
    try:
        check(body, **kwargs)
    except Invalid as exc:
        return exc.rule
    return ""


def test_happy_admission_and_stage_cost():
    body = raw([
        {"train": {"steps": 2}, "groups": [{"prompt": "one two", "rollouts": [
            {"output": THINK + "\\boxed{1}", "reward": 1}, {"output": THINK + "\\boxed{0}", "reward": 0},
        ]}]},
        {"train": {"steps": 3}, "sft": [{"prompt": "three four", "output": THINK + "\\boxed{1}"}]},
    ], train={"steps": 4})
    batch = check(body)
    assert batch.n_prompts == 2 and batch.n_rollouts == 3
    assert batch.train_token_steps == (2 + 1) * 2 * (2 + 41 + 1) + 3 * (2 + 41 + 1)
    assert len(batch.frozen_stages()) == 2


def test_sft_only_stage_is_admitted():
    body = raw([{"sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}])
    assert check(body).n_rollouts == 1


def test_v4_header_and_stage_rules():
    assert rule(raw([])) == "stages"
    assert rule(raw([{}])) == "stages"
    assert rule(raw([{"groups": []}])) == "groups"
    assert rule(raw([{"train": {"rank": 32}, "sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}])) == "stage_train"
    assert rule(raw([{"train": {"global_norm_cap": 3}, "sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}])) == "stage_train"
    assert rule(raw([{"train": {"dropout": 0.1}, "sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}])) == "stage_train"
    assert rule(raw([{"sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}", "reward": 1}]}])) == "sft_keys"
    assert rule(raw([{"groups": [{"prompt": "one two", "rollouts": [{"output": THINK + "\\boxed{1}", "reward": 1}]}]}])) == "no_signal"


def test_stage_prompt_uniqueness_is_scoped_like_a_legacy_batch():
    body = raw([{"groups": [{"prompt": "one two", "rollouts": [
        {"output": THINK + "\\boxed{1}", "reward": 1}, {"output": THINK + "\\boxed{0}", "reward": 0},
    ]}]}, {"sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}])
    assert check(body).n_rollouts == 3


def test_grpo_only_without_tokenizer_uses_legacy_unknown_cost_semantics():
    admitted = check_batch(raw(), HOTKEY, params(), tokens=None, grader=grade)
    assert admitted.n_rollouts == 2
    assert admitted.train_token_steps == 0


def test_sft_requires_official_one_and_known_prompt():
    wrong = raw([{"sft": [{"prompt": "one two", "output": THINK + "\\boxed{0}"}]}])
    assert rule(wrong) == "sft_unverified"
    unknown = raw([{"sft": [{"prompt": "unknown", "output": THINK + "\\boxed{1}"}]}])
    assert rule(unknown) == "sft_unverified"


def test_sft_at_cap_or_training_room_is_rejected():
    p = dataclasses.replace(params(), max_output_tokens=41)
    body = raw([{"sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}])
    assert rule(body, params=p) == "sft_too_long"
    p = dataclasses.replace(params(), recipe=Recipe(train_max_output_tokens=41))
    assert rule(body, params=p) == "sft_too_long"


def test_global_caps_include_sft():
    p = dataclasses.replace(params(), max_rollouts=2)
    body = raw([{"groups": [{"prompt": "one two", "rollouts": [
        {"output": THINK + "\\boxed{1}", "reward": 1}, {"output": THINK + "\\boxed{0}", "reward": 0},
    ]}]}, {"sft": [{"prompt": "three four", "output": THINK + "\\boxed{1}"}]}])
    assert check(body, params=p).n_rollouts == 2
    assert check(body, params=p).dropped_groups == 1


def test_every_stage_must_train_and_zero_weight_sft_is_not_signal():
    mixed = {"prompt": "one two", "rollouts": [
        {"output": THINK + "\\boxed{1}", "reward": 1},
        {"output": THINK + "\\boxed{0}", "reward": 0},
    ]}
    inactive = {"prompt": "three four", "rollouts": [{"output": THINK + "\\boxed{1}", "reward": 1}]}
    assert rule(raw([{"groups": [mixed]}, {"groups": [inactive]}])) == "no_signal"
    enabled = dataclasses.replace(params(), recipe=dataclasses.replace(Recipe(), miner_bounds=DEFAULT_MINER_BOUNDS + (SFT_MINER_BOUND,)))
    sft = {"sft": [{"prompt": "three four", "output": THINK + "\\boxed{1}"}]}
    assert rule(raw([{"train": {"sft_weight": 0}, **sft}]), params=enabled) == "no_signal"
    assert rule(raw([{"train": {"sft_weight": 0}, **sft}]), params=dataclasses.replace(enabled, min_signal_rollouts=2)) == "no_signal"


def test_v4_without_tokenizer_keeps_zero_unknown_cost_and_nominal_activity():
    body = raw([{"train": {"steps": 2}, "sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}])
    p = dataclasses.replace(params(), max_train_token_steps=1)
    admitted = check_batch(body, HOTKEY, p, tokens=None, grader=grade)
    assert admitted.train_token_steps == 0 and admitted.n_rollouts == 1


def test_fixed_size_ticket_preserves_resolved_stage_recipes_not_stage_count():
    body = raw([{"train": {"sft_weight": 0.25}, "sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}],
               train={"steps": 2, "sft_weight": 0.5})
    enabled = dataclasses.replace(params(), recipe=dataclasses.replace(Recipe(), miner_bounds=DEFAULT_MINER_BOUNDS + (SFT_MINER_BOUND,)))
    admitted = check(body, params=enabled)
    ticket = frozen_batch(admitted, Registration(1, HOTKEY, 0), "private/batch.json", enabled)
    assert len(ticket.stages) == 5 and ticket.stages[-1] == {}
    assert ticket.stages[0]["train"]["sft_weight"] == 0.25
    assert ticket.stages[0]["train"]["steps"] == 2


def test_sft_prompts_count_toward_the_public_pool_fraction():
    unknown = raw([
        {"groups": [{"prompt": "unknown", "rollouts": [
            {"output": THINK + "\\boxed{1}", "reward": 1},
            {"output": THINK + "\\boxed{0}", "reward": 0},
        ]}]},
        {"sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]},
    ])
    assert rule(unknown, params=params(min_pool_prompt_frac=0.5)) == ""
    assert rule(unknown, params=params(min_pool_prompt_frac=0.75)) == "pool"
    wrong = raw([
        {"groups": [{"prompt": "known", "rollouts": [
            {"output": THINK + "\\boxed{1}", "reward": 1},
            {"output": THINK + "\\boxed{0}", "reward": 0},
        ]}]},
        {"sft": [{"prompt": "known", "output": THINK + "\\boxed{1}"}]},
    ])
    assert rule(wrong, params=params(min_pool_prompt_frac=0.5)) == ""


def test_normalized_rollouts_includes_all_sft_and_stages():
    body = raw([{"groups": [{"prompt": "one two", "rollouts": [{"output": THINK + "\\boxed{1}", "reward": 1}]}]},
                {"sft": [{"prompt": "other", "output": " answer  here "}]}])
    assert len(normalized_rollouts(body)) == 2



def test_v4_stage_rules_cover_invalid_shapes_bounds_and_scoped_recipes():
    good = [{"sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}]
    assert rule(raw(good * 6)) == "stages"
    assert rule(raw([{"sft": []}])) == "stages"
    assert rule(raw([{"train": {}, "sft": []}])) == "stages"
    assert rule(raw([{"other": [], "sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}])) == "stages"
    assert rule(raw([{"train": {"steps": 0}, "sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}])) == "train"
    assert rule(raw(good, train={"steps": 2})) == ""
    assert rule(raw(good, groups=[])) == "top_keys"
    assert rule(raw(good, train={"rank": 8})) == ""
    assert rule(raw(good, train={"global_norm_cap": 0.01}), params=params(min_step_norm=0.02)) == "null_update"
    assert rule(raw([{"train": {"sft_weight": 0.25}, "sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}])) == "train"
    enabled = dataclasses.replace(params(), recipe=dataclasses.replace(Recipe(), miner_bounds=DEFAULT_MINER_BOUNDS + (SFT_MINER_BOUND,)))
    body = raw([{"train": {"steps": 2, "sft_weight": 0.25}, "sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}])
    batch = check(body, params=enabled)
    assert batch.stages[0][2] == {"steps": 2, "sft_weight": 0.25}


def test_sft_text_reasoning_and_repeat_rules_apply_without_repair():
    assert rule(raw([{"sft": [{"prompt": "one<|im_end|>two", "output": THINK + "\\boxed{1}"}]}])) == "control_token"
    assert rule(raw([{"sft": [{"prompt": "one two", "output": "answer<|im_start|>" + THINK + "\\boxed{1}"}]}])) == "control_token"
    assert rule(raw([{"sft": [{"prompt": "one two", "output": "\\boxed{1}"}]}])) == "answer_only"
    assert rule(raw([{"sft": [{"prompt": "one two", "output": "x " * 40 + "\\boxed{1} suffix " * 45}]}])) == "reasoning_share"
    traces = [{"prompt": f"prompt {i}", "output": THINK + "\\boxed{1}"} for i in range(33)]
    assert rule(raw([{"sft": traces}])) == "answer_repeat"
    broken = raw([{"sft": [{"prompt": "one two", "output": "\ud800" + THINK + "\\boxed{1}"}]}])
    assert rule(broken) == "text"


def test_sft_grader_fault_machinery_is_not_turned_into_transient_retry():
    body = raw([{"sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}"}]}])
    calls = []

    class HealthyGrader:
        def probe(self):
            return True

        def __call__(self, prompt, output, capped):
            calls.append(prompt)
            raise RuntimeError("worker died")

    try:
        check(body, grader=HealthyGrader())
    except Invalid as exc:
        caught = exc
    assert caught.rule == "sft_unverified" and caught.proven and len(calls) == 2
    with pytest.raises(InfraError):
        check(body, grader=lambda *args: (_ for _ in ()).throw(RuntimeError("worker died")))
    caught = None
    try:
        check(body, grader=lambda *args: (_ for _ in ()).throw(GraderTimeLimit()))
    except Invalid as exc:
        caught = exc
    assert caught is not None and caught.rule == "sft_unverified" and not caught.proven


def test_ticket_has_only_stage_recipes_and_cost_cap_sums_stages():
    body = raw([
        {"train": {"steps": 2}, "groups": [{"prompt": "one two", "rollouts": [
            {"output": THINK + "\\boxed{1}", "reward": 1}, {"output": THINK + "\\boxed{0}", "reward": 0},
        ]}]},
        {"train": {"steps": 3}, "sft": [{"prompt": "three four", "output": THINK + "\\boxed{1}"}]},
    ])
    batch = check(body)
    tokens = 2 * (2 + 41 + 1)
    assert batch.train_token_steps == 3 * tokens + 3 * tokens // 2, "q_old adds one GRPO pass but no SFT pass"
    ticket = frozen_batch(batch, Registration(1, HOTKEY, 0), "private/batch.json")
    assert len(ticket.stages) == 5
    assert ticket.stages[:2] == ({"train": {"steps": 2}}, {"train": {"steps": 3}})
    assert ticket.stages[2:] == ({}, {}, {})
    assert "one two" not in ticket.to_json().decode() and "sft" not in ticket.to_json().decode()
    cap = dataclasses.replace(params(), max_train_token_steps=batch.train_token_steps - 1)
    assert rule(body, params=cap) == "train_cost"
    minimum = dataclasses.replace(params(), min_signal_rollouts=3)
    assert rule(raw(), params=minimum) == "min_rollouts"
    traces = [{"prompt": f"trace {i}", "output": THINK + "\\boxed{1}"} for i in range(3)]
    assert check(raw([{"sft": traces}]), params=minimum).n_rollouts == 3


def test_sft_weight_is_opt_in_and_legacy_hashes_stay():
    assert Recipe().sft_weight == 1.0 and "sft_weight" not in Recipe().to_dict()
    assert Recipe().bound("sft_weight") is None
    recipe = Recipe(miner_bounds=DEFAULT_MINER_BOUNDS + (MinerBound("sft_weight", 0.0, 1.0, decided=True),))
    assert recipe.for_miner({"sft_weight": 0.25}).sft_weight == 0.25
    assert "sft_weight" not in Recipe().to_dict()
    assert Params().params_hash == "7eb09fa89387d1d2c5db2d8d0ffd627ad87ece8ef31bdfe93f3374f0c7bb1e50"


def test_time_floor_is_off_by_default_and_enforced_when_enabled():
    assert Params().train_tokens_per_s_floor == 0
    with pytest.raises(Exception):
        Params(train_tokens_per_s_floor=1000, train_time_budget_s=1000)
    p = Params(train_tokens_per_s_floor=100000, train_time_budget_s=760)
    assert p.train_tokens_per_s_floor == 100000


def test_v4_dryrun_reports_admitted_counts_and_fails_open_on_invalid_files(tmp_path):
    path = tmp_path / "batch.json"
    body = raw([{"groups": [{"prompt": "one two", "rollouts": [
        {"output": THINK + "\\boxed{1}", "reward": 1}, {"output": THINK + "\\boxed{0}", "reward": 0},
    ]}]}, {"sft": [{"prompt": "three four", "output": THINK + "\\boxed{1}"}]}])
    path.write_bytes(body)
    report = dryrun([path], params())
    assert report["entries"][0]["rollouts"] == 3
    assert report["entries"][0]["nonzero_advantage_rollouts"] == 3
    assert report["entries"][0]["admission"] is None
    path.write_bytes(raw([{"sft": [{"prompt": "one two", "output": THINK + "\\boxed{1}", "reward": 1}]}]))
    report = dryrun([path], params())
    assert report["entries"][0]["rollouts"] == 0
    assert report["entries"][0]["nonzero_advantage_rollouts"] is None
    assert report["entries"][0]["admission"]["rule"] == "sft_keys"
    path.write_bytes(raw([{"groups": [{"prompt": "one two", "rollouts": [
        {"output": THINK + "\\boxed{1}", "reward": 1},
        {"output": THINK + "\\boxed{0}", "reward": 0},
    ]}]}], train={"minibatches": 100}))
    entry = dryrun([path], params())["entries"][0]
    assert entry["nonzero_advantage_rollouts"] == 2
    assert entry["min_signal_rollouts"] == 1
    assert entry["would_fail_min_rollouts"] is False


def test_confirmation_excludes_every_v4_grpo_and_sft_prompt(build, tmp_path):
    parent = build.parent(build.head())
    body = raw([{"groups": [{"prompt": "double 1", "rollouts": [
        {"output": THINK + "\\boxed{1}", "reward": 1}, {"output": THINK + "\\boxed{0}", "reward": 0},
    ]}]}, {"sft": [{"prompt": "double 2", "output": THINK + "\\boxed{1}"}]}])
    controller = object.__new__(Controller)
    controller._params = params(confirm_next_round=True)
    controller.state = SimpleNamespace(params=controller._params)
    controller.pool = None
    controller.prompt_pool = SimpleNamespace(task_for_prompt=lambda prompt: SimpleNamespace(task_id="task-" + prompt.split()[-1]))
    controller.store = LocalStore(tmp_path / "store")
    controller.store.put(winner_batch(parent.pending.crown_round_id), body)
    parent = dataclasses.replace(parent, pending=dataclasses.replace(parent.pending, batch_sha256=hashlib.sha256(body).hexdigest()))
    assert controller._training_task_exclusions(parent) == {"task-1", "task-2"}


def test_live_params_json_round_trip_keeps_hashes_and_opt_in_sft_bound():
    live = Path(__file__).parents[1] / "ops" / "train64k_params_t3600_prune_open_kl8k_confirm.json"
    data = json.loads(live.read_text())
    restored = Params.from_dict(data)
    assert restored.to_dict() | {"kl_rollouts": 128} == data
    assert restored.params_hash == "46791df11c9b8b55a83bfed44f9c579900aa2253b3ae6716c667e2f01582f876"
    override = Path(__file__).parents[1] / "ops" / "train64k_params_v4_sft.json"
    enabled = Params.from_dict(json.loads(override.read_text()))
    assert enabled.recipe.bound("sft_weight") == SFT_MINER_BOUND
    assert enabled.recipe.sft_weight == 1.0
    assert Params.from_dict(enabled.to_dict()) == enabled
    assert enabled.params_hash == "ccfd78857c3d322653dad4ea38140030676a6425b9e50ecea847e3e3beff2d84"
    assert enabled.recipe_id == "f45ab0bd7e200186219578ea43e07f651809102a27a8f0bcb854acc6d153c61f"
    with pytest.raises(ConfigError):
        Params.from_dict(enabled.to_dict() | {"max_stages": 0})
