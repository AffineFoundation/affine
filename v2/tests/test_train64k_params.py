"""Opt-in full-eval-length training caps, without changing the production recipe."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sn120_v2.admission import check_batch, trained_tokens
from sn120_v2.gpu.batch import fit_tokens
from sn120_v2.params import load_params
from sn120_v2.public import snapshot
from sn120_v2.publisher import check_pair, render
from sn120_v2.sim import SimWorld

ROOT = Path(__file__).parents[1]
PRODUCTION = ROOT / "tools" / "dedup-deploy" / "new-params.json"
TRAIN64K = ROOT / "ops" / "train64k_params.json"
TRAIN64K_T3600 = ROOT / "ops" / "train64k_params_t3600.json"
TRAIN64K_T3600_PRUNE = ROOT / "ops" / "train64k_params_t3600_prune.json"
LIVE_WITH_PRUNE = ROOT / "ops" / "null_prune_params.json"
CAP_CHANGES = {
    "recipe.train_max_output_tokens": (32768, 65536),
    "recipe.train_max_context_tokens": (36864, 69632),
    "max_train_token_steps": (16000000, 22130000),
}


@pytest.mark.parametrize("path,extra", [
    (TRAIN64K, {}),
    (TRAIN64K_T3600, {"train_time_budget_s": (2400, 3600)}),
])
def test_train64k_loads_and_changes_only_the_approved_values(path, extra):
    expected = CAP_CHANGES | extra
    production_raw = json.loads(PRODUCTION.read_text())
    train64k_raw = json.loads(path.read_text())
    production_flat = production_raw | {f"recipe.{key}": value for key, value in production_raw.pop("recipe").items()}
    train64k_flat = train64k_raw | {f"recipe.{key}": value for key, value in train64k_raw.pop("recipe").items()}
    assert set(production_flat) == set(train64k_flat)
    differences = {key for key in production_flat if production_flat[key] != train64k_flat[key]}
    assert differences == set(expected)
    assert len(differences) == len(expected) == (4 if extra else 3)
    for key, (before, after) in expected.items():
        assert (production_flat[key], train64k_flat[key]) == (before, after)

    production, train64k = load_params(PRODUCTION), load_params(path)
    assert production.params_hash == "ab4e228c961058a29ed062bc67f6da6881cf3c13396eb03730d54badb4f261c0"
    assert train64k.to_dict() == json.loads(path.read_text())
    assert train64k.recipe.code_digest == production.recipe.code_digest == "22b2d191b32e394d4621ce477e702b50410a6ef7706cce6cfd20c60ff64e42e1"
    assert train64k.recipe_id == "61c2a8e642fbeb953d921d4fd2b4d0f388ce32d166ac091323e26e3964a3efe6"
    assert train64k.params_hash == {
        TRAIN64K: "a8d0f5fe50a1d47de9cb5bac618453321ce58d460f75618290a0db3001d695ac",
        TRAIN64K_T3600: "05b688fcc0ceddb4bc78b56cef471110a9b9b107cc82bf2b6318c7d1ad355f40",
    }[path]
    assert train64k.train_time_budget_s < train64k.round_time_limit_s - train64k.promote_reserve_s == 10200


def test_50000_token_output_is_whole_at_64k_and_truncated_at_32k():
    production = load_params(PRODUCTION)
    train64k = load_params(TRAIN64K)
    n_prompt, n_output, eos = 3_000, 50_000, -1
    output = list(range(n_output))

    kept, capped, truncated = fit_tokens(
        n_prompt, output, eos, train64k.recipe.train_max_output_tokens, train64k.recipe.train_max_context_tokens,
    )
    assert (kept, capped, truncated) == (output + [eos], False, False)
    assert trained_tokens(n_prompt, n_output, train64k.recipe) == n_prompt + len(kept) == 53_001

    kept, capped, truncated = fit_tokens(
        n_prompt, output, eos, production.recipe.train_max_output_tokens, production.recipe.train_max_context_tokens,
    )
    assert (kept, capped, truncated) == (output[:32768], True, True)
    assert trained_tokens(n_prompt, n_output, production.recipe) == n_prompt + len(kept) == 35_768


class WordTokens:
    def prompt_tokens(self, prompt: str) -> int:
        return 2_998 + len(prompt.split())

    def output_tokens(self, text: str) -> int:
        return len(text.split())


def test_two_60000_token_outputs_fit_admission_room_and_cost_cap():
    params = load_params(TRAIN64K)
    prompt = "synthetic prompt"
    output = "word " * 60_000
    raw = json.dumps({
        "version": "affine-batch-v3", "hotkey": "5Train64k",
        "groups": [{"prompt": prompt, "rollouts": [
            {"output": output, "reward": 1.0}, {"output": output, "reward": 0.0},
        ]}],
    }).encode()
    tokens = WordTokens()
    n_prompt = tokens.prompt_tokens(prompt)
    room = min(params.recipe.train_max_output_tokens, params.recipe.train_max_context_tokens - n_prompt)
    assert (n_prompt, room) == (3000, 65536)
    assert trained_tokens(n_prompt, 60_000, params.recipe) == 63_001
    batch = check_batch(raw, "5Train64k", params, tokens=tokens)
    assert batch.checked == ("format", "tokens")
    assert batch.train_token_steps == 2 * 63_001 == 126_002
    assert batch.train_token_steps <= params.max_train_token_steps == 22_130_000


def test_train64k_prune_retains_live_params_and_changes_only_four_values(tmp_path):
    live_raw = json.loads(LIVE_WITH_PRUNE.read_text())
    selected_raw = json.loads(TRAIN64K_T3600_PRUNE.read_text())
    assert live_raw["null_prune_from_round"] == selected_raw["null_prune_from_round"] == "r000098"
    assert selected_raw == json.loads(TRAIN64K_T3600.read_text()) | {"null_prune_from_round": "r000098"}
    live, selected = load_params(LIVE_WITH_PRUNE), load_params(TRAIN64K_T3600_PRUNE)
    assert live.params_hash == "1449210f89f30a65dc3f6e8d4ac956f5f2c1c8d37364a46b20e5cfb01c24eace"
    assert selected.params_hash == "c5d4405bcfb01b68766f97409f7ce1bfc98f875cf77af661421a8a3a28df2057"
    assert selected.recipe_id == "61c2a8e642fbeb953d921d4fd2b4d0f388ce32d166ac091323e26e3964a3efe6"
    assert selected.recipe.code_digest == live.recipe.code_digest == "22b2d191b32e394d4621ce477e702b50410a6ef7706cce6cfd20c60ff64e42e1"
    from sn120_v2.cli import changed_params, main as cli_main
    assert set(changed_params(live, selected)) == set(CAP_CHANGES) | {"train_time_budget_s"}

    # Exercise the real CLI against a disposable simulated state only.
    world = SimWorld(tmp_path / "world", live)
    assert cli_main([
        "set-params", "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack", "--params", str(TRAIN64K_T3600_PRUNE),
    ]) == 0
    from sn120_v2.state import load
    assert load(world.root / "journal.jsonl").params.params_hash == selected.params_hash
    assert cli_main([
        "set-params", "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack", "--params", str(LIVE_WITH_PRUNE),
    ]) == 0
    assert load(world.root / "journal.jsonl").params.params_hash == live.params_hash


@pytest.mark.parametrize("path,minutes", [(TRAIN64K, 40), (TRAIN64K_T3600, 60), (TRAIN64K_T3600_PRUNE, 60)])
def test_publisher_shows_the_budget_and_training_time(tmp_path, path, minutes):
    params = load_params(path)
    world = SimWorld(tmp_path / "world", params)
    snap = snapshot(world.controller.state, params, 1_800_000_000)
    mining, llms = render(snap, "https://example.org", "https://example.org/links")
    text = llms.decode()
    assert "22130000" in text
    assert f"{minutes} minutes" in text
    assert f"| `train_time_budget_s` | {minutes * 60} |" in text
    assert "| `train_max_output_tokens` | 65536 |" in text
    assert "| `train_max_context_tokens` | 69632 |" in text
    assert check_pair(mining, llms) == []
