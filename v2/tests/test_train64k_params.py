"""Opt-in full-eval-length training caps, without changing the production recipe."""
from __future__ import annotations

import json
from pathlib import Path

from sn120_v2.admission import trained_tokens
from sn120_v2.gpu.batch import fit_tokens
from sn120_v2.params import load_params

ROOT = Path(__file__).parents[1]
PRODUCTION = ROOT / "tools" / "dedup-deploy" / "new-params.json"
TRAIN64K = ROOT / "ops" / "train64k_params.json"


def test_train64k_loads_and_changes_only_the_two_recipe_caps():
    production_raw = json.loads(PRODUCTION.read_text())
    train64k_raw = json.loads(TRAIN64K.read_text())
    production_recipe = production_raw.pop("recipe")
    train64k_recipe = train64k_raw.pop("recipe")
    assert production_raw == train64k_raw
    assert {key for key in production_recipe.keys() | train64k_recipe.keys()
            if production_recipe.get(key) != train64k_recipe.get(key)} == {
        "train_max_output_tokens", "train_max_context_tokens",
    }
    assert (train64k_recipe["train_max_output_tokens"], train64k_recipe["train_max_context_tokens"]) == (65536, 69632)

    production = load_params(PRODUCTION)
    train64k = load_params(TRAIN64K)
    assert production.params_hash.startswith("ab4e228c")
    assert train64k.to_dict() == (production_raw | {"recipe": train64k_recipe})
    assert train64k.max_train_token_steps == production.max_train_token_steps == 16_000_000
    assert train64k.recipe.code_digest == production.recipe.code_digest == "22b2d191b32e394d4621ce477e702b50410a6ef7706cce6cfd20c60ff64e42e1"
    assert train64k.recipe_id != production.recipe_id


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
