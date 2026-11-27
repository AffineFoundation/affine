"""The Generator (mechanism 1) task-check rule in the main miner contract.

Contract text only: mining.json carries ``contract.generator_task_check`` and
llms.txt states the same rule beside the ``gen:`` reserve bullet. It reads no
parameter, so it can change neither ``params_hash`` nor any rotation input.
"""
from __future__ import annotations

import json

from sn120_v2 import public, publisher, sim
from sn120_v2.guard import judge
from sn120_v2.params import Params

ARGMAX = dict(crown_gate="argmax_symmetric", z_bar=0.5, z_field_bar=99.0, min_scored_tasks=2,
              confirm_next_round=True)
DEFAULT_PARAMS_HASH = "7eb09fa89387d1d2c5db2d8d0ffd627ad87ece8ef31bdfe93f3374f0c7bb1e50"

EXPECTED = {
    "rule": "every scored Generator task gets deterministic checks; every task that passes the 2-6/8 king rule gets an "
            "LLM judge; if any task a miner submitted in an epoch is judged broken (wrong answer/logical error, or "
            "ambiguous/underspecified with no single answer, missing figure, multi-answer, ill-posed, format mismatch, "
            "proof, truncated), that miner's Generator score for that epoch is 0 and none of its tasks enter the pool",
    "broken_categories": ["bad_key", "ambiguous", "ill_posed", "multi_answer", "format_mismatch", "truncated",
                          "needs_figure", "proof"],
    "published": "verdict category only, never reasoning or answers",
    "sidelined_ledger": "https://affine.io/generator/sidelined.jsonl",
    "guide": "https://affine.io/generator/generator-guide.md",
}


def render(tmp_path, name="world", **changes):
    params = sim.sim_params(**(ARGMAX | changes))
    snap = public.snapshot(sim.SimWorld(tmp_path / name, params).controller.state, params, 1_800_000_000)
    mining, llms = publisher.render(snap, "https://files.example")
    return json.loads(mining), llms.decode(), params


def test_mining_json_carries_the_generator_task_check_object(tmp_path):
    doc, text, params = render(tmp_path)
    assert doc["contract"]["generator_task_check"] == EXPECTED
    assert publisher.check_pair(json.dumps(doc).encode(), text.encode(), params) == []


def test_the_broken_categories_are_lowercase_guard_reject_verdicts():
    categories = publisher.GENERATOR_TASK_CHECK["broken_categories"]
    assert all(category == category.lower() for category in categories)
    assert {category.upper() for category in categories} <= set(judge.VERDICTS) - {"OK", "TRIVIAL"}


def test_llms_txt_states_the_rule_in_the_generator_area_and_the_changelog(tmp_path):
    _doc, text, _params = render(tmp_path)
    flat = " ".join(text.split())
    assert "- **Generator task check (mechanism 1):** every scored Generator task gets deterministic checks" in flat
    assert "passes the 2-6/8 king rule also gets an LLM judge" in flat
    assert "that miner's Generator score for that epoch is 0 and none of its tasks enter the pool" in flat
    assert "Only the verdict category is published, never reasoning or answers" in flat
    assert "`https://affine.io/generator/sidelined.jsonl`" in flat
    assert "`https://affine.io/generator/generator-guide.md`" in flat
    # Beside the gen: reserve bullet, inside the live task bodies section.
    section = text.index("### Live task bodies and rotation")
    gen = text.index("- **`gen:` / reserve IDs:**", section)
    check = text.index("- **Generator task check (mechanism 1):**", section)
    assert gen < check < text.index("- **Rotation-v2:**", section)
    assert flat.count("- 2026-11-27 / Generator task check / unpublished →") == 1
    # Both other 2026-11-27 entries are still there, once each.
    assert flat.count("- 2026-11-27 / `rotation_rule` / `info` → `info-guard`") == 1
    assert flat.count("- 2026-11-27 / `z_crown` /") == 1


def test_the_rule_is_contract_text_and_moves_no_hash(tmp_path):
    assert Params().params_hash == DEFAULT_PARAMS_HASH
    first, _text, params = render(tmp_path, "one")
    second, _text, moved = render(tmp_path, "two", z_crown=1.5)
    assert params.params_hash != moved.params_hash  # the params do differ ...
    # ... but the published rule does not read them.
    assert first["contract"]["generator_task_check"] == second["contract"]["generator_task_check"]
    assert "generator_task_check" not in json.dumps(params.to_dict())
    assert first["contract"]["params_hash"] == params.params_hash
    # The contract object is a copy: editing a rendered document cannot change the next one.
    first["contract"]["generator_task_check"]["broken_categories"].append("x")
    assert publisher.contract(params, "https://files.example")["generator_task_check"] == EXPECTED
