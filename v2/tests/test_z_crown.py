"""The crown bar (``z_crown``, else ``z_bar``) is separate from the confirmation bar (``z_bar``)."""
from __future__ import annotations

import dataclasses
import json
import math

import pytest

from sn120_v2 import public, publisher, sim
from sn120_v2.errors import ConfigError
from sn120_v2.judging import score_confirmation, score_entry
from sn120_v2.params import Params

# The default digest from before the split. Unset z_crown must leave it, and the JSON that it hashes, unchanged.
DEFAULT_PARAMS_HASH = "7eb09fa89387d1d2c5db2d8d0ffd627ad87ece8ef31bdfe93f3374f0c7bb1e50"
ARGMAX = dict(crown_gate="argmax_symmetric", z_bar=0.5, z_field_bar=99.0, min_scored_tasks=2)
LORA, NULL = "PP" + "F" * 10, "F" * 12  # z is about 1.483: above 0.5 and 1.0, not above 1.5


def score(build, lora: str, null: str, **changes):
    params = build.params(n_tasks=len(null), **(ARGMAX | changes))
    return score_entry((1.0,) * len(null), (null,), (lora,), params, n_total=len(null))


def test_unset_z_crown_keeps_the_default_digest_and_json():
    assert Params().params_hash == DEFAULT_PARAMS_HASH
    assert Params(z_crown=None).params_hash == DEFAULT_PARAMS_HASH
    assert "z_crown" not in Params().to_dict()


def test_set_z_crown_changes_the_digest_and_roundtrips():
    params = dataclasses.replace(Params(), crown_gate="argmax_symmetric", z_crown=1.5)
    assert params.params_hash != Params().params_hash
    assert params.to_dict()["z_crown"] == 1.5
    restored = Params.from_dict(params.to_dict())
    assert restored.z_crown == 1.5 and restored.params_hash == params.params_hash
    assert Params.from_dict(Params().to_dict()).z_crown is None


def test_crown_bar_is_z_crown_when_set_and_z_bar_otherwise():
    symmetric = Params(crown_gate="argmax_symmetric")
    assert symmetric.crown_bar == symmetric.z_bar == 1.0
    assert dataclasses.replace(symmetric, z_bar=0.5, z_crown=1.5).crown_bar == 1.5
    assert dataclasses.replace(symmetric, z_bar=0.5, z_crown=0.0).crown_bar == 0.0


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_z_crown_must_be_finite_and_nonnegative(bad):
    with pytest.raises(ConfigError):
        dataclasses.replace(Params(crown_gate="argmax_symmetric"), z_crown=bad)


def test_z_crown_is_refused_without_the_symmetric_gate():
    with pytest.raises(ConfigError):
        Params(z_crown=1.5)  # the default gate is field_median, which never reads it


def test_crown_needs_strict_z_above_z_crown_and_unset_z_crown_uses_z_bar(build):
    base = score(build, LORA, NULL)
    assert 1.0 < base.z <= 1.5 and base.eligible, "unset z_crown: the crown bar is z_bar = 0.5"
    gated = score(build, LORA, NULL, z_crown=1.5)
    assert not gated.eligible and gated.reasons == (f"z {base.z:.3f} is not above 1.5",)
    assert score(build, "PPP" + "F" * 9, NULL, z_crown=1.5).eligible, "z is about 1.915, above 1.5"


def test_the_crown_bar_is_strict_at_the_exact_z(build):
    z = score(build, LORA, NULL, z_bar=0.0).z
    assert not score(build, LORA, NULL, z_bar=0.0, z_crown=z).eligible, "z equal to the crown bar is not a win"
    assert score(build, LORA, NULL, z_bar=0.0, z_crown=math.nextafter(z, 0.0)).eligible


def test_confirmation_keeps_its_own_bar_and_ignores_the_crown_bar():
    weights, new, old = (1.0,) * 12, (LORA,), (NULL,)
    # The controller passes ticket.params.z_bar here, so a crown bar of 1.5 changes nothing for the fresh test.
    passed = score_confirmation(weights, new, old, z_bar=0.5, crown_z=1.9)
    assert passed["confirmed"] and passed["reason"] == "fresh paired sample-mean delta is positive"
    failed = score_confirmation(weights, new, old, z_bar=1.5, crown_z=1.9)
    assert not failed["confirmed"] and failed["reason"] == f"fresh paired z {failed['z']:.3f} is not above 1.5"
    old_crown = score_confirmation(weights, new, old, z_bar=0.5, crown_z=0.4)
    assert not old_crown["confirmed"] and old_crown["reason"] == "crown-round z 0.400 is not above 0.5"


def _render(tmp_path, **changes):
    params = sim.sim_params(**(ARGMAX | {"confirm_next_round": True}) | changes)
    snap = public.snapshot(sim.SimWorld(tmp_path / "world", params).controller.state, params, 1_800_000_000)
    mining, llms = publisher.render(snap, "https://files.example")
    return json.loads(mining), llms.decode(), params


def test_unset_z_crown_publishes_the_existing_wording_and_no_z_crown_row(tmp_path):
    doc, text, params = _render(tmp_path)
    flat = " ".join(text.split())
    assert "z strictly above `z_bar` (0.5)" in flat
    assert "fresh paired z is strictly above `z_bar` (0.5)" in flat
    assert "| `z_crown` |" not in text and "z_crown" not in doc["params"]
    assert publisher.check_pair(json.dumps(doc).encode(), text.encode(), params) == []


def test_set_z_crown_publishes_the_crown_bar_and_the_confirmation_bar_apart(tmp_path):
    doc, text, params = _render(tmp_path, z_crown=1.5)
    flat = " ".join(text.split())
    assert "z strictly above `z_crown` (1.5)" in flat
    assert "`z_crown=1.5`, so `z > 1.5`" in flat
    assert "fresh paired z is strictly above `z_bar` (0.5)" in flat
    assert "recorded crown-round z is strictly above `z_bar` (0.5)" in flat
    assert "| `z_crown` | 1.5 |" in text
    assert doc["params"]["z_crown"] == 1.5 and doc["params"]["z_bar"] == 0.5
    confirmation = doc["contract"]["crown_confirmation"]
    assert confirmation["z_bar"] == 0.5 and "fresh paired z > z_bar (0.5)" in confirmation["confirm_rule"]
    assert publisher.check_pair(json.dumps(doc).encode(), text.encode(), params) == []


def test_changelog_names_the_split_on_its_own_line(tmp_path):
    _, text, _ = _render(tmp_path)
    flat = " ".join(text.split())  # llms.txt wraps long entries; the entry is one line in the source
    assert flat.count("- 2026-11-27 / `z_crown` / crown and confirmation shared `z_bar`") == 1
    assert "the crown uses `z_crown` (strict `z > 1.5` in the approved params)" in flat
    assert "confirmation keeps `z_bar` (fresh paired `z > 0.5` and recorded crown-round `z > 0.5`)" in flat
    assert "`confirm_next_round` stays on; existing params hashes are unchanged." in flat
