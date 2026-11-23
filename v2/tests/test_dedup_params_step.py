"""The dedup operator step preserves a king-x2 journal's full Params record."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import sn120_v2
from sn120_v2.canonical import file_sha256, manifest_digest
from sn120_v2.cli import changed_params, main as cli_main
from sn120_v2.journal import JOURNAL_FILE, read_journal
from sn120_v2.params import Params
from sn120_v2.sim import SimWorld, sim_params
from sn120_v2.state import load

SCRIPT = Path(__file__).parents[1] / "ops" / "dedup_params_step.py"
_spec = importlib.util.spec_from_file_location("dedup_params_step", SCRIPT)
assert _spec is not None and _spec.loader is not None
step = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step)
assert sys.modules["sn120_v2"] is sn120_v2
GPU_DIGEST = manifest_digest({
    path.name: file_sha256(path)
    for path in sorted((SCRIPT.parents[1] / "sn120_v2/gpu").glob("*.py"))
})


@pytest.mark.parametrize("king_samples", [None, 2])
def test_full_step_changes_release_keys_on_a_kingx2_sim_journal(tmp_path, capsys, king_samples):
    # Simulate a legacy journal with nondefault historical dedup settings:
    # every requested value differs, plus merged eval and the release GPU code.
    # Explicit replay bookkeeping keeps the approved no-op values visible.
    old = sim_params(
        irt_draw_rule="sqrt_fisher", null_samples_per_task=king_samples,
        dedup_enabled=False, train_seed_scope="batch", z_bar=0.75,
        min_signal_rollouts=64, min_step_norm=0.02,
        dedup_rollout_overlap=0.75, dedup_cos_shared_seed=0.2, dedup_cos_own_seed=0.01,
    )
    world = SimWorld(tmp_path / "world", old)
    pinned = load(world.root / JOURNAL_FILE).params
    assert pinned is not None and pinned.irt_draw_rule == "sqrt_fisher"
    assert pinned.null_samples_per_task == king_samples
    assert all(name in pinned.to_dict() for name in (
        "dedup_rollout_overlap", "dedup_cos_shared_seed", "dedup_cos_own_seed",
        "min_signal_rollouts", "min_step_norm", "z_bar",
    ))
    before_bytes = (world.root / JOURNAL_FILE).read_bytes()
    before_records = len(read_journal(world.root / JOURNAL_FILE))
    output = tmp_path / f"dedup-{king_samples}.json"

    # Exercise the real standalone-script entry point and its printed preview.
    assert step.main(["--state-dir", str(world.root), "--output", str(output)]) == 0
    preview = json.loads(capsys.readouterr().out)
    full = json.loads(output.read_text())
    assert full == pinned.to_dict() | step.DEDUP_FIELDS | {
        "recipe": pinned.to_dict()["recipe"] | {"code_digest": GPU_DIGEST},
    }
    assert full["irt_draw_rule"] == "sqrt_fisher"
    assert full["recipe"]["code_digest"] == GPU_DIGEST
    if king_samples is not None:
        assert full["null_samples_per_task"] == king_samples
    else:
        assert "null_samples_per_task" not in full
    assert (world.root / JOURNAL_FILE).read_bytes() == before_bytes
    assert len(read_journal(world.root / JOURNAL_FILE)) == before_records

    expected = {
        "train_seed_scope": {"before": "batch", "after": "round"},
        "dedup_enabled": {"before": False, "after": True},
        "dedup_rollout_overlap": {"before": 0.75, "after": 0.5},
        "dedup_cos_shared_seed": {"before": 0.2, "after": 0.135},
        "dedup_cos_own_seed": {"before": 0.01, "after": 0.002},
        "min_signal_rollouts": {"before": 64, "after": 1},
        "min_step_norm": {"before": 0.02, "after": 0.0},
        "z_bar": {"before": 0.75, "after": 1.5},
        "eval_merged": {"before": False, "after": True},
        "recipe.code_digest": {"before": "", "after": GPU_DIGEST},
    }
    assert preview["changed"] == expected
    assert {name for name in preview["changed"]} == set(Params.LEGACY_FIELDS) - {"crown_gate", "z_field_bar"} | {
        "z_bar", "eval_merged", "recipe.code_digest",
    }

    # Feed the resulting file to the actual set-params command, not to_dict or
    # a synthetic replacement of the command's input parser.
    assert cli_main([
        "set-params", "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack",
        "--params", str(output),
    ]) == 0
    applied = json.loads(capsys.readouterr().out)
    after = load(world.root / JOURNAL_FILE).params
    assert after is not None
    assert changed_params(pinned, after) == expected
    assert applied["changed"] == expected
    assert applied["params_hash_before"] == preview["params_hash_before"] == pinned.params_hash
    assert applied["params_hash"] == preview["params_hash"] == after.params_hash
    assert after.to_dict() == full
    assert after.null_samples_per_task == king_samples
    assert after.irt_draw_rule == "sqrt_fisher"
    assert after.recipe == dataclasses.replace(pinned.recipe, code_digest=GPU_DIGEST)
    # All other effective fields and their defaults are unchanged too.
    for field in dataclasses.fields(Params):
        if not field.name.startswith("_") and field.name not in step.DEDUP_FIELDS.keys() | {"recipe"}:
            assert getattr(after, field.name) == getattr(pinned, field.name), field.name


def test_legacy_kingx2_fields_become_explicit_without_other_effective_changes(tmp_path, capsys):
    world = SimWorld(tmp_path / "world", sim_params(irt_draw_rule="sqrt_fisher", null_samples_per_task=2))
    pinned = load(world.root / JOURNAL_FILE).params
    assert pinned is not None and all(name not in pinned.to_dict() for name in Params.LEGACY_FIELDS)
    output = tmp_path / "dedup.json"
    assert step.main(["--state-dir", str(world.root), "--output", str(output)]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert set(preview["changed"]) == {"train_seed_scope", "dedup_enabled", "z_bar", "eval_merged", "recipe.code_digest"}
    assert cli_main([
        "set-params", "--state-dir", str(world.root), "--stack", "sn120_v2.sim:stack", "--params", str(output),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["changed"] == preview["changed"]
    after = load(world.root / JOURNAL_FILE).params
    assert after is not None and after.to_dict() == json.loads(output.read_text())
    assert all(name in after.to_dict() for name in Params.LEGACY_FIELDS)
    assert after.null_samples_per_task == 2 and after.irt_draw_rule == "sqrt_fisher"
    assert after.recipe == dataclasses.replace(pinned.recipe, code_digest=GPU_DIGEST)
    for field in dataclasses.fields(Params):
        if not field.name.startswith("_") and field.name not in step.DEDUP_FIELDS.keys() | {"recipe"}:
            assert getattr(after, field.name) == getattr(pinned, field.name), field.name


def test_step_refuses_to_overwrite_a_reviewed_file_and_leaves_journal_untouched(tmp_path, capsys):
    world = SimWorld(tmp_path / "world", sim_params(null_samples_per_task=2, irt_draw_rule="sqrt_fisher"))
    output = tmp_path / "reviewed.json"
    assert step.main(["--state-dir", str(world.root), "--output", str(output)]) == 0
    capsys.readouterr()
    output.write_text("operator review\n")
    before = (world.root / JOURNAL_FILE).read_bytes()
    with pytest.raises(FileExistsError):
        step.main(["--state-dir", str(world.root), "--output", str(output)])
    assert output.read_text() == "operator review\n"
    assert (world.root / JOURNAL_FILE).read_bytes() == before
