"""The final-look, field-relative per-task-median crown gate."""
from __future__ import annotations

import dataclasses
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from sn120_v2.judging import decide, field_differences, score_entry
from sn120_v2.params import Params
from sn120_v2.types import LOST, WINNER

FIELD = dict(crown_gate="field_median", z_bar=1.0, z_field_bar=1.0, min_scored_tasks=2)
LOOKS = dict(looks=(0.5, 1.0))


def settle(build, null, loras, weights=None, n_tasks=None, **changes):
    params = build.params(n_tasks=n_tasks or len(null), **FIELD, **changes)
    ticket = build.ticket(len(loras), len(null), params, weights=weights)
    return decide([(ticket, build.result(ticket, null, loras))])


def test_all_null_field_never_crowns(build):
    outcomes = "PFPFPFPFPFPF"
    decision = settle(build, outcomes, {0: outcomes, 1: outcomes, 2: outcomes})
    assert decision.winner_queue_pos is None
    assert [entry.status for entry in decision.entries] == [LOST] * 3
    for entry in decision.entries:
        assert entry.score.field_delta == entry.score.field_se == entry.score.field_z == 0.0
        assert entry.score.reasons == (
            "z 0.000 is not above 1", "field z 0.000 is not above 1",
        )


def test_one_clear_winner_among_nulls_crowns(build):
    # All differences are +.5, hence king z and field z are both infinite.
    # A nonzero standard error is required; one early failure makes both finite.
    null = "F" * 20
    winner = "PF" + "P" * 18
    near_null = "F" * 20
    decision = settle(build, null, {0: near_null, 1: winner, 2: near_null})
    assert decision.winner_queue_pos == 1
    assert decision.winner.status == WINNER
    assert decision.entries[1].score.field_z > 1
    assert "field's per-task median by more than 1 SE" in decision.reason


def test_field_below_king_does_not_crown_even_its_best_member(build):
    null = "PPFF" * 10
    best = "P" * 21 + "F" * 19
    worse = "P" * 10 + "F" * 30
    decision = settle(build, null, {0: best, 1: worse, 2: worse})
    assert decision.winner_queue_pos is None
    assert decision.entries[0].score.delta > 0
    assert decision.entries[0].score.z < 1
    assert "z 0.227 is not above 1" in decision.entries[0].reason
    assert decision.entries[0].score.field_z > 1  # field-relative is not enough


def test_junk_flood_and_mediocre_batch_must_clear_both_bars(build):
    # Six junk batches fail most tasks. The mediocre batch barely beats the
    # king, and neither batch has enough consistent signal to clear both bars.
    null = ("P" * 4 + "F" * 12) * 4
    mediocre = ("P" * 5 + "F" * 11) * 4
    junk = ("P" * 6 + "F" * 10) * 4
    decision = settle(build, null, {0: mediocre, **{i: junk for i in range(1, 7)}})
    assert decision.winner_queue_pos is None
    assert all(entry.status == LOST for entry in decision.entries)
    assert decision.entries[0].score.field_z <= 1
    assert all(entry.score.field_z <= 0 for entry in decision.entries[1:])


def test_even_member_median_is_half_on_zero_one_split():
    # Four members at each task, including the king: the two middle values
    # are averaged rather than choosing either of them.
    assert field_differences([1, 0, 1], [[1, 0, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]], 3) == [0.5, 0.0, 0.5]


def test_n1_field_z_is_exactly_king_z(build):
    null = "PP" + "F" * 10
    winner = "PPPP" + "F" * 8
    params = build.params(n_tasks=12, **FIELD)
    ticket = build.ticket(1, 12, params)
    decision = decide([(ticket, build.result(ticket, null, {0: winner}))])
    score = decision.entries[0].score
    assert score.field_z == score.z
    assert score.field_delta == score.delta / 2
    assert score.field_se == score.se / 2
    assert score.reasons == ()


def test_n2_field_medians_and_z(build):
    null = "F" * 30
    first = "P" * 20 + "F" * 10
    second = "F" * 20 + "P" * 5 + "F" * 5
    decision = settle(build, null, {0: first, 1: second})
    assert decision.winner_queue_pos == 0
    for entry in decision.entries:
        score = entry.score
        assert score.field_z == pytest.approx(score.field_delta / score.field_se)
        assert score.field_z > 1


def test_king_gate_reproduces_legacy_decide_bit_for_bit(build):
    null = "P" * 10 + "F" * 10
    loras = {0: "P" * 15 + "F" * 5, 1: null, 2: "F" * 20}
    params = build.params(n_tasks=20)
    ticket = build.ticket(3, 20, params)
    result = build.result(ticket, null, loras)
    new = decide([(ticket, result)], duplicate_evidence=())
    source = subprocess.check_output([
        "git", "show", "9dc952b32e2eb937fb5050eef641b841a1d0f2b2:v2/sn120_v2/judging.py",
    ], cwd=Path(__file__).parents[2], text=True)
    name = "sn120_v2._field_gate_pinned_base_judging"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "sn120_v2"
    sys.modules[name] = module
    exec(compile(source, "9dc952b3:v2/sn120_v2/judging.py", "exec"), module.__dict__)
    legacy = module.decide([(ticket, result)])
    assert new == legacy
    assert all(entry.score is None or entry.score.field_z is None for entry in new.entries)
    unscored = build.ticket(1, 20, params, lease=1, first_pos=3)
    leases = [(ticket, result), (unscored, build.result(unscored, (), {3: "no_decision"}))]
    assert decide(leases) == module.decide(leases)
    assert Params.from_dict(params.to_dict()).crown_gate == "king"


def test_legacy_params_record_decodes_with_unchanged_hash():
    data = Params().to_dict()
    del data["crown_gate"]
    del data["z_field_bar"]
    restored = Params.from_dict(data)
    assert restored.crown_gate == "king" and restored.z_field_bar == 1.0
    assert restored.to_dict() == data
    assert Params.from_dict(restored.to_dict()).params_hash == restored.params_hash


def test_early_dropped_member_contributes_only_on_its_scored_tasks(build):
    # At look 0 the second entry is far enough behind to be dropped. It is
    # still in the field on those first 50 tasks, but not on later tasks.
    null = "P" * 40 + "F" * 60
    first = "P" * 20 + "F" * 20 + "P" * 40 + "F" * 20
    dropped = "P" * 20 + "F" * 30 + "-" * 50
    params = build.params(n_tasks=100, **FIELD, **LOOKS)
    ticket = build.ticket(2, 100, params)
    result = build.result(ticket, null, {
        0: first, 1: build.dropped(ticket, 1, dropped, 0),
    })
    decision = decide([(ticket, result)])
    assert decision.winner_queue_pos == 0
    assert decision.entries[0].score.field_z > 1
    assert decision.entries[1].score.field_delta is not None
    assert decision.entries[1].score.field_z == 0.0  # all-null on its prefix


def test_king_samples_are_averaged_before_the_field_median(build):
    # N=1 is exact with multiple king samples too: median(a, (k1+k2)/2).
    params = build.params(n_tasks=12, null_samples_per_task=2, **FIELD)
    ticket = build.ticket(1, 12, params)
    result = build.result(ticket, ("PFPFPFPFPFPF", "PPPFPFPFPFPF"), {0: "PPPFPFPFPFPF"})
    decision = decide([(ticket, result)])
    score = decision.entries[0].score
    assert score.field_z == score.z
    assert score.field_delta == score.delta / 2
    assert score.field_se == score.se / 2


def test_publisher_states_new_rule_and_mining_json_has_both_params(tmp_path):
    from sn120_v2 import publisher, sim

    world = sim.SimWorld(tmp_path / "world", sim.sim_params(**(FIELD | {"z_field_bar": 1.25})))
    snapshot = __import__("sn120_v2.public", fromlist=["snapshot"]).snapshot(
        world.controller.state, world.params, world.clock(), {}, 0,
    )
    mining, llms = publisher.render(snapshot)
    text = llms.decode()
    assert "The old z 1.5 king-only bar is retired." in text
    assert "field_z` > `z_field_bar` (1.25) AND paired z against the king > `z_bar`" in text
    assert "provisional and will be recalibrated." in text
    floors = ("There are no minimum-signal or update-norm floors (`min_signal_rollouts` 1,\n"
              "  `min_step_norm` 0); only non-finite or stored null updates and duplicates are removed.")
    assert text.count(floors) == 2
    import json
    doc = json.loads(mining)
    assert doc["params"]["crown_gate"] == "field_median"
    assert doc["params"]["z_field_bar"] == 1.25
    for change in ({"min_signal_rollouts": 2}, {"min_step_norm": 0.1}):
        snapshot["params"] = dataclasses.replace(world.params, **change).to_dict()
        changed = publisher.render(snapshot)[1].decode()
        assert "There are no minimum-signal or update-norm floors" not in changed


def test_new_score_fields_are_absent_when_not_used(build):
    params = build.params(n_tasks=8, crown_gate="king")
    score = score_entry((1.0,) * 8, ("PFPFPFPF",), ("PFPFPFPF",), params, n_total=8)
    assert all(key not in score.to_dict() for key in ("field_delta", "field_se", "field_z"))
    assert type(score).from_dict(score.to_dict()) == score
