"""Symmetric mean-pass-rate crown gate: the king is another competitor."""
from __future__ import annotations

import dataclasses
import json

import pytest

from sn120_v2.judging import TIE_TOLERANCE, decide, drop_at_look, score_entry
from sn120_v2.types import LOST, WINNER

ARGMAX = dict(crown_gate="argmax_symmetric", z_bar=0.0, z_field_bar=99.0, min_scored_tasks=2)
LOOKS = dict(looks=(0.5, 1.0))


def settle(build, null, loras, weights=None, **changes):
    n_tasks = len(null[0]) if isinstance(null, tuple) else len(null)
    params = build.params(n_tasks=n_tasks, **(ARGMAX | changes))
    ticket = build.ticket(len(loras), n_tasks, params, weights=weights)
    return decide([(ticket, build.result(ticket, null, loras))])


def test_king_best_stays_with_decided_reason(build):
    decision = settle(build, "PPPP" * 5, {0: "PPFF" * 5, 1: "FFFF" * 5})
    assert decision.winner_queue_pos is None
    assert decision.reason == "king had the highest mean pass rate"
    assert all(entry.status == LOST for entry in decision.entries)


def test_clear_winner_crowns_without_z(build):
    decision = settle(build, "F" * 20, {0: "P" * 20, 1: "F" * 20})
    assert decision.winner_queue_pos == 0
    assert decision.winner.status == WINNER
    assert decision.winner.score.z == 0  # identical nonzero deltas have zero SE
    assert decision.winner.score.field_delta is None
    assert decision.winner.score.field_se is None
    assert decision.winner.score.field_z is None
    assert decision.reason == "entry 0 had the highest mean pass rate, above the king"


def test_tiny_above_tolerance_crowns_and_tie_with_king_stays(build):
    assert settle(build, "PPFF", {0: "PPPF"}).winner_queue_pos == 0
    tied = settle(build, "PPFF", {0: "PPFF", 1: "PFPF"})
    assert tied.winner_queue_pos is None
    assert tied.reason == "king had the highest mean pass rate"
    assert all(entry.score.delta <= TIE_TOLERANCE for entry in tied.entries)


def test_tie_between_candidates_goes_to_earlier_queue(build):
    decision = settle(build, "F" * 8, {0: "P" * 4 + "F" * 4, 1: "PPFFPPFF"})
    assert decision.winner_queue_pos == 0
    assert decision.entries[0].status == WINNER
    assert decision.entries[1].status == LOST


def test_all_null_above_king_crowns(build):
    decision = settle(build, "PFFF" * 5, {0: "PFFF" * 5, 1: "PPFF" * 5, 2: "FFFF" * 5})
    assert decision.winner_queue_pos == 1


def test_whole_field_below_king_stays(build):
    decision = settle(build, "PPPF" * 5, {0: "PPFF" * 5, 1: "PFFF" * 5, 2: "FFFF" * 5})
    assert decision.winner_queue_pos is None
    assert decision.reason == "king had the highest mean pass rate"


def test_argmax_zero_bar_ignores_z_and_positive_bar_is_strict(build):
    for null, lora in (("F" * 20, "P" * 20), ("P" * 10 + "F" * 10, "P" * 11 + "F" * 9)):
        params = build.params(**(ARGMAX | {"n_tasks": 20}))
        old = score_entry((1.0,) * 20, (null,), (lora,), dataclasses.replace(params, z_bar=0.0), n_total=20)
        new = score_entry((1.0,) * 20, (null,), (lora,), dataclasses.replace(params, z_bar=1.0), n_total=20)
        assert not old.reasons or "z " not in " ".join(old.reasons)
        if old.se == 0:
            assert old.eligible and not new.eligible and new.reasons == ("z 0.000 is not above 1",)
        elif old.z < 1:
            assert old.eligible and not new.eligible and new.reasons == (f"z {old.z:.3f} is not above 1",)
    exact = score_entry((1.0,) * 2, ("FF",), ("PF",), build.params(n_tasks=2, **ARGMAX), n_total=2)
    assert exact.z == 1.0 and exact.se > 0
    assert exact.eligible
    assert not score_entry((1.0,) * 2, ("FF",), ("PF",), build.params(n_tasks=2, **(ARGMAX | {"z_bar": 1.0})), n_total=2).eligible
    above = score_entry((1.0,) * 3, ("FFF",), ("PPF",), build.params(n_tasks=3, **(ARGMAX | {"z_bar": 1.0})), n_total=3)
    assert above.z > 1 and above.eligible
    decision = settle(build, "PPFF" * 5, {0: "PPPF" * 5}, z_field_bar=-1e9)
    assert decision.winner_queue_pos == 0, "z_field_bar stays ignored"


def test_r000095_like_weighted_king_and_cap_outcomes(build):
    # Two king runs are averaged per task before applying the unequal HT weights.
    # The king's weighted pass rate is .60; the candidate's is .66. Its seven C
    # outcomes count as not-pass, but remain below the separate cap-hit filter.
    decision = settle(
        build,
        ("P" * 60 + "F" * 40, "P" * 20 + "F" * 20 + "P" * 40 + "F" * 20),
        {0: "P" * 66 + "F" * 27 + "C" * 7, 1: "P" * 64 + "F" * 36},
        weights=tuple(1.0 if task % 2 == 0 else 2.0 for task in range(100)),
        null_samples_per_task=2,
        z_bar=0.0,
        z_field_bar=1e9,
    )
    assert decision.winner_queue_pos == 0
    assert decision.winner.score.pass_rate == pytest.approx(0.66)
    assert decision.winner.score.null_pass_rate == pytest.approx(0.60)
    assert decision.winner.score.delta == pytest.approx(0.06)
    assert decision.winner.score.cap_rate == pytest.approx(0.07)
    assert decision.winner.score.field_z is None


def test_publisher_renders_argmax_rule(tmp_path):
    from sn120_v2 import publisher, sim

    world = sim.SimWorld(tmp_path / "world", sim.sim_params(**ARGMAX))
    snapshot = __import__("sn120_v2.public", fromlist=["snapshot"]).snapshot(
        world.controller.state, world.params, world.clock(), {}, 0,
    )
    mining, llms = publisher.render(snapshot)
    text = llms.decode()
    assert "The highest mean pass rate among the king and the candidates wins; the king stays if" in text
    assert "stays if it is best (ties go to the king, then to the earlier queue position). There is no z threshold; the" in text
    assert "old z bars are retired." in text
    assert "duplicates are removed first" in text
    assert "round-wide training seed" in text
    assert "There are no minimum-signal or update-norm floors" in text
    assert "Merged evaluation is unchanged." in text
    assert "Early drops are disabled" in text
    assert "anti-gaming pre-filters, not z gates" in " ".join(text.split())
    assert "mean pass rate is the HT-weighted mean of per-task pass rates" in " ".join(text.split())
    assert "each king task rate is the mean of its null runs" in " ".join(text.split())
    assert "`C` counts as not-pass" in " ".join(text.split())
    assert publisher.check_pair(mining, llms, world.params) == []


def test_publisher_shows_enabled_crown_and_confirmation_z_bar(tmp_path):
    from sn120_v2 import publisher, sim

    world = sim.SimWorld(tmp_path / "world", sim.sim_params(**ARGMAX, confirm_next_round=True))
    world.controller.set_params(dataclasses.replace(world.params, z_bar=1.0), world.clock())
    mining, llms = publisher.render(__import__("sn120_v2.public", fromlist=["snapshot"]).snapshot(
        world.controller.state, world.controller.params, world.clock(), {}, 0,
    ))
    text = " ".join(llms.decode().split())
    assert "delta above the tie tolerance (1e-12) AND z strictly above `z_bar` (1)" in text
    assert "fresh paired z is strictly above `z_bar` (1)" in text
    assert "pending crown's recorded crown-round z is strictly above `z_bar` (1)" in text
    assert "z == `z_bar` fails on both rounds" in text
    assert "`z_bar` and `z_field_bar` are ignored" not in text
    assert "There is no z threshold" not in text
    assert "`z_field_bar` is ignored" in text
    confirmation = json.loads(mining)["contract"]["crown_confirmation"]
    assert confirmation["z_bar"] == 1.0
    assert "fresh paired z > z_bar (1)" in confirmation["confirm_rule"]
    assert "recorded crown-round z > z_bar (1)" in confirmation["confirm_rule"]
    assert publisher.check_pair(mining, llms, world.controller.params) == []


def test_argmax_never_drops_at_early_look(build):
    params = build.params(n_tasks=100, **ARGMAX, **LOOKS)
    score = score_entry((1.0,) * 50, ("P" * 50,), ("F" * 50,), params)
    assert drop_at_look(score, 0, 100, params) is False
    ticket = build.ticket(1, 100, params)
    result = build.result(ticket, "P" * 50 + "F" * 50, {
        0: build.dropped(ticket, 0, "F" * 50 + "-" * 50, 0),
    })
    with pytest.raises(Exception, match="argmax_symmetric has no early drops"):
        decide([(ticket, result)])
    # Other modes retain their existing early-drop predicate.
    median_params = build.params(n_tasks=100, crown_gate="field_median", z_bar=0.0, z_field_bar=0.0, **LOOKS)
    median_score = score_entry((1.0,) * 50, ("PPFF" * 12 + "PP",), ("FFPP" * 12 + "FF",), median_params)
    assert median_score.se > 0 and drop_at_look(median_score, 0, 100, median_params) is False
    median_score = score_entry((1.0,) * 50, ("P" * 50,), ("PPFF" * 12 + "PP",), median_params)
    assert median_score.se > 0 and drop_at_look(median_score, 0, 100, median_params) is True
