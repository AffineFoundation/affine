from __future__ import annotations

import math
import random
import statistics
from statistics import NormalDist

import pytest

from sn120_v2.errors import ContractError, HeadMismatchError
from sn120_v2.judging import (
    check_result,
    decide,
    drop_at_look,
    estimate,
    pick_winner,
    score_entry,
    scored_prefix,
    upper_final_z,
)
from sn120_v2.types import LOST, NO_DECISION, SKIP, WINNER, EntryResult, Score


def runs(*parts: tuple[str, int]) -> str:
    """Build an outcome string from (code, count) runs."""
    return "".join(code * count for code, count in parts)


def one_score(build, null: str, lora: str, weights=None, **param_changes) -> Score:
    params = build.params(n_tasks=len(null), **param_changes)
    return score_entry(weights or (1.0,) * len(null), (null,), (lora,), params, n_total=len(null))


def settle(build, null, loras, n_entries=None, **param_changes):
    n_tasks = len(null) if isinstance(null, str) else len(null[0])
    params = build.params(n_tasks=n_tasks, **param_changes)
    ticket = build.ticket(n_entries or len(loras), n_tasks, params)
    return decide([(ticket, build.result(ticket, null, loras))])


# ---------------------------------------------------------------- estimator


def test_equal_weights_give_the_paired_mean_and_standard_error():
    diffs = [-1, 0, 0, 1, 0, 1, 0, 0]
    delta, se = estimate([1.0] * 8, diffs)
    assert delta == pytest.approx(statistics.mean(diffs))
    assert se == pytest.approx(statistics.stdev(diffs) / math.sqrt(8))


def test_the_estimator_does_not_change_when_all_weights_are_scaled():
    diffs = [1, 0, -1, 1, 0, 0]
    assert estimate([2.5] * 6, diffs) == pytest.approx(estimate([1.0] * 6, diffs))


def test_weighted_estimate_matches_a_hand_computation():
    # weights 1 and 3, differences 1 and 0:
    # delta = (1*1 + 3*0) / 4 = 0.25
    # var = 2/1 * ((1*0.75)^2 + (3*-0.25)^2) / 4^2 = 0.140625, so se = 0.375
    delta, se = estimate([1.0, 3.0], [1.0, 0.0])
    assert delta == pytest.approx(0.25)
    assert se == pytest.approx(0.375)


def test_a_task_with_more_weight_moves_the_estimate_more():
    light, _ = estimate([1.0, 1.0, 1.0], [1.0, 0.0, 0.0])
    heavy, _ = estimate([5.0, 1.0, 1.0], [1.0, 0.0, 0.0])
    assert heavy > light


def test_one_task_has_no_standard_error():
    assert estimate([1.0], [1.0]) == (1.0, 0.0)


def test_score_uses_the_difference_of_pass_rates_on_the_same_tasks(build):
    score = one_score(build, null="PFPFPFPF", lora="FFPPPPPF")
    diffs = [-1, 0, 0, 1, 0, 1, 0, 0]
    assert score.delta == pytest.approx(0.125)
    assert score.se == pytest.approx(statistics.stdev(diffs) / math.sqrt(8))
    assert score.z == pytest.approx(score.delta / score.se)
    assert (score.pass_rate, score.null_pass_rate) == (pytest.approx(0.625), pytest.approx(0.5))
    assert score.n_tasks == 8


def test_sampling_weights_enter_the_score(build):
    even = one_score(build, null="FFFF", lora="PFFF")
    tilted = one_score(build, null="FFFF", lora="PFFF", weights=(9.0, 1.0, 1.0, 1.0))
    assert even.delta == pytest.approx(0.25)
    assert tilted.delta == pytest.approx(0.75)


def test_two_samples_per_task_give_half_step_pass_rates(build):
    params = build.params(n_tasks=4, samples_per_task=2)
    score = score_entry((1.0,) * 4, ("PPFF", "PFFF"), ("PPPF", "PPFF"), params, n_total=4)
    # per-task rates: LoRA 1, 1, .5, 0 and null 1, .5, 0, 0 -> differences 0, .5, .5, 0
    assert score.delta == pytest.approx(0.25)
    assert score.pass_rate == pytest.approx(0.625) and score.null_pass_rate == pytest.approx(0.375)


def test_a_capped_reply_is_not_passed_and_counts_in_the_cap_rate(build):
    score = one_score(build, null="PPPP", lora="PCPP")
    assert score.delta == pytest.approx(-0.25)
    assert score.cap_rate == pytest.approx(0.25) and score.null_cap_rate == 0.0


def test_an_infrastructure_error_is_not_passed_and_stays_in_the_count(build):
    score = one_score(build, null="PPPP", lora="PEPP")
    assert score.delta == pytest.approx(-0.25)
    assert score.pass_rate == pytest.approx(0.75)
    assert score.cover == pytest.approx(0.75)


# ---------------------------------------------------------------- win rule


def test_z_must_be_strictly_above_the_bar(build):
    null, lora = runs(("F", 20)), runs(("P", 6), ("F", 14))
    z = one_score(build, null, lora).z
    assert z > 1
    assert one_score(build, null, lora).eligible
    assert not one_score(build, null, lora, z_bar=z).eligible, "z equal to the bar is not a win"
    assert one_score(build, null, lora, z_bar=z - 1e-9).eligible


def test_a_lora_below_one_standard_error_is_not_eligible(build):
    # differences: -1 on 2 tasks, +1 on 3 tasks, 0 on the other 15
    score = one_score(build, null=runs(("P", 5), ("F", 15)), lora=runs(("P", 3), ("F", 2), ("P", 3), ("F", 12)))
    assert score.delta == pytest.approx(0.05)
    assert 0 < score.z < 1 and not score.eligible
    assert any("is not above 1" in reason for reason in score.reasons)


def test_a_lora_behind_m_is_not_eligible(build):
    score = one_score(build, null=runs(("P", 10), ("F", 10)), lora=runs(("F", 20)))
    assert score.z < 0 and not score.eligible


def test_identical_outcomes_give_z_zero(build):
    score = one_score(build, null="PFPFPFPF", lora="PFPFPFPF")
    assert (score.delta, score.se, score.z) == (0.0, 0.0, 0.0) and not score.eligible


def test_a_zero_standard_error_fails_closed_even_with_a_positive_delta(build):
    score = one_score(build, null=runs(("F", 8)), lora=runs(("P", 8)))
    assert score.delta == 1.0 and score.se == 0.0
    assert score.z == 0.0 and not score.eligible


def test_fewer_tasks_than_the_minimum_can_never_win(build):
    null, lora = runs(("F", 20)), runs(("P", 6), ("F", 14))
    assert one_score(build, null, lora, min_scored_tasks=20).eligible
    assert not one_score(build, null, lora, min_scored_tasks=21).eligible


def test_the_bar_has_no_correction_for_the_number_of_candidates(build):
    null, strong = runs(("F", 20)), runs(("P", 6), ("F", 14))
    alone = settle(build, null, {0: strong})
    crowd = settle(build, null, {0: strong, 1: null, 2: null, 3: null})
    assert alone.entries[0].score == crowd.entries[0].score
    assert alone.winner_queue_pos == crowd.winner_queue_pos == 0


@pytest.mark.parametrize("caps,ok", [(32, True), (33, True), (34, False)])
def test_cap_hit_on_the_floor_passes_and_only_a_share_above_it_is_rejected(build, caps, ok):
    null = runs(("C", 10), ("F", 50), ("P", 40))
    lora = runs(("C", caps), ("P", 100 - caps))
    score = one_score(build, null, lora)
    assert score.cap_rate == pytest.approx(caps / 100) and score.null_cap_rate == pytest.approx(0.10)
    assert score.z > 1, "limit is max(0.33, 0.10 + 0.02) = 0.33; a share of exactly 0.33 is not greater than it"
    assert score.eligible is ok


@pytest.mark.parametrize("caps,ok", [(41, True), (42, True), (43, False)])
def test_cap_hit_on_the_head_relative_limit_passes_and_only_above_it_is_rejected(build, caps, ok):
    null = runs(("C", 40), ("F", 40), ("P", 20))
    lora = runs(("C", caps), ("P", 100 - caps))
    score = one_score(build, null, lora)
    assert score.null_cap_rate == pytest.approx(0.40), "limit is max(0.33, 0.40 + 0.02) = 0.42"
    assert score.eligible is ok
    if not ok:
        assert any("cap-hit rate 0.4300 is above 0.4200" in reason for reason in score.reasons)


def test_cap_hit_settings_are_parameters(build):
    null, lora = runs(("F", 100)), runs(("C", 20), ("P", 80))
    assert one_score(build, null, lora).eligible
    assert not one_score(build, null, lora, cap_hit_floor=0.15, cap_hit_margin=0.02).eligible


# ---------------------------------------------------------------- winner


NULL_100 = runs(("P", 10), ("F", 90))
HIGH_DELTA = runs(("F", 10), ("P", 40), ("F", 50))  # +1 on 40 tasks, -1 on 10: delta 0.30
HIGH_Z = runs(("P", 10), ("P", 20), ("F", 70))  # +1 on 20 tasks, no -1: delta 0.20


def test_the_highest_delta_wins_not_the_highest_z(build):
    decision = settle(build, NULL_100, {0: HIGH_Z, 1: HIGH_DELTA})
    by_pos = {entry.queue_pos: entry for entry in decision.entries}
    assert by_pos[0].score.z > by_pos[1].score.z > 1
    assert by_pos[1].score.delta > by_pos[0].score.delta
    assert decision.winner_queue_pos == 1
    assert (by_pos[1].status, by_pos[0].status) == (WINNER, LOST)


def test_a_tie_goes_to_the_earlier_queued_entry(build):
    decision = settle(build, NULL_100, {0: HIGH_Z, 1: HIGH_DELTA, 2: HIGH_DELTA})
    assert decision.winner_queue_pos == 1
    assert [entry.status for entry in decision.entries] == [LOST, WINNER, LOST]


def test_only_one_entry_wins_even_when_several_clear_the_bar(build):
    decision = settle(build, NULL_100, {0: HIGH_DELTA, 1: HIGH_Z, 2: HIGH_Z})
    assert [entry.status for entry in decision.entries].count(WINNER) == 1
    assert all(entry.score.eligible for entry in decision.entries)


def test_an_ineligible_entry_with_the_highest_delta_does_not_win(build):
    capped_but_high = runs(("C", 35), ("P", 65))
    decision = settle(build, NULL_100, {0: capped_but_high, 1: HIGH_Z})
    assert decision.entries[0].score.delta > decision.entries[1].score.delta
    assert decision.winner_queue_pos == 1


def score_with(delta: float, eligible: bool = True) -> Score:
    return Score(100, delta, 0.01, delta / 0.01, 0.5, 0.4, 0.0, 0.0, 1.0, eligible)


def test_deltas_that_differ_only_by_rounding_noise_are_a_tie():
    assert pick_winner([(3, score_with(0.2)), (1, score_with(0.2 + 1e-15)), (2, score_with(0.2 - 1e-15))]) == 1


def test_a_real_difference_beats_queue_order():
    assert pick_winner([(0, score_with(0.2)), (5, score_with(0.2 + 1e-6))]) == 5


def test_no_eligible_entry_means_no_winner():
    assert pick_winner([(0, score_with(0.5, eligible=False))]) is None
    assert pick_winner([]) is None


# ---------------------------------------------------------------- decision


def test_a_winner_and_the_losers_are_charged(build):
    decision = settle(build, NULL_100, {0: HIGH_DELTA, 1: NULL_100})
    assert decision.winner.uid == 10 and decision.winner.lora_hash
    assert [entry.charged for entry in decision.entries] == [True, True]
    assert "beat M" in decision.reason


def test_a_null_round_keeps_m_and_still_charges_scored_entries(build):
    decision = settle(build, NULL_100, {0: NULL_100, 1: runs(("F", 100))})
    assert decision.winner_queue_pos is None and decision.winner is None
    assert [entry.status for entry in decision.entries] == [LOST, LOST]
    assert all(entry.charged for entry in decision.entries)
    assert decision.reason.startswith("null round")


def test_no_decision_and_skip_entries_are_not_charged(build):
    decision = settle(build, NULL_100, {0: HIGH_DELTA, 1: NO_DECISION, 2: SKIP})
    assert [entry.status for entry in decision.entries] == [WINNER, NO_DECISION, SKIP]
    assert [entry.charged for entry in decision.entries] == [True, False, False]
    assert decision.entries[1].score is None and decision.entries[1].reason


def test_a_round_where_nothing_was_scored_has_no_winner(build):
    params = build.params(n_tasks=4)
    ticket = build.ticket(2, 4, params)
    decision = decide([(ticket, build.result(ticket, (), {0: NO_DECISION, 1: NO_DECISION}))])
    assert decision.winner_queue_pos is None and decision.reason == "no entry was scored"
    assert not any(entry.charged for entry in decision.entries)


def test_an_incomplete_null_m_makes_every_scored_entry_no_decision(build):
    null = runs(("E", 10), ("F", 90))
    decision = settle(build, null, {0: HIGH_DELTA, 1: NO_DECISION})
    assert decision.winner_queue_pos is None
    assert [entry.status for entry in decision.entries] == [NO_DECISION, NO_DECISION]
    assert "null M" in decision.entries[0].reason
    assert not any(entry.charged for entry in decision.entries)


def test_an_incomplete_lora_is_no_decision_and_the_others_are_still_judged(build):
    broken = runs(("E", 10), ("P", 90))
    decision = settle(build, NULL_100, {0: broken, 1: HIGH_DELTA})
    assert [entry.status for entry in decision.entries] == [NO_DECISION, WINNER]
    assert not decision.entries[0].charged


def test_a_few_errors_inside_the_cover_limit_still_count_as_failures(build):
    with_errors = runs(("E", 5), ("F", 5), ("P", 40), ("F", 50))
    decision = settle(build, NULL_100, {0: with_errors})
    assert decision.entries[0].status == WINNER, "5 errors in 100 is exactly the 0.95 cover limit: still judged"
    assert decision.entries[0].score.cover == pytest.approx(0.95)


def test_one_error_past_the_cover_limit_is_no_decision(build):
    with_errors = runs(("E", 6), ("F", 4), ("P", 40), ("F", 50))
    decision = settle(build, NULL_100, {0: with_errors})
    assert decision.entries[0].status == NO_DECISION


def test_the_decision_is_a_pure_function_of_its_inputs(build):
    one = settle(build, NULL_100, {0: HIGH_Z, 1: HIGH_DELTA})
    two = settle(build, NULL_100, {0: HIGH_Z, 1: HIGH_DELTA})
    assert one.digest() == two.digest()


# ---------------------------------------------------------------- early looks

LOOKS = dict(looks=(0.25, 0.5, 1.0), min_scored_tasks=20)


def at_look(z: float, n_tasks: int) -> Score:
    """A score at an early look with a given z."""
    return Score(n_tasks, z * 0.01, 0.01, z, 0.5, 0.5, 0.0, 0.0, 1.0, False)


def test_a_lora_is_dropped_only_when_its_upper_bound_cannot_reach_the_bar(build):
    params = build.params(n_tasks=100, **LOOKS)
    spent = lambda share: 0.05 * math.log(1 + (math.e - 1) * share)
    for look, share, earlier in ((0, 0.25, 0.0), (1, 0.5, 0.25)):
        margin = NormalDist().inv_cdf(1 - (spent(share) - spent(earlier)))
        # upper bound of the final z is (z + margin) / sqrt(share); it reaches the bar of 1 at this z:
        edge = math.sqrt(share) - margin
        n = int(share * 100)
        assert upper_final_z(at_look(edge, n), look, 100, params) == pytest.approx(1.0)
        assert drop_at_look(at_look(edge - 1e-6, n), look, 100, params)
        assert not drop_at_look(at_look(edge + 1e-6, n), look, 100, params)
        assert edge < -1.4, "a LoRA must be well behind M before it is dropped"


def test_only_an_early_look_with_enough_tasks_can_drop(build):
    params = build.params(n_tasks=100, **LOOKS)
    assert drop_at_look(at_look(-9.0, 25), 0, 100, params)
    assert not drop_at_look(at_look(-9.0, 100), 2, 100, params), "never at the last look"
    assert not drop_at_look(at_look(-9.0, 30), 0, 100, params), "the score must be the score of that look"
    few = build.params(n_tasks=100, looks=(0.1, 1.0), min_scored_tasks=20)
    assert not drop_at_look(at_look(-9.0, 10), 0, 100, few), "too few tasks to judge"
    no_spread = Score(25, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, False)
    assert not drop_at_look(no_spread, 0, 100, params), "no standard error, no drop"


def early_drop_rate(build, final_z: float, paths: int = 8000) -> float:
    """Share of LoRAs dropped at an early look, when the z they are truly heading for is ``final_z``."""
    params = build.params(n_tasks=1000, **LOOKS)
    rng = random.Random(7)
    dropped = 0
    for _ in range(paths):
        walk = 0.0
        for look, (start, stop) in enumerate(((0.0, 0.25), (0.25, 0.5))):
            walk += final_z * (stop - start) + rng.gauss(0.0, math.sqrt(stop - start))
            if drop_at_look(at_look(walk / math.sqrt(stop), int(stop * 1000)), look, 1000, params):
                dropped += 1
                break
    return dropped / paths


def test_a_lora_on_the_bar_is_dropped_early_less_often_than_drop_alpha(build):
    assert early_drop_rate(build, final_z=1.0) < 0.05
    assert early_drop_rate(build, final_z=2.0) < 0.015, "a real winner is almost never dropped"


def test_a_lora_far_behind_is_dropped_early_and_one_level_with_m_rarely(build):
    assert early_drop_rate(build, final_z=-4.0) > 0.85
    assert early_drop_rate(build, final_z=0.0) < 0.15


def test_the_win_test_is_never_used_at_an_early_look(build):
    params = build.params(n_tasks=100, **LOOKS)
    ahead = score_entry((1.0,) * 100, (runs(("F", 100)),), (runs(("P", 10), ("F", 15), ("-", 75)),), params)
    assert ahead.z > 1, "far ahead at the look"
    assert not ahead.eligible, "but a part of the draw can never make a winner"
    assert not drop_at_look(ahead, 0, 100, params)


def test_drop_alpha_is_a_parameter(build):
    loose = build.params(n_tasks=100, drop_alpha=0.5, **LOOKS)
    strict = build.params(n_tasks=100, drop_alpha=0.001, **LOOKS)
    assert drop_at_look(at_look(-1.0, 25), 0, 100, loose)
    assert not drop_at_look(at_look(-1.0, 25), 0, 100, strict)


def looks_case(build, dropped_outcomes: str, look: int):
    params = build.params(n_tasks=100, **LOOKS)
    ticket = build.ticket(2, 100, params)
    null = runs(("P", 20), ("F", 80))
    winner = runs(("P", 20), ("P", 35), ("F", 45))
    return ticket, build.result(ticket, null, {0: build.dropped(ticket, 0, dropped_outcomes, look), 1: winner})


def test_a_dropped_entry_lost_and_is_charged(build):
    ticket, result = looks_case(build, runs(("F", 25), ("-", 75)), look=0)
    decision = decide([(ticket, result)])
    first = decision.entries[0]
    assert (first.status, first.eliminated_at_look, first.charged) == (LOST, 0, True)
    assert first.score.n_tasks == 25
    assert decision.winner_queue_pos == 1


def test_a_drop_that_the_rule_does_not_allow_is_rejected(build):
    ticket, result = looks_case(build, runs(("P", 25), ("-", 75)), look=0)
    with pytest.raises(ContractError, match="drop rule"):
        check_result(ticket, result)


def test_a_drop_must_match_the_size_of_its_look(build):
    ticket, result = looks_case(build, runs(("F", 30), ("-", 70)), look=0)
    with pytest.raises(ContractError, match="dropped at look 0"):
        check_result(ticket, result)


def test_nothing_can_be_dropped_at_the_last_look(build):
    ticket, result = looks_case(build, runs(("F", 100)), look=2)
    with pytest.raises(ContractError, match="early look"):
        check_result(ticket, result)


def test_an_entry_that_was_not_dropped_must_be_scored_on_every_task(build):
    params = build.params(n_tasks=8)
    ticket = build.ticket(1, 8, params)
    with pytest.raises(ContractError, match="every task"):
        check_result(ticket, build.result(ticket, "PPPPFFFF", {0: "PPPP----"}))


def test_scored_prefix_rejects_gaps_and_ragged_samples():
    assert scored_prefix(("PF--", "FP--")) == 2
    assert scored_prefix(("PFPF",)) == 4
    assert scored_prefix(("----",)) == 0
    with pytest.raises(ContractError):
        scored_prefix(("P-F-",))
    with pytest.raises(ContractError):
        scored_prefix(("PF--", "PFP-"))
    with pytest.raises(ContractError):
        scored_prefix(())


# ---------------------------------------------------------------- sealed result checks


def good_case(build):
    params = build.params(n_tasks=8)
    ticket = build.ticket(2, 8, params)
    return ticket, build.result(ticket, "PFPFPFPF", {0: "PPPPPFPF", 1: "PFPFPFPF"})


def test_a_well_formed_result_passes(build):
    ticket, result = good_case(build)
    check_result(ticket, result)


def test_a_result_against_another_head_is_refused(build):
    ticket, result = good_case(build)
    with pytest.raises(HeadMismatchError):
        check_result(ticket, build.replace(result, head_version_id="0" * 64))


@pytest.mark.parametrize(
    "change,match",
    [
        (dict(ticket_digest="0" * 64), "does not answer this ticket"),
        (dict(image_digest="unknown-image"), "pinned set"),
        (dict(lease_id="r000001-l9"), "another round, lease or box"),
        (dict(round_id="r000002"), "another round, lease or box"),
        (dict(null_outcomes=("PFPF",)), "outcome string has 4 codes"),
        (dict(null_outcomes=("PFPFPF--",)), "scored on more tasks than null M"),
        (dict(null_outcomes=("PFPFPFPF", "PFPFPFPF")), "one outcome string per sample"),
    ],
)
def test_a_malformed_result_is_refused(build, change, match):
    ticket, result = good_case(build)
    with pytest.raises(ContractError, match=match):
        check_result(ticket, build.replace(result, **change))


def test_a_result_must_report_every_entry_in_queue_order(build):
    ticket, result = good_case(build)
    with pytest.raises(ContractError, match="every ticket entry"):
        check_result(ticket, build.replace(result, entries=result.entries[:1]))
    with pytest.raises(ContractError, match="every ticket entry"):
        check_result(ticket, build.replace(result, entries=tuple(reversed(result.entries))))


def with_lora(build, result, **lora_changes):
    first = result.entries[0]
    changed = build.replace(first, lora=build.replace(first.lora, **lora_changes))
    return build.replace(result, entries=(changed,) + result.entries[1:])


@pytest.mark.parametrize(
    "change,match",
    [
        (dict(head_version_id="0" * 64), "another head"),
        (dict(batch_sha256="0" * 64), "another batch"),
        (dict(recipe_id="0" * 64), "another recipe"),
        (dict(rank=64), "rank, alpha or target modules"),
        (dict(target_modules=("q_proj",)), "rank, alpha or target modules"),
    ],
)
def test_a_lora_that_does_not_belong_to_its_entry_is_refused(build, change, match):
    ticket, result = good_case(build)
    with pytest.raises(ContractError, match=match):
        check_result(ticket, with_lora(build, result, **change))


def test_a_lora_must_have_the_shape_the_miner_asked_for(build):
    params = build.params(n_tasks=8)
    ticket = build.ticket(1, 8, params, train={0: {"rank": 32, "alpha": 64, "target_modules": ["q_proj", "v_proj"]}})
    result = build.result(ticket, "PFPFPFPF", {0: "PPPPPFPF"})
    lora = result.entries[0].lora
    assert (lora.rank, lora.alpha, lora.target_modules) == (32, 64.0, ("q_proj", "v_proj"))
    check_result(ticket, result)
    with pytest.raises(ContractError, match="rank, alpha or target modules"):
        check_result(ticket, with_lora(build, result, rank=params.recipe.rank))


def test_a_lora_from_an_unpinned_trainer_image_or_with_a_non_finite_report_is_refused(build):
    ticket, result = good_case(build)
    report = result.entries[0].lora.report
    with pytest.raises(ContractError, match="trainer image"):
        check_result(ticket, with_lora(build, result, report=build.replace(report, image_digest="other")))
    with pytest.raises(ContractError, match="non-finite"):
        check_result(ticket, with_lora(build, result, report=build.replace(report, loss=float("nan"))))


def test_an_unscored_entry_may_not_carry_outcomes(build):
    ticket, result = good_case(build)
    sneaky = EntryResult(1, NO_DECISION, reason="oom", outcomes=("PPPPPPPP",))
    with pytest.raises(ContractError, match="unscored entry"):
        check_result(ticket, build.replace(result, entries=(result.entries[0], sneaky)))


def test_with_no_pinned_image_every_result_is_refused(build):
    params = build.params(n_tasks=8, recipe=build.params().recipe.__class__())
    ticket = build.ticket(1, 8, params)
    with pytest.raises(ContractError, match="pinned set"):
        check_result(ticket, build.result(ticket, "PFPFPFPF", {0: "PPPPPFPF"}))


# ---------------------------------------------------------------- more than one box


def two_boxes(build, null_a, loras_a, null_b, loras_b, **param_changes):
    params = build.params(n_tasks=100, **param_changes)
    ticket_a = build.ticket(2, 100, params, lease=0, first_pos=0)
    ticket_b = build.ticket(2, 100, params, lease=1, first_pos=2)
    return decide([(ticket_a, build.result(ticket_a, null_a, loras_a)), (ticket_b, build.result(ticket_b, null_b, loras_b))])


def test_two_winners_against_the_same_head_the_larger_margin_is_promoted(build):
    decision = two_boxes(build, NULL_100, {0: HIGH_Z, 1: NULL_100}, NULL_100, {2: HIGH_DELTA, 3: NULL_100})
    assert decision.winner_queue_pos == 2
    assert [entry.status for entry in decision.entries] == [LOST, LOST, WINNER, LOST]
    assert decision.entries[0].charged, "the other box's winner loses its entry; it is not requeued"


def test_two_boxes_with_equal_margins_the_earlier_queued_entry_wins(build):
    decision = two_boxes(build, NULL_100, {0: NULL_100, 1: HIGH_DELTA}, NULL_100, {2: HIGH_DELTA, 3: NULL_100})
    assert decision.winner_queue_pos == 1


def test_boxes_whose_null_m_rates_disagree_are_not_merged(build):
    other_null = runs(("P", 30), ("F", 70))
    decision = two_boxes(build, NULL_100, {0: HIGH_DELTA, 1: NULL_100}, other_null, {2: HIGH_DELTA, 3: NO_DECISION})
    assert decision.winner_queue_pos is None
    assert [entry.status for entry in decision.entries] == [NO_DECISION] * 4
    assert "calibration_tol" in decision.reason


def test_every_lease_of_a_round_must_share_head_draw_and_parameters(build):
    ticket_a = build.ticket(1, 8, build.params(n_tasks=8), lease=0)
    ticket_b = build.ticket(1, 8, build.params(n_tasks=8, z_bar=2.0), lease=1, first_pos=1)
    pair = lambda ticket: (ticket, build.result(ticket, "PFPFPFPF", {ticket.entries[0].queue_pos: "PPPPPFPF"}))
    with pytest.raises(ContractError, match="shares one head"):
        decide([pair(ticket_a), pair(ticket_b)])
