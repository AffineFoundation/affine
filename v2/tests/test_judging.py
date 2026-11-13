from __future__ import annotations

import math
import random
import statistics
from statistics import NormalDist

import pytest

from sn120_v2.errors import ContractError, HeadMismatchError
from sn120_v2.judging import check_result, decide, drop_at_look, estimate, pick_winner, score_entry, scored_prefix, upper_final_z
from sn120_v2.types import LOST, NO_DECISION, SKIP, WINNER, EntryResult, Score


def runs(*parts: tuple[str, int]) -> str:
    """Build an outcome string from (code, count) runs."""
    return "".join(code * count for code, count in parts)


def one_score(build, null: str, lora: str, weights=None, **param_changes) -> Score:
    params = build.params(n_tasks=len(null), **param_changes)
    return score_entry(weights or (1.0,) * len(null), (null,), (lora,), params, n_total=len(null))


def settle(build, null, loras, **param_changes):
    params = build.params(n_tasks=len(null), **param_changes)
    ticket = build.ticket(len(loras), len(null), params)
    return decide([(ticket, build.result(ticket, null, loras))])


NULL_100 = runs(("P", 10), ("F", 90))
HIGH_DELTA = runs(("F", 10), ("P", 40), ("F", 50))  # +1 on 40 tasks, -1 on 10: delta 0.30
HIGH_Z = runs(("P", 10), ("P", 20), ("F", 70))  # +1 on 20 tasks, no -1: delta 0.20, higher z
LOOKS = dict(looks=(0.25, 0.5, 1.0), min_scored_tasks=20)

# ---------------------------------------------------------------- estimator


def test_equal_weights_give_the_paired_mean_and_standard_error():
    diffs = [-1, 0, 0, 1, 0, 1, 0, 0]
    delta, se = estimate([2.5] * 8, diffs)
    assert delta == pytest.approx(statistics.mean(diffs))
    assert se == pytest.approx(statistics.stdev(diffs) / math.sqrt(8))
    assert estimate([1.0], [1.0]) == (1.0, 0.0), "one task has no standard error"


def test_weighted_estimate_matches_a_hand_computation():
    # weights 1 and 3, differences 1 and 0: delta = 1/4.
    # var = 2/1 * ((1 * 0.75)^2 + (3 * -0.25)^2) / 4^2 = 0.140625, so se = 0.375.
    assert estimate([1.0, 3.0], [1.0, 0.0]) == (pytest.approx(0.25), pytest.approx(0.375))


def test_score_is_the_weighted_difference_of_pass_rates_on_the_same_tasks(build):
    score = one_score(build, null="PFPFPFPF", lora="FFPPPPPF")
    assert (score.delta, score.pass_rate, score.null_pass_rate) == (pytest.approx(0.125), pytest.approx(0.625), pytest.approx(0.5))
    assert score.z == pytest.approx(score.delta / score.se) and score.n_tasks == 8
    assert one_score(build, null="FFFF", lora="PFFF", weights=(9.0, 1.0, 1.0, 1.0)).delta == pytest.approx(0.75)
    two = score_entry((1.0,) * 4, ("PPFF", "PFFF"), ("PPPF", "PPFF"), build.params(n_tasks=4, samples_per_task=2), n_total=4)
    assert two.delta == pytest.approx(0.25), "two samples per task: rates 1, 1, .5, 0 against 1, .5, 0, 0"


def test_a_capped_reply_and_an_infrastructure_error_are_not_passed_and_stay_in_the_count(build):
    capped = one_score(build, null="PPPP", lora="PCPP")
    assert (capped.delta, capped.cap_rate, capped.null_cap_rate) == (pytest.approx(-0.25), pytest.approx(0.25), 0.0)
    failed = one_score(build, null="PPPP", lora="PEPP")
    assert (failed.delta, failed.pass_rate, failed.cover) == (pytest.approx(-0.25), pytest.approx(0.75), pytest.approx(0.75))


# ---------------------------------------------------------------- win rule


def test_z_must_be_strictly_above_the_bar(build):
    null, lora = runs(("F", 20)), runs(("P", 6), ("F", 14))
    z = one_score(build, null, lora).z
    assert z > 1 and one_score(build, null, lora).eligible
    assert not one_score(build, null, lora, z_bar=z).eligible, "z equal to the bar is not a win"
    assert one_score(build, null, lora, z_bar=z - 1e-9).eligible
    assert not one_score(build, null, lora, min_scored_tasks=21).eligible, "too few scored tasks can never win"


def test_a_lora_below_one_standard_error_and_a_zero_standard_error_are_not_eligible(build):
    # differences: -1 on 2 tasks, +1 on 3 tasks, 0 on the other 15
    close = one_score(build, null=runs(("P", 5), ("F", 15)), lora=runs(("P", 3), ("F", 2), ("P", 3), ("F", 12)))
    assert close.delta == pytest.approx(0.05) and 0 < close.z < 1 and not close.eligible
    same = one_score(build, null="PFPFPFPF", lora="PFPFPFPF")
    assert (same.delta, same.se, same.z, same.eligible) == (0.0, 0.0, 0.0, False)
    perfect = one_score(build, null=runs(("F", 8)), lora=runs(("P", 8)))
    assert (perfect.delta, perfect.se, perfect.z, perfect.eligible) == (1.0, 0.0, 0.0, False), "no spread: fail closed"


def test_the_bar_has_no_correction_for_the_number_of_candidates(build):
    null, strong = runs(("F", 20)), runs(("P", 6), ("F", 14))
    alone = settle(build, null, {0: strong})
    crowd = settle(build, null, {0: strong, 1: null, 2: null, 3: null})
    assert alone.entries[0].score == crowd.entries[0].score and alone.winner_queue_pos == crowd.winner_queue_pos == 0


@pytest.mark.parametrize("caps,ok", [(32, True), (33, True), (34, False)])
def test_cap_hit_on_the_floor_passes_and_only_a_share_above_it_is_rejected(build, caps, ok):
    score = one_score(build, runs(("C", 10), ("F", 50), ("P", 40)), runs(("C", caps), ("P", 100 - caps)))
    assert score.null_cap_rate == pytest.approx(0.10) and score.z > 1, "limit is max(0.33, 0.10 + 0.02) = 0.33"
    assert score.eligible is ok


@pytest.mark.parametrize("caps,ok", [(41, True), (42, True), (43, False)])
def test_cap_hit_on_the_head_relative_limit_passes_and_only_above_it_is_rejected(build, caps, ok):
    score = one_score(build, runs(("C", 40), ("F", 40), ("P", 20)), runs(("C", caps), ("P", 100 - caps)))
    assert score.null_cap_rate == pytest.approx(0.40), "limit is max(0.33, 0.40 + 0.02) = 0.42"
    assert score.eligible is ok
    assert ok or any("cap-hit rate 0.4300 is above 0.4200" in reason for reason in score.reasons)


# ---------------------------------------------------------------- winner and decision


def test_the_highest_delta_wins_and_a_tie_goes_to_the_earlier_queued_entry(build):
    decision = settle(build, NULL_100, {0: HIGH_Z, 1: HIGH_DELTA, 2: HIGH_DELTA})
    first, second = decision.entries[0].score, decision.entries[1].score
    assert first.z > second.z > 1 and second.delta > first.delta, "the winner is not the entry with the highest z"
    assert [entry.status for entry in decision.entries] == [LOST, WINNER, LOST]
    capped_but_high = runs(("C", 35), ("P", 65))
    assert settle(build, NULL_100, {0: capped_but_high, 1: HIGH_Z}).winner_queue_pos == 1, "an ineligible entry cannot win"


def test_rounding_noise_is_a_tie_and_a_real_difference_beats_queue_order():
    score = lambda delta, eligible=True: Score(100, delta, 0.01, delta / 0.01, 0.5, 0.4, 0.0, 0.0, 1.0, eligible)
    assert pick_winner([(3, score(0.2)), (1, score(0.2 + 1e-15)), (2, score(0.2 - 1e-15))]) == 1
    assert pick_winner([(0, score(0.2)), (5, score(0.2 + 1e-6))]) == 5
    assert pick_winner([(0, score(0.5, eligible=False))]) is None and pick_winner([]) is None


def test_scored_entries_are_charged_and_no_decision_and_skip_entries_are_not(build):
    decision = settle(build, NULL_100, {0: HIGH_DELTA, 1: NULL_100, 2: NO_DECISION, 3: SKIP})
    assert [entry.status for entry in decision.entries] == [WINNER, LOST, NO_DECISION, SKIP]
    assert [entry.charged for entry in decision.entries] == [True, True, False, False]
    assert decision.winner.uid == 10 and decision.winner.lora_hash and "beat M" in decision.reason
    null_round = settle(build, NULL_100, {0: NULL_100, 1: runs(("F", 100))})
    assert null_round.winner is None and all(entry.charged for entry in null_round.entries)
    assert null_round.reason.startswith("null round")
    assert settle(build, NULL_100, {0: HIGH_Z, 1: HIGH_DELTA}).digest() == settle(build, NULL_100, {0: HIGH_Z, 1: HIGH_DELTA}).digest()


def test_a_round_where_nothing_was_scored_has_no_winner_and_charges_nobody(build):
    ticket = build.ticket(2, 4, build.params(n_tasks=4))
    decision = decide([(ticket, build.result(ticket, (), {0: NO_DECISION, 1: NO_DECISION}))])
    assert decision.winner_queue_pos is None and decision.reason == "no entry was scored"
    assert not any(entry.charged for entry in decision.entries)


def test_too_many_unresolved_rollouts_are_no_decision_not_a_loss(build):
    broken_null = settle(build, runs(("E", 10), ("F", 90)), {0: HIGH_DELTA, 1: NO_DECISION})
    assert [entry.status for entry in broken_null.entries] == [NO_DECISION, NO_DECISION] and "null M" in broken_null.entries[0].reason
    broken_lora = settle(build, NULL_100, {0: runs(("E", 10), ("P", 90)), 1: HIGH_DELTA})
    assert [entry.status for entry in broken_lora.entries] == [NO_DECISION, WINNER]
    on_the_limit = runs(("E", 5), ("F", 5), ("P", 40), ("F", 50))
    assert settle(build, NULL_100, {0: on_the_limit}).entries[0].status == WINNER, "5 errors in 100 is exactly the 0.95 limit"
    past_the_limit = runs(("E", 6), ("F", 4), ("P", 40), ("F", 50))
    assert settle(build, NULL_100, {0: past_the_limit}).entries[0].status == NO_DECISION


# ---------------------------------------------------------------- early looks


def at_look(z: float, n_tasks: int) -> Score:
    return Score(n_tasks, z * 0.01, 0.01, z, 0.5, 0.5, 0.0, 0.0, 1.0, False)


def test_a_lora_is_dropped_only_when_its_upper_bound_cannot_reach_the_bar(build):
    params = build.params(n_tasks=100, **LOOKS)
    spent = lambda share: 0.05 * math.log(1 + (math.e - 1) * share)
    for look, share, earlier in ((0, 0.25, 0.0), (1, 0.5, 0.25)):
        margin = NormalDist().inv_cdf(1 - (spent(share) - spent(earlier)))
        edge = math.sqrt(share) - margin  # here the upper bound (z + margin) / sqrt(share) is exactly the bar of 1
        n = int(share * 100)
        assert upper_final_z(at_look(edge, n), look, 100, params) == pytest.approx(1.0)
        assert drop_at_look(at_look(edge - 1e-6, n), look, 100, params)
        assert not drop_at_look(at_look(edge + 1e-6, n), look, 100, params)
        assert edge < -1.4, "a LoRA must be well behind M before it is dropped"
    assert drop_at_look(at_look(-1.0, 25), 0, 100, build.params(n_tasks=100, drop_alpha=0.5, **LOOKS))
    assert not drop_at_look(at_look(-1.0, 25), 0, 100, build.params(n_tasks=100, drop_alpha=0.001, **LOOKS))


def test_only_an_early_look_with_enough_tasks_can_drop_and_it_never_names_a_winner(build):
    params = build.params(n_tasks=100, **LOOKS)
    assert drop_at_look(at_look(-9.0, 25), 0, 100, params)
    assert not drop_at_look(at_look(-9.0, 100), 2, 100, params), "never at the last look"
    assert not drop_at_look(at_look(-9.0, 30), 0, 100, params), "the score must be the score of that look"
    few = build.params(n_tasks=100, looks=(0.1, 1.0), min_scored_tasks=20)
    assert not drop_at_look(at_look(-9.0, 10), 0, 100, few), "too few tasks to judge"
    assert not drop_at_look(Score(25, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, False), 0, 100, params), "no standard error"
    ahead = score_entry((1.0,) * 100, (runs(("F", 100)),), (runs(("P", 10), ("F", 15), ("-", 75)),), params)
    assert ahead.z > 1 and not ahead.eligible and not drop_at_look(ahead, 0, 100, params)


def early_drop_rate(build, final_z: float, paths: int = 8000) -> float:
    """Share of LoRAs dropped at an early look, when the z they are truly heading for is ``final_z``."""
    params = build.params(n_tasks=1000, **LOOKS)
    rng, dropped = random.Random(7), 0
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
    assert early_drop_rate(build, final_z=0.0) < 0.15
    assert early_drop_rate(build, final_z=-4.0) > 0.85, "a LoRA far behind is dropped early"


def looks_case(build, dropped_outcomes: str, look: int):
    ticket = build.ticket(2, 100, build.params(n_tasks=100, **LOOKS))
    null, winner = runs(("P", 20), ("F", 80)), runs(("P", 20), ("P", 35), ("F", 45))
    return ticket, build.result(ticket, null, {0: build.dropped(ticket, 0, dropped_outcomes, look), 1: winner})


def test_a_dropped_entry_lost_and_is_charged(build):
    decision = decide([looks_case(build, runs(("F", 25), ("-", 75)), look=0)])
    first = decision.entries[0]
    assert (first.status, first.eliminated_at_look, first.charged, first.score.n_tasks) == (LOST, 0, True, 25)
    assert decision.winner_queue_pos == 1


@pytest.mark.parametrize(
    "outcomes,look,match",
    [
        (runs(("P", 25), ("-", 75)), 0, "drop rule"),
        (runs(("F", 30), ("-", 70)), 0, "dropped at look 0"),
        (runs(("F", 100)), 2, "early look"),
    ],
)
def test_a_drop_that_the_rule_does_not_allow_is_rejected(build, outcomes, look, match):
    with pytest.raises(ContractError, match=match):
        check_result(*looks_case(build, outcomes, look))


def test_scored_prefix_rejects_gaps_and_ragged_samples():
    assert (scored_prefix(("PF--", "FP--")), scored_prefix(("PFPF",)), scored_prefix(("----",))) == (2, 4, 0)
    for bad in (("P-F-",), ("PF--", "PFP-"), ()):
        with pytest.raises(ContractError):
            scored_prefix(bad)


# ---------------------------------------------------------------- sealed result checks


def good_case(build):
    ticket = build.ticket(2, 8, build.params(n_tasks=8))
    return ticket, build.result(ticket, "PFPFPFPF", {0: "PPPPPFPF", 1: "PFPFPFPF"})


@pytest.mark.parametrize(
    "change,match",
    [
        (dict(ticket_digest="0" * 64), "does not answer this ticket"),
        (dict(image_digest="unknown-image"), "pinned set"),
        (dict(lease_id="r000001-l9"), "another round, lease or box"),
        (dict(null_outcomes=("PFPF",)), "outcome string has 4 codes"),
        (dict(null_outcomes=("PFPFPF--",)), "scored on more tasks than null M"),
        (dict(null_outcomes=("PFPFPFPF", "PFPFPFPF")), "one outcome string per sample"),
    ],
)
def test_a_malformed_result_is_refused(build, change, match):
    ticket, result = good_case(build)
    check_result(ticket, result)
    with pytest.raises(ContractError, match=match):
        check_result(ticket, build.replace(result, **change))


def test_a_result_against_another_head_or_with_wrong_entries_is_refused(build):
    ticket, result = good_case(build)
    with pytest.raises(HeadMismatchError):
        check_result(ticket, build.replace(result, head_version_id="0" * 64))
    sneaky = EntryResult(1, NO_DECISION, reason="oom", outcomes=("PPPPPPPP",))
    for entries, match in (
        (result.entries[:1], "every ticket entry"),
        (tuple(reversed(result.entries)), "every ticket entry"),
        ((result.entries[0], sneaky), "unscored entry"),
    ):
        with pytest.raises(ContractError, match=match):
            check_result(ticket, build.replace(result, entries=entries))
    unpinned = build.ticket(1, 8, build.params(n_tasks=8, recipe=type(ticket.params.recipe)()))
    with pytest.raises(ContractError, match="pinned set"):
        check_result(unpinned, build.result(unpinned, "PFPFPFPF", {0: "PPPPPFPF"}))


def with_lora(build, result, **lora_changes):
    first = result.entries[0]
    return build.replace(result, entries=(build.replace(first, lora=build.replace(first.lora, **lora_changes)),) + result.entries[1:])


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


def test_a_lora_must_have_the_shape_the_miner_asked_for_and_a_finite_report(build):
    params = build.params(n_tasks=8)
    ticket = build.ticket(1, 8, params, train={0: {"rank": 32, "alpha": 64, "target_modules": ["q_proj", "v_proj"]}})
    result = build.result(ticket, "PFPFPFPF", {0: "PPPPPFPF"})
    lora = result.entries[0].lora
    assert (lora.rank, lora.alpha, lora.target_modules) == (32, 64.0, ("q_proj", "v_proj"))
    check_result(ticket, result)
    with pytest.raises(ContractError, match="rank, alpha or target modules"):
        check_result(ticket, with_lora(build, result, rank=params.recipe.rank))
    with pytest.raises(ContractError, match="non-finite"):
        check_result(ticket, with_lora(build, result, report=build.replace(lora.report, loss=float("nan"))))


# ---------------------------------------------------------------- more than one box


def two_boxes(build, null_a, loras_a, null_b, loras_b):
    params = build.params(n_tasks=100)
    ticket_a = build.ticket(2, 100, params, lease=0, first_pos=0)
    ticket_b = build.ticket(2, 100, params, lease=1, first_pos=2)
    return decide([(ticket_a, build.result(ticket_a, null_a, loras_a)), (ticket_b, build.result(ticket_b, null_b, loras_b))])


def test_two_winners_against_the_same_head_the_larger_margin_is_promoted(build):
    decision = two_boxes(build, NULL_100, {0: HIGH_Z, 1: NULL_100}, NULL_100, {2: HIGH_DELTA, 3: NULL_100})
    assert [entry.status for entry in decision.entries] == [LOST, LOST, WINNER, LOST]
    assert decision.entries[0].charged, "the other box's winner loses its entry; it is not requeued"
    tie = two_boxes(build, NULL_100, {0: NULL_100, 1: HIGH_DELTA}, NULL_100, {2: HIGH_DELTA, 3: NULL_100})
    assert tie.winner_queue_pos == 1, "equal margins: the earlier-queued entry wins"


def test_boxes_whose_null_m_rates_disagree_are_not_merged(build):
    other_null = runs(("P", 30), ("F", 70))
    decision = two_boxes(build, NULL_100, {0: HIGH_DELTA, 1: NULL_100}, other_null, {2: HIGH_DELTA, 3: NO_DECISION})
    assert decision.winner_queue_pos is None and [entry.status for entry in decision.entries] == [NO_DECISION] * 4
    assert "calibration_tol" in decision.reason
    ticket_a = build.ticket(1, 8, build.params(n_tasks=8), lease=0)
    ticket_b = build.ticket(1, 8, build.params(n_tasks=8, z_bar=2.0), lease=1, first_pos=1)
    pair = lambda ticket: (ticket, build.result(ticket, "PFPFPFPF", {ticket.entries[0].queue_pos: "PPPPPFPF"}))
    with pytest.raises(ContractError, match="shares one head"):
        decide([pair(ticket_a), pair(ticket_b)])
