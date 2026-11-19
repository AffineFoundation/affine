"""Exact q values and every one of the 8/16/24/32 pass-count outcomes."""

from __future__ import annotations

from fractions import Fraction
from math import comb

import pytest

from sn120_v2.errors import ContractError
from sn120_v2.generator import constants, decisions

EXPECTED = {
    8: {"accept": set(range(3, 6)), "reject": {0, 8}},
    16: {"accept": set(range(6, 11)), "reject": {0, 1, 2, 14, 15, 16}},
    24: {"accept": set(range(8, 17)), "reject": {0, 1, 2, 3, 21, 22, 23, 24}},
    32: {"accept": set(range(8, 25)), "reject": set(range(0, 8)) | set(range(25, 33))},
}


def binomial_q(s, n):
    # Independent direct polynomial calculation; not the implementation's helper.
    def tail(x):
        return sum(comb(n + 1, j) * x**j * (1 - x) ** (n + 1 - j) for j in range(s + 1, n + 2))

    return tail(0.75) - tail(0.25)


@pytest.mark.parametrize("total", [0, 1, 8, 16, 24, 32])
def test_q_matches_direct_binomial_identity_at_every_pass_count(total):
    for passes in range(total + 1):
        expected = binomial_q(passes, total)
        assert decisions.q(passes, total) == pytest.approx(expected, rel=0, abs=1e-15)
        assert decisions.q_independent(passes, total) == pytest.approx(expected, rel=0, abs=1e-15)
        assert decisions.q_exact(passes, total) == sum(
            Fraction(comb(total + 1, j), 4 ** (total + 1)) * (3**j - 3 ** (total + 1 - j))
            for j in range(passes + 1, total + 2)
        )


def test_known_uniform_posterior_values_and_symmetry():
    assert decisions.q(0, 0) == 0.5
    assert decisions.q(3, 8) == pytest.approx(0.82427978515625)
    assert decisions.q(0, 8) == pytest.approx(0.07508087158203125)
    for n in (8, 16, 24, 32):
        for s in range(n + 1):
            assert decisions.q(s, n) == pytest.approx(decisions.q(n - s, n), abs=1e-15)


@pytest.mark.parametrize("total", [8, 16, 24, 32])
def test_complete_decision_table_is_exact(total):
    for passes in range(total + 1):
        result = decisions.decision(passes, total)
        assert result.q == decisions.q(passes, total)
        if passes in EXPECTED[total]["accept"]:
            assert result.action == constants.ACCEPT
            assert result.accepted
            assert result.next_total is None
        elif passes in EXPECTED[total]["reject"]:
            assert result.action == constants.REJECT
            assert not result.accepted
            assert result.next_total is None
        elif result.final:
            assert result.total == 32 and result.action == constants.REJECT
            assert result.q < constants.FINAL_Q_ACCEPT
        else:
            assert result.action == constants.CONTINUE
            assert result.next_total == total + 8
            assert not result.final


def test_exact_threshold_comparisons_are_inclusive():
    rows = decisions.decision_table()
    for result in rows:
        if result.action == constants.ACCEPT:
            assert result.q >= (constants.FINAL_Q_ACCEPT if result.final else constants.Q_ACCEPT)
        elif result.action == constants.REJECT:
            assert result.q < constants.FINAL_Q_ACCEPT if result.final else result.q <= constants.Q_REJECT
        else:
            assert constants.Q_REJECT < result.q < constants.Q_ACCEPT


def test_final_at_32_uses_half_not_point_eight():
    assert decisions.q(8, 32) > 0.5
    assert decisions.q(8, 32) < 0.8
    assert decisions.decision(8, 32).action == constants.ACCEPT
    assert decisions.decision(8, 32).final
    assert decisions.decision(24, 32).action == constants.ACCEPT
    assert decisions.decision(25, 32).action == constants.REJECT
    assert decisions.decision(7, 32).action == constants.REJECT


@pytest.mark.parametrize("total", [0, 1, 2, 7, 8, 9, 16, 17, 24, 25, 31, 32])
def test_budget_exhaustion_at_any_count_uses_explicit_half_threshold(total):
    for passes in range(total + 1):
        result = decisions.decision(passes, total, final=True)
        assert result.final
        assert result.next_total is None
        assert result.action == (constants.ACCEPT if decisions.q(passes, total) >= 0.5 else constants.REJECT)
        assert "budget exhausted" in result.reason
    if total < 32:
        assert any(decisions.decision(s, total).action == constants.CONTINUE for s in range(total + 1))


def test_budget_zero_uses_the_same_half_threshold():
    result = decisions.decision(0, 0, final=True)
    assert result.action == constants.ACCEPT and result.q == 0.5
    assert decisions.decision(0, 0) == decisions.Decision(
        constants.CONTINUE, 0, 0, 0.5, 8, False, "sample to 8"
    )


def test_partial_nonfinal_decision_never_silently_ends_budget():
    for total in range(0, 32):
        for passes in range(total + 1):
            result = decisions.decision(passes, total)
            if result.action == constants.CONTINUE:
                assert not result.final
                assert result.next_total == min(32, max(8, ((total // 8) + 1) * 8))


def test_constants_centralize_sampling_king_prior_and_increments():
    assert constants.BETA_PRIOR == (1, 1)
    assert constants.MIDDLE_BAND == (0.25, 0.75)
    assert constants.SAMPLE_COUNTS == (8, 16, 24, 32)
    assert constants.SAMPLING.temperature == 1.0
    assert constants.SAMPLING.top_p == 0.95
    assert constants.SAMPLING.top_k == 20
    assert constants.SAMPLING.max_tokens == 65_536
    assert constants.SAMPLING.thinking is True
    assert constants.KING.frozen_at == "deadline"
    assert constants.KING.dtype == "BF16"
    assert constants.KING.kv_cache_dtype == "BF16"
    assert constants.KING.fp8 is False


@pytest.mark.parametrize(
    "passes,total",
    [
        (-1, 8),
        (0, -1),
        (9, 8),
        (0, 33),
        (True, 8),
        (0, True),
        (1.0, 8),
        (1, 8.0),
        ("1", 8),
        (None, 8),
    ],
)
def test_invalid_counts_fail_closed(passes, total):
    if total != 33:
        with pytest.raises(ContractError):
            decisions.q(passes, total)
    with pytest.raises(ContractError):
        decisions.decision(passes, total)


def test_final_flag_is_strict_boolean_and_tail_arguments_checked():
    with pytest.raises(ContractError):
        decisions.decision(3, 8, final=1)
    for threshold in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
        with pytest.raises(ContractError):
            decisions.independent_tail(3, 8, threshold)
