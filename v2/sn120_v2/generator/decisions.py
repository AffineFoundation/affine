"""Sequential Generator decisions from an exact uniform-prior posterior.

With ``s`` passes in ``n`` Bernoulli trials, Beta(1,1) gives
``p ~ Beta(s+1, n-s+1)``. The informativeness statistic is
``q = P(0.25 <= p <= 0.75)``. This module uses the binomial identity

    P(p <= x) = P(Binomial(n+1, x) >= s+1)
              = sum(comb(n+1,j) * x**j * (1-x)**(n+1-j), j=s+1..n+1).

No SciPy import, sampling, numerical integration, or approximation is needed.
The calculation is an exact float evaluation of rational polynomial sums; use
``q_exact`` for a deterministic rational result when required by an audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, isfinite
from ..errors import ContractError
from .constants import (
    ACCEPT,
    BETA_PRIOR,
    CONTINUE,
    FINAL_Q_ACCEPT,
    MAX_SAMPLES,
    MIDDLE_BAND,
    MIN_SAMPLES,
    Q_ACCEPT,
    Q_REJECT,
    REJECT,
    SAMPLE_COUNTS,
    SAMPLE_INCREMENT,
)


def q_exact(passes: int, total: int) -> Fraction:
    """Exact ``P(1/4 <= p <= 3/4)`` under the Beta(1,1) prior."""
    if BETA_PRIOR != (1, 1) or MIDDLE_BAND != (0.25, 0.75):
        raise ContractError("this binomial identity is pinned to Beta(1,1) and the [1/4,3/4] band")
    if type(total) is not int or type(passes) is not int:
        raise ContractError("passes and total must be integers, not booleans")
    if total < 0 or not 0 <= passes <= total:
        raise ContractError("need 0 <= passes <= total")

    def upper_tail(success_probability: Fraction) -> Fraction:
        return sum(
            (
                Fraction(comb(total + 1, successes))
                * success_probability**successes
                * (1 - success_probability) ** (total + 1 - successes)
                for successes in range(passes + 1, total + 2)
            ),
            Fraction(0),
        )

    return upper_tail(Fraction(3, 4)) - upper_tail(Fraction(1, 4))


def q(passes: int, total: int) -> float:
    """The probability that the true task pass rate is in the middle band."""
    return float(q_exact(passes, total))


@dataclass(frozen=True, slots=True)
class Decision:
    """The protocol's next action and the sufficient statistics behind it."""

    action: str
    passes: int
    total: int
    q: float
    next_total: int | None
    final: bool
    reason: str

    @property
    def accepted(self) -> bool:
        return self.action == ACCEPT

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "passes": self.passes,
            "total": self.total,
            "q": self.q,
            "next_total": self.next_total,
            "final": self.final,
            "reason": self.reason,
        }


def decision(passes: int, total: int, *, final: bool = False) -> Decision:
    """Decide accept/reject/continue at any completed sample count.

    Before 32 samples, q>=0.8 accepts and q<=0.2 rejects. Both boundaries are
    inclusive. Otherwise ask for exactly eight more samples, up to 32.

    At 32, or when the caller marks the sample budget exhausted before 32, the
    final rule is explicit: q>=0.5 accepts, q<0.5 rejects. There is no early
    final decision without ``final=True``. ``total=0`` has no observed passes
    and is accepted only under an explicitly exhausted budget (q=0.5).
    """
    if type(total) is not int or type(passes) is not int:
        raise ContractError("Generator passes and total must be integers, not booleans")
    if total < 0 or not 0 <= passes <= total or total > MAX_SAMPLES or MAX_SAMPLES != 32:
        raise ContractError(f"Generator counts must satisfy 0 <= passes <= total <= {MAX_SAMPLES}")
    if type(final) is not bool:
        raise ContractError("Generator final must be a boolean")
    probability = q(passes, total)
    if final:
        accepted = q_exact(passes, total) >= Fraction(1, 2)
        return Decision(
            ACCEPT if accepted else REJECT,
            passes,
            total,
            probability,
            None,
            True,
            f"budget exhausted at {total}: q {'>=' if accepted else '<'} {FINAL_Q_ACCEPT}",
        )
    if total < MAX_SAMPLES:
        if q_exact(passes, total) >= Fraction(4, 5):
            return Decision(ACCEPT, passes, total, probability, None, False, f"q >= {Q_ACCEPT}")
        if q_exact(passes, total) <= Fraction(1, 5):
            return Decision(REJECT, passes, total, probability, None, False, f"q <= {Q_REJECT}")
    if total == MAX_SAMPLES:
        accepted = q_exact(passes, total) >= Fraction(1, 2)
        return Decision(
            ACCEPT if accepted else REJECT,
            passes,
            total,
            probability,
            None,
            True,
            f"32 samples: q {'>=' if accepted else '<'} {FINAL_Q_ACCEPT}",
        )
    if total < MIN_SAMPLES:
        next_total = MIN_SAMPLES
    else:
        next_total = min(MAX_SAMPLES, ((total // SAMPLE_INCREMENT) + 1) * SAMPLE_INCREMENT)
    return Decision(
        CONTINUE,
        passes,
        total,
        probability,
        next_total,
        False,
        f"sample to {next_total}",
    )


def decision_table(*, final: bool = False) -> tuple[Decision, ...]:
    """All pass counts at each scheduled checkpoint, for auditing or display."""
    return tuple(decision(passes, total, final=final) for total in SAMPLE_COUNTS for passes in range(total + 1))


def independent_tail(passes: int, total: int, threshold: float) -> float:
    """Optional independent primitive: posterior ``P(p > threshold)``.

    This is a small cross-check for q consumers, not a second decision rule.
    """
    if type(passes) is not int or type(total) is not int or total < 0 or not 0 <= passes <= total:
        raise ContractError("need integer counts with 0 <= passes <= total")
    if type(threshold) not in (int, float) or not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ContractError("threshold must be a finite probability in [0,1]")
    return sum(
        comb(total + 1, successes)
        * threshold**successes
        * (1 - threshold) ** (total + 1 - successes)
        for successes in range(passes + 1, total + 2)
    )


def q_independent(passes: int, total: int) -> float:
    """Alternative direct binomial-tail calculation for tests and auditors."""
    return independent_tail(passes, total, 0.75) - independent_tail(passes, total, 0.25)


# Backward-friendly descriptive aliases for components that prefer them.
informativeness = q
admission_decision = decision
