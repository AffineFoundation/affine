"""Never emit an all-zero flip / incentive weight vector.

Hard constraint (Jacob, 2026-10-23): authenticity gates, ``unknown_policy``, and
verify-fail stripping must never produce or submit all-zero on-chain weights.
When candidate shares have no positive mass, retain the last good shares (or
refuse submission and keep on-chain weights). Never burn and never fall back to
owner-only weights.
"""
from __future__ import annotations

import math

VERSION = 'sn120-flip-share-floor-v1'
FLOOR_POLICY = 'last-good-shares-v1'
# Refuse to invent weights when neither candidate nor last-good has mass.
REFUSE_ALL_ZERO = 'retain-onchain-no-submit-v1'


def need(value, message):
    if not value:
        raise ValueError(message)


def _number(value):
    need(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'nonnegative finite share')
    return float(value)


def positive_mass(shares):
    """True when at least one share is strictly positive."""
    need(type(shares) is dict, 'share map')
    total = 0.0
    for value in shares.values():
        total += _number(value)
    return total > 0.0


def normalize_shares(shares):
    """Return a copy normalized to sum 1. Refuses an all-zero input."""
    need(type(shares) is dict and shares, 'nonempty share map')
    cleaned = {key: _number(value) for key, value in shares.items()}
    total = math.fsum(cleaned.values())
    need(total > 0.0, 'refusing to normalize all-zero shares')
    return {key: value / total for key, value in cleaned.items()}


def apply_share_floor(candidate_shares, last_good_shares=None, *, policy=FLOOR_POLICY):
    """Apply the never-all-zero floor to a candidate flip share vector.

    Returns ``(shares, meta)`` where ``shares`` is safe to publish only when
    ``meta['submit']`` is True. When ``submit`` is False the caller must retain
    existing on-chain weights and must not submit an all-zero vector.
    """
    need(policy == FLOOR_POLICY, 'explicit last-good share floor policy')
    need(type(candidate_shares) is dict, 'candidate shares')
    meta = dict(version=VERSION, policy=policy, floor_applied=False, submit=True,
                reason='candidate-positive-mass',
                no_authenticity_bonus=True,
                never_burn=True, never_owner_only=True)
    if positive_mass(candidate_shares):
        return normalize_shares(candidate_shares), meta
    if last_good_shares is not None and positive_mass(last_good_shares):
        meta.update(floor_applied=True, reason='all-zero-candidate-retained-last-good',
                    submit=True)
        return normalize_shares(last_good_shares), meta
    meta.update(floor_applied=True, submit=False, reason='no-last-good-refuse-all-zero',
                refuse_policy=REFUSE_ALL_ZERO)
    return dict(candidate_shares), meta


def gate_unknown_policy(unknown_policy, *, credit_requires_verify_pass):
    """Document-safe unknown_policy values for authenticity-coupled flip pay.

    ``pass_through`` may temporarily credit unverified rows while the verify
    ledger catches up, but any switch that strips unknown/unverified mass must
    still run ``apply_share_floor`` before submission.
    """
    need(type(unknown_policy) is str and unknown_policy, 'unknown_policy')
    need(type(credit_requires_verify_pass) is bool, 'credit_requires_verify_pass')
    allowed = {
        # Current live default: unverified rows pass through until joined.
        'pass_through',
        # Strip unverified from credit; MUST pair with apply_share_floor.
        'strip_unverified',
        # Treat unverified as zero credit but keep last-good floor mandatory.
        'zero_unverified',
    }
    need(unknown_policy in allowed, 'unknown_policy must keep a never-all-zero floor path')
    return dict(version=VERSION, unknown_policy=unknown_policy,
                credit_requires_verify_pass=credit_requires_verify_pass,
                share_floor_required=True, floor_policy=FLOOR_POLICY,
                never_all_zero_weights=True)
