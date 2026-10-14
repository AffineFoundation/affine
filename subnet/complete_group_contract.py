"""Resolve batch semantics from an already authenticated manifest payload.

This deliberately does not verify signatures, source pins, calibration evidence,
task identity or rollout sampling. Callers must retain those existing checks.
Legacy parsing is delegated to the historical source's original validators.
The new sampling version and feature declaration below are PROPOSED wire fields,
not claims about compatibility with existing live miners or verifiers.
"""
from collections.abc import Callable, Collection
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

GROUP_VERSION = 'complete-canonical-group-v1'
COMPLETE_GROUP_SAMPLING_VERSION = 'forced-inverse-cdf-complete-group-v1'
MIMO_SAMPLING_VERSION = 'forced-inverse-cdf-complete-group-mimo-v1'
COMPLETE_GROUP_SAMPLING_VERSIONS = frozenset({COMPLETE_GROUP_SAMPLING_VERSION, MIMO_SAMPLING_VERSION})
# v1: only completed wrong answers ('negative') satisfy min_completed_failure;
# unresolved/runaway rollouts ('neutral') are retained but never count.
LEGACY_OUTCOME_POLICY = 'completed-eos-math-unresolved-masked-v1'
# v2: unresolved (budget-capped or EOS without a gradable answer) rollouts are
# trained with reward 0 and count as failures for group admission, so a group
# of 4 correct + 4 runaways is admitted. Zero-variance groups (no success or
# no failure of either kind) are still rejected by the two minima.
UNRESOLVED_FAILURE_POLICY = 'completed-eos-math-unresolved-failure-v2'
OUTCOME_POLICIES = frozenset({LEGACY_OUTCOME_POLICY, UNRESOLVED_FAILURE_POLICY})
OUTCOME_POLICY = UNRESOLVED_FAILURE_POLICY


def failure_count(counts, policy):
    """Failures that satisfy min_completed_failure under the signed policy.

    counts maps native classification -> count (Counter/dict; missing = 0).
    """
    if policy == LEGACY_OUTCOME_POLICY:
        return counts.get('negative', 0)
    if policy == UNRESOLVED_FAILURE_POLICY:
        return counts.get('negative', 0) + counts.get('neutral', 0)
    raise ValueError('unknown completed/unresolved outcome policy')


def meets_minima(counts, group):
    """Mixed-outcome admission for a signed group_contract dict or contract."""
    get = (lambda k: group[k]) if isinstance(group, dict) else (lambda k: getattr(group, k))
    return (counts.get('positive', 0) >= get('min_success') and
            failure_count(counts, get('outcome_policy')) >= get('min_completed_failure'))
LEGACY_SAMPLING_VERSIONS = frozenset({
    'forced-inverse-cdf-replay-v1',
    'forced-inverse-cdf-prefill-v2',
    'forced-inverse-cdf-prefill-support-v3',
    'forced-inverse-cdf-prefill-miner-bound-v5',
})
LEGACY_MINER_VERSION = 'forced-inverse-cdf-prefill-miner-bound-v5'
REQUIRED_CLIENT_FEATURES = frozenset({
    'complete-canonical-groups-v1',
    'canonical-group-nonces-v1',
    'retained-unresolved-attempts-v1',
    'sampler-selected-behavior-logprobs-v1',
})
GROUP_FIELDS = frozenset({
    'version', 'group_size', 'min_success', 'min_completed_failure', 'outcome_policy',
})
NEW_SAMPLING_FIELDS = frozenset({
    'version', 'randomness', 'verification', 'generation', 'calibration',
    'support_adjudication',
})
# These are not fields on the authenticated legacy payload. Reject alternate
# spellings instead of allowing a future caller to use an unbound second limit.
ROOT_ALIASES = frozenset({
    'group_size', 'min_success', 'min_completed_failure', 'max_attempts',
    'nonce_count', 'nonce_start', 'nonce_end', 'nonce_range', 'attempt_limit',
})


@dataclass(frozen=True)
class ValidationHooks:
    """Supply functions from the manifest's trusted historical/composed source.

    Signature/source selection happens before this call. The legacy functions
    are configured_quotas, controller.class_quotas, forced_sampling.validate and
    fast_prefill_audit.calibration; no model or network work belongs in them.
    Calibration validation checks shape/bounds only, not executed evidence.
    """
    configured_quotas: Callable[[dict], tuple[int, int]]
    class_quotas: Callable[[Any, Any, dict | None], tuple[int, int]]
    validate_sampling: Callable[[dict], dict]
    validate_calibration: Callable[[dict], dict]


@dataclass(frozen=True)
class LegacyQuotaContract:
    success_quota: int
    failure_quota: int
    max_batches: int
    sampling_version: str | None
    attempt_count: int | None
    kind: Literal['legacy-quota'] = 'legacy-quota'

    @property
    def required_rollouts(self) -> int:
        return self.success_quota + self.failure_quota

    @property
    def allowed_nonces(self) -> range | None:
        """Search budget only: legacy miners need not return every attempt."""
        return None if self.attempt_count is None else range(self.attempt_count)


@dataclass(frozen=True)
class CompleteGroupContract:
    group_size: int
    min_success: int
    min_completed_failure: int
    max_batches: int
    required_client_features: tuple[str, ...]
    group_version: str = GROUP_VERSION
    sampling_version: str = COMPLETE_GROUP_SAMPLING_VERSION
    outcome_policy: str = OUTCOME_POLICY
    kind: Literal['complete-group'] = 'complete-group'

    @property
    def required_rollouts(self) -> int:
        return self.group_size

    @property
    def required_nonces(self) -> range:
        """All these nonces exactly once, in canonical serialized order."""
        return range(self.group_size)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, low: int, high: int, name: str) -> int:
    _need(type(value) is int and low <= value <= high, f'{name}: bounded integer required')
    return value


def _digest(value: Any, name: str) -> None:
    _need(type(value) is str and len(value) == 64
          and set(value) <= set('0123456789abcdef'), f'{name}: lowercase 32-byte digest required')


def for_manifest(
    manifest: dict,
    *,
    validators: ValidationHooks,
    supported_features: Collection[str] = (),
) -> LegacyQuotaContract | CompleteGroupContract:
    """Resolve signed payload semantics, before checkpoint download or mining.

    Caller authentication is mandatory but is outside this pure structural
    function. Unknown versions fail closed. A new group has one size field;
    there is no K/L projection or independently configurable attempt budget.
    Whole-manifest unrelated fields are left to their original validators.
    """
    _need(type(manifest) is dict, 'manifest must be a decoded authenticated payload')
    _need(not {'payload', 'signature', 'signer'} & manifest.keys(),
          'pass the authenticated payload, not an unverified signed envelope')
    _need(not ROOT_ALIASES & manifest.keys(), 'ambiguous root quota/nonce aliases')
    _need(isinstance(validators, ValidationHooks), 'explicit trusted validation hooks required')
    # A defensive copy prevents caller-owned validator code from changing the
    # object whose signed bytes the caller already authenticated.
    value = deepcopy(manifest)
    sampling = value.get('sampling_contract')
    _need(sampling is None or type(sampling) is dict, 'sampling_contract must be an object')
    version = sampling.get('version') if sampling is not None else None
    _need(version is None or type(version) is str, 'sampling version must be a string')

    if 'group_contract' not in value:
        _need('required_client_features' not in value, 'group features without group contract')
        _need(version is None or version in LEGACY_SAMPLING_VERSIONS,
              'unknown sampler or missing complete-group contract; upgrade required')
        if sampling is None:
            _need('sampling_source_hash' not in value, 'sampling contract missing')
        else:
            validators.validate_sampling(sampling)
        # Preserve the original distinction: samples_per_batch is an optional
        # consistency/config projection, not a replacement for signed v5 K/L.
        configured = validators.configured_quotas(value)
        k, l = ((value.get('K'), value.get('L')) if version == LEGACY_MINER_VERSION else configured)
        k, l = validators.class_quotas(k, l, sampling)
        # The v5 binding enforces this bound. Do not impose it retroactively on
        # older manifests: their original admission layer owns that field.
        limit = value.get('max_batches') if version == LEGACY_MINER_VERSION else value.get('max_batches', 4)
        if version == LEGACY_MINER_VERSION:
            limit = _integer(limit, 1, 256, 'max_batches')
        return LegacyQuotaContract(k, l, limit, version,
                                   sampling['max_attempts'] if sampling is not None else None)

    _need(not {'K', 'L', 'samples_per_batch'} & value.keys(),
          'complete groups cannot carry legacy quota fields, even agreeing ones')
    group = value['group_contract']
    _need(type(group) is dict and set(group) == GROUP_FIELDS, 'exact group_contract fields required')
    _need(group['version'] == GROUP_VERSION, 'unknown group contract version; upgrade required')
    n = _integer(group['group_size'], 2, 128, 'group_size')
    good = _integer(group['min_success'], 1, n, 'min_success')
    wrong = _integer(group['min_completed_failure'], 1, n, 'min_completed_failure')
    _need(good + wrong <= n, 'independent outcome minima exceed group size')
    _need(group['outcome_policy'] in OUTCOME_POLICIES, 'unknown completed/unresolved outcome policy')
    _need(type(sampling) is dict and set(sampling) == NEW_SAMPLING_FIELDS,
          'exact group sampling fields required; derive nonce range from group_size only')
    _need(version in COMPLETE_GROUP_SAMPLING_VERSIONS,
          'complete groups need their new sampling version, not legacy quota sampling')
    _digest(sampling['randomness'], 'sampling randomness')
    _need(sampling['verification'] == 'prefill-cdf-calibrated'
          and sampling['generation'] == 'cached-eager-inverse-cdf'
          and sampling['support_adjudication'] == 'exact-cached-replay-v1',
          'unsupported group sampling/verification profile')
    validators.validate_calibration(sampling['calibration'])
    limit = _integer(value.get('max_batches'), 1, 256, 'max_batches')
    features = value.get('required_client_features')
    _need(type(features) is list and all(type(f) is str and f for f in features)
          and len(features) == len(set(features)), 'unique required_client_features list required')
    _need(REQUIRED_CLIENT_FEATURES <= set(features), 'mandatory complete-group client features missing')
    _need(isinstance(supported_features, Collection) and not isinstance(supported_features, (str, bytes))
          and all(type(f) is str for f in supported_features), 'supported features must be a string collection')
    _need(set(features) <= set(supported_features), 'unsupported required client feature; upgrade required')
    return CompleteGroupContract(n, good, wrong, limit, tuple(sorted(features)), sampling_version=version,
                                 outcome_policy=group['outcome_policy'])
