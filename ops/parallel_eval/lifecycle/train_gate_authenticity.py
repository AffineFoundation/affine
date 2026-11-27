"""Gate GRPO packing on TOPLOC / sampling authenticity.

Jacob rule (2026-10-23): a batch or trajectory that fails TOPLOC or sampling
verification must not be packed into GRPO training and must not earn flip
credit. Unverified (pending / unaudited) data also earns nothing and must not
enter the train pack. There is no authenticity *bonus*;
``no_authenticity_guard`` stays true on the flip formula.

Native grading receipts never set ``sampling_verified=True`` — that field stays
false because native grades do not certify sampling. Authenticity for training
comes from the sampling-verify ledger (or an explicit train denylist of fails)
applied when building the train pack.
"""
from __future__ import annotations

VERSION = 'mimo-training-authenticity-gate-v1'
ASSURANCE_UNAUDITED = 'natively-graded-sampling-unaudited'
ASSURANCE_VERIFIED = 'natively-graded-sampling-audit-verified'
ASSURANCE_REJECTED = 'native-claim-rejected-sampling-unaudited'

# Outcomes that may enter the GRPO pack. Everything else is excluded.
TRAIN_ELIGIBLE_OUTCOMES = frozenset({'verified_valid'})

# Explicit fails (and structural rejects) strip train + flip credit.
TRAIN_DENY_OUTCOMES = frozenset({
    'confirmed_invalid',
    'structural_reject',
    'structural_invalid',
    'invalid_sample',
})

# Pending / unknown / unaudited: not eligible for train or credit.
TRAIN_PENDING_OUTCOMES = frozenset({
    'pending',
    'unaudited',
    'unknown',
    'numerical_ambiguous',
    'infrastructure_error',
    None,
})


def need(value, message):
    if not value:
        raise ValueError(message)


def batch_key(miner, batch_sha256):
    """Stable identity for one committed group in the verify ledger / denylist."""
    need(type(miner) is str and len(miner) == 64, 'miner public key')
    need(type(batch_sha256) is str and len(batch_sha256) == 64 and
         all(c in '0123456789abcdef' for c in batch_sha256), 'batch digest')
    return miner + ':' + batch_sha256


def normalize_outcome(value):
    if value is None:
        return None
    need(type(value) is str and value, 'typed sampling-verify outcome')
    return value


def train_eligible_outcome(outcome):
    """True only when sampling verify has passed for this group."""
    return normalize_outcome(outcome) in TRAIN_ELIGIBLE_OUTCOMES


def deny_outcome(outcome):
    return normalize_outcome(outcome) in TRAIN_DENY_OUTCOMES


def classify_for_train(outcome):
    """Return train disposition: eligible | deny | pending."""
    value = normalize_outcome(outcome)
    if value in TRAIN_ELIGIBLE_OUTCOMES:
        return 'eligible'
    if value in TRAIN_DENY_OUTCOMES:
        return 'deny'
    return 'pending'


def filter_train_pack(selected, outcomes_by_key, *, denylist_keys=()):
    """Keep only verify-pass groups for GRPO; strip fails and unverified.

    ``selected`` is the sealed coordinator ``selected`` list (native-eligible).
    ``outcomes_by_key`` maps ``batch_key(miner, batch_sha256)`` → outcome string
    from the sampling-verify ledger (``verified_valid``, ``confirmed_invalid``,
    ``pending``, …). Missing keys are treated as pending/unaudited.

    ``denylist_keys`` is the staged ops train denylist (explicit verify-fail
    identities). A key on the denylist is always excluded, even if the ledger
    row is missing or stale.
    """
    need(type(selected) is list, 'selected groups')
    need(type(outcomes_by_key) is dict, 'sampling verify outcomes by batch key')
    deny = set(denylist_keys or ())
    need(all(type(k) is str and ':' in k for k in deny), 'denylist batch keys')
    eligible, excluded = [], []
    for row in selected:
        need(type(row) is dict and type(row.get('miner')) is str and
             type(row.get('batch_sha256')) is str, 'selected train row')
        key = batch_key(row['miner'], row['batch_sha256'])
        if key in deny:
            excluded.append(dict(row, reason='sampling-verify-denylist',
                                 sampling_audit_outcome=outcomes_by_key.get(key, 'confirmed_invalid')))
            continue
        outcome = outcomes_by_key.get(key)
        disposition = classify_for_train(outcome)
        if disposition == 'eligible':
            packed = dict(row)
            packed['assurance'] = ASSURANCE_VERIFIED
            packed['sampling_audit_outcome'] = 'verified_valid'
            # Native grades still do not certify sampling; leave their flag alone.
            eligible.append(packed)
        elif disposition == 'deny':
            excluded.append(dict(row, reason='sampling-verify-fail',
                                 sampling_audit_outcome=normalize_outcome(outcome)))
        else:
            excluded.append(dict(row, reason='sampling-unverified',
                                 sampling_audit_outcome=normalize_outcome(outcome) or 'pending'))
    return eligible, excluded


def build_train_pack_payload(dispatch_payload, outcomes_by_key, *, denylist_keys=()):
    """Build a filtered train-pack payload from a sealed coordinator dispatch.

    The sealed dispatch (credit / audit join surface) is unchanged. The returned
    payload is what the trainer may consume: only verify-pass groups, with
    ``training_requires_sampling_audit=True``.
    """
    need(type(dispatch_payload) is dict, 'sealed dispatch payload')
    need(dispatch_payload.get('version') == 'mimo-complete-group-coordinator-v1',
         'complete-group coordinator dispatch')
    selected = list(dispatch_payload.get('selected') or [])
    eligible, stripped = filter_train_pack(selected, outcomes_by_key, denylist_keys=denylist_keys)
    group_size = None
    if dispatch_payload.get('selected_rollout_count') and dispatch_payload.get('selected_task_count'):
        need(dispatch_payload['selected_task_count'] == len(selected), 'sealed selected count')
        if selected:
            group_size = dispatch_payload['selected_rollout_count'] // dispatch_payload['selected_task_count']
    if group_size is None:
        # Fall back when sealing an empty epoch or tests omit rollout counts.
        group_size = 8
    excluded = list(dispatch_payload.get('excluded') or []) + stripped
    return dict(
        version=dispatch_payload['version'],
        epoch=dispatch_payload['epoch'],
        run_id=dispatch_payload['run_id'],
        checkpoint=dispatch_payload['checkpoint'],
        manifest_sha256=dispatch_payload['manifest_sha256'],
        population_sha256=dispatch_payload.get('population_sha256'),
        selection=dispatch_payload.get('selection'),
        selected=eligible,
        excluded=excluded,
        training_requires_sampling_audit=True,
        authenticity_gate=VERSION,
        sealed_dispatch_selected_task_count=len(selected),
        selected_task_count=len(eligible),
        selected_rollout_count=len(eligible) * group_size,
        sampling_verify_stripped=len(stripped),
    )


def outcomes_from_ledger(ledger):
    """Parse ``sn120-sampling-verify-ledger-v1``-shaped rows into a key→outcome map.

    Accepts either a list of observation dicts or a document with ``rows`` /
    ``observations`` / ``entries``. Each row needs ``miner``, ``batch_sha256``
    (or ``batch``) and ``outcome`` (or ``status``).
    """
    need(type(ledger) is dict or type(ledger) is list, 'verify ledger')
    if type(ledger) is list:
        rows = ledger
    else:
        rows = ledger.get('rows') or ledger.get('observations') or ledger.get('entries') or []
        need(type(rows) is list, 'ledger rows')
    out = {}
    for row in rows:
        need(type(row) is dict, 'ledger row')
        miner = row.get('miner')
        batch = row.get('batch_sha256') or row.get('batch')
        outcome = row.get('outcome') or row.get('status')
        key = batch_key(miner, batch)
        need(key not in out or out[key] == outcome, 'conflicting verify ledger row')
        out[key] = normalize_outcome(outcome)
    return out


def denylist_keys_from_document(document):
    """Parse a staged train denylist document into batch keys."""
    need(type(document) is dict, 'train denylist document')
    keys = document.get('keys') or document.get('denylist') or document.get('batch_keys') or []
    need(type(keys) is list, 'denylist keys list')
    rows = document.get('rows') or document.get('entries') or []
    need(type(rows) is list, 'denylist rows')
    result = set()
    for key in keys:
        need(type(key) is str and ':' in key, 'denylist key')
        result.add(key)
    for row in rows:
        need(type(row) is dict, 'denylist row')
        result.add(batch_key(row['miner'], row.get('batch_sha256') or row['batch']))
    return result
