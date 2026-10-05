"""Explicit future epoch policy and immutable post-freeze coverage context."""
import hashlib
import math
import re

from .backend_jobs import FIXED_POLICY, COVERED_POLICY, canonical
from .persistent_cpu_adamw import POLICY as PERSISTENT_POLICY

COVERAGE_REVISION = 'frozen-verified-pairs-v1'


def epoch_policy(config):
    policy = config.get('training_policy', FIXED_POLICY)
    if policy not in (FIXED_POLICY, COVERED_POLICY, PERSISTENT_POLICY):
        raise ValueError('approved epoch training policy')
    if policy in (COVERED_POLICY,PERSISTENT_POLICY) and config.get('balanced_replay'):
        raise ValueError('covered historical replay requires separate admission')
    return policy


def coverage_manifest(manifest, receipts, challenge):
    if epoch_policy(manifest) not in (COVERED_POLICY,PERSISTENT_POLICY):
        raise ValueError('explicit covered training manifest required')
    timestamp = challenge.get('generated_after_freeze_at')
    if (type(timestamp) not in (int, float) or not math.isfinite(timestamp) or
            timestamp < manifest['deadline'] or challenge.get('receipts') != receipts or
            not receipts or not isinstance(challenge.get('seed'), str) or
            re.fullmatch('[0-9a-f]{64}', challenge['seed']) is None):
        raise ValueError('original frozen coverage challenge')
    if ('audit_frozen_receipts' in manifest and manifest['audit_frozen_receipts'] != receipts
            or 'audit_seed' in manifest and manifest['audit_seed'] != challenge['seed']):
        raise ValueError('coverage and original audit population binding')
    context = dict(version=COVERAGE_REVISION, epoch=manifest['epoch'],
                   checkpoint=manifest['checkpoint']['id'], seed=challenge['seed'],
                   receipts_sha256=hashlib.sha256(canonical(receipts)).hexdigest(),
                   generated_after_freeze_at=timestamp)
    if 'training_coverage' in manifest and manifest['training_coverage'] != context:
        raise ValueError('immutable covered training context')
    return dict(manifest, audit_seed=challenge['seed'], audit_frozen_receipts=receipts,
                training_coverage=context)


def validate_coverage(manifest, submissions):
    """Validate signed claims before checkpoint/artifact reads or GPU execution."""
    context = manifest.get('training_coverage', {})
    receipts = manifest.get('audit_frozen_receipts')
    if not isinstance(context, dict) or not isinstance(receipts, dict):
        raise ValueError('signed covered training context required')
    challenge = dict(seed=context.get('seed'), receipts=receipts,
                     generated_after_freeze_at=context.get('generated_after_freeze_at'))
    expected = coverage_manifest(manifest, receipts, challenge)
    if expected['training_coverage'] != context:
        raise ValueError('signed covered training context fields')
    hashes = {b['sha256']for r in receipts.values()for b in r['artifacts']}if manifest.get('submission_transport_policy')else{r['sha256'] for r in receipts.values()}
    if (not submissions or any(r.get('sha256') not in hashes for r in submissions)):
        raise ValueError('training submissions outside frozen coverage population')
    return context
