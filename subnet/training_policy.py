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


OBJECTIVE_VERSION = 'sequence-mean-preference-positive-nll-v1'


def objective_config(value=None):
    if value is None:
        return dict(version=OBJECTIVE_VERSION, positive_nll_weight=0.0)
    if (not isinstance(value,dict) or set(value)!={'version','positive_nll_weight'}
            or value['version']!=OBJECTIVE_VERSION
            or type(value['positive_nll_weight'])not in(int,float)
            or not math.isfinite(value['positive_nll_weight'])
            or value['positive_nll_weight'] not in (0,1)):
        raise ValueError('explicit zero-or-unit positive-NLL objective')
    return dict(value)



HORIZON_VERSION = 'retained-adam-unit-positive-nll-16-v1'


def objective_horizon(value):
    """Fixed successful-update dose; no wall-clock or outcome-dependent selection."""
    from .training_receipts import digest
    keys={'version','first_optimizer_step','updates','initial_checkpoint',
          'initial_parent_descriptor_sha256','coefficient_during','coefficient_after'}
    if (not isinstance(value,dict) or set(value)!=keys or value['version']!=HORIZON_VERSION
            or type(value['first_optimizer_step']) is not int or value['first_optimizer_step']<1
            or type(value['updates']) is not int or value['updates']!=16
            or type(value['coefficient_during']) not in (int,float) or value['coefficient_during']!=1
            or type(value['coefficient_after']) not in (int,float) or value['coefficient_after']!=0):
        raise ValueError('exact retained-Adam sixteen-update objective horizon')
    digest(value['initial_checkpoint']);digest(value['initial_parent_descriptor_sha256'])
    return dict(value)


def objective_at_step(horizon,optimizer_step_before):
    h=objective_horizon(horizon)
    if type(optimizer_step_before) is not int or optimizer_step_before<h['first_optimizer_step']:
        raise ValueError('objective cannot precede its retained parent')
    return objective_config(dict(version=OBJECTIVE_VERSION,positive_nll_weight=(
        1.0 if optimizer_step_before<h['first_optimizer_step']+h['updates'] else 0.0)))
