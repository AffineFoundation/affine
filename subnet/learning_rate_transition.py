"""Explicit signed learning-rate changes for an existing persistent Adam state.

This changes only the effective learning rate of the authorized update range.
The original optimizer genesis, master weights, moments and counters persist.
"""
import copy
import math
import time

from .training_receipts import authenticate, digest, sha

VERSION = 'persistent-adamw-effective-learning-rate-v1'
GENESIS_AUTH_VERSION = 'persistent-adamw-effective-learning-rate-genesis-v1'
GENESIS_DOCUMENT_VERSION = 'explicit-fp32-master-genesis-v2-effective-lr'
STATE_VERSION = 'persistent-fp32-trainer-state-v2-effective-lr'
FIELDS = frozenset(('version', 'epoch', 'job_id', 'input_checkpoint',
    'parent_descriptor_sha256', 'genesis_sha256', 'optimizer_step_before',
    'steps', 'parameters_sha256', 'base_hyperparameters_sha256',
    'effective_learning_rate', 'created_at', 'expires_at',
    'execution_release_sha256'))


def genesis_document(parameters_sha256, input_checkpoint, run_id, rate):
    """Unique explicit restart identity; the enclosing ROOT job authorizes it."""
    from .persistent_cpu_adamw import HYPERPARAMETERS, POLICY
    for item in (parameters_sha256, input_checkpoint, run_id):
        digest(item)
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= HYPERPARAMETERS['lr']:
        raise ValueError('finite conservative genesis learning rate')
    return dict(version=GENESIS_DOCUMENT_VERSION, policy=POLICY,
        hyperparameters=copy.deepcopy(HYPERPARAMETERS), parameters_sha256=parameters_sha256,
        input_checkpoint=input_checkpoint, explicit_optimizer_genesis=True,
        run_id=run_id, initial_effective_learning_rate=rate)


def validate_genesis_document(value, parameters_sha256, input_checkpoint):
    if not isinstance(value, dict):
        raise ValueError('explicit unique learning-rate genesis document')
    expected = genesis_document(parameters_sha256, input_checkpoint,
        value.get('run_id'), value.get('initial_effective_learning_rate'))
    if sha(value) != sha(expected):
        raise ValueError('exact unique learning-rate genesis document')
    return copy.deepcopy(value)


def validate_authorization(document, authority, *, epoch, job_id,
        input_checkpoint, parent_descriptor_sha256, genesis_sha256,
        optimizer_step_before, steps, parameters_sha256, now=None,
        historical=False):
    from .persistent_cpu_adamw import HYPERPARAMETERS
    value = authenticate(document, authority)
    initial = value.get('version') == GENESIS_AUTH_VERSION
    if set(value) != (FIELDS | {'run_id'} if initial else FIELDS) or value['version'] not in (VERSION, GENESIS_AUTH_VERSION):
        raise ValueError('exact signed effective learning-rate authorization')
    if type(value['optimizer_step_before']) is not int or type(value['steps']) is not int:
        raise ValueError('signed learning-rate counters must be integers, not booleans')
    expected = dict(epoch=epoch, job_id=job_id, input_checkpoint=input_checkpoint,
        parent_descriptor_sha256=parent_descriptor_sha256,
        genesis_sha256=genesis_sha256, optimizer_step_before=optimizer_step_before,
        steps=steps, parameters_sha256=parameters_sha256,
        base_hyperparameters_sha256=sha(HYPERPARAMETERS))
    if any(value[key] != wanted for key, wanted in expected.items()):
        raise ValueError('learning-rate authorization parent/job/input/counter binding')
    for key in ('input_checkpoint', 'genesis_sha256',
                'parameters_sha256', 'base_hyperparameters_sha256', 'execution_release_sha256'):
        digest(value[key])
    if initial:
        if parent_descriptor_sha256 is not None or type(optimizer_step_before) is not int or optimizer_step_before != 0:
            raise ValueError('explicit genesis grant requires absent parent and zero counter')
        expected_genesis = genesis_document(parameters_sha256, input_checkpoint,
            value['run_id'], value['effective_learning_rate'])
        if sha(expected_genesis) != genesis_sha256:
            raise ValueError('genesis grant exact unique run/rate/checkpoint/inventory binding')
    else:
        digest(parent_descriptor_sha256)
    if (not isinstance(epoch, str) or not epoch or len(epoch) > 200 or
            not isinstance(job_id, str) or not job_id or len(job_id) > 200 or
            type(optimizer_step_before) is not int or optimizer_step_before < (0 if initial else 1) or
            type(steps) is not int or not 1 <= steps <= 32 or
            optimizer_step_before + steps >= 2**31):
        raise ValueError('bounded existing optimizer continuation')
    rate = value['effective_learning_rate']
    if (type(rate) not in (int, float) or not math.isfinite(rate) or
            not 0 < rate <= HYPERPARAMETERS['lr']):
        raise ValueError('finite positive learning rate no larger than original rate')
    start, end = value['created_at'], value['expires_at']
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end))
            or not 0 <= start < end):
        raise ValueError('learning-rate authorization finite time bounds')
    if not historical:
        observed = time.time() if now is None else now
        if type(observed) not in (int, float) or not math.isfinite(observed) or not start <= observed < end:
            raise ValueError('learning-rate execution authorization not currently valid')
    return copy.deepcopy(value)


def effective_hyperparameters(value):
    from .persistent_cpu_adamw import HYPERPARAMETERS
    result = copy.deepcopy(HYPERPARAMETERS)
    result['lr'] = value['effective_learning_rate']
    return result


def validate_state_transition(descriptor):
    """Descriptor itself must already be bound by the caller's ROOT digest.

The historical grant remains authentic after expiry. It explains exactly the
LR used for this completed state; it does not authorize any subsequent step.
"""
    document = descriptor['learning_rate_authorization']
    authority = descriptor['learning_rate_authority']
    value = authenticate(document, authority)
    if type(value.get('steps')) is not int or not 1 <= value['steps'] <= 32:
        raise ValueError('historical learning-rate bounded step count')
    value = validate_authorization(document, authority,
        epoch=descriptor['epoch'], job_id=value.get('job_id'),
        input_checkpoint=descriptor['input_checkpoint'],
        parent_descriptor_sha256=descriptor['parent_state_sha256'],
        genesis_sha256=descriptor['genesis_sha256'],
        optimizer_step_before=descriptor['optimizer_steps'] - value.get('steps', 0),
        steps=value.get('steps'), parameters_sha256=descriptor['parameters_sha256'],
        historical=True)
    if sha(descriptor['hyperparameters']) != sha(effective_hyperparameters(value)):
        raise ValueError('state must report the actual authorized effective hyperparameters')
    return value
