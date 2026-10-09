"""Exact ROOT-authorized retry of one proven pre-update resource failure.

Original job, reset grant and failure evidence stay immutable. This authorizes
no optimizer reset, candidate deletion, scientific change, or reserve bypass.
"""
import hashlib
import math
from pathlib import Path
import re
import time

VERSION = 'fresh-genesis-pre-update-resource-retry-v1'
FAILURE = 'alternating optimizer memory and model disk budget'


def validate(envelope, reset_envelope, authority, workspace, *, historical=False, now=None):
    import trainer_reset_lifecycle as reset
    from subnet.distributed_roles import authenticate
    from subnet.optimizer_state_cache import sha
    from subnet.persistent_training_protocol import validate_job
    from subnet.unaudited_training_execution import validate as validate_execution
    from subnet.trainer_cache_lifecycle import live_original
    p, root = reset._ready(reset_envelope, authority, workspace)
    grant = authenticate(envelope, authority)
    fields = {'version', 'original_job', 'original_status', 'original_worker_log_sha256',
        'new_job', 'reset_envelope_sha256', 'created_at', 'expires_at'}
    if set(grant) != fields or grant['version'] != VERSION or grant['reset_envelope_sha256'] != sha(reset_envelope):
        raise ValueError('exact original reset resource retry grant')
    start, end = grant['created_at'], grant['expires_at']
    if (any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end))
            or not 0 < end-start <= 86400
            or not historical and not start <= (time.time() if now is None else now) < end):
        raise ValueError('bounded resource retry lifetime')
    old = authenticate(grant['original_job'], authority)
    new = authenticate(grant['new_job'], authority)
    from subnet.cache_lifecycle import identifier
    for job in (old, new): identifier(job['job_id'])
    if old['job_id'] != p['first_job_id'] or old['job_id'] == new['job_id']:
        raise ValueError('resource retry binds failed first job and distinct successor')
    for job, document in ((old, grant['original_job']), (new, grant['new_job'])):
        if reset._read(root/(job['job_id']+'.json')) != document:
            raise ValueError('original immutable local signed retry jobs')
        manifest = authenticate(job['manifest'], authority)
        binding, descriptor = validate_job(job, manifest, authority)
        if (descriptor is not None or binding['parent'] is not None or binding['global_step_before'] != 0
                or binding['genesis'] != p['next_genesis'] or binding['genesis_sha256'] != p['to_genesis_sha256']
                or binding['input_checkpoint'] != p['input_checkpoint']):
            raise ValueError('retry cannot change zero-step initial optimizer lineage')
    mutable = {'job_id', 'created_at', 'expires_at', 'persistent_training',
        'unaudited_training_execution', 'learner_selection_operator_admission'}
    if {k:v for k,v in old.items() if k not in mutable} != {k:v for k,v in new.items() if k not in mutable}:
        raise ValueError('retry changes scientific source, manifest, inputs or execution')
    def transport(job):
        value = job['persistent_training']
        return {k:v for k,v in value.items() if k != 'output_namespace'}
    if transport(old) != transport(new):
        raise ValueError('retry changes optimizer transport beyond job namespace')
    executions = [validate_execution(doc, authority, now=job['created_at'])
        for job, doc in ((old, grant['original_job']), (new, grant['new_job']))]
    def fixed_execution(value):
        row = {k:v for k,v in value.items() if k not in {'job_id','created_at','expires_at','learning_rate_authorization'}}
        lr = authenticate(value['learning_rate_authorization'], authority)
        row['learning_rate_authorization'] = {k:v for k,v in lr.items() if k not in {'job_id','created_at','expires_at'}}
        return row
    if fixed_execution(executions[0]) != fixed_execution(executions[1]):
        raise ValueError('retry changes qualified execution or effective learning rate')
    status = grant['original_status']
    if (reset._read(root/'runner-status'/(old['job_id']+'.json')) != status
            or status.get('job_id') != old['job_id'] or status.get('phase') != 'failed'
            or type(status.get('exit_code')) is not int or status['exit_code'] == 0
            or status.get('actual_wait') is not True or live_original(status)):
        raise ValueError('original retry failure is not exact terminal child evidence')
    log = root/(old['job_id']+'-worker.log')
    from subnet.cache_lifecycle import snapshot
    before = snapshot(log); data = log.read_bytes()
    if (snapshot(log) != before or re.fullmatch('[0-9a-f]{64}', grant['original_worker_log_sha256']) is None
            or hashlib.sha256(data).hexdigest() != grant['original_worker_log_sha256']
            or not data.rstrip().endswith(('ValueError: '+FAILURE).encode())
            or b'cache_budget=local_cache.admit(' not in data
            or (root/'jobs'/old['job_id']/'report.json').exists()):
        raise ValueError('exact pre-update memory-admission failure evidence')
    return grant, old, new


def require_zero_state(cache, reset_envelope, old, new):
    import trainer_reset_lifecycle as reset
    if cache.fd is None or cache.job != new:
        raise ValueError('retry requires exact new job and held optimizer lease')
    if reset._read(cache.workspace/'.cache-lifecycle/trainer-current-state.json') != reset._fence(reset_envelope, 'retention'):
        raise ValueError('retry requires unpromoted first-ACK retention fence')
    if any((cache.root/name).exists() for name in ('current.json','pending.json')):
        raise ValueError('retry cannot reset existing optimizer state')
    for job in (old, new):
        if cache.directory(job['job_id']).exists() or (cache.workspace/'jobs'/job['job_id']/'report.json').exists():
            raise ValueError('retry cannot replace candidate or reported training')
