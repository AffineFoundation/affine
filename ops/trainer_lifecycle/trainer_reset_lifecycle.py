"""Explicit fresh-genesis retirement and CPU-only compatibility adapter.

This never mutates the numerical backend. Both catalogues are fenced before
old optimizer bytes are removed; an older cleanup executable fails closed.
The original signed job, ACK, report and reset history remain on disk.
"""
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import time

VERSION = 'explicit-fresh-genesis-local-optimizer-reset-v1'
FENCE = 'fresh-genesis-reset-pending-first-ACK-v1'
CURRENT = 'reset-bound-trainer-retention-v1'

RECOVERY_SHA256 = '0132315f21d59b67a3af8ad3261337e1d766e595c16ee12033112b4c435cb1cb'

def _recovery_module():
    import hashlib
    import trainer_reset_recovery as retry
    if hashlib.sha256(Path(retry.__file__).read_bytes()).hexdigest() != RECOVERY_SHA256:
        raise ValueError('pinned first-job recovery adapter changed')
    return retry


def original_ack(ack, authority, workspace):
    """Read the original ROOT job/report, independent of legacy CPU helpers."""
    from subnet.distributed_roles import authenticate
    from subnet.optimizer_state_cache import sha
    from subnet.cache_lifecycle import identifier
    value=authenticate(ack,authority);root=Path(workspace)
    fields={'version','job_id','job_sha256','report_sha256','input_checkpoint','input_cache','new_checkpoint','trainer_state','authority_state_committed'}
    if set(value)!=fields or value['version']!='durable-original-trainer-cache-ACK-v1' or value['authority_state_committed']is not True:
        raise ValueError('exact original committed optimizer ACK')
    jobid=identifier(value['job_id']);job=authenticate(_read(root/(jobid+'.json')),authority)
    report=_read(root/'jobs'/jobid/'report.json');manifest=authenticate(job['manifest'],authority)
    state=report['persistent_training_state'];descriptor=state['descriptor'];pointer=value['trainer_state']
    genesis=descriptor.get('genesis_sha256');step=descriptor.get('optimizer_steps')
    if (job.get('role')!='train' or report.get('success')is not True or report.get('job_id')!=jobid
            or sha(job)!=value['job_sha256'] or report.get('job_sha256')!=sha(job)
            or sha(report)!=value['report_sha256'] or report.get('new_checkpoint')!=value['new_checkpoint']
            or manifest['checkpoint']!=value['input_checkpoint']
            or sha(descriptor)!=state['descriptor_sha256'] or state['descriptor_sha256']!=pointer['descriptor_sha256']
            or state['namespace']!=pointer['namespace'] or type(step)is not int or step<0 or step!=pointer['optimizer_steps']
            or not isinstance(genesis,str) or re.fullmatch('[0-9a-f]{64}',genesis)is None
            or genesis!=pointer.get('genesis_sha256') or genesis!=manifest.get('trainer_state_binding',{}).get('genesis_sha256')
            or descriptor.get('inference_checkpoint')!=value['new_checkpoint']['id']):
        raise ValueError('authenticated original optimizer job/report lineage')
    return value,manifest,report,dict(genesis_sha256=genesis,optimizer_steps=step,
        descriptor_sha256=sha(descriptor),inference_checkpoint=descriptor['inference_checkpoint'])


def previous_lineage(value, ack, authority, root, lineage):
    """Accept authenticated current v2 or the exact old two-field catalogue."""
    if set(value)=={'optimizer_steps','descriptor_sha256'}:
        if any(value[k]!=lineage[k]for k in value):raise ValueError('exact legacy old lineage')
        return lineage,[],ack
    if value.get('version')!='genesis-bound-trainer-retention-v2':
        raise ValueError('unknown old trainer retention catalogue')
    _,_,_,before=original_ack(value['ROOT_ack'],authority,root)
    retired=value['retired_geneses']
    if (set(value)!={'version','genesis_sha256','optimizer_steps','descriptor_sha256','inference_checkpoint','ROOT_ack','retired_geneses'}
            or any(value[k]!=v for k,v in before.items()) or not isinstance(retired,list)
            or len(retired)!=len(set(retired)) or before['genesis_sha256']in retired):
        raise ValueError('authenticated old retention lineage')
    return before,retired,value['ROOT_ack']


def _read(path):
    from subnet.cache_lifecycle import snapshot
    if snapshot(path)['mode'] & 0o077:
        raise ValueError('private owned reset metadata required')
    return json.loads(path.read_bytes())


def _save(path, value):
    from subnet.storage import canonical
    tmp = path.with_name(path.name + '.reset-tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _scope(envelope, authority, workspace, *, now=None, historical=False):
    from subnet.distributed_roles import authenticate
    from subnet.optimizer_state_cache import sha
    p = authenticate(envelope, authority); root = Path(workspace).absolute()
    fields = {'version', 'workspace', 'created_at', 'expires_at', 'previous_current_sha256',
        'previous_retention_sha256', 'previous_ack', 'from_genesis_sha256',
        'previous_descriptor_sha256', 'next_genesis', 'to_genesis_sha256',
        'input_checkpoint', 'first_job_id', 'durable_transition_receipt',
        'extra_lease_paths', 'retire_old_optimizer'}
    if (set(p) != fields or p['version'] != VERSION or p['workspace'] != str(root)
            or root != root.resolve() or not root.is_dir()
            or p['retire_old_optimizer'] is not True):
        raise ValueError('exact explicit optimizer reset scope')
    for name in ('previous_current_sha256', 'previous_retention_sha256',
            'from_genesis_sha256', 'previous_descriptor_sha256', 'to_genesis_sha256', 'input_checkpoint'):
        if not isinstance(p[name], str) or re.fullmatch('[0-9a-f]{64}', p[name]) is None:
            raise ValueError('exact reset digest')
    from subnet.cache_lifecycle import identifier
    identifier(p['first_job_id'])
    start, end = p['created_at'], p['expires_at']
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end))
            or not 0 < end - start <= 86400):
        raise ValueError('bounded explicit reset lifetime')
    if not historical and not start <= (time.time() if now is None else now) < end:
        raise ValueError('reset grant expired')
    genesis = p['next_genesis']; receipt = p['durable_transition_receipt']
    if (sha(genesis) != p['to_genesis_sha256'] or genesis.get('input_checkpoint') != p['input_checkpoint']
            or genesis.get('explicit_optimizer_genesis') is not True
            or p['from_genesis_sha256'] == p['to_genesis_sha256']
            or not isinstance(receipt, dict) or receipt.get('full_readback_verified') is not True
            or not isinstance(receipt.get('key'), str) or not receipt['key']
            or re.fullmatch('[0-9a-f]{64}', receipt.get('sha256', '')) is None):
        raise ValueError('explicit new genesis and durable history required')
    leases = p['extra_lease_paths']
    if not isinstance(leases, list) or len(leases) > 3 or len(leases) != len(set(leases)):
        raise ValueError('bounded exact extra leases')
    for name in leases:
        path = Path(name)
        if not path.is_absolute() or path != path.resolve():
            raise ValueError('canonical existing private-study lease')
    journal = root / '.cache-lifecycle' / ('fresh-genesis-reset-' + sha(envelope) + '.json')
    return p, root, journal


def _fence(envelope, role):
    return dict(version=FENCE, role=role, transition=envelope)


def _ready(envelope, authority, workspace):
    p, root, journal = _scope(envelope, authority, workspace, historical=True)
    from subnet.optimizer_state_cache import sha
    result = _read(journal.with_suffix('.complete.json'))
    if (result.get('transition_sha256') != sha(envelope) or result.get('status') != 'complete'
            or result.get('to_genesis_sha256') != p['to_genesis_sha256']):
        raise ValueError('completed explicit optimizer retirement required')
    return p, root


def plan_reset(first_job_envelope, authority, workspace, durable_transition_receipt, *, extra_lease_paths=(), now=None):
    """Build an unsigned exact plan from an already signed initial train job.

    This is read-only: ROOT still signs the result before execute_reset. There
    is no guessed job identity and no reset when constructing ordinary jobs.
    """
    from subnet.distributed_roles import authenticate
    from subnet.optimizer_state_cache import sha
    root=Path(workspace).absolute();job=authenticate(first_job_envelope,authority)
    manifest=authenticate(job['manifest'],authority);binding=manifest['trainer_state_binding']
    if (job.get('role')!='train' or binding.get('parent')is not None
            or type(binding.get('global_step_before'))is not int or binding['global_step_before']!=0
            or not isinstance(binding.get('genesis'),dict)
            or sha(binding['genesis'])!=binding['genesis_sha256']
            or binding['input_checkpoint']!=manifest['checkpoint']['id']):
        raise ValueError('original signed initial train job required')
    current=_read(root/'.optimizer-state-cache/current.json')
    retention=_read(root/'.cache-lifecycle/trainer-current-state.json')
    _,_,_,before=original_ack(current['ROOT_ack'],authority,root)
    previous_lineage(retention,current['ROOT_ack'],authority,root,before)
    start=time.time()if now is None else now
    return dict(version=VERSION,workspace=str(root),created_at=start,expires_at=start+3600,
        previous_current_sha256=sha(current),previous_retention_sha256=sha(retention),
        previous_ack=current['ROOT_ack'],from_genesis_sha256=before['genesis_sha256'],
        previous_descriptor_sha256=before['descriptor_sha256'],next_genesis=binding['genesis'],
        to_genesis_sha256=binding['genesis_sha256'],input_checkpoint=binding['input_checkpoint'],
        first_job_id=job['job_id'],durable_transition_receipt=durable_transition_receipt,
        extra_lease_paths=list(extra_lease_paths),retire_old_optimizer=True)


def execute_reset(envelope, authority, workspace, *, now=None, idle_guard=None):
    """Retire only the signed exact old optimizer after both leases are held.

    Safe to retry an interrupted operation with the same envelope. No wildcard
    deletion, arbitrary directory traversal, model deletion or optimizer upload.
    """
    from subnet.cache_lifecycle import CacheLifecycle, snapshot
    from subnet.optimizer_state_cache import StateCache, sha, member
    p, root, journal = _scope(envelope, authority, workspace, now=now)
    ack, manifest, report, lineage = original_ack(p['previous_ack'], authority, root)
    if (lineage['genesis_sha256'] != p['from_genesis_sha256']
            or lineage['descriptor_sha256'] != p['previous_descriptor_sha256']):
        raise ValueError('exact original committed optimizer lineage')
    from subnet.distributed_roles import authenticate
    job = authenticate(_read(root / (ack['job_id'] + '.json')), authority)
    lifecycle = CacheLifecycle(root)
    if idle_guard is None:
        from ops.training_retention import unreferenced
        idle_guard = unreferenced
    with contextlib.ExitStack() as stack:
        stack.enter_context(lifecycle.lease_checkpoint('trainer-state-retention', blocking=False))
        cache = stack.enter_context(StateCache(root, job, manifest, authority))
        for name in p['extra_lease_paths']:
            snapshot(name)
            fd = os.open(name, os.O_RDWR | os.O_NOFOLLOW)
            stack.callback(os.close, fd); fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current_path = cache.root / 'current.json'
        retention_path = lifecycle.meta / 'trainer-current-state.json'
        if journal.exists():
            saved = _read(journal)
            if saved['transition'] != envelope:
                raise ValueError('reset journal identity changed')
            current, retention = saved['current'], saved['retention']
        else:
            current, retention = _read(current_path), _read(retention_path)
        if sha(current) != p['previous_current_sha256'] or sha(retention) != p['previous_retention_sha256']:
            raise ValueError('exact original current and retention catalogues')
        descriptor = report['persistent_training_state']['descriptor']
        before, retired, previous_ack = previous_lineage(retention, p['previous_ack'], authority, root, lineage)
        if (sha(descriptor) != p['previous_descriptor_sha256'] or before != lineage
                or previous_ack != p['previous_ack'] or current['ROOT_ack'] != p['previous_ack']
                or current['job_id']!=ack['job_id'] or current['job_sha256']!=ack['job_sha256']
                or current['descriptor_sha256']!=p['previous_descriptor_sha256']):
            raise ValueError('both catalogues must bind the same original ACK')
        directory = cache.directory(current['job_id'])
        if directory.parent != cache.root:
            raise ValueError('reset currently supports the exact owned disk candidate only')
        completion = journal.with_suffix('.complete.json')
        if completion.exists():
            _ready(envelope, authority, root)
            return _read(completion)
        for path, old, role in ((current_path, current, 'optimizer'), (retention_path, retention, 'retention')):
            if _read(path) not in (old, _fence(envelope, role)):
                raise ValueError('catalogue changed during reset')
        if (cache.root / 'pending.json').exists():
            raise ValueError('pending candidate must be resolved before reset')
        promotion = cache.root / 'promotion.json'
        if promotion.exists() and _read(promotion).get('phase') != 'complete':
            raise ValueError('original promotion must be terminal')
        files = current['files']; shards = {r['name']: r for r in descriptor['shards']}
        if set(files) != set(shards) or not 1 <= len(files) <= 256:
            raise ValueError('complete bounded original optimizer inventory')
        paths = []
        if directory.exists() and set(x.name for x in directory.iterdir()) - set(files):
            raise ValueError('unowned file in retired optimizer candidate')
        for name, row in files.items():
            path = directory / member(name)
            if (row['size'], row['sha256']) != (shards[name]['size'], shards[name]['sha256']):
                raise ValueError('exact original shard descriptor binding')
            if path.exists():
                if snapshot(path) != row['stat']:
                    raise ValueError('old optimizer shard changed')
                paths.append(path)
            elif not journal.exists():
                raise ValueError('missing old optimizer before reset journal')
        idle_guard(paths)
        if not journal.exists():
            _save(journal, dict(transition=envelope, current=current, retention=retention, retired_geneses=retired))
        # Fence old executables before the first destructive operation.
        _save(retention_path, _fence(envelope, 'retention'))
        _save(current_path, _fence(envelope, 'optimizer'))
        for path in paths:
            if snapshot(path) != files[path.name]['stat']:
                raise ValueError('old optimizer changed immediately before retirement')
            path.unlink()
        if directory.exists(): directory.rmdir()
        result = dict(status='complete', transition_sha256=sha(envelope),
            to_genesis_sha256=p['to_genesis_sha256'], retired_bytes=sum(r['size'] for r in files.values()),
            retired_files=len(files), old_models_and_original_evidence_preserved=True)
        _save(completion, result)
        return result


def install_for_train(envelope, authority, workspace, *, recovery_envelope=None):
    """Install only in the separately pinned private CPU worker entrypoint."""
    import importlib
    cache = importlib.import_module('subnet.optimizer_state_cache')
    p, root = _ready(envelope, authority, workspace)
    first_job_id = p['first_job_id']
    if recovery_envelope is not None:
        retry = _recovery_module()
        _, failed_job, retry_job = retry.validate(recovery_envelope, envelope, authority, root, historical=True)
        first_job_id = retry_job['job_id']
    original = cache.StateCache.prepare_parent
    def prepare(self, descriptor, source):
        if self.workspace != root: return original(self, descriptor, source)
        binding = self.manifest.get('trainer_state_binding', {})
        if binding.get('genesis_sha256') != p['to_genesis_sha256']:
            raise ValueError('retired optimizer genesis cannot resume')
        if descriptor is None:
            if (self.job['job_id'] != first_job_id or binding.get('parent') is not None
                    or type(binding.get('global_step_before')) is not int or binding['global_step_before'] != 0
                    or binding.get('input_checkpoint') != p['input_checkpoint']
                    or binding.get('genesis') != p['next_genesis']):
                raise ValueError('exact authorized zero-step first training job')
            if recovery_envelope is not None:
                retry.validate(recovery_envelope, envelope, authority, root)
                retry.require_zero_state(self, envelope, failed_job, retry_job)
        elif descriptor.get('genesis_sha256') != p['to_genesis_sha256']:
            raise ValueError('new run continuation descriptor binding')
        marker = self.root / 'current.json'
        if marker.exists() and _read(marker) == _fence(envelope, 'optimizer'):
            if self.fd is None or descriptor is not None or (self.root / 'pending.json').exists():
                raise ValueError('initial reset fence requires held cache lease and absent candidate')
            marker.unlink()
        return original(self, descriptor, source)
    cache.StateCache.prepare_parent = prepare
    return original


def install_for_retirement(envelope, authority, workspace, *, recovery_envelope=None):
    """Admit the first genuine new-run ACK, then use ordinary lifecycle rules."""
    import importlib
    cache = importlib.import_module('subnet.optimizer_state_cache')
    retention = importlib.import_module('subnet.trainer_cache_lifecycle')
    p, root = _ready(envelope, authority, workspace)
    first_job_id = p['first_job_id']
    if recovery_envelope is not None:
        retry = _recovery_module()
        _, _, retry_job = retry.validate(recovery_envelope, envelope, authority, root, historical=True)
        first_job_id = retry_job['job_id']
    original_retire, original_promote = retention.retire, cache.promote
    def promote(ack, actor, location):
        if Path(location) == root:
            _, _, _, lineage = original_ack(ack, actor, location)
            if actor != authority or lineage['genesis_sha256'] != p['to_genesis_sha256']:
                raise ValueError('retired-genesis optimizer ACK rejected')
        return original_promote(ack, actor, location)
    def retire(ack, actor, location):
        if Path(location)!=root:return original_retire(ack,actor,location)
        value,manifest,report,lineage=original_ack(ack,actor,location)
        if actor!=authority or lineage['genesis_sha256']!=p['to_genesis_sha256']:
            return dict(status='superseded',reason='retired-genesis',removed_checkpoints=[],retired_downloads=[])
        status=_read(root/'runner-status'/(value['job_id']+'.json'))
        if status.get('phase')!='complete' or status.get('exit_code')!=0 or retention.live_original(status):
            return dict(status='deferred',reason='original-child-not-terminal')
        if any(retention.live_original(_read(path))for path in(root/'runner-status').glob('*.json')):
            return dict(status='deferred',reason='workspace-role-in-flight')
        from subnet.cache_lifecycle import CacheLifecycle
        from subnet.optimizer_state_cache import sha
        lifecycle=CacheLifecycle(root)
        with lifecycle.lease_checkpoint('trainer-state-retention',blocking=False):
            marker=lifecycle.meta/'trainer-current-state.json';prior=_read(marker)
            if prior==_fence(envelope,'retention'):
                if value['job_id']!=first_job_id:raise ValueError('exact first new optimizer ACK')
                saved=_read(_scope(envelope,authority,root,historical=True)[2])
                retired=saved['retired_geneses']+[p['from_genesis_sha256']]
            elif (prior.get('version')==CURRENT and prior.get('transition_sha256')==sha(envelope)):
                _,_,_,bound=original_ack(prior['ROOT_ack'],authority,root)
                if bound!=prior['lineage'] or bound['genesis_sha256']!=p['to_genesis_sha256']:
                    raise ValueError('original acknowledged new optimizer lineage')
                if bound['optimizer_steps']>lineage['optimizer_steps']:
                    return dict(status='superseded',reason='older-step')
                if bound['optimizer_steps']==lineage['optimizer_steps'] and bound!=lineage:
                    raise ValueError('same counter different optimizer lineage')
                retired=prior['retired_geneses']
            else:raise ValueError('unrecognized new-run retention marker')
            promoted=promote(ack,actor,location)
            if not isinstance(promoted,dict)or promoted.get('promoted')is not True:
                raise ValueError('real optimizer candidate promotion required')
            result=retention._retire_owned(lifecycle,value,ack,value['job_id'],root)
            # Keep counters nested so even legacy counter-only cleanup refuses
            # this marker before touching checkpoints on a delayed old ACK.
            _save(marker,dict(version=CURRENT,lineage=lineage,ROOT_ack=ack,
                transition_sha256=sha(envelope),retired_geneses=retired))
            result['optimizer_cache_promotion']=promoted
            return result
    retention.retire,cache.promote=retire,promote
    return original_retire,original_promote
