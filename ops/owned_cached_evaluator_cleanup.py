"""Post-durable-ACK disposal of one exact owned cached evaluator checkpoint."""
import hashlib,json,time
from pathlib import Path

VERSION='owned-cached-evaluation-durable-ack-v1'

def retire(ack_envelope,authority,workspace,expected_source_files):
    from subnet.backend_jobs import signed,canonical
    from subnet.cache_lifecycle import CacheLifecycle
    from subnet.evaluator_cache_lifecycle import live_original
    ack=signed(ack_envelope,authority);job=signed(ack['original_job'],authority);manifest=signed(job['manifest'],authority);report=ack['original_report'];cp=manifest['checkpoint']
    if ack.get('version')!=VERSION or ack.get('durable_report_full_readback')is not True:raise ValueError('actual durable evaluation ACK')
    if ack.get('workspace')!=str(Path(workspace).absolute()) or Path(workspace).absolute()!=Path(workspace).resolve():raise ValueError('owned diagnostic namespace')
    if (job['role']!='evaluate' or job['source_files']!=expected_source_files or manifest['source_bundle']['sha256']!='4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373' or job.get('owned_evaluation_policy',{}).get('version')!='owned-cached-native-evaluation-v1'):raise ValueError('exact approved owned evaluation source/job')
    if (ack['job_sha256']!=hashlib.sha256(canonical(job)).hexdigest() or report['job_sha256']!=ack['job_sha256'] or ack['report_sha256']!=hashlib.sha256(canonical(report)).hexdigest() or ack['checkpoint']!=cp or report['checkpoint']!=cp['id'] or report['job_id']!=job['job_id'] or report['success']is not True):raise ValueError('original model/report ACK binding')
    root=Path(workspace);markers=root/'runner-status'
    status=json.loads((markers/(job['job_id']+'.json')).read_bytes())
    if status.get('job_id')!=job['job_id'] or status.get('phase')!='complete' or status.get('exit_code')!=0:raise ValueError('genuine original completed terminal')
    if live_original(status):return dict(status='deferred',reason='original-process-still-live',removed=[])
    # Any live job on this owned namespace protects models, including a reader.
    if any(live_original(json.loads(p.read_bytes()))for p in markers.glob('*.json')):return dict(status='deferred',reason='owned-namespace-active-process',removed=[])
    cache=CacheLifecycle(root);directory=root/'checkpoints'/cp['id']
    if not directory.exists():return dict(status='complete',removed=[],already_absent=True)
    receipt=cache._receipt(cp['id'])
    if not receipt.is_file():raise ValueError('backend authenticated hydration receipt required')
    value=json.loads(receipt.read_bytes())
    if value.get('path',str(Path('checkpoints')/cp['id']))!=str(Path('checkpoints')/cp['id']) or value.get('files')!=cp['files'] or set(value.get('members',{}))!=set(cp['files']):raise ValueError('exact owned complete hydration inventory')
    # Existing eviction checks single-link owner/inode metadata and acquires
    # inherited checkpoint leases nonblocking. No unowned discovery/adoption.
    removed=cache.evict_checkpoints(keep=0,only=[cp['id']])
    return dict(status='complete'if removed else 'deferred',removed=removed,reason=None if removed else 'lease-or-inode-guard',job_id=job['job_id'],checkpoint=cp['id'])

def retire_completed(controller,jobs,config):
    """ACK verified reports durably before retiring exact disposable model bytes."""
    from subnet.backend_jobs import signed,canonical
    from subnet.remote_backend import save
    from botocore.exceptions import ClientError
    import inspect,shlex
    state=controller.state
    for path in (state/'checkpoint-evaluations').glob('*.json'):
        record=json.loads(path.read_bytes())
        if record.get('status')!='complete':continue
        label=record['request']['label'];prior,job,manifest=jobs.original(state/'roles'/(label+'.json'))
        remote=jobs.instance(config['source_sha256'],original=job)
        report=remote.checked(json.loads((state/'roles'/(job['job_id']+'-report.json')).read_bytes()),prior,manifest)
        if job.get('checkpoint_cache')is not None or remote.config.get('checkpoint_caches',{}).get(manifest['checkpoint']['id']):raise ValueError('external mapped caches cannot be retired')
        payload=dict(version=VERSION,workspace=remote.workspace,original_job=json.loads((state/'roles'/(job['job_id']+'-job.json')).read_bytes()),original_report=report,job_sha256=prior['job_sha256'],report_sha256=hashlib.sha256(canonical(report)).hexdigest(),checkpoint=manifest['checkpoint'],durable_report_full_readback=True)
        envelope=controller.signed(payload);raw=canonical(envelope);key='private/owned-cached-evaluator/'+prior['job_sha256']+'/durable-ack.json'
        disposition=state/'cache-disposal'/(job['job_id']+'.json')
        if disposition.exists():
            previous=json.loads(disposition.read_bytes())
            if previous.get('durable_ack_sha256')==hashlib.sha256(raw).hexdigest()and previous.get('result',{}).get('status')=='complete':continue
        try:existing=controller.bucket.get(key)
        except ClientError as error:
            if str(error.response.get('Error',{}).get('Code'))not in ('NoSuchKey','404','NotFound'):raise
            existing=None
        if existing is None:controller.bucket.put(key,raw,content_type='application/json')
        if controller.bucket.get(key)!=raw:raise ValueError('durable complete original report ACK readback')
        local=state/'durable-evaluation-acks'/(job['job_id']+'.json')
        if local.exists()and local.read_bytes()!=raw:raise ValueError('original durable ACK changed')
        if not local.exists():save(local,envelope)
        ackpath=remote.workspace+'/durable-evaluation-acks/'+prior['job_sha256']+'.json'
        prepare="import json,hashlib;from pathlib import Path;p=Path("+repr(ackpath)+");assert not p.parent.is_symlink()and not p.is_symlink();p.parent.mkdir(mode=0o700,exist_ok=True);print(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest()if p.exists()else None}))"
        observed=json.loads(remote.command(shlex.quote(remote.python)+' -I -B -c '+shlex.quote(prepare),timeout=30))
        rawsha=hashlib.sha256(raw).hexdigest()
        if observed['sha256']is None:remote.copy_to(local,ackpath)
        elif observed['sha256']!=rawsha:raise ValueError('remote original durable ACK changed')
        # Stage the potentially large original job/ACK as a file; never exceed
        # exec argument limits or put capability URLs in command arguments.
        # Authenticate all pinned runtime bytes before importing cleanup APIs.
        code="import sys,hashlib,json;from pathlib import Path;root=Path("+repr(remote.code)+");files="+repr(job['source_files'])+";assert {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in(root/'subnet').glob('*.py')}==files;sys.path.insert(0,str(root));ackpath=Path("+repr(ackpath)+");assert not ackpath.is_symlink();raw=ackpath.read_bytes();assert hashlib.sha256(raw).hexdigest()=="+repr(rawsha)+";envelope=json.loads(raw);VERSION="+repr(VERSION)+"\n"+inspect.getsource(retire)+"\nprint(json.dumps(retire(envelope,"+repr(controller.authority.id)+","+repr(remote.workspace)+",files)))"
        result=json.loads(remote.command(shlex.quote(remote.python)+' -I -B -c '+shlex.quote(code),timeout=60))
        save(disposition,dict(durable_ack_key=key,durable_ack_sha256=hashlib.sha256(raw).hexdigest(),result=result,observed_at=time.time()))
