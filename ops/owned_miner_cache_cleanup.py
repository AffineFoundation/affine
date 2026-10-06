"""Post-durable-ACK disposal of exact owned miner checkpoints."""
import hashlib,json,time
from pathlib import Path

VERSION='owned-miner-durable-terminal-ack-v1'

def retire(ack_envelope,authority,workspace,expected_source_files,approved_source):
    from subnet.backend_jobs import signed,canonical
    from subnet.cache_lifecycle import CacheLifecycle
    from subnet.evaluator_cache_lifecycle import live_original
    ack=signed(ack_envelope,authority);job=signed(ack['original_job'],authority);manifest=signed(job['manifest'],authority);report=ack['original_report'];cp=manifest['checkpoint']
    if ack.get('version')!=VERSION or ack.get('durable_report_full_readback')is not True:raise ValueError('actual durable miner ACK')
    if ack.get('workspace')!=str(Path(workspace).absolute()) or Path(workspace).absolute()!=Path(workspace).resolve():raise ValueError('owned diagnostic namespace')
    if (job['role']!='mine' or job['source_files']!=expected_source_files or manifest['source_bundle']['sha256']!=approved_source or job.get('miner_id')!=ack.get('miner_id') or report.get('miner_id')!=job.get('miner_id')):raise ValueError('exact approved owned miner source/job')
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

def retire_completed(controller,remote,sources,*,output,owned_miners,max_jobs=8):
    """Archive genuine original mine reports, then retire only owned models.

    Scientific jobs, reports, upload commitments and source files remain intact.
    Missing/failed reports and active namespaces defer without creating credit.
    """
    from subnet.backend_jobs import signed,canonical
    from subnet.remote_backend import save
    from botocore.exceptions import ClientError
    import inspect,shlex
    output=Path(output);output.mkdir(parents=True,exist_ok=True,mode=0o700)
    results=[]
    for path in sorted((controller.state/'roles').glob('*-job.json'),reverse=True):
        envelope=json.loads(path.read_bytes());job=signed(envelope,controller.authority.id)
        if job.get('role')!='mine' or job.get('miner_id')not in owned_miners:continue
        manifest=signed(job['manifest'],controller.authority.id);source=manifest['source_bundle']['sha256']
        if source not in sources or job['source_files']!=sources[source]:continue
        jobid=job['job_id'];target=output/(jobid+'.json')
        if target.exists()and json.loads(target.read_bytes()).get('result',{}).get('status')=='complete':continue
        if len(results)>=max_jobs:break
        # Observation only: never reserve, launch, retry or adopt a compute job.
        status=remote.remote_status(jobid,timeout=20,physical=True)
        if status.get('phase')!='complete' or status.get('exit_code')!=0:
            result=dict(status='deferred',reason='original-mine-not-successfully-terminal')
            save(target,dict(job_id=jobid,result=result,observed_at=time.time()));results.append(result);continue
        reportpath=output/(jobid+'-original-report.json')
        if not reportpath.exists():remote.copy_from(remote.workspace+'/jobs/'+jobid+'/report.json',reportpath)
        prior=dict(job_id=jobid,role='mine',epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],job_sha256=hashlib.sha256(canonical(job)).hexdigest(),manifest_sha256=hashlib.sha256(canonical(manifest)).hexdigest(),source_files=job['source_files'],runtime_versions=job['runtime_versions'])
        report=remote.checked(json.loads(reportpath.read_bytes()),prior,manifest)
        if report.get('success')is not True or report.get('miner_id')!=job['miner_id']:raise ValueError('original successful owned miner report')
        payload=dict(version=VERSION,workspace=remote.workspace,original_job=envelope,original_report=report,job_sha256=prior['job_sha256'],report_sha256=hashlib.sha256(canonical(report)).hexdigest(),checkpoint=manifest['checkpoint'],miner_id=job['miner_id'],durable_report_full_readback=True)
        ack=controller.signed(payload);raw=canonical(ack);key='private/owned-miner-terminal/'+prior['job_sha256']+'/durable-ack.json'
        try:existing=controller.bucket.get(key)
        except ClientError as error:
            if str(error.response.get('Error',{}).get('Code'))not in ('NoSuchKey','404','NotFound'):raise
            existing=None
        if existing is None:controller.bucket.put(key,raw,content_type='application/json')
        if controller.bucket.get(key)!=raw:raise ValueError('durable complete original miner ACK full readback')
        local=output/(jobid+'-durable-ack.json')
        if local.exists()and local.read_bytes()!=raw:raise ValueError('original miner ACK changed')
        if not local.exists():save(local,ack)
        ackpath=remote.workspace+'/durable-miner-acks/'+prior['job_sha256']+'.json'
        prepare="import json,hashlib;from pathlib import Path;p=Path("+repr(ackpath)+");assert not p.parent.is_symlink()and not p.is_symlink();p.parent.mkdir(mode=0o700,exist_ok=True);print(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest()if p.exists()else None}))"
        observed=json.loads(remote.command(shlex.quote(remote.python)+' -I -B -c '+shlex.quote(prepare),timeout=30));rawsha=hashlib.sha256(raw).hexdigest()
        if observed['sha256']is None:remote.copy_to(local,ackpath)
        elif observed['sha256']!=rawsha:raise ValueError('remote original miner ACK changed')
        code="import sys,hashlib,json;from pathlib import Path;root=Path("+repr(remote.code)+");pins="+repr(remote.metadata['source_files'])+";assert {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in(root/'subnet').glob('*.py')}==pins;sys.path.insert(0,str(root));ackpath=Path("+repr(ackpath)+");assert not ackpath.is_symlink();raw=ackpath.read_bytes();assert hashlib.sha256(raw).hexdigest()=="+repr(rawsha)+";envelope=json.loads(raw);VERSION="+repr(VERSION)+"\n"+inspect.getsource(retire)+"\nprint(json.dumps(retire(envelope,"+repr(controller.authority.id)+","+repr(remote.workspace)+","+repr(job['source_files'])+","+repr(source)+")))"
        result=json.loads(remote.command(shlex.quote(remote.python)+' -I -B -c '+shlex.quote(code),timeout=60))
        save(target,dict(job_id=jobid,durable_ack_key=key,durable_ack_sha256=rawsha,result=result,observed_at=time.time()));results.append(result)
    return results
