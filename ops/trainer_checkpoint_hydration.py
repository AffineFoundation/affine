"""CPU-only bounded trainer checkpoint prefetch before capacity and GPU execution.

The scientific evaluator still authenticates the complete default checkpoint.
Pinned helpers resume their own retained partials, with no model or GPU work.
"""
import hashlib,json,shlex,time
from pathlib import Path

VERSION='owned-trainer-bounded-checkpoint-hydration-v1'

def materialize_evaluator_helper(source,destination):
    """Add only typed expiry to the measured transport; public helper stays exact."""
    source=Path(source);destination=Path(destination);raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!='80769c7d739fb8c1d3b7e31d8ffaeaf3c1f3a578650fe67a5914c3fe90ec5e55':
        raise ValueError('qualified original bounded hydration helper')
    text=raw.decode();anchor='MAX_FILE=20*1024**3;MAX_TOTAL=64*1024**3'
    text=text.replace(anchor,'class HydrationReadPlanExpired(ValueError):pass\n'+anchor,1)
    for message in ('read plan expired; preserve partial','read plan expired during transfer'):
        old="raise ValueError("+repr(message)+")"
        if text.count(old)!=1:raise ValueError('reviewed typed expiry shape')
        text=text.replace(old,"raise HydrationReadPlanExpired("+repr(message)+")",1)
    rendered=text.encode()
    if destination.exists() and destination.read_bytes()!=rendered:raise ValueError('immutable evaluator helper collision')
    if not destination.exists():destination.write_bytes(rendered);destination.chmod(0o600)
    return hashlib.sha256(rendered).hexdigest()

def transient_transfer(diagnostic,started_at):
    return (isinstance(diagnostic,dict)and isinstance(diagnostic.get('observed_at'),(int,float))
        and diagnostic['observed_at']>=started_at and diagnostic.get('error_type')in
        ('HydrationReadPlanExpired','ChunkedEncodingError','ConnectionError','ConnectTimeout','ReadTimeout','Timeout'))

def policy_admission(policy):
    if (not isinstance(policy,dict) or set(policy)!={'version','helper_path','helper_sha256','remote_directory','retained_UUID'}
            or policy['version']!=VERSION):raise ValueError('signed evaluator hydration policy')
    for name in ('helper_path','remote_directory'):
        p=Path(policy[name])
        if not p.is_absolute()or '..'in p.parts:raise ValueError('exact hydration helper paths')
    helper=Path(policy['helper_path'])
    if helper.is_symlink()or hashlib.sha256(helper.read_bytes()).hexdigest()!=policy['helper_sha256']:
        raise ValueError('pinned evaluator hydration helper bytes')
    if not isinstance(policy['retained_UUID'],str)or not policy['retained_UUID']:raise ValueError('exact hydration node')

def create_plan(controller,remote,checkpoint,policy,*,now=None):
    from subnet.backend_jobs import canonical,file_map
    policy_admission(policy);now=time.time()if now is None else now
    if file_map(checkpoint['files'])!=checkpoint['id']:raise ValueError('learned checkpoint full inventory')
    descriptor=controller.signed(dict(id=checkpoint['id'],files=checkpoint['files']))
    objects={};urls={};bucket=controller.bucket
    for name,digest in checkpoint['files'].items():
        key='public/checkpoints/'+checkpoint['id']+'/'+name
        head=bucket.client.head_object(Bucket=bucket.name,Key=key)
        objects[name]=dict(bytes=head['ContentLength'],sha256=digest)
        urls[name]=bucket.client.generate_presigned_url('get_object',Params={'Bucket':bucket.name,'Key':key},ExpiresIn=3600)
    return controller.signed(dict(kind='immutable-checkpoint-read-hydration-v1',role='train-input',
        retained_UUID=policy['retained_UUID'],helper_sha256=policy['helper_sha256'],GPU_runs=0,optimizer_runs=0,
        chain_transactions=0,publication_writes=0,created_at=now,expires_at=now+3500,
        checkpoint=checkpoint['id'],checkpoint_authority=controller.authority.id,checkpoint_descriptor=descriptor,
        checkpoint_descriptor_sha256=hashlib.sha256(canonical(descriptor)).hexdigest(),objects=objects,read_urls=urls,
        destination=remote.workspace+'/checkpoints/'+checkpoint['id'],allows_concurrent_scientific_reads=True))

def plan_identity(plan):
    # Only read capabilities and their validity interval may change on renewal.
    return {k:v for k,v in plan.items() if k not in ('created_at','expires_at','read_urls')}

def admit_plan(envelope,controller,remote,cp,policy):
    from subnet.backend_jobs import signed,canonical,file_map
    plan=signed(envelope,controller.authority.id)
    descriptor=signed(plan['checkpoint_descriptor'],controller.authority.id)
    if (file_map(cp['files'])!=cp['id'] or descriptor!=dict(id=cp['id'],files=cp['files'])
            or plan['checkpoint']!=cp['id'] or plan['destination']!=remote.workspace+'/checkpoints/'+cp['id']
            or plan['helper_sha256']!=policy['helper_sha256'] or plan['retained_UUID']!=policy['retained_UUID']
            or plan['role']!='train-input' or plan['checkpoint_authority']!=controller.authority.id
            or plan['checkpoint_descriptor_sha256']!=hashlib.sha256(canonical(plan['checkpoint_descriptor'])).hexdigest()
            or set(plan['objects'])!=set(cp['files']) or set(plan['read_urls'])!=set(cp['files'])
            or any(m['sha256']!=cp['files'][n] or type(m['bytes'])is not int or m['bytes']<=0 for n,m in plan['objects'].items())):
        raise ValueError('original evaluator hydration plan changed')
    return plan

def admit_receipt(result,digest,plan,cp):
    if (not isinstance(result,dict) or result.get('read_plan_sha256')!=digest or result.get('actual_files')!=cp['files']
            or result.get('objects')!=plan['objects'] or result.get('actual_local_destination')!=plan['destination']
            or result.get('checkpoint')!=cp['id'] or result.get('role')!='train-input'
            or result.get('retained_UUID')!=plan['retained_UUID'] or result.get('CPU_only')is not True
            or result.get('GPU_runs')!=0):raise ValueError('complete original hydration receipt')
    return result

def adopt_partials(helper,envelopes,authority):
    """Move owned partial bytes under the same kernel lock, retaining all plans.

    The original and renewed ROOT plans must differ only in read capabilities
    and deadlines. Original binding/progress/receipt records are never edited.
    Interrupted adoption is retryable: the object-directory rename is atomic.
    """
    from pathlib import Path
    import hashlib,json
    plans=[]
    for envelope in envelopes:
        raw=helper.canonical(envelope);digest=hashlib.sha256(raw).hexdigest()
        value=helper.authenticate(envelope,authority)
        # Validate expired originals at their issue time; never use expired URLs.
        helper.validate(envelope,authority,'train-input',value['retained_UUID'],now=value['created_at'])
        plans.append((digest,value))
    current,job=plans[-1]
    invariant=lambda p:{k:v for k,v in p.items()if k not in ('created_at','expires_at','read_urls')}
    if any(invariant(value)!=invariant(job)for _,value in plans):raise ValueError('read renewal identity changed')
    dest=Path(job['destination']);cp=job['checkpoint']
    if dest.is_symlink()or any(p.is_symlink()for p in dest.parents):raise ValueError('exact renewal destination')
    lock=helper.acquire_lock(dest.parent,cp)
    try:
        if dest.exists():return {'adopted':False,'destination_exists':True}
        def location(digest,value):
            stage=dest.parent/('.'+cp+'.hydrate-'+digest[:16])
            body=dict(plan_sha256=digest,checkpoint=cp,role='train-input',retained_UUID=value['retained_UUID'],descriptor_sha256=value['checkpoint_descriptor_sha256'])
            return stage,body
        target,binding=location(current,job)
        if target.exists():
            if target.is_symlink()or not (target/'binding.json').is_file()or json.loads((target/'binding.json').read_text())!=binding:raise ValueError('renewed staging identity collision')
            if (target/'objects').exists():return {'adopted':False,'current_objects_exist':True}
        for digest,value in reversed(plans[:-1]):
            stage,oldbinding=location(digest,value);work=stage/'objects'
            if not stage.exists():continue
            if stage.is_symlink()or not (stage/'binding.json').is_file()or json.loads((stage/'binding.json').read_text())!=oldbinding:raise ValueError('original staging identity collision')
            if not work.exists():continue
            if work.is_symlink()or not work.is_dir():raise ValueError('regular original staged objects')
            for member in work.iterdir():
                name=member.name[:-8]if member.name.endswith('.partial')else member.name
                if member.is_symlink()or not member.is_file()or name not in job['objects'] or member.stat().st_size>job['objects'][name]['bytes']:raise ValueError('original staged object bounds')
            if not target.exists():target.mkdir(mode=0o700);helper.save(target/'binding.json',binding)
            work.rename(target/'objects')
            return {'adopted':True,'original_plan_sha256':digest,'renewed_plan_sha256':current}
        return {'adopted':False,'prior_objects_absent':True}
    finally:lock.close()

def prefetch(controller,remote,manifest,policy):
    from subnet.backend_jobs import canonical
    from subnet.remote_backend import save
    import inspect
    policy_admission(policy);cp=manifest['checkpoint']
    namespace=hashlib.sha256(remote.workspace.encode()).hexdigest()[:16]
    folder=controller.state/'checkpoint-hydration'/namespace;folder.mkdir(parents=True,exist_ok=True)
    revisions=folder/(cp['id']+'-plans');revisions.mkdir(exist_ok=True)
    paths=list(revisions.glob('*.json'))
    legacy=folder/(cp['id']+'-plan.json')
    if legacy.exists():paths.append(legacy)
    records={}
    for path in paths:
        envelope=json.loads(path.read_bytes());plan=admit_plan(envelope,controller,remote,cp,policy)
        digest=hashlib.sha256(canonical(envelope)).hexdigest()
        records[digest]=(path,envelope,plan)
    ordered=sorted(records.items(),key=lambda item:(item[1][2]['created_at'],item[0]))
    if ordered and any(plan_identity(row[2])!=plan_identity(ordered[0][1][2])for _,row in ordered):raise ValueError('read renewal identity changed')
    def location(digest):return policy['remote_directory']+'/'+namespace+'/'+cp['id']+'/'+digest
    def command(script,timeout=30):return remote.command(shlex.quote(remote.python)+' -I -B -c '+shlex.quote(script),timeout=timeout)
    # A completed receipt remains useful after its URL expires. Confirm that the
    # immutable cache still exists; scientific evaluation independently full-hashes.
    for digest,(path,envelope,plan) in reversed(ordered):
        receipt=location(digest)+'/receipt.json'
        localreceipt=folder/(cp['id']+'-'+digest+'-receipt.json')
        check="import json;from pathlib import Path;p=Path("+repr(receipt)+");assert not p.is_symlink();r=json.loads(p.read_text())if p.exists()else None;d=Path("+repr(plan['destination'])+");o="+repr(plan['objects'])+";ok=d.is_dir()and not d.is_symlink()and all((d/n).is_file()and not(d/n).is_symlink()and(d/n).stat().st_size==m['bytes']for n,m in o.items());print(json.dumps(dict(receipt=r,cache_present=ok)))"
        observed=json.loads(command(check))
        if observed['receipt'] is not None:
            result=admit_receipt(observed['receipt'],digest,plan,cp)
            if observed['cache_present']:save(localreceipt,result);return result
    # No usable receipt: renew only capabilities, keeping exact authenticated
    # inventory/actor/workspace/helper and every previous plan immutable.
    if not ordered or time.time()>=ordered[-1][1][2]['expires_at'] or observed.get('receipt')is not None:
        envelope=create_plan(controller,remote,cp,policy)
        plan=admit_plan(envelope,controller,remote,cp,policy)
        if ordered and plan_identity(plan)!=plan_identity(ordered[-1][1][2]):raise ValueError('read renewal identity changed')
        digest=hashlib.sha256(canonical(envelope)).hexdigest();path=revisions/(digest+'.json')
        if not path.exists():save(path,envelope);path.chmod(0o600)
        ordered.append((digest,(path,envelope,plan)))
    digest,(path,envelope,plan)=ordered[-1]
    directory=location(digest);helper=directory+'/helper.py';planpath=directory+'/plan.json';receipt=directory+'/receipt.json'
    inventory={helper:policy['helper_sha256'],planpath:digest}
    script="from pathlib import Path;import json,hashlib;p=Path("+repr(directory)+");assert p.absolute()==p.resolve();p.mkdir(parents=True,exist_ok=True,mode=0o700);files="+repr(inventory)+";out={}\nfor name,digest in files.items():\n p=Path(name);assert not p.is_symlink();out[name]=hashlib.sha256(p.read_bytes()).hexdigest()if p.exists()else None\nprint(json.dumps(out))"
    observed=json.loads(command(script))
    for source,target in [(policy['helper_path'],helper),(path,planpath)]:
        if observed[target]is None:remote.copy_to(source,target)
        elif observed[target]!=inventory[target]:raise ValueError('original remote hydration helper/plan changed')
    if len(ordered)>1:
        code=inspect.getsource(adopt_partials)
        renewal_envelopes=[row[1]for _,row in ordered]
        renewal_digest=hashlib.sha256(canonical(renewal_envelopes)).hexdigest()
        renewal_local=folder/(cp['id']+'-'+renewal_digest+'-renewal-plans.json')
        save(renewal_local,renewal_envelopes)
        renewal_remote=directory+'/renewal-plans-'+renewal_digest+'.json'
        remote.copy_to(renewal_local,renewal_remote)
        script="import importlib.util,hashlib,json;from pathlib import Path;p="+repr(helper)+";assert hashlib.sha256(Path(p).read_bytes()).hexdigest()=="+repr(policy['helper_sha256'])+";s=importlib.util.spec_from_file_location('hydration_helper',p);h=importlib.util.module_from_spec(s);s.loader.exec_module(h);raw=Path("+repr(renewal_remote)+").read_bytes();assert hashlib.sha256(raw).hexdigest()=="+repr(renewal_digest)+";envelopes=json.loads(raw)\n"+code+"\nprint(json.dumps(adopt_partials(h,envelopes,"+repr(controller.authority.id)+")))"
        command(script)
    args=[remote.python,'-B',helper,'--plan',planpath,'--plan-sha256',digest,'--operator',controller.authority.id,
        '--role','train-input','--retained-UUID',policy['retained_UUID'],'--destination',plan['destination'],'--output',receipt]
    started_at=time.time()
    try:remote.command(' '.join(shlex.quote(str(a))for a in args),timeout=3600)
    except Exception:
        stage=Path(plan['destination']).parent/('.'+cp['id']+'.hydrate-'+digest[:16])
        failure=stage/'last-failure.json'
        check="import json;from pathlib import Path;p=Path("+repr(str(failure))+");assert not p.is_symlink();print(p.read_text()if p.exists()else'null')"
        try:diagnostic=json.loads(command(check))
        except Exception:diagnostic=None
        if transient_transfer(diagnostic,started_at)or(diagnostic is None and time.time()>=plan['expires_at']):
            from subnet.checkpoint_evaluator import EvaluationHydrationDeferred
            reason=diagnostic['error_type']if diagnostic else'expired-before-dispatch'
            save(folder/(cp['id']+'-'+digest+'-transfer-deferral.json'),dict(reason=reason,observed_at=time.time(),read_plan_sha256=digest,remote_job_started=False))
            raise EvaluationHydrationDeferred(reason,digest)from None
        raise

    check="import json;from pathlib import Path;p=Path("+repr(receipt)+");assert not p.is_symlink();print(p.read_text())"
    result=admit_receipt(json.loads(command(check)),digest,plan,cp)
    save(folder/(cp['id']+'-'+digest+'-receipt.json'),result)
    return result


def ensure_checkpoint(router,manifest,policy):
    """Repair a missing working copy, never a malformed or unauthenticated one."""
    from subnet.remote_backend import save
    checkpoint=manifest['checkpoint']['id'];trainer=router.roles['train']
    cache=router.caches.get('train',{}).get(checkpoint)
    if cache is not None:
        target=Path(cache)
        workspace=Path(trainer.workspace)
        if not target.is_absolute()or '..'in target.parts or not target.is_relative_to(workspace):
            raise ValueError('owned trainer checkpoint cache path')
        script="import json,stat;from pathlib import Path;p=Path("+repr(cache)+");present=p.exists()or p.is_symlink();print(json.dumps(dict(present=present,regular=(not p.is_symlink()and p.is_dir()and all(x.is_file()and not x.is_symlink()for x in p.iterdir()))if present else False)))"
        observation=json.loads(trainer.command(shlex.quote(trainer.python)+' -I -B -c '+shlex.quote(script),timeout=30))
        if observation['present']:
            if observation['regular'] is not True:raise ValueError('malformed present trainer checkpoint cache')
            return dict(status='working-copy-present',checkpoint=checkpoint,path=cache)
    receipt=prefetch(router.controller,trainer,manifest,policy)
    target=trainer.workspace+'/checkpoints/'+checkpoint
    if receipt['actual_local_destination']!=target or receipt['actual_files']!=manifest['checkpoint']['files']:
        raise ValueError('full authenticated trainer hydration before capacity')
    with router.cache_lock:
        router.caches.setdefault('train',{})[checkpoint]=target
        router.owners[target]='train'
        save(router.cache_path,router.caches);save(router.owner_path,router.owners)
    return dict(status='full-SHA-hydrated',checkpoint=checkpoint,path=target)


def install(role_router,policy):
    policy_admission(policy)
    original=role_router.RoutedJobs.training_capacity
    def capacity(self,manifest,steps,submission_bytes=None):
        # Prepared-job reset has already completed; only then hydrate its input.
        ensure_checkpoint(self,manifest,policy)
        return original(self,manifest,steps,submission_bytes=submission_bytes)
    role_router.RoutedJobs.training_capacity=capacity
