"""Operator SSH dispatch of signed GPU roles; all checkpoints stay remote/R2."""
import base64
import hashlib
import json
import math
import secrets
import shlex
import subprocess
import time
from pathlib import Path
import requests
from nacl.signing import VerifyKey
from .backend_jobs import canonical,file_map,FIXED_POLICY as FULL_POLICY,SOURCE_FILES,signed
from .controller import Controller
from .scoring import score


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_bytes(canonical(value));tmp.chmod(0o600);tmp.replace(path)

def training_submission_bytes(receipts,reports,manifest):
    """Count every frozen ZIP that this training request actually downloads."""
    from .artifact_budget import for_manifest
    limit=for_manifest(manifest)['compressed_bytes'];total=0;count=0
    for miner,report in reports.items():
        if not report['accepted']:continue
        receipt=receipts[miner];size=receipt.get('size')
        if (type(size) is not int or not 0<size<=limit or
                report.get('submission_sha256')!=receipt['sha256']):
            raise ValueError('training frozen receipt size/hash binding')
        count+=1;total+=size
    if not 1<=count<=256:raise ValueError('planned training submission count')
    return total

def role_time_budget(config,role):
    budgets=config.get('job_ttl_seconds_by_role',{})
    roles={'mine','verify','train','evaluate','upload'}
    if not isinstance(budgets,dict) or set(budgets)-roles:
        raise ValueError('remote role time budget configuration')
    if any(type(seconds) is not int or not 60<=seconds<=86400 for seconds in budgets.values()):
        raise ValueError('remote role time budget bounds')
    if role not in roles:raise ValueError('remote role')
    return budgets.get(role,3600)

class RemoteJobs:
    def __init__(self,config,controller):
        role_time_budget(config,'evaluate')
        self.config=config;self.controller=controller;self.state=controller.state/'roles';self.state.mkdir(exist_ok=True)
        self.peer=config.get('user','root')+'@'+config['host']
        shared=['-o','BatchMode=yes','-o','UserKnownHostsFile='+config['known_hosts']]
        self.ssh=['ssh',*shared,'-p',str(config['port']),self.peer]
        self.scp=['scp','-q',*shared,'-P',str(config['port'])]
        self.code=config['code'];self.workspace=config['workspace'];self.python=config.get('python','/root/miner-venv/bin/python')
        self.metadata=json.loads(self.command('cd '+shlex.quote(self.code)+' && '+shlex.quote(self.python)+' -B -c '+shlex.quote("import json,hashlib;from pathlib import Path;from importlib.metadata import version;print(json.dumps(dict(source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('subnet').glob('*.py')},runtime_versions={n:version(n) for n in ['torch','transformers','toploc']})))")))
        if not set(SOURCE_FILES)<=set(self.metadata['source_files']):raise ValueError('remote source inventory incomplete')
    def command(self,command,timeout=1800):
        return subprocess.check_output(self.ssh+[command],text=True,timeout=timeout)
    def copy_to(self,local,remote):
        subprocess.run(self.scp+[str(local),self.peer+':'+remote],check=True,timeout=600)
    def copy_from(self,remote,local):
        subprocess.run(self.scp+[self.peer+':'+remote,str(local)],check=True,timeout=600)
    def capacity(self,cache):
        code="import json,os;from pathlib import Path;p=Path("+repr(cache)+");s=os.statvfs(p);print(json.dumps(dict(free_bytes=s.f_bavail*s.f_frsize,checkpoint_bytes=sum(q.stat().st_size for q in p.iterdir() if q.is_file()))))"
        result=json.loads(self.command(shlex.quote(self.python)+' -c '+shlex.quote(code)))
        required=2*result['checkpoint_bytes']+2*1024**3
        if result['free_bytes']<required:raise ValueError('remote checkpoint history disk reserve')
        return dict(result,required_bytes=required)
    def publication_capacity(self,cache):
        # Upload streams an existing export; it does not allocate another model
        # or a new checkpoint. Keep the training reserve for new training only.
        code="import json,os;from pathlib import Path;p=Path("+repr(cache)+");s=os.statvfs(p);print(json.dumps(dict(free_bytes=s.f_bavail*s.f_frsize,checkpoint_bytes=sum(q.stat().st_size for q in p.iterdir() if q.is_file()))))"
        result=json.loads(self.command(shlex.quote(self.python)+' -I -B -c '+shlex.quote(code)))
        if result['checkpoint_bytes']<=0 or result['free_bytes']<1024**2:raise ValueError('existing checkpoint upload metadata reserve')
        return dict(result,required_bytes=1024**2,purpose='existing-checkpoint-streaming-upload')
    def training_resume(self,label,manifest,submissions,steps,replay):
        record=self.state/(label+'.json')
        if not record.exists():return None
        prior=json.loads(record.read_text())
        job=signed(json.loads((self.state/(prior['job_id']+'-job.json')).read_text()),self.controller.authority.id)
        if (job.get('role')!='train' or prior.get('role')!='train' or job.get('job_id')!=prior['job_id']
                or hashlib.sha256(canonical(job)).hexdigest()!=prior['job_sha256']
                or signed(job['manifest'],self.controller.authority.id)!=manifest
                or prior['manifest_sha256']!=hashlib.sha256(canonical(manifest)).hexdigest()
                or job.get('steps')!=steps or type(job.get('steps'))is not int or type(steps)is not int or not 1<=steps<=32
                or job.get('training_policy')!=FULL_POLICY or job.get('replay')!=replay
                or [r['sha256'] for r in job.get('submissions',[])]!=[r['sha256'] for r in submissions]):
            raise ValueError('original training request changed')
        reportpath=self.state/(prior['job_id']+'-report.json')
        if reportpath.exists():
            self.checked(json.loads(reportpath.read_text()),prior,manifest);phase='complete'
        else:
            phase=self.remote_status(prior['job_id'])['phase']
            if phase not in ('running','complete'):raise ValueError('original training not resumable; inspect without relaunch')
        return dict(resuming_original_training=True,original_job_id=prior['job_id'],original_phase=phase,
                    new_training_started=False,new_disk_reserve_not_required=True)
    def remote_status(self,identifier):
        code="from subnet.remote_runner import probe;import json;print(json.dumps(probe("+repr(self.workspace)+","+repr(identifier)+")))"
        return json.loads(self.command('cd '+shlex.quote(self.code)+' && '+shlex.quote(self.python)+' -B -c '+shlex.quote(code)))
    def run(self,label,role,manifest,cache=None,**fields):
        if not label.replace('-','').replace('_','').isalnum():raise ValueError('job label')
        record=self.state/(label+'.json');prior=None
        if record.exists():
            prior=json.loads(record.read_text());reportpath=self.state/(prior['job_id']+'-report.json')
            if reportpath.exists():return self.checked(json.loads(reportpath.read_text()),prior,manifest)
            status=self.remote_status(prior['job_id'])
            if status['phase']=='complete':
                self.copy_from(self.workspace+'/jobs/'+prior['job_id']+'/report.json',reportpath)
                return self.checked(json.loads(reportpath.read_text()),prior,manifest)
            if status['phase'] in ('failed','not_launched'):
                if role=='train':raise RuntimeError('original training terminal or absent; refuse automatic relaunch')
                save(self.state/(prior['job_id']+'-failure.json'),status);prior=None
            elif status['phase']!='running':raise ValueError('unknown authoritative remote job status')
        if prior is None:
            identifier=label+'-'+secrets.token_hex(4);now=time.time()
            payload=dict(schema=1,job_id=identifier,role=role,created_at=now,expires_at=now+role_time_budget(self.config,role),manifest=self.controller.signed(manifest),**self.metadata,**fields)
            jobpath=self.state/(identifier+'-job.json');save(jobpath,self.controller.signed(payload))
            prior=dict(job_id=identifier,role=role,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],job_sha256=hashlib.sha256(canonical(payload)).hexdigest(),manifest_sha256=hashlib.sha256(canonical(manifest)).hexdigest(),source_files=self.metadata['source_files'],runtime_versions=self.metadata['runtime_versions'])
            save(record,prior);remotejob=self.workspace+'/'+identifier+'.json'
            self.command('mkdir -p '+shlex.quote(self.workspace));self.copy_to(jobpath,remotejob)
            command='cd '+shlex.quote(self.code)+' && CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup '+shlex.quote(self.python)+' -B -m subnet.remote_runner '+shlex.quote(remotejob)+' --authority '+self.controller.authority.id+' --workspace '+shlex.quote(self.workspace)
            if cache:command+=' --checkpoint-cache '+shlex.quote(cache)
            self.command(command+' > '+shlex.quote(self.workspace+'/'+identifier+'-runner.log')+' 2>&1 < /dev/null &')
        reportpath=self.state/(prior['job_id']+'-report.json')
        started=time.time()
        while True:
            status=self.remote_status(prior['job_id'])
            if status['phase']=='complete':
                self.copy_from(self.workspace+'/jobs/'+prior['job_id']+'/report.json',reportpath)
                return self.checked(json.loads(reportpath.read_text()),prior,manifest)
            if status['phase']=='failed':
                save(self.state/(prior['job_id']+'-failure.json'),status);raise RuntimeError('remote role exited: '+str(status.get('exit_code',status.get('reason'))))
            if status['phase']=='not_launched' and time.time()-started>30:raise RuntimeError('remote launcher marker absent; retain job record for authoritative recovery')
            if time.time()-started>1800:raise TimeoutError('same remote role remains active; retain job identity')
            time.sleep(5)
    def checked(self,report,prior,manifest):
        from .backend_profiles import resolve
        _,profile,policy=resolve(manifest)
        if report.get('job_id')!=prior['job_id'] or report.get('operator')!=self.controller.authority.id or report.get('job_sha256')!=prior['job_sha256'] or report.get('checkpoint')!=manifest['checkpoint']['id'] or report.get('epoch')!=manifest['epoch'] or report.get('role')!=prior['role'] or report.get('success') is not True or report.get('chain_transactions') is not False or canonical(report.get('backend_profile'))!=canonical(profile) or canonical(report.get('numerical_policy'))!=canonical(policy) or report.get('source_files')!=prior['source_files'] or report.get('runtime_versions')!=prior['runtime_versions'] or hashlib.sha256(canonical(manifest)).hexdigest()!=prior['manifest_sha256']:raise ValueError('remote role report binding')
        # Older dispatch records omit timestamps; reconstruct them from their
        # original signed job, never from mutable local configuration or a new
        # signature. A live observation timeout does not affect this check.
        job=signed(json.loads((self.state/(prior['job_id']+'-job.json')).read_text()),self.controller.authority.id)
        if hashlib.sha256(canonical(job)).hexdigest()!=prior['job_sha256']:
            raise ValueError('remote role original signed job binding')
        created,expires,completed=job.get('created_at'),job.get('expires_at'),report.get('completed_at')
        if any(type(value) not in (int,float) or not math.isfinite(value) for value in (created,expires,completed)) or not created<=completed<expires or not 0<expires-created<=86400:
            raise ValueError('remote role report outside signed time budget')
        return report

class RemoteController(Controller):
    def __init__(self,bucket,gateway,state,remote):
        super().__init__(bucket,gateway,state)
        if 'roles' in remote:
            from .role_router import RoutedJobs
            self.jobs=RoutedJobs(remote,self)
        else:self.jobs=RemoteJobs(remote,self)
    def open(self,*args,max_batches=3,**kwargs):
        if type(max_batches)is not int or not 1<=max_batches<=256:raise ValueError('per UID batch quota')
        heldouts=kwargs.pop('heldout_indices',None)
        operator_test_policy=kwargs.pop('operator_test_policy',None)
        if operator_test_policy is not None:
            from .empty_epoch_policy import validate
            validate(operator_test_policy)
            if not args or not args[0].startswith('nonpayable-'):raise ValueError('nonpayable controlled window')
        if heldouts is not None:
            from .verified_replay_pool import definitions,heldout_registry
            raw={'heldout_indices':heldouts,'environments':[dict(env_id=r['spec']['id'],spec=r['spec'],indices=r.get('indices',[]),harness=r.get('harness')) for r in kwargs.get('environments',[])]}
            heldout_registry(raw,definitions(raw))
        # Delay base-controller publication so the approved quota is included in
        # the first public manifest rather than overwriting a signed challenge.
        original=self.bucket
        class Buffered:
            def __init__(self):self.writes=[]
            def __getattr__(self,name):return getattr(original,name)
            def json(self,key,value):self.writes.append((key,value))
        buffered=Buffered();self.bucket=buffered
        try:manifest=super().open(*args,**kwargs)
        finally:self.bucket=original
        manifest['max_batches']=max_batches
        if operator_test_policy is not None:manifest['operator_test_policy']=operator_test_policy
        if heldouts is not None:manifest['heldout_indices']=heldouts
        save(self.state/(manifest['epoch']+'-manifest.json'),manifest)
        live_registrations=kwargs.get('live_reward_registration_snapshot')
        for key,value in buffered.writes:
            if key=='public/'+manifest['epoch']+'/manifest.json':value=self.signed(manifest)
            original.json(key,value)
        if manifest.get('live_reward_contract') is not None:
            from .live_reward_bridge import emit_opening_documents
            emit_opening_documents(self,manifest,live_registrations)
        return manifest
    def checkpoint_with_reads(self,checkpoint):
        files=checkpoint['files']
        if checkpoint['id']!=file_map(files):raise ValueError('checkpoint read capability file-map binding')
        return dict(checkpoint,read_urls={name:self.bucket.presign('public/checkpoints/'+checkpoint['id']+'/'+name) for name in files})

    def publish_remote_checkpoint(self,manifest,remote_path):
        cp=manifest['checkpoint'];capacity=(self.jobs.publication_capacity(remote_path) if hasattr(self.jobs,'publication_capacity') else self.jobs.capacity(remote_path))
        report=self.jobs.run(manifest['epoch']+'-publish-'+cp['id'][:8],'upload',manifest,remote_path,
            put_urls={n:self.bucket.presign('public/checkpoints/'+cp['id']+'/'+n,'put_object',3600) for n in cp['files']})
        observed={}
        for name,expected in cp['files'].items():
            h=hashlib.sha256();size=0
            with requests.get(self.bucket.presign('public/checkpoints/'+cp['id']+'/'+name),stream=True,timeout=180,allow_redirects=False) as response:
                if response.status_code!=200:raise ValueError('operator R2 checkpoint read status')
                for part in response.iter_content(1024*1024):h.update(part);size+=len(part)
            if h.hexdigest()!=expected:raise ValueError('operator independent checkpoint integrity')
            observed[name]=dict(sha256=h.hexdigest(),bytes=size)
        descriptor=dict(id=cp['id'],files=cp['files']);key='public/checkpoints/'+cp['id']+'/authorities/'+self.authority.id+'/checkpoint.json'
        from botocore.exceptions import ClientError
        try:existing=json.loads(self.bucket.get(key))
        except ClientError as error:
            if str(error.response.get('Error',{}).get('Code')) not in ('NoSuchKey','404','NotFound'):raise
            self.bucket.json(key,self.signed(descriptor))
        else:
            if existing['signer']!=self.authority.id:raise ValueError('checkpoint descriptor signer')
            VerifyKey(bytes.fromhex(self.authority.id)).verify(canonical(existing['payload']),base64.b64decode(existing['signature'],validate=True))
            if existing['payload']!=descriptor:raise ValueError('immutable checkpoint descriptor collision')
        save(self.state/(manifest['epoch']+'-checkpoint-publication.json'),dict(checkpoint=cp['id'],objects=observed,capacity=capacity,operator_independent_hashes=True))
        return self.checkpoint_with_reads(dict(descriptor,descriptor_key=key))
    def finalize(self,manifest,checkpoint_path):
        if manifest.get('payable') is not False:raise ValueError('remote experimental controller is nonpayable only')
        epoch=manifest['epoch'];saved=self.state/(epoch+'-scores.json')
        if saved.exists():
            result=json.loads(saved.read_text())
            if result.get('checkpoint')!=manifest['checkpoint']['id'] or result.get('payable') is not False or result['receipts']!=self.gateway.freeze(epoch):raise ValueError('cached finalized manifest/receipt binding')
            reports={m:json.loads((self.state/(epoch+'-'+m+'-report.json')).read_text()) for m in result['receipts']}
            if any(r.get('epoch')!=epoch or r.get('submission_sha256')!=result['receipts'][m]['sha256'] for m,r in reports.items()):raise ValueError('cached audit report binding')
            self.bucket.json('public/'+epoch+'/scores.json',self.signed(result))
            if manifest.get('live_reward_contract') is not None:
                from .live_reward_bridge import persist_signed_compute_evidence
                persist_signed_compute_evidence(self,manifest,result,reports)
            return result,reports
        receipts=self.gateway.freeze(epoch);challengepath=self.state/(epoch+'-audit-challenge.json')
        if challengepath.exists():challenge=json.loads(challengepath.read_text())
        else:
            challenge=dict(seed=secrets.token_hex(32),generated_after_freeze_at=time.time(),receipts=receipts);save(challengepath,challenge)
        if challenge['receipts']!=receipts:raise ValueError('frozen audit challenge binding')
        self.bucket.json('public/'+epoch+'/audit-challenge.json',self.signed(challenge));audit_manifest=dict(manifest,audit_seed=challenge['seed'],audit_frozen_receipts=receipts)
        bounded=manifest.get('audit_policy',{}).get('version')=='bounded-random-v1'
        if bounded:
            from .audit_policy import validate,allocate
            policy=validate(manifest['audit_policy'])
            allocations=allocate({m:manifest['max_batches'] for m in receipts},policy,challenge['seed'])
            from collections import Counter
            multiplicity=Counter(r['sha256'] for r in receipts.values())
            # Byte-identical cross-miner submissions cannot earn unique points.
            # Do not let one shared hash apply another UID's allocation.
            allocations={m:(n if multiplicity[receipts[m]['sha256']]==1 else 0) for m,n in allocations.items()}
            audit_manifest['audit_policy']=dict(policy,submission_counts={r['sha256']:allocations[m] for m,r in receipts.items()})
            save(self.state/(epoch+'-audit-plan.json'),dict(allocations=allocations,population_basis='signed-per-miner-upper-bound',policy=policy))
        save(self.state/(epoch+'-audit-manifest.json'),audit_manifest);reports={}
        def verify_one(item):
            miner,receipt=item
            remote=self.jobs.run(epoch+'-verify-'+miner[:8],'verify',audit_manifest,checkpoint_path,
                submissions=[dict(url=self.bucket.presign(receipt['frozen_key']),sha256=receipt['sha256'])])
            return miner,receipt,remote
        if hasattr(self.jobs,'queue'):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(receipts) or 1,len(self.jobs.verifiers))) as pool:
                verified=list(pool.map(verify_one,receipts.items()))
        else:verified=[verify_one(item) for item in receipts.items()]
        for miner,receipt,remote in verified:
            report=remote['audits'][0]
            if report['submission_sha256']!=receipt['sha256']:raise ValueError('frozen artifact report binding')
            report.update(remote_job_id=remote['job_id'],backend_profile=remote['backend_profile'],execution_resources_enforced=remote['execution_resources_enforced'])
            save(self.state/(epoch+'-'+miner+'-report.json'),report);reports[miner]=report
            self.bucket.json('public/'+epoch+'/audits/'+miner+'.json',self.signed(report))
            artifact=self.state/(epoch+'-'+miner+'.zip')
            if not hasattr(self.jobs,'queue') and not artifact.exists():self.bucket.download(receipt['frozen_key'],artifact)
        if bounded:
            from .audit_policy import escalation_allocations,penalty_count
            remaining={m:manifest['max_batches']
                       for m,r in reports.items() if penalty_count(r,policy['penalties'])}
            additions=escalation_allocations(remaining,allocations,policy,challenge['seed'])
            counts=dict(audit_manifest['audit_policy']['submission_counts'])
            for miner,extra in additions.items():
                if not extra:continue
                receipt=receipts[miner];counts[receipt['sha256']]+=extra
            expanded=dict(audit_manifest,audit_policy=dict(audit_manifest['audit_policy'],submission_counts=counts))
            for miner,extra in additions.items():
                if not extra:continue
                receipt=receipts[miner]
                remote=self.jobs.run(epoch+'-verify-expanded-'+miner[:8],'verify',expanded,checkpoint_path,
                    submissions=[dict(url=self.bucket.presign(receipt['frozen_key']),sha256=receipt['sha256'])])
                report=remote['audits'][0]
                if report['submission_sha256']!=receipt['sha256']:raise ValueError('expanded frozen artifact binding')
                previous={o['batch']:o for o in reports[miner]['outcomes'] if o.get('fully_audited') is True}
                current={o['batch']:o for o in report['outcomes']}
                if any(current.get(b,{}).get('valid')!=o.get('valid') for b,o in previous.items()):raise ValueError('expanded audit inconsistent with initial audited outcomes')
                report.update(remote_job_id=remote['job_id'],backend_profile=remote['backend_profile'],execution_resources_enforced=remote['execution_resources_enforced'])
                save(self.state/(epoch+'-'+miner+'-initial-report.json'),reports[miner])
                save(self.state/(epoch+'-'+miner+'-report.json'),report);reports[miner]=report
                self.bucket.json('public/'+epoch+'/audits/'+miner+'.json',self.signed(report))
            audit_manifest=expanded
            save(self.state/(epoch+'-audit-manifest.json'),audit_manifest)
            self.bucket.json('public/'+epoch+'/audit-plan.json',self.signed(dict(allocations=allocations,escalations=additions,policy=policy,population_basis='signed-per-miner-upper-bound')))
        result=score(reports,policy['penalties'] if bounded else None);result.update(payable=False,epoch_id=epoch,finalized_at=time.time(),receipts=receipts,checkpoint=manifest['checkpoint']['id'])
        save(saved,result);self.bucket.json('public/'+epoch+'/scores.json',self.signed(result))
        if manifest.get('live_reward_contract') is not None:
            from .live_reward_bridge import persist_signed_compute_evidence
            persist_signed_compute_evidence(self,manifest,result,reports)
        return result,reports
    def train(self,manifest,reports,checkpoint_path,destination=None,steps=1,replay=None,**ignored):
        epoch=manifest['epoch'];cached=self.state/(epoch+'-training-metrics.json')
        if cached.exists():
            metrics=json.loads(cached.read_text())
            if metrics.get('source_epoch')!=epoch or metrics.get('input_checkpoint')!=manifest['checkpoint']['id'] or metrics.get('training_policy')!=FULL_POLICY or metrics.get('weights_changed') is not True or metrics['checkpoint']!=file_map(metrics['new_checkpoint']['files']) or metrics['checkpoint']==manifest['checkpoint']['id']:raise ValueError('cached GPU training checkpoint binding')
            expected_replay=hashlib.sha256(canonical(replay)).hexdigest() if replay is not None else None
            if metrics.get('replay_inputs_sha256')!=expected_replay or metrics['steps']!=steps:raise ValueError('cached replay/current request binding')
            publication=json.loads((self.state/(epoch+'-checkpoint-publication.json')).read_text())
            if publication.get('checkpoint')!=metrics['checkpoint'] or publication.get('operator_independent_hashes') is not True or {name:value['sha256'] for name,value in publication.get('objects',{}).items()}!=metrics['new_checkpoint']['files']:raise ValueError('cached checkpoint publication receipt binding')
            output=self.checkpoint_with_reads(metrics['new_checkpoint'])
            current=dict(metrics,new_checkpoint=output)
            self.bucket.json('public/'+epoch+'/training.json',self.signed(current))
            return output,current
        receipts=json.loads((self.state/(epoch+'-scores.json')).read_text())['receipts']
        submissions=[dict(url=self.bucket.presign(receipts[m]['frozen_key']),sha256=receipts[m]['sha256']) for m,r in reports.items() if r['accepted']]
        if not submissions:raise ValueError('no independently verified training data')
        extra={'replay':replay} if replay is not None else {}
        training_manifest=manifest
        if manifest.get('audit_policy',{}).get('version')=='bounded-random-v1':
            training_manifest=json.loads((self.state/(epoch+'-audit-manifest.json')).read_text())
        planned_bytes=training_submission_bytes(receipts,reports,manifest)
        capacity=(self.jobs.training_resume(epoch+'-train',training_manifest,submissions,steps,replay) if hasattr(self.jobs,'training_resume') else None)
        if capacity is None:
            capacity=(self.jobs.training_capacity(manifest,steps,submission_bytes=planned_bytes) if hasattr(self.jobs,'training_capacity') else self.jobs.capacity(checkpoint_path))
        remote=self.jobs.run(epoch+'-train','train',training_manifest,checkpoint_path,submissions=submissions,steps=steps,training_policy=FULL_POLICY,**extra)
        new=dict(remote['new_checkpoint']);path=new.pop('path')
        if new['id']==manifest['checkpoint']['id']:raise ValueError('unchanged trained checkpoint')
        output=self.publish_remote_checkpoint(dict(manifest,checkpoint=new),path)
        metrics=dict(steps=steps,weights_changed=True,full_model_finetune=True,training_policy=FULL_POLICY,updates=remote['training']['updates'],source_epoch=epoch,input_checkpoint=manifest['checkpoint']['id'],input_pairs=sum(len(r['accepted']) for r in reports.values()),checkpoint=new['id'],new_checkpoint=output,checkpoint_path=path,capacity_preflight=capacity,remote_job_id=remote['job_id'])
        if replay is not None:
            metrics['replay_training']=remote['replay_training'];metrics['replay_inputs_sha256']=hashlib.sha256(canonical(replay)).hexdigest()
        save(cached,metrics);self.bucket.json('public/'+epoch+'/training.json',self.signed(metrics));return output,metrics
