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
from .backend_jobs import COVERED_POLICY,PERSISTENT_POLICY as PERSISTENT_RECEIPT_POLICY
from .training_policy import epoch_policy
from .controller import Controller
from .scoring import score
RECEIPT_TRAINING_POLICIES=(COVERED_POLICY,PERSISTENT_RECEIPT_POLICY)


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

def dispatch_verifications(jobs, items, operation):
    """Keep initial and escalation work bounded by the admitted worker roster.

    Ordered results preserve scoring and original labels; queued job recovery
    remains responsible for reusing completed or still-live signed requests.
    """
    items = list(items)
    if hasattr(jobs, 'queue') and items:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(items), len(jobs.verifiers))) as pool:
            return list(pool.map(operation, items))
    return [operation(item) for item in items]

class RemoteJobTerminalError(RuntimeError):
    terminal_job=True

class RemoteObservationTimeout(TimeoutError):
    """An observation window elapsed; the original remote job is still live."""
    def __init__(self,job_id,role):
        self.job_id=job_id;self.role=role
        super().__init__('same remote role remains active; retain job identity: '+job_id)


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
    def persistent_training_capacity(self,manifest,steps,submission_bytes=None):
        from .persistent_cpu_adamw import POLICY as PERSISTENT_POLICY
        if manifest.get('training_policy')!=PERSISTENT_POLICY:raise ValueError('single-host prospective persistent capacity only')
        from .persistent_training_worker import capacity_requirement
        cache=self.workspace+'/checkpoints/'+manifest['checkpoint']['id']
        code="import json;from subnet.persistent_training_worker import capacity_probe;print(json.dumps(capacity_probe("+repr(self.workspace)+","+repr(cache)+")))"
        probe=json.loads(self.command('cd '+shlex.quote(self.code)+' && '+shlex.quote(self.python)+' -I -B -c '+shlex.quote("import sys;sys.path.insert(0,"+repr(self.code)+");"+code)))
        return capacity_requirement(manifest,probe,checkpoint_bytes=probe['checkpoint_bytes'],missing_input=False)
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
                or job.get('training_policy')!=epoch_policy(manifest) or job.get('replay')!=replay
                or [r['sha256'] for r in job.get('submissions',[])]!=[r['sha256'] for r in submissions]):
            raise ValueError('original training request changed')
        from .persistent_cpu_adamw import POLICY as PERSISTENT_POLICY
        if job.get('training_policy')==PERSISTENT_POLICY and [r.get('accepted_batch_sha256')for r in job['submissions']]!=[r.get('accepted_batch_sha256')for r in submissions]:
            raise ValueError('original persistent training accepted pair population changed')
        if job.get('training_policy') in RECEIPT_TRAINING_POLICIES:
            if manifest.get('training_input_policy')=='committed-unaudited-training-v1':
                from .committed_training_inputs import receipt_inventory
            elif (manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2'):
                if 'subnet/compact_training_inputs.py' not in job.get('source_files',{}):
                    raise ValueError('compact original job source pin required')
                from .compact_training_inputs import receipt_inventory
            else:
                from .training_receipts import receipt_inventory
            if receipt_inventory(job['submissions'])!=receipt_inventory(submissions):
                raise ValueError('original training verifier admission receipts changed')
        reportpath=self.state/(prior['job_id']+'-report.json')
        if reportpath.exists():
            self.checked(json.loads(reportpath.read_text()),prior,manifest);phase='complete'
        else:
            phase=self.remote_status(prior['job_id'])['phase']
            if phase not in ('running','complete'):raise ValueError('original training not resumable; inspect without relaunch')
        return dict(resuming_original_training=True,original_job_id=prior['job_id'],original_phase=phase,
                    new_training_started=False,new_disk_reserve_not_required=True)
    def remote_status(self,identifier,timeout=1800,physical=False):
        code="from subnet.remote_runner import probe;import json;print(json.dumps(probe("+repr(self.workspace)+","+repr(identifier)+(",physical=True"if physical else "")+")))"
        return json.loads(self.command('cd '+shlex.quote(self.code)+' && '+shlex.quote(self.python)+' -B -c '+shlex.quote(code),timeout=timeout))
    def mine_reservation(self,label):
        """A collection deadline never releases the original physical miner."""
        for path in sorted(self.state.glob('*.json')):
            prior=json.loads(path.read_text())
            if not isinstance(prior,dict) or prior.get('role')!='mine' or 'manifest_sha256'not in prior or path.name==label+'.json':continue
            original=signed(json.loads((self.state/(prior['job_id']+'-job.json')).read_text()),self.controller.authority.id)
            if original.get('role')!='mine' or original.get('job_id')!=prior['job_id'] or hashlib.sha256(canonical(original)).hexdigest()!=prior['job_sha256']:
                raise ValueError('original miner reservation binding')
            if signed(original['manifest'],self.controller.authority.id).get('hourly_execution_policy')is None:continue
            if prior.get('physical_workspace',self.workspace)!=self.workspace:raise RemoteMinerReserved(prior['job_id'])
            terminal=self.state/(prior['job_id']+'-physical-terminal.json')
            if terminal.exists():
                value=json.loads(terminal.read_text())
                if value.get('job_sha256')!=prior['job_sha256'] or value.get('phase')not in ('complete','failed'):raise ValueError('miner terminal reservation evidence')
                continue
            try:status=self.remote_status(prior['job_id'],timeout=20,physical=True)
            except Exception as exc:raise RemoteMinerReserved(prior['job_id'])from exc
            if status.get('phase')not in ('complete','failed'):raise RemoteMinerReserved(prior['job_id'])
            save(terminal,dict(job_id=prior['job_id'],job_sha256=prior['job_sha256'],phase=status['phase'],observed_at=time.time(),original_status=status))
    def run(self,label,role,manifest,cache=None,dispatch_only=False,**fields):
        if type(dispatch_only)is not bool or dispatch_only and (role!='mine' or manifest.get('hourly_execution_policy')is None):raise ValueError('dispatch-only requires signed hourly miner')
        if not label.replace('-','').replace('_','').isalnum():raise ValueError('job label')
        record=self.state/(label+'.json');prior=None
        if dispatch_only:self.mine_reservation(label)
        if record.exists():
            prior=json.loads(record.read_text());reportpath=self.state/(prior['job_id']+'-report.json')
            if dispatch_only:
                original=signed(json.loads((self.state/(prior['job_id']+'-job.json')).read_text()),self.controller.authority.id)
                if (original.get('role')!='mine' or original.get('miner_id')!=fields.get('miner_id') or signed(original['manifest'],self.controller.authority.id)!=manifest or hashlib.sha256(canonical(original)).hexdigest()!=prior['job_sha256']):raise ValueError('original miner request changed')
                return dict(dispatch_only=True,original_job_id=prior['job_id'],job_sha256=prior['job_sha256'],terminal_observed=False,new_job_started=False)
            if getattr(self,'config',{}).get('retain_original_jobs',False) and role=='evaluate':
                original=signed(json.loads((self.state/(prior['job_id']+'-job.json')).read_text()),self.controller.authority.id)
                if (original.get('role')!=role or original.get('heldout')!=fields.get('heldout')
                        or original.get('successor_calibration')!=fields.get('successor_calibration')
                        or signed(original['manifest'],self.controller.authority.id)!=manifest
                        or hashlib.sha256(canonical(original)).hexdigest()!=prior['job_sha256']):
                    raise ValueError('original evaluation request changed')
            if reportpath.exists():return self.checked(json.loads(reportpath.read_text()),prior,manifest)
            status=self.remote_status(prior['job_id'])
            if status['phase']=='complete':
                self.copy_from(self.workspace+'/jobs/'+prior['job_id']+'/report.json',reportpath)
                return self.checked(json.loads(reportpath.read_text()),prior,manifest)
            if status['phase'] in ('failed','not_launched'):
                if role=='train' or getattr(self,'config',{}).get('retain_original_jobs',False):raise RemoteJobTerminalError('original role terminal or absent; refuse automatic relaunch')
                save(self.state/(prior['job_id']+'-failure.json'),status);prior=None
            elif status['phase']!='running':raise ValueError('unknown authoritative remote job status')
        if prior is None:
            identifier=label+'-'+secrets.token_hex(4);now=time.time()
            payload=dict(schema=1,job_id=identifier,role=role,created_at=now,expires_at=now+role_time_budget(self.config,role),manifest=self.controller.signed(manifest),**self.metadata,**fields)
            if role=='train' and manifest.get('training_startup_recovery') is not None:
                recovery=signed(manifest['training_startup_recovery'],self.controller.authority.id)
                payload['expires_at']=min(payload['expires_at'],recovery['expires_at'])
            if role=='train' and fields.get('training_policy') in RECEIPT_TRAINING_POLICIES:
                if manifest.get('training_input_policy')=='committed-unaudited-training-v1':
                    from .committed_training_inputs import VERSION,validate_job as validate_receipt_job
                elif (manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2'):
                    if 'subnet/compact_training_inputs.py' not in payload.get('source_files',{}):
                        raise ValueError('compact dispatch source pin required')
                    from .compact_training_inputs import VERSION,validate_job as validate_receipt_job
                else:
                    from .training_receipts import VERSION,validate_job as validate_receipt_job
                payload['training_input_policy']=VERSION
                validate_receipt_job(payload,manifest,self.controller.authority.id)
            from .persistent_cpu_adamw import POLICY as PERSISTENT_POLICY
            if role=='train'and fields.get('training_policy')==PERSISTENT_POLICY:
                if 'persistent_training'in fields:raise ValueError('persistent capabilities are original-job scoped')
                from .persistent_training_protocol import prepare_job,validate_job
                transport_ttl=role_time_budget(self.config,role)
                if manifest.get('training_startup_recovery')is not None:
                    transport_ttl=int(payload['expires_at']-time.time())
                    if transport_ttl<=0:raise ValueError('startup recovery authorization expired before capability issue')
                payload['persistent_training']=prepare_job(self.controller,manifest,identifier,fields['steps'],transport_ttl)
                validate_job(payload,manifest,self.controller.authority.id)
            jobpath=self.state/(identifier+'-job.json');save(jobpath,self.controller.signed(payload))
            prior=dict(job_id=identifier,role=role,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],job_sha256=hashlib.sha256(canonical(payload)).hexdigest(),manifest_sha256=hashlib.sha256(canonical(manifest)).hexdigest(),source_files=self.metadata['source_files'],runtime_versions=self.metadata['runtime_versions'])
            if dispatch_only:prior['physical_workspace']=self.workspace
            save(record,prior);remotejob=self.workspace+'/'+identifier+'.json'
            self.command('mkdir -p '+shlex.quote(self.workspace));self.copy_to(jobpath,remotejob)
            command='cd '+shlex.quote(self.code)+' && CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup '+shlex.quote(self.python)+' -B -m subnet.remote_runner '+shlex.quote(remotejob)+' --authority '+self.controller.authority.id+' --workspace '+shlex.quote(self.workspace)
            if cache:command+=' --checkpoint-cache '+shlex.quote(cache)
            self.command(command+' > '+shlex.quote(self.workspace+'/'+identifier+'-runner.log')+' 2>&1 < /dev/null &')
        if dispatch_only:
            return dict(dispatch_only=True,original_job_id=prior['job_id'],job_sha256=prior['job_sha256'],terminal_observed=False,new_job_started=True)
        reportpath=self.state/(prior['job_id']+'-report.json')
        started=time.time()
        while True:
            status=self.remote_status(prior['job_id'])
            if status['phase']=='complete':
                self.copy_from(self.workspace+'/jobs/'+prior['job_id']+'/report.json',reportpath)
                return self.checked(json.loads(reportpath.read_text()),prior,manifest)
            if status['phase']=='failed':
                save(self.state/(prior['job_id']+'-failure.json'),status);raise RemoteJobTerminalError('remote role exited: '+str(status.get('exit_code',status.get('reason'))))
            if status['phase']=='not_launched' and time.time()-started>30:raise RuntimeError('remote launcher marker absent; retain job record for authoritative recovery')
            if time.time()-started>1800:raise RemoteObservationTimeout(prior['job_id'],role)
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
        if job.get('role')=='train' and job.get('training_policy') in RECEIPT_TRAINING_POLICIES:
            if manifest.get('training_input_policy')=='committed-unaudited-training-v1':
                from .committed_training_inputs import validate_report as validate_receipt_report
            elif (manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2'):
                if 'subnet/compact_training_inputs.py' not in job.get('source_files',{}):
                    raise ValueError('compact original report source pin required')
                from .compact_training_inputs import validate_report as validate_receipt_report
            else:
                from .training_receipts import validate_report as validate_receipt_report
            validate_receipt_report(report,job,manifest,self.controller.authority.id)
        created,expires,completed=job.get('created_at'),job.get('expires_at'),report.get('completed_at')
        if any(type(value) not in (int,float) or not math.isfinite(value) for value in (created,expires,completed)) or not created<=completed<expires or not 0<expires-created<=86400:
            raise ValueError('remote role report outside signed time budget')
        from .persistent_cpu_adamw import POLICY as PERSISTENT_POLICY
        if job.get('training_policy')==PERSISTENT_POLICY:
            from .persistent_training_protocol import validate_job,validate_report
            validate_job(job,manifest,self.controller.authority.id);validate_report(report,job,manifest)
        return report


class RemoteController(Controller):
    def __init__(self,bucket,gateway,state,remote):
        super().__init__(bucket,gateway,state)
        self.independent_state_reader=None
        if remote.get('independent_state_reader') is not None:
            from .independent_state_dispatch import IndependentStateReader
            self.independent_state_reader=IndependentStateReader(remote['independent_state_reader'],self)
        self.training_startup_recovery_files=dict(remote.get('training_startup_recovery_files',{}))
        self.training_execution_amendment_files=dict(remote.get('training_execution_amendment_files',{}))
        self.training_execution_amendment_required_epochs=list(remote.get('training_execution_amendment_required_epochs',[]))
        if 'roles' in remote:
            from .role_router import RoutedJobs
            self.jobs=RoutedJobs(remote,self)
        else:self.jobs=RemoteJobs(remote,self)
    def open(self,*args,max_batches=3,**kwargs):
        if type(max_batches)is not int or not 1<=max_batches<=256:raise ValueError('per UID batch quota')
        if kwargs.get('submission_transport_policy')is not None:kwargs['commitment_max_batches']=max_batches
        exclusion_policy=kwargs.pop('temporary_exclusion_policy',None)
        exclusion_snapshot=None
        if exclusion_policy is not None:
            from .audit_exclusion import validate,snapshot
            exclusion_policy=validate(exclusion_policy)
            history_path=self.state/'confirmed-invalid-history.json'
            history=json.loads(history_path.read_text())if history_path.exists()else dict(version='authenticated-confirmed-invalid-history-v1',epochs=[])
            exclusion_snapshot=dict(policy=exclusion_policy,history=self.signed(history),excluded_miners=snapshot(history,exclusion_policy))
        input_policy=kwargs.pop('training_input_policy',None)
        if input_policy is not None:
            if input_policy not in ('authenticated-verifier-receipts-v1','authenticated-verifier-compact-inputs-v2','committed-unaudited-training-v1'):
                raise ValueError('unapproved training input policy')
            if kwargs.get('training_policy') not in RECEIPT_TRAINING_POLICIES:
                raise ValueError('explicit receipt input requires covered/persistent objective')
        if kwargs.get('submission_transport_policy')and input_policy not in ('authenticated-verifier-compact-inputs-v2','committed-unaudited-training-v1'):raise ValueError('per-pair transport requires compact audited-only training')
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
        if input_policy=='committed-unaudited-training-v1':kwargs['training_input_policy']=input_policy
        try:manifest=super().open(*args,**kwargs)
        finally:self.bucket=original
        manifest['max_batches']=max_batches
        if exclusion_snapshot is not None:manifest['audit_exclusion_snapshot']=exclusion_snapshot
        if input_policy is not None:manifest['training_input_policy']=input_policy
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

    def stage_remote_checkpoint(self,manifest,remote_path):
        """Upload and independently hash bytes without signing an authority descriptor."""
        cp=manifest['checkpoint'];capacity=(self.jobs.publication_capacity(remote_path) if hasattr(self.jobs,'publication_capacity') else self.jobs.capacity(remote_path))
        report=self.jobs.run(manifest['epoch']+'-publish-'+cp['id'][:8],'upload',manifest,remote_path,
            put_urls={n:self.bucket.presign('public/checkpoints/'+cp['id']+'/'+n,'put_object',3600) for n in cp['files']})
        workers=1
        if manifest.get('persistent_publication_policy') is not None:
            from .persistent_publication import validate_policy
            workers=validate_policy(manifest['persistent_publication_policy'])['checkpoint_readback_workers']
        def check(row):
            name,expected=row;h=hashlib.sha256();size=0
            with requests.get(self.bucket.presign('public/checkpoints/'+cp['id']+'/'+name),
                    stream=True,timeout=180,allow_redirects=False,
                    headers={'Accept-Encoding':'identity'}) as response:
                if response.status_code!=200 or response.headers.get('Content-Encoding','identity')!='identity':
                    raise ValueError('operator R2 checkpoint read status/encoding')
                declared=response.headers.get('Content-Length')
                for part in response.iter_content(1024*1024):
                    if not part:continue
                    if not isinstance(part,bytes) or len(part)>1024*1024:
                        raise ValueError('bounded checkpoint read chunk')
                    h.update(part);size+=len(part)
                if declared is not None and size!=int(declared):raise ValueError('checkpoint full object size')
            if h.hexdigest()!=expected:raise ValueError('operator independent checkpoint integrity')
            return name,dict(sha256=h.hexdigest(),bytes=size)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers)as pool:observed=dict(pool.map(check,cp['files'].items()))
        staged=dict(checkpoint=cp['id'],objects=observed,capacity=capacity,operator_independent_hashes=True)
        save(self.state/(manifest['epoch']+'-checkpoint-staging.json'),staged)
        return staged

    def commit_remote_checkpoint(self,manifest,staged):
        """Commit only the exact locally journaled independently hashed objects."""
        cp=manifest['checkpoint'];path=self.state/(manifest['epoch']+'-checkpoint-staging.json')
        if not path.is_file()or canonical(json.loads(path.read_text()))!=canonical(staged):raise ValueError('original checkpoint staging journal')
        if (staged.get('checkpoint')!=cp['id']or staged.get('operator_independent_hashes')is not True
                or {n:v.get('sha256')for n,v in staged.get('objects',{}).items()}!=cp['files']
                or any(type(v.get('bytes'))is not int or v['bytes']<=0 for v in staged['objects'].values())):
            raise ValueError('actual checkpoint staging full integrity receipt')
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
        save(self.state/(manifest['epoch']+'-checkpoint-publication.json'),staged)
        return self.checkpoint_with_reads(dict(descriptor,descriptor_key=key))
    def publish_remote_checkpoint(self,manifest,remote_path):
        return self.commit_remote_checkpoint(manifest,self.stage_remote_checkpoint(manifest,remote_path))
    def collect_learner_inputs(self,manifest,*,round_number=None):
        from .committed_training_inputs import collect
        return collect(self,manifest,round_number=round_number)
    def finalize(self,manifest,checkpoint_path):
        from .forced_sampling import require_report
        if manifest.get('payable') is not False:raise ValueError('remote experimental controller is nonpayable only')
        epoch=manifest['epoch'];saved=self.state/(epoch+'-scores.json')
        from .capture_status import InfrastructureSkipped,terminal
        if (self.state/(epoch+'-capture-status.json')).exists():raise InfrastructureSkipped(terminal(self,manifest))
        if saved.exists():
            result=json.loads(saved.read_text())
            if result.get('checkpoint')!=manifest['checkpoint']['id'] or result.get('payable') is not False or result['receipts']!=self.gateway.freeze(epoch):raise ValueError('cached finalized manifest/receipt binding')
            reports={m:json.loads((self.state/(epoch+'-'+m+'-report.json')).read_text()) for m in result['receipts']}
            if any(r.get('epoch')!=epoch or r.get('submission_sha256')!=result['receipts'][m]['sha256'] for m,r in reports.items()):raise ValueError('cached audit report binding')
            for report in reports.values():require_report(manifest,report)
            self.bucket.json('public/'+epoch+'/scores.json',self.signed(result))
            if manifest.get('live_reward_contract') is not None:
                from .live_reward_bridge import persist_signed_compute_evidence
                persist_signed_compute_evidence(self,manifest,result,reports)
            return result,reports
        timings_path=self.state/(epoch+'-finalize-timings.json');timings=json.loads(timings_path.read_text())if timings_path.exists()else {}
        timings.setdefault('freeze_started_at',time.time());save(timings_path,timings)
        from .commitment_transport import FreezeMetadataIncomplete
        try:receipts=self.gateway.freeze(epoch)
        except FreezeMetadataIncomplete:
            from .hourly_policy import cutoff
            until=cutoff(manifest,'freeze')
            if until is None or time.time()<until:raise
            raise InfrastructureSkipped(terminal(self,manifest))
        timings.setdefault('freeze_completed_at',time.time());save(timings_path,timings)
        challengepath=self.state/(epoch+'-audit-challenge.json')
        if challengepath.exists():challenge=json.loads(challengepath.read_text())
        else:
            challenge=dict(seed=secrets.token_hex(32),generated_after_freeze_at=time.time(),receipts=receipts);save(challengepath,challenge)
        if challenge['receipts']!=receipts:raise ValueError('frozen audit challenge binding')
        self.bucket.json('public/'+epoch+'/audit-challenge.json',self.signed(challenge));audit_manifest=dict(manifest,audit_seed=challenge['seed'],audit_frozen_receipts=receipts)
        bounded=manifest.get('audit_policy',{}).get('version')=='bounded-random-v1'
        if bounded:
            from .audit_policy import validate,allocate
            policy=validate(manifest['audit_policy'])
            allocations=allocate({m:(0 if m in manifest.get('audit_exclusion_snapshot',{}).get('excluded_miners',[])else len(r['artifacts'])if manifest.get('submission_transport_policy')else manifest['max_batches']) for m,r in receipts.items()},policy,challenge['seed'])
            from collections import Counter
            multiplicity=Counter(r['sha256'] for r in receipts.values())
            # Byte-identical cross-miner submissions cannot earn unique points.
            # Do not let one shared hash apply another UID's allocation.
            allocations={m:(n if multiplicity[receipts[m]['sha256']]==1 else 0) for m,n in allocations.items()}
            audit_manifest['audit_policy']=dict(policy,submission_counts={r['sha256']:allocations[m] for m,r in receipts.items()})
            if manifest.get('submission_transport_policy'):
                from .audit_policy import selection
                selected_slots={m:selection(len(r['artifacts']),allocations[m],challenge['seed'],r['sha256'])for m,r in receipts.items()}
                audit_manifest['audit_policy']['submission_counts'].update({b['sha256']:1 for m,r in receipts.items()for b in r['artifacts']if b['slot']in selected_slots[m]})
            save(self.state/(epoch+'-audit-plan.json'),dict(allocations=allocations,population_basis='actual-committed-pair-count'if manifest.get('submission_transport_policy')else'signed-per-miner-upper-bound',policy=policy))
        save(self.state/(epoch+'-audit-manifest.json'),audit_manifest);reports={}
        from .hourly_policy import cutoff
        audit_until=cutoff(manifest,'audit')
        def verify_one(item):
            miner,receipt=item
            submissions=[dict(url=self.bucket.presign(b['frozen_key']),sha256=b['sha256'],commitment_miner=miner,commitment_ref=dict(miner=miner,commitment_sha256=receipt['sha256'],**{k:b[k]for k in ('slot','env_id','index','batch_sha256','size','frozen_key')}))for b in receipt['artifacts']if b['slot']in selected_slots[miner]]if manifest.get('submission_transport_policy')else[dict(url=self.bucket.presign(receipt['frozen_key']),sha256=receipt['sha256'])]
            try:
                if audit_until is not None and time.time()>=audit_until:raise TimeoutError('signed audit cutoff elapsed')
                extra={'observe_until':audit_until}if audit_until is not None and hasattr(self.jobs,'queue')else {}
                remote=self.jobs.run(epoch+'-verify-'+miner[:8],'verify',audit_manifest,checkpoint_path,submissions=submissions,**extra)
                if audit_until is not None and time.time()>=audit_until:raise TimeoutError('original report arrived after cutoff; retained for forensic audit')
                return miner,receipt,remote
            except (TimeoutError,RuntimeError)as exc:
                if audit_until is None:raise
                from .commitment_transport import deferred
                reason='budget_deferred'if time.time()>=audit_until else 'infrastructure_deferred'
                report=deferred(audit_manifest,receipt,reason,time.time())
                save(self.state/(epoch+'-'+miner+'-deferral.json'),dict(report=report,error_type=type(exc).__name__,observed_at=time.time()))
                return miner,receipt,{'deferred_report':report}
        selected_items=list(receipts.items())
        if manifest.get('submission_transport_policy'):
            from .commitment_transport import unchecked
            selected_items=[(m,r)for m,r in receipts.items()if selected_slots[m]]
            for m,r in receipts.items():
                if selected_slots[m]:continue
                report=unchecked(audit_manifest,r);reports[m]=report
                save(self.state/(epoch+'-'+m+'-report.json'),report);self.bucket.json('public/'+epoch+'/audits/'+m+'.json',self.signed(report))
        timings.setdefault('audits_started_at',time.time());save(timings_path,timings)
        if manifest.get('proof_copy_policy') is not None:
            from .selected_proof_copy import copy_selected
            copy_errors=copy_selected(self.gateway,manifest,receipts,selected_slots,audit_until)
            from .commitment_transport import deferred
            for miner,error in copy_errors.items():
                report=deferred(audit_manifest,receipts[miner],'budget_deferred' if audit_until is not None and time.time()>=audit_until else 'infrastructure_deferred',time.time());reports[miner]=report
                save(self.state/(epoch+'-'+miner+'-report.json'),report);self.bucket.json('public/'+epoch+'/audits/'+miner+'.json',self.signed(report))
            selected_items=[item for item in selected_items if item[0]not in copy_errors]
            from .selected_proof_copy import signed_copy_inventory
            audit_manifest['proof_copy_receipts']=signed_copy_inventory(self.gateway,epoch)
            save(self.state/(epoch+'-audit-manifest.json'),audit_manifest)
        verified=dispatch_verifications(self.jobs,selected_items,verify_one)
        timings.setdefault('audits_completed_at',time.time());timings['selected_miner_jobs']=len(selected_items);save(timings_path,timings)
        for miner,receipt,remote in verified:
            if 'deferred_report'in remote:
                report=remote['deferred_report'];reports[miner]=report;save(self.state/(epoch+'-'+miner+'-report.json'),report);self.bucket.json('public/'+epoch+'/audits/'+miner+'.json',self.signed(report));continue
            if manifest.get('submission_transport_policy'):
                from .commitment_transport import combine
                report=combine(audit_manifest,receipt,remote)
            else:report=remote['audits'][0]
            if report['submission_sha256']!=receipt['sha256']:raise ValueError('frozen artifact report binding')
            require_report(manifest,report)
            report.update(remote_job_id=remote['job_id'],backend_profile=remote['backend_profile'],execution_resources_enforced=remote['execution_resources_enforced'])
            save(self.state/(epoch+'-'+miner+'-report.json'),report);reports[miner]=report
            self.bucket.json('public/'+epoch+'/audits/'+miner+'.json',self.signed(report))
            artifact=self.state/(epoch+'-'+miner+'.zip')
            if not manifest.get('submission_transport_policy')and not hasattr(self.jobs,'queue') and not artifact.exists():self.bucket.download(receipt['frozen_key'],artifact)
        if bounded and not manifest.get('submission_transport_policy'):
            from .audit_policy import escalation_allocations,penalty_count
            remaining={m:manifest['max_batches']
                       for m,r in reports.items() if penalty_count(r,policy['penalties'])}
            additions=escalation_allocations(remaining,allocations,policy,challenge['seed'])
            counts=dict(audit_manifest['audit_policy']['submission_counts'])
            for miner,extra in additions.items():
                if not extra:continue
                receipt=receipts[miner];counts[receipt['sha256']]+=extra
            expanded=dict(audit_manifest,audit_policy=dict(audit_manifest['audit_policy'],submission_counts=counts))
            def expand_one(item):
                miner,extra=item
                receipt=receipts[miner]
                remote=self.jobs.run(epoch+'-verify-expanded-'+miner[:8],'verify',expanded,checkpoint_path,
                    submissions=[dict(url=self.bucket.presign(receipt['frozen_key']),sha256=receipt['sha256'])])
                return miner,receipt,remote
            expanded_reports=dispatch_verifications(self.jobs,
                [(miner,extra) for miner,extra in additions.items() if extra],expand_one)
            for miner,receipt,remote in expanded_reports:
                report=remote['audits'][0]
                if report['submission_sha256']!=receipt['sha256']:raise ValueError('expanded frozen artifact binding')
                require_report(manifest,report)
                previous={o['batch']:o for o in reports[miner]['outcomes'] if o.get('fully_audited') is True}
                current={o['batch']:o for o in report['outcomes']}
                if any(current.get(b,{}).get('valid')!=o.get('valid') for b,o in previous.items()):raise ValueError('expanded audit inconsistent with initial audited outcomes')
                report.update(remote_job_id=remote['job_id'],backend_profile=remote['backend_profile'],execution_resources_enforced=remote['execution_resources_enforced'])
                save(self.state/(epoch+'-'+miner+'-initial-report.json'),reports[miner])
                save(self.state/(epoch+'-'+miner+'-report.json'),report);reports[miner]=report
                self.bucket.json('public/'+epoch+'/audits/'+miner+'.json',self.signed(report))
            audit_manifest=expanded
            save(self.state/(epoch+'-audit-manifest.json'),audit_manifest)
            self.bucket.json('public/'+epoch+'/audit-plan.json',self.signed(dict(allocations=allocations,escalations=additions,policy=policy,population_basis='actual-committed-pair-count'if manifest.get('submission_transport_policy')else'signed-per-miner-upper-bound')))
        result=score(reports,policy['penalties'] if bounded else None);result.update(payable=False,epoch_id=epoch,finalized_at=time.time(),receipts=receipts,checkpoint=manifest['checkpoint']['id'])
        if manifest.get('audit_exclusion_snapshot')is not None:
            from .audit_exclusion import confirmed
            history_path=self.state/'confirmed-invalid-history.json'
            history=json.loads(history_path.read_text())if history_path.exists()else dict(version='authenticated-confirmed-invalid-history-v1',epochs=[])
            if not any(row['epoch']==epoch for row in history['epochs']):
                history['epochs'].append(dict(epoch=epoch,reports={m:r for m,r in reports.items()if confirmed(r)}));save(history_path,history)
                self.bucket.json('public/audits/confirmed-invalid-history.json',self.signed(history))
        save(saved,result);self.bucket.json('public/'+epoch+'/scores.json',self.signed(result))
        if manifest.get('live_reward_contract') is not None:
            from .live_reward_bridge import persist_signed_compute_evidence
            persist_signed_compute_evidence(self,manifest,result,reports)
        return result,reports
    def train(self,manifest,reports,checkpoint_path,destination=None,steps=1,replay=None,**ignored):
        from .training_receipts import require_execution_amendment
        require_execution_amendment(self,manifest)
        from .persistent_cpu_adamw import POLICY as PERSISTENT_POLICY
        if epoch_policy(manifest)==PERSISTENT_POLICY:
            from .persistent_training_controller import train
            return train(self,manifest,reports,checkpoint_path,steps=steps,replay=replay)
        epoch=manifest['epoch'];cached=self.state/(epoch+'-training-metrics.json')
        policy=epoch_policy(manifest)
        from .backend_jobs import COVERED_POLICY
        training_manifest=manifest
        if policy==COVERED_POLICY and manifest.get('audit_policy',{}).get('version')=='bounded-random-v1':
            training_manifest=json.loads((self.state/(epoch+'-audit-manifest.json')).read_text())
        if policy==COVERED_POLICY:
            from .training_policy import coverage_manifest
            receipts=json.loads((self.state/(epoch+'-scores.json')).read_text())['receipts']
            challenge=json.loads((self.state/(epoch+'-audit-challenge.json')).read_text())
            if (training_manifest.get('training_policy')!=policy or training_manifest.get('epoch')!=epoch or training_manifest.get('checkpoint')!=manifest['checkpoint']):raise ValueError('original covered audit manifest binding')
            training_manifest=coverage_manifest(training_manifest,receipts,challenge)
            if replay is not None:raise ValueError('covered historical replay requires separate admission')
        if policy==COVERED_POLICY:
            if (training_manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2'):
                from .compact_training_inputs import prepare_submissions,receipt_inventory
                from .compact_training_inputs import validate_report as validate_receipt_report
                submissions=prepare_submissions(self,training_manifest,reports,receipts)
            else:
                from .training_receipts import prepare_submissions,amend_manifest,receipt_inventory
                from .training_receipts import validate_report as validate_receipt_report
                submissions=prepare_submissions(self,training_manifest,reports,receipts)
                training_manifest=amend_manifest(self,training_manifest,submissions,steps)
        if cached.exists():
            metrics=json.loads(cached.read_text())
            if policy==COVERED_POLICY:
                record=json.loads((self.state/'roles'/(epoch+'-train.json')).read_bytes())
                job=signed(json.loads((self.state/'roles'/(record['job_id']+'-job.json')).read_bytes()),self.authority.id)
                if (hashlib.sha256(canonical(job)).hexdigest()!=record['job_sha256'] or
                        signed(job['manifest'],self.authority.id)!=training_manifest or job['steps']!=steps or
                        receipt_inventory(job['submissions'])!=receipt_inventory(submissions) or
                        metrics.get('verifier_receipt_inventory')!=receipt_inventory(submissions)):
                    raise ValueError('cached original verifier-receipt training request changed')
                validate_receipt_report(json.loads((self.state/'roles'/(record['job_id']+'-report.json')).read_bytes()),job,training_manifest,self.authority.id)
            if metrics.get('source_epoch')!=epoch or metrics.get('input_checkpoint')!=manifest['checkpoint']['id'] or metrics.get('training_policy')!=policy or metrics.get('weights_changed') is not True or metrics['checkpoint']!=file_map(metrics['new_checkpoint']['files']) or metrics['checkpoint']==manifest['checkpoint']['id']:raise ValueError('cached GPU training checkpoint binding')
            if policy==COVERED_POLICY and metrics.get('training_coverage')!=training_manifest['training_coverage']:raise ValueError('cached covered training context changed')
            expected_replay=hashlib.sha256(canonical(replay)).hexdigest() if replay is not None else None
            if metrics.get('replay_inputs_sha256')!=expected_replay or metrics['steps']!=steps:raise ValueError('cached replay/current request binding')
            publication=json.loads((self.state/(epoch+'-checkpoint-publication.json')).read_text())
            if publication.get('checkpoint')!=metrics['checkpoint'] or publication.get('operator_independent_hashes') is not True or {name:value['sha256'] for name,value in publication.get('objects',{}).items()}!=metrics['new_checkpoint']['files']:raise ValueError('cached checkpoint publication receipt binding')
            output=self.checkpoint_with_reads(metrics['new_checkpoint'])
            current=dict(metrics,new_checkpoint=output)
            self.bucket.json('public/'+epoch+'/training.json',self.signed(current))
            return output,current
        receipts=json.loads((self.state/(epoch+'-scores.json')).read_text())['receipts']
        if policy!=COVERED_POLICY:
            submissions=[dict(url=self.bucket.presign(receipts[m]['frozen_key']),sha256=receipts[m]['sha256']) for m,r in reports.items() if r['accepted']]
        if not submissions:raise ValueError('no independently verified training data')
        extra={'replay':replay} if replay is not None else {}
        if policy!=COVERED_POLICY and manifest.get('audit_policy',{}).get('version')=='bounded-random-v1':
            training_manifest=json.loads((self.state/(epoch+'-audit-manifest.json')).read_text())
        planned_bytes=(sum(obj['size'] for obj in submissions) if (training_manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2')
                       else training_submission_bytes(receipts,reports,manifest))
        capacity=(self.jobs.training_resume(epoch+'-train',training_manifest,submissions,steps,replay) if hasattr(self.jobs,'training_resume') else None)
        if capacity is None:
            capacity=(self.jobs.training_capacity(training_manifest,steps,submission_bytes=planned_bytes) if hasattr(self.jobs,'training_capacity') else self.jobs.capacity(checkpoint_path))
        remote=self.jobs.run(epoch+'-train','train',training_manifest,checkpoint_path,submissions=submissions,steps=steps,training_policy=policy,**extra)
        if policy==COVERED_POLICY:
            if remote['training'].get('training_policy')!=policy or remote['training'].get('training_coverage')!=training_manifest['training_coverage']:raise ValueError('remote covered training context changed')
        new=dict(remote['new_checkpoint']);path=new.pop('path')
        if new['id']==manifest['checkpoint']['id']:raise ValueError('unchanged trained checkpoint')
        output=self.publish_remote_checkpoint(dict(manifest,checkpoint=new),path)
        metrics=dict(steps=steps,weights_changed=True,full_model_finetune=True,training_policy=policy,updates=remote['training']['updates'],source_epoch=epoch,input_checkpoint=manifest['checkpoint']['id'],input_pairs=sum(len(r['accepted']) for r in reports.values()),checkpoint=new['id'],new_checkpoint=output,checkpoint_path=path,capacity_preflight=capacity,remote_job_id=remote['job_id'])
        if policy==COVERED_POLICY:
            metrics['training_coverage']=remote['training']['training_coverage'];metrics['covered_training_inputs']=remote['covered_training_inputs']
            metrics.update(training_input_policy=remote['training']['training_input_policy'],
                trainer_verification_performed=False,all_pairs_authenticated_verifier_receipts=True,
                verifier_receipt_inventory=receipt_inventory(submissions))
        if replay is not None:
            metrics['replay_training']=remote['replay_training'];metrics['replay_inputs_sha256']=hashlib.sha256(canonical(replay)).hexdigest()
        save(cached,metrics);self.bucket.json('public/'+epoch+'/training.json',self.signed(metrics));return output,metrics

class RemoteMinerReserved(RuntimeError):
    def __init__(self,job_id):
        self.job_id=job_id
        super().__init__("original physical miner remains reserved: "+job_id)
