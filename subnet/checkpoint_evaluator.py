"""Independent durable checkpoint evaluation, never a training prerequisite.

The queue keeps the original complete manifest and exact cohort configuration.
No checkpoint is called evaluated until its authenticated remote report passes
all existing heldout completeness, source, seed and worker-time checks.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import logging
import importlib
import inspect
import textwrap
import sys
import types
import time
from pathlib import Path
from .remote_backend import save, RemoteObservationTimeout
from .storage import canonical

VERSION='independent-checkpoints-v1'
CONFIG_FIELDS=('heldout','environment','environments','evaluation_experiment_id','evaluation_seed',
               'model_id','evaluation_state','trusted_evaluation_policy')

SOURCE_ROUTES_VERSION='independent-evaluator-source-routes-v1'

def detached_historical_run(module):
    """Replace only the reviewed historical SSH launch, preserving its ABI."""
    source=textwrap.dedent(inspect.getsource(module.RemoteJobs.run))
    old="""        command='cd '+shlex.quote(self.code)+' && CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup '+shlex.quote(self.python)+' -B -m subnet.remote_runner '+shlex.quote(remotejob)+' --authority '+self.controller.authority.id+' --workspace '+shlex.quote(self.workspace)
        if cache:command+=' --checkpoint-cache '+shlex.quote(cache)
        self.command(command+' > '+shlex.quote(self.workspace+'/'+identifier+'-runner.log')+' 2>&1 < /dev/null &')"""
    if source.count(old)==1:
        source=source.replace(old,'        self.launch_runner(identifier,remotejob,cache)')
        namespace={}
        exec(compile(source,module.__file__,'exec'),module.__dict__,namespace)
        return namespace['run']
    if 'nohup'in source or source.count('self.launch_runner(identifier,remotejob,cache)')!=1:
        raise ValueError('unreviewed historical evaluator SSH launch shape')
    return module.RemoteJobs.run

def qualified_dispatcher(row, controller):
    """Use the exact historical CPU admission ABI, without changing GPU code."""
    root=Path(row['local_source_path'])
    if root.is_symlink() or (root/'subnet').is_symlink():raise ValueError('qualified evaluator source tree changed')
    if {str(p.relative_to(root))for p in(root/'subnet').glob('*.py')}!=set(row['source_files']):
        raise ValueError('qualified evaluator source inventory changed')
    for name,digest in row['source_files'].items():
        if (root/name).is_symlink() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('qualified evaluator source bytes changed before import')
    namespace='_affine_evaluator_'+hashlib.sha256(canonical(row)).hexdigest()
    if namespace not in sys.modules:
        package=types.ModuleType(namespace);package.__path__=[str(root)]
        sys.modules[namespace]=package
    module=importlib.import_module(namespace+'.subnet.remote_backend')
    remote=module.RemoteJobs(row['endpoint'],controller)
    # Only the reviewed descriptor-closing transport is shared. Historical
    # request validation, source inventories and report checks remain original.
    from .remote_backend import RemoteJobs
    remote.launch_runner=types.MethodType(RemoteJobs.launch_runner,remote)
    remote.run=types.MethodType(detached_historical_run(module),remote)
    remote.observation_timeout_type=module.RemoteObservationTimeout
    return remote

class QualifiedEvaluationJobs:
    """ROOT-signed source routes for a single physical evaluator and old ledger."""
    def __init__(self, controller, document, factory=qualified_dispatcher):
        from .backend_jobs import signed
        self.controller=controller;self.factory=factory;self.instances={}
        if isinstance(document,(str,Path)):document=json.loads(Path(document).read_bytes())
        value=signed(document,controller.authority.id)
        if (set(value)!={'version','physical_id','sources'} or value['version']!=SOURCE_ROUTES_VERSION
                or not isinstance(value['physical_id'],str) or not value['physical_id']
                or not isinstance(value['sources'],dict) or not value['sources']):
            raise ValueError('independent evaluator signed source routes')
        self.rows=copy.deepcopy(value['sources']);physical=None
        for sha,row in self.rows.items():
            if (not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha)
                    or set(row)!={'endpoint','local_source_path','source_files','runtime_versions','new_dispatch_approved'}
                    or type(row['new_dispatch_approved'])is not bool):
                raise ValueError('independent evaluator exact source route')
            endpoint=row['endpoint'];root=Path(row['local_source_path'])
            if not root.is_absolute() or root.is_symlink() or not root.is_dir() or (root/'subnet').is_symlink():
                raise ValueError('qualified local evaluator source tree')
            files=row['source_files']
            if (not isinstance(files,dict) or not {'subnet/__init__.py','subnet/remote_backend.py','subnet/backend_jobs.py'}<=set(files)
                    or {str(p.relative_to(root)) for p in (root/'subnet').glob('*.py')}!=set(files)):
                raise ValueError('qualified evaluator complete runtime inventory')
            for name,digest in files.items():
                if (not name.startswith('subnet/') or name.count('/')!=1 or not name.endswith('.py')
                        or (root/name).is_symlink() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest):
                    raise ValueError('qualified evaluator local source hash')
            if set(row['runtime_versions'])!={'torch','transformers','toploc'}:
                raise ValueError('qualified evaluator runtime versions')
            host=(endpoint['host'],endpoint['port'],endpoint.get('user','root'),endpoint['workspace'])
            if physical is None:physical=host
            elif host!=physical:raise ValueError('source routes must preserve one physical evaluator and workspace')
            endpoint['retain_original_jobs']=True
        self.state=controller.state/'roles'
    def instance(self,sha):
        if sha not in self.rows:raise ValueError('unapproved original evaluation source')
        if sha not in self.instances:
            row=self.rows[sha];remote=self.factory(row,self.controller)
            if remote.metadata!={'source_files':row['source_files'],'runtime_versions':row['runtime_versions']}:
                raise ValueError('qualified evaluator actual metadata mismatch')
            self.instances[sha]=remote
        return self.instances[sha]
    def original(self,path):
        from .backend_jobs import signed
        prior=json.loads(path.read_bytes())
        job=signed(json.loads((self.state/(prior['job_id']+'-job.json')).read_bytes()),self.controller.authority.id)
        if (job['role']!='evaluate' or job['job_id']!=prior['job_id']
                or hashlib.sha256(canonical(job)).hexdigest()!=prior['job_sha256']):
            raise ValueError('original evaluator role ledger binding')
        return prior,job,signed(job['manifest'],self.controller.authority.id)
    def busy(self):
        # Probe only, using any qualified CPU endpoint. An old unknown source
        # does not permit bypassing its live physical reservation.
        probe=self.instance(next(iter(self.rows)));seen=set()
        for path in self.state.glob('*.json'):
            try:prior=json.loads(path.read_bytes())
            except ValueError:continue
            if prior.get('role')!='evaluate' or 'job_id'not in prior or 'job_sha256'not in prior:continue
            self.original(path)
            if prior['job_id']in seen:continue
            seen.add(prior['job_id'])
            phase=probe.remote_status(prior['job_id'],timeout=30,physical=True)['phase']
            if phase=='running':return True
            if phase not in ('complete','failed','not_launched'):raise ValueError('unknown original evaluator physical liveness')
        return False
    def dispatch_eligible(self,request):
        sha=request['manifest'].get('source_bundle',{}).get('sha256')
        if sha not in self.rows:raise ValueError('unapproved original evaluation source')
        record=self.state/(request['label']+'.json')
        if record.exists():
            self.original(record)
            return True  # Observe the same issued original, including terminal reports.
        return self.rows[sha]['new_dispatch_approved']

    def run(self,label,role,manifest,cache=None,**fields):
        if role!='evaluate':raise ValueError('source router only dispatches evaluation')
        sha=manifest.get('source_bundle',{}).get('sha256')
        if sha not in self.rows:raise ValueError('unapproved original evaluation source')
        row=self.rows[sha];record=self.state/(label+'.json')
        if record.exists():
            _,job,original=self.original(record)
            if original!=manifest or job['source_files']!=row['source_files'] or job['runtime_versions']!=row['runtime_versions']:
                raise ValueError('original evaluation source route changed')
            if any(job.get(key)!=value for key,value in fields.items()):
                raise ValueError('original evaluation source route changed')
        elif not row['new_dispatch_approved']:
            raise ValueError('evaluation source not approved for new GPU dispatch')
        elif self.busy():
            raise RemoteObservationTimeout(label,role)
        remote=self.instance(sha)
        if not record.exists() and 'evaluation_min_free_disk_bytes'in row['endpoint']:
            import shlex
            minimum=row['endpoint']['evaluation_min_free_disk_bytes']
            if type(minimum)is not int or minimum<=0:raise ValueError('signed evaluator disk admission')
            script='import json,shutil;print(json.dumps({"free":shutil.disk_usage('+repr(row['endpoint']['workspace'])+').free}))'
            capacity=json.loads(remote.command(shlex.quote(remote.python)+' -I -B -c '+shlex.quote(script),timeout=30))
            if type(capacity.get('free'))is not int or capacity['free']<minimum:
                raise OSError('independent evaluator disk admission deferred')
        cache=row['endpoint'].get('checkpoint_caches',{}).get(manifest['checkpoint']['id'])
        # Queue cache hints may name a trainer filesystem; only this evaluator's
        # explicitly approved map can cross the physical-role boundary. With
        # no map, sealed checkpoint() uses the owned, fully authenticated cache.
        try:return remote.run(label,role,manifest,cache,**fields)
        except remote.observation_timeout_type as error:
            raise RemoteObservationTimeout(error.job_id,error.role)from error

def evaluation_mode(config):
    mode=config.get('evaluation_mode','synchronous-v1')
    if mode not in ('synchronous-v1',VERSION):raise ValueError('evaluation mode')
    return mode

def fingerprint(manifest,config,plan):
    """Checkpoint, cohort and runtime identity; epoch/URLs/step labels are not identity."""
    bundle=manifest.get('source_bundle')
    source_identity=({k:bundle.get(k) for k in ('sha256','format')}
                     if isinstance(bundle,dict) else bundle)
    return hashlib.sha256(canonical(dict(version=VERSION,
        checkpoint=dict(id=manifest['checkpoint']['id'],files=manifest['checkpoint'].get('files')),
        heldout=plan,environments=[dict(env_id=r['env_id'],spec=r['spec']) for r in manifest['environments'] if any(v['env_id']==r['env_id'] for v in plan)],
        runtime={k:manifest.get(k) for k in ('model_runtime_revision','backend_profile','numerical_policy','harness_source_hash')},
        source_bundle=source_identity,model=config.get('model_id','HuggingFaceTB/SmolLM2-1.7B-Instruct'),
        experiment_id=config.get('evaluation_experiment_id','gpu-continuous-fixed128'),
        evaluation_seed=config.get('evaluation_seed',20260930),**({'trusted_evaluation_policy':config['trusted_evaluation_policy']}if 'trusted_evaluation_policy'in config else {})))).hexdigest()

def historical_request(controller,identity,config,plan):
    """Reference only a locally retained authenticated completed original job.

    Source/runtime/cohort differences fail matching. No status probe, dispatch,
    new signature or invented training counter is performed during this scan.
    """
    from .backend_jobs import signed
    jobs=getattr(controller.jobs,'roles',{}).get('evaluate',controller.jobs)
    checker=getattr(jobs,'checked',None)
    if checker is None:return None
    for path in (controller.state/'roles').glob('*-eval-*.json'):
        try:
            prior=json.loads(path.read_text())
            if prior.get('role')!='evaluate' or not path.stem.endswith(('-eval-before','-eval-after')):continue
            reportpath=path.parent/(prior['job_id']+'-report.json')
            if not reportpath.exists():continue
            job=signed(json.loads((path.parent/(prior['job_id']+'-job.json')).read_text()),controller.authority.id)
            manifest=signed(job['manifest'],controller.authority.id)
            if job.get('trusted_evaluation_policy')!=config.get('trusted_evaluation_policy'):continue
            if job.get('heldout')!=plan or fingerprint(manifest,config,plan)!=identity:continue
            report=checker(json.loads(reportpath.read_text()),prior,manifest)
            label=path.stem;phase=label.rsplit('-eval-',1)[1]
            records=[json.loads((Path(config.get('evaluation_state','state/evaluations'))/(label+'-'+suite['env_id']+'.json')).read_text()) for suite in plan]
            if any(r.get('remote_job_id')!=report['job_id'] or r.get('checkpoint')!=manifest['checkpoint']['id'] or r.get('timestamp')!=report['completed_at'] for r in records):continue
            steps={r['training_steps'] for r in records}
            if len(steps)!=1:continue
            counters={r.get('public_optimizer_steps') for r in records}
            if len(counters)!=1:continue
            return dict(version=VERSION,label=label,manifest=manifest,cache=None,phase=phase,
                        training_steps=steps.pop(),public_optimizer_steps=counters.pop(),
                        config={k:config[k] for k in CONFIG_FIELDS if k in config},heldout_plan=plan)
        except (ValueError,KeyError,TypeError,OSError):continue
    return None

def enqueue(controller,manifest,cache,phase,steps,config,*,public_optimizer_steps=None):
    from .gpu_service import heldout
    if evaluation_mode(config)!=VERSION:raise ValueError('independent evaluation admission')
    # Validate the complete fixed plan before the controller advances. No
    # filtering, fresh seeds or relabeling of the configured comparison cohort.
    plan=heldout(config,manifest)
    if not plan or phase not in ('before','after') or type(steps) is not int or steps<0:
        raise ValueError('bounded checkpoint evaluation request')
    if public_optimizer_steps is not None and (type(public_optimizer_steps) is not int or public_optimizer_steps<0):
        raise ValueError('public optimizer counter')
    label=manifest['epoch']+'-eval-'+phase
    selected={k:config[k] for k in CONFIG_FIELDS if k in config}
    request=dict(version=VERSION,label=label,manifest=manifest,cache=cache,
                 phase=phase,training_steps=steps,public_optimizer_steps=public_optimizer_steps,config=selected,heldout_plan=plan)
    digest=hashlib.sha256(canonical(request)).hexdigest()
    identity=fingerprint(manifest,config,plan)
    path=controller.state/'checkpoint-evaluations'/(identity+'.json')
    reference=dict(version=VERSION,evaluation_id=identity,original_label=label,requested_epoch=manifest['epoch'],phase=phase,checkpoint=manifest['checkpoint']['id'])
    if path.exists():
        previous=json.loads(path.read_text())
        original=previous['request']
        if (hashlib.sha256(canonical(original)).hexdigest()!=previous['request_sha256'] or
                fingerprint(original['manifest'],original['config'],original['heldout_plan'])!=identity):
            raise ValueError('immutable checkpoint evaluation request changed')
        reference.update(original_label=original['label'],original_training_steps=original['training_steps'],reuses_original_execution=True)
        save(controller.state/'checkpoint-evaluation-references'/(label+'.json'),reference)
        return previous
    historical=historical_request(controller,identity,config,plan)
    if historical is not None:
        request=historical;digest=hashlib.sha256(canonical(request)).hexdigest()
        reference.update(original_label=request['label'],original_training_steps=request['training_steps'],reuses_original_execution=True)
    record=dict(evaluation_id=identity,request=request,request_sha256=digest,queued_at=time.time(),status='queued')
    save(controller.state/'checkpoint-evaluation-references'/(label+'.json'),reference)
    save(path,record)
    return record

def evaluate_one(controller,path):
    from .gpu_service import heldout,evaluate
    path=Path(path);record=json.loads(path.read_text());request=record['request']
    if (request.get('version')!=VERSION or
            hashlib.sha256(canonical(request)).hexdigest()!=record['request_sha256'] or
            heldout(request['config'],request['manifest'])!=request['heldout_plan']):
        raise ValueError('checkpoint evaluation queue binding')
    if record['status'] in ('complete','failed','unresolved'):return record
    try:
        records=evaluate(controller,request['manifest'],request['cache'],
                         request['phase'],request['training_steps'],request['config'],label_override=request['label'])
    except RemoteObservationTimeout as error:
        # RemoteJobs persists and reuses the SAME signed job and original
        # expiry on the next pass. Never generate replacement evaluation.
        record.update(status='observing_original_job',remote_job_id=error.job_id,
                      last_observed_at=time.time())
        save(path,record)
        return record
    for result in records:
        result['public_optimizer_steps']=request['public_optimizer_steps']
        save(Path(request['config'].get('evaluation_state','state/evaluations'))/(result['run_id']+'.json'),result)
    record.update(status='complete',records=records,
                  completed_at=max(r['timestamp'] for r in records))
    public=dict(version=VERSION,epoch=request['manifest']['epoch'],checkpoint=request['manifest']['checkpoint']['id'],
                request_sha256=record['request_sha256'],phase=request['phase'],training_steps=request['training_steps'],
                public_optimizer_steps=request['public_optimizer_steps'],status=record['status'],
                records=records,queued_at=record['queued_at'],completed_at=record['completed_at'])
    controller.bucket.json('public/'+request['manifest']['epoch']+'/evaluation-'+request['phase']+'.json',controller.signed(public))
    save(path,record)
    return record

def pending_pass(controller,now=None,*,dispatch_order=None):
    """Advance terminal bad requests; never run another job while GPU liveness is unknown."""
    now=time.time() if now is None else now
    if dispatch_order not in (None,'latest-approved-source-first-v1'):raise ValueError('independent evaluation dispatch order')
    files=list((controller.state/'checkpoint-evaluations').glob('*.json'))
    def order(path):
        try:
            record=json.loads(path.read_text());queued=record.get('queued_at',0)
            if dispatch_order is None:return (0,queued)
            original=controller.state/'roles'/(record['request']['label']+'.json')
            return (0 if original.exists() else 1,-queued)
        except (ValueError,OSError,AttributeError,KeyError):return (-1,0)
    for path in sorted(files,key=order):
        fault=controller.state/'checkpoint-evaluation-faults'/path.name
        if fault.exists() and json.loads(fault.read_text()).get('status') in ('failed','unresolved'):continue
        try:
            record=json.loads(path.read_text())
            if record.get('status') in ('complete','failed','unresolved') or record.get('retry_after',0)>now:continue
            eligible=getattr(controller.jobs,'dispatch_eligible',None)
            if eligible is not None:
                from .gpu_service import heldout
                request=record['request']
                if (request.get('version')!=VERSION or hashlib.sha256(canonical(request)).hexdigest()!=record['request_sha256']or heldout(request['config'],request['manifest'])!=request['heldout_plan']):
                    raise ValueError('checkpoint evaluation queue binding')
                if not eligible(request):
                    save(controller.state/'checkpoint-evaluation-deferrals'/path.name,dict(status='waiting_source_dispatch_approval',original_request_sha256=record['request_sha256'],source_sha256=request['manifest']['source_bundle']['sha256'],observed_at=now,remote_job_started=False))
                    continue
            result=evaluate_one(controller,path)
            if result['status']=='observing_original_job':return result
            return result
        except Exception as error:
            # Check authoritative original role liveness before any other GPU
            # dispatch. A broken SSH probe is not permission to launch another.
            try:busy=controller.jobs.busy()
            except Exception:busy=None
            attempts=(json.loads(fault.read_text()).get('attempts',0) if fault.exists() else 0)+1
            terminal=isinstance(error,(ValueError,KeyError,TypeError,AttributeError)) or getattr(error,'terminal_job',False)
            status='failed' if terminal and busy is False else ('unresolved' if attempts>=8 and busy is False else 'retry_original_request')
            evidence=dict(status=status,error_type=type(error).__name__,attempts=attempts,
                          original_request=path.name,gpu_busy=busy,time=now,retry_after=now+min(300,10*2**min(attempts,5)))
            save(fault,evidence)
            if busy is not False:return evidence
            if status=='retry_original_request':
                # Retry same request later without monopolizing a known-idle GPU.
                try:
                    record=json.loads(path.read_text());record['retry_after']=evidence['retry_after'];save(path,record)
                except (ValueError,OSError):pass
            # A terminal corrupt/failed request remains forensic evidence and
            # never receives a fabricated completed result or zero model reward.
            continue
    return None

def progress(state):
    """Latest training and evaluation are independent, explicitly labeled."""
    state=Path(state);status=json.loads((state/'controller.json').read_text())
    rows=[];corrupt=0
    for path in (state/'checkpoint-evaluations').glob('*.json'):
        try:
            row=json.loads(path.read_text())
            if not isinstance(row,dict) or 'status' not in row or 'request' not in row:
                raise ValueError('malformed evaluation queue')
            rows.append(row)
        except (ValueError,OSError):corrupt+=1
    faults=[json.loads(p.read_text()) for p in (state/'checkpoint-evaluation-faults').glob('*.json')]
    completed=[r for r in rows if r['status']=='complete' and all(v['status']=='complete' for v in r['records'])]
    latest=max(completed,key=lambda r:(r['request']['training_steps'],r['completed_at'])) if completed else None
    return dict(version=VERSION,latest_training_checkpoint=status['checkpoint']['id'],
                latest_evaluated_checkpoint=latest['request']['manifest']['checkpoint']['id'] if latest else None,
                latest_evaluation_report_ids=[r['run_id'] for r in latest['records']] if latest else [],
                pending_checkpoints=sum(r['status']!='complete' for r in rows)+corrupt,
                unresolved_requests=sum(r['status'] in ('failed','unresolved') for r in faults),
                public_optimizer_steps=status.get('public_optimizer_steps'),
                evaluation_caught_up=bool(latest and latest['request']['manifest']['checkpoint']['id']==status['checkpoint']['id']))

def run(config,once=False):
    from .remote_backend import RemoteJobs
    from .controller import Controller
    from .storage import Bucket,Gateway
    if evaluation_mode(config)!=VERSION:raise ValueError('independent evaluator configuration')
    if config.get('preparation_only',False) or not config.get('activation_allowed',True):
        raise ValueError('evaluator activation is not allowed')
    state=Path(config['state']);state.mkdir(parents=True,exist_ok=True)
    # A single evaluator process owns one physical role. A second service fails
    # closed instead of racing launches or overwriting progress receipts.
    with (state/'checkpoint-evaluator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bucket=Bucket(config['bucket']);gateway=Gateway(bucket,state_path=state/'gateway.json',public_url='http://unused-gpu-operator.invalid',direct_r2=True)
        if not (state/'authority.seed').exists():raise ValueError('existing controller authority required')
        controller=Controller(bucket,gateway,state)
        remote=config['remote']
        endpoint=dict(remote.get('roles',{}).get('evaluate',remote),
                      job_ttl_seconds_by_role=remote.get('job_ttl_seconds_by_role',{}),retain_original_jobs=True)
        class EvaluationJobs:
            def busy(self):
                for path in remote_jobs.state.glob('*.json'):
                    try:prior=json.loads(path.read_text())
                    except ValueError:continue
                    if prior.get('role')!='evaluate' or 'job_id' not in prior or 'job_sha256' not in prior:continue
                    if remote_jobs.remote_status(prior['job_id'])['phase']=='running':return True
                return False
            def run(self,label,role,manifest,cache=None,**fields):
                if role!='evaluate':raise ValueError('independent evaluator cannot dispatch other roles')
                local=endpoint.get('checkpoint_caches',{}).get(manifest['checkpoint']['id']) if 'roles' in remote else cache
                return remote_jobs.run(label,role,manifest,local,**fields)
        if config.get('evaluation_source_routes')is not None:
            controller.jobs=QualifiedEvaluationJobs(controller,config['evaluation_source_routes'])
        else:
            remote_jobs=RemoteJobs(endpoint,controller)
            controller.jobs=EvaluationJobs()
        while True:
            pending_pass(controller,dispatch_order=config.get('evaluation_dispatch_order'))
            value=progress(state);save(state/'checkpoint-evaluation-progress.json',value)
            bucket.json('public/streams/'+config['epoch_prefix']+'/evaluation-progress.json',controller.signed(value))
            if once:return
            time.sleep(10)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true')
    args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.once)

if __name__=='__main__':main()
