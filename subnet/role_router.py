"""Separate-host dispatch with an operator-side atomic verifier queue."""
import hashlib
import json
import secrets
import shlex
import threading
import time
from pathlib import Path
from .distributed_roles import Coordinator, CoordinatorServer
from .storage import canonical


class RoutedJobs:
    def __init__(self, config, controller):
        from .remote_backend import RemoteJobs,save
        self.config=config; self.controller=controller; self.state=controller.state/'roles'
        self.state.mkdir(exist_ok=True); self.roles={}
        endpoints=config['roles']
        if not {'mine','train','evaluate','verify'} <= set(endpoints) or len(endpoints['verify']) < 2:
            raise ValueError('separate miner/trainer/evaluator and two verifiers required')
        hosts=[]
        for role in ('mine','train','evaluate'):
            self.roles[role]=RemoteJobs({'job_ttl_seconds_by_role':config.get('job_ttl_seconds_by_role',{}),**endpoints[role]},controller)
            hosts.append((endpoints[role]['host'],endpoints[role]['port']))
        self.verifiers=[RemoteJobs(endpoint,controller) for endpoint in endpoints['verify']]
        hosts.extend((endpoint['host'],endpoint['port']) for endpoint in endpoints['verify'])
        if len(set(hosts)) != len(hosts): raise ValueError('separate role endpoints required')
        self.metadata=self.verifiers[0].metadata
        if any(worker.metadata != self.metadata for worker in self.verifiers[1:]):
            raise ValueError('verifier source/runtime pins must agree')
        worker_ids=[endpoint['worker_identity'] for endpoint in endpoints['verify']]
        if len(set(worker_ids)) != len(worker_ids): raise ValueError('distinct verifier identities required')
        q=config['verifier_queue']; workers={identity:['verify'] for identity in worker_ids}
        self.queue=Coordinator(self.state/'verifier-queue.sqlite3',controller.authority.id,workers,
                               lease_seconds=q.get('lease_seconds',300),max_attempts=q.get('max_attempts',3))
        self.server=CoordinatorServer((q.get('host','127.0.0.1'),q['port']),self.queue,controller.signed)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True); self.thread.start()
        self.owner_path=self.state/'checkpoint-owners.json'
        self.owners=json.loads(self.owner_path.read_text()) if self.owner_path.exists() else {}
        self.cache_lock=threading.RLock()
        self.cache_path=self.state/'role-checkpoint-caches.json'
        self.caches=json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {role:dict(endpoint.get('checkpoint_caches',{})) for role,endpoint in endpoints.items() if role!='verify'}
        self.initial_role=config.get('initial_checkpoint_role','mine')
        if self.initial_role not in self.roles: raise ValueError('initial checkpoint role')
        self.history_prefix=q.get('history_prefix','private/distributed-role-history')
        if not self.history_prefix.startswith('private/'): raise ValueError('job capability history must remain private')

    def checkpoint_path(self, role, checkpoint):
        return self.roles[role].workspace+'/checkpoints/'+checkpoint

    def capacity(self, cache):
        # Inspect the actual owning host. Other roles do not inherit this path.
        owner=self.owners.get(cache,self.initial_role)
        source=self.roles[owner].capacity(cache)
        if owner=='train':return source
        trainer=self.roles['train']
        code="import json,os;from pathlib import Path;p=Path("+repr(trainer.workspace)+");p.mkdir(parents=True,exist_ok=True);s=os.statvfs(p);print(json.dumps(dict(free_bytes=s.f_bavail*s.f_frsize)))"
        target=json.loads(trainer.command(shlex.quote(trainer.python)+' -c '+shlex.quote(code)))
        if target['free_bytes']<source['required_bytes']:raise ValueError('trainer checkpoint download/output disk reserve')
        return dict(source,trainer_capacity=target)

    def publication_capacity(self,cache):
        return self.roles[self.owners.get(cache,self.initial_role)].publication_capacity(cache)

    def retire_training_cache(self,job,report,pointer):
        from .backend_jobs import signed
        from .remote_backend import save
        manifest=signed(job['manifest'],self.controller.authority.id)
        result=self.roles['train'].retire_training_cache(job,report,pointer,
            self.caches.get('train',{}).get(manifest['checkpoint']['id']))
        with self.cache_lock:
            removed=set(result.get('removed_checkpoints',[]));train=self.caches.setdefault('train',{})
            for cp in removed:
                path=train.pop(cp,None)
                if path and self.owners.get(path)=='train':self.owners.pop(path,None)
            if removed:save(self.cache_path,self.caches);save(self.owner_path,self.owners)
        return result

    def training_resume(self,label,manifest,submissions,steps,replay):
        return self.roles['train'].training_resume(label,manifest,submissions,steps,replay)

    def training_capacity(self,manifest,steps,submission_bytes=None):
        """Free bytes already account for installed packages and existing caches.

        Legacy full training retains one complete checkpoint per optimizer step
        plus its final export. Covered training saves one final checkpoint and
        reserves an additional full temporary export. Count missing input caches.
        Signed compressed/raw artifact budgets cover download and decoding room.
        """
        from .artifact_budget import for_manifest
        if type(steps) is not int or not 1<=steps<=32:raise ValueError('training capacity steps')
        checkpoint=manifest['checkpoint']['id'];trainer=self.roles['train']
        cache=self.caches.get('train',{}).get(checkpoint)
        source=self.roles[self.initial_role]
        source_path=self.caches.get(self.initial_role,{}).get(checkpoint) or self.checkpoint_path(self.initial_role,checkpoint)
        from .persistent_cpu_adamw import POLICY as PERSISTENT_POLICY
        if manifest.get('training_policy')==PERSISTENT_POLICY:
            from .persistent_training_worker import capacity_requirement
            code="import json;from subnet.persistent_training_worker import capacity_probe;print(json.dumps(capacity_probe("+repr(trainer.workspace)+","+repr(cache)+")))"
            probe=json.loads(trainer.command('cd '+shlex.quote(trainer.code)+' && '+shlex.quote(trainer.python)+' -I -B -c '+shlex.quote("import sys;sys.path.insert(0,"+repr(trainer.code)+");"+code)))
            if cache:checkpoint_bytes=probe['checkpoint_bytes']
            else:
                owner=self.owners.get(source_path,self.initial_role)
                measured=self.roles[owner].publication_capacity(source_path)
                checkpoint_bytes=measured['checkpoint_bytes']
            return capacity_requirement(manifest,probe,checkpoint_bytes=checkpoint_bytes,missing_input=cache is None)
        if cache:
            measured=trainer.capacity(cache);checkpoint_bytes=measured['checkpoint_bytes'];free=measured['free_bytes']
        else:
            checkpoint_bytes=source.capacity(source_path)['checkpoint_bytes']
            code="import json,os;from pathlib import Path;p=Path("+repr(trainer.workspace)+");p.mkdir(parents=True,exist_ok=True);s=os.statvfs(p);print(json.dumps(dict(free_bytes=s.f_bavail*s.f_frsize)))"
            free=json.loads(trainer.command(shlex.quote(trainer.python)+' -c '+shlex.quote(code)))['free_bytes']
        budget=for_manifest(manifest)
        compact_input=(manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2')
        if compact_input:
            from .compact_training_inputs import MAX_BYTES
            budget=dict(compressed_bytes=MAX_BYTES,raw_bytes=MAX_BYTES)
        if submission_bytes is not None and (type(submission_bytes) is not int or not 0<submission_bytes<=256*budget['compressed_bytes']):
            raise ValueError('planned training submission bytes')
        # Download ZIPs are retained for the entire job. Raw tensors are decoded
        # one submission at a time, so retain one raw-artifact working reserve.
        downloads=max(budget['compressed_bytes'],submission_bytes or 0)
        from .backend_jobs import COVERED_POLICY
        covered=manifest.get('training_policy')==COVERED_POLICY
        retained_steps=0 if covered else steps
        temporary_exports=1 if covered else 0
        required=checkpoint_bytes*(retained_steps+temporary_exports+1+(0 if cache else 1))+downloads+budget['raw_bytes']+2*1024**3
        if free<required:raise ValueError('trainer input/snapshot/export/artifact disk reserve')
        return dict(free_bytes=free,checkpoint_bytes=checkpoint_bytes,required_bytes=required,input_cache=bool(cache),retained_step_checkpoints=retained_steps,final_exports=1,temporary_export_copies=temporary_exports,
                    planned_submission_bytes=submission_bytes,download_reserve_bytes=downloads,raw_working_reserve_bytes=budget['raw_bytes'])

    def run(self,label,role,manifest,cache=None,observe_until=None,dispatch_only=False,**fields):
        from .remote_backend import save,role_time_budget
        if type(dispatch_only)is not bool or dispatch_only and (role!='mine' or manifest.get('hourly_execution_policy')is None):raise ValueError('dispatch-only requires signed hourly miner')
        if manifest.get('payable') is not False: raise ValueError('distributed nonpayable only')
        if role != 'verify':
            selected=self.owners.get(cache,self.initial_role) if role=='upload' else role
            if selected not in self.roles: raise ValueError('distributed role')
            # Only upload reads an explicitly owned local checkpoint. Every
            # compute role resolves the exact signed R2 map on its own host.
            local_cache=cache if role=='upload' else self.caches.get(selected,{}).get(manifest['checkpoint']['id'])
            kwargs=dict(fields)
            if dispatch_only:kwargs['dispatch_only']=True
            report=self.roles[selected].run(label,role,manifest,local_cache,**kwargs)
            with self.cache_lock:
                if role!='upload':
                    self.caches.setdefault(selected,{})[manifest['checkpoint']['id']]=local_cache or self.checkpoint_path(selected,manifest['checkpoint']['id'])
                    save(self.cache_path,self.caches)
                if role=='train':
                    self.owners[report['new_checkpoint']['path']]='train'; save(self.owner_path,self.owners)
                    self.caches.setdefault('train',{})[report['new_checkpoint']['id']]=report['new_checkpoint']['path'];save(self.cache_path,self.caches)
            return report
        if not label.replace('-','').replace('_','').isalnum(): raise ValueError('job label')
        record=self.state/(label+'.json')
        if record.exists():
            prior=json.loads(record.read_text())
            if prior['manifest_sha256'] != hashlib.sha256(canonical(manifest)).hexdigest(): raise ValueError('immutable queue manifest')
            envelope=json.loads((self.state/(prior['job_id']+'-job.json')).read_text())
            original=envelope['payload']
            if original.get('role')!=role or [r['sha256'] for r in original.get('submissions',[])] != [r['sha256'] for r in fields.get('submissions',[])]:
                raise ValueError('immutable queue role/frozen inputs')
            if manifest.get('submission_transport_policy') is not None:
                fields_without_url=lambda rows:[{k:v for k,v in r.items()if k!='url'}for r in rows]
                if fields_without_url(original['submissions'])!=fields_without_url(fields.get('submissions',[])):
                    raise ValueError('immutable selected child metadata')
        else:
            now=time.time(); identifier=label+'-'+secrets.token_hex(4)
            payload=dict(schema=1,job_id=identifier,role=role,created_at=now,
                         expires_at=now+role_time_budget(self.config,role),manifest=self.controller.signed(manifest),**self.metadata,**fields)
            envelope=self.controller.signed(payload)
            prior=dict(job_id=identifier,role=role,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],
                       job_sha256=hashlib.sha256(canonical(payload)).hexdigest(),manifest_sha256=hashlib.sha256(canonical(manifest)).hexdigest(),**self.metadata)
            save(self.state/(identifier+'-job.json'),envelope); save(record,prior)
        if observe_until is not None and time.time()>=observe_until:raise TimeoutError('audit budget closed; original request retained')
        self.queue.enqueue(envelope); self.queue.archive(prior['job_id'],self.controller.bucket,self.history_prefix)
        started=time.time()
        while True:
            if observe_until is not None and time.time()>=observe_until:raise TimeoutError('audit budget closed; original lease retained')
            status=self.queue.status(prior['job_id'])
            if status['status']=='complete':
                report=status['report']; self.verifiers[0].checked(report,prior,manifest)
                self.queue.archive(prior['job_id'],self.controller.bucket,self.history_prefix)
                save(self.state/(prior['job_id']+'-report.json'),report)
                return report
            if status['status'] in ('failed','expired') or time.time()>=envelope['payload']['expires_at']:
                raise RuntimeError('verifier job exhausted or expired; original commitment retained')
            if time.time()-started>1800: raise TimeoutError('verifier job observation timeout; original lease retained')
            time.sleep(2)

    def stop(self):
        self.server.shutdown(); self.server.server_close()
