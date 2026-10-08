"""Signed trainer-only storage cutover atop an unchanged admitted learner.

The inference source, miners, verifiers, evaluator and weight writer retain
their contracts. A separately pinned trainer execution tree owns Adam state.
"""
import argparse
import hashlib
import json
import shlex
from pathlib import Path
import sys

VERSION='trainer-local-state-learner-service-v1'


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    root=Path(root)
    if root!=root.resolve() or root.is_symlink():raise ValueError('ordinary pinned source root')
    rows={}
    for p in root.rglob('*'):
        if p.is_symlink():raise ValueError('source symlink')
        if p.is_file():rows[str(p.relative_to(root))]=file_hash(p)
    return rows


def main():
    a=argparse.ArgumentParser();a.add_argument('--policy',required=True);a.add_argument('--check',action='store_true');args=a.parse_args()
    document=json.loads(Path(args.policy).read_bytes());p=document['payload']
    runtime=Path(p['baseline_runtime']);sys.path.insert(0,str(runtime))
    from ops import durable_learner_service as baseline
    guards=baseline.guards
    p=guards.signed(document,guards.AUTHORITY)
    if p['version']!=VERSION or file_hash(__file__)!=p['runner_sha256']:raise ValueError('signed local trainer service')
    original=guards.read(p['baseline_policy']['path'])
    if file_hash(p['baseline_policy']['path'])!=p['baseline_policy']['sha256']:raise ValueError('baseline policy drift')
    # Composite admission imports only the baseline's pinned scalar schema.
    signed_original=guards.signed(original,guards.AUTHORITY)
    sys.path.insert(1,signed_original['source_root'])
    old=baseline.validate_policy(original)
    cfg=guards.read(p['config']['path'])
    if file_hash(p['config']['path'])!=p['config']['sha256']:raise ValueError('local trainer config drift')
    original_cfg=guards.read(old['config']['path'])
    # The new config changes only the trainer endpoint/code/peer and explicit
    # witnessed recovery mapping. Never alter inference, rewards or chain flags.
    normalized=json.loads(json.dumps(cfg))
    normalized['remote']['roles']['train']=original_cfg['remote']['roles']['train']
    normalized['remote'].pop('training_startup_recovery_files',None)
    if cfg['remote'].get('publication_manifest_projection')!={'version':'train-recovery-publication-projection-v1'}:
        raise ValueError('explicit model-only recovery publication scope')
    if 'publication_manifest_projection' in original_cfg['remote']:
        normalized['remote']['publication_manifest_projection']=original_cfg['remote']['publication_manifest_projection']
    else:normalized['remote'].pop('publication_manifest_projection',None)
    original_normalized=json.loads(json.dumps(original_cfg));original_normalized['remote'].pop('training_startup_recovery_files',None)
    if normalized!=original_normalized:raise ValueError('trainer-only configuration scope')
    train=cfg['remote']['roles']['train']
    if train.get('optimizer_state_lifecycle')!='trainer-local-only-v1':raise ValueError('explicit local trainer lifecycle')
    if any(train[k]!=original_cfg['remote']['roles']['train'][k]for k in ('host','port','user','workspace','python','known_hosts')):
        raise ValueError('retain actual trainer host and optimizer workspace')
    for field in ('cpu_overlay','trainer_source'):
        if inventory(p[field]['root'])!=p[field]['files']:raise ValueError('complete pinned '+field+' inventory')
    old_science=inventory(old['source_root'])
    protected=['model','gpu_runtime','proofs','forced_sampling','sampling_uniqueness','harness','environments',
               'task_normalized_training','persistent_cpu_adamw','epoch_optimizer','covered_epoch_optimizer']
    if any(p['trainer_source']['files'].get('subnet/'+n+'.py')!=old_science.get('subnet/'+n+'.py')for n in protected):
        raise ValueError('storage cutover cannot change sampling, model or optimizer mathematics')
    peers=p['peer_admissions']
    for source,peer in peers.items():
        grant=guards.signed(peer['authorization_document'],old['authority'])
        if grant['source_sha256']!=source or grant['scientific_source_files']!=p['trainer_runtime_files'] or not grant['backend_execution_allowed']:
            raise ValueError('explicit trainer execution and CPU peer grant')
    changed=json.loads(json.dumps(old));changed['operator_overlay']['root']=p['cpu_overlay']['root']
    changed['native_training_eligibility']['authorization']=p['native_authorization']
    with guards.singleton(old['singleton_lock']):
        guards.no_predecessors(old)
        service=baseline.prepare_runtime(changed)
        from subnet import remote_backend
        qualified=remote_backend.RemoteJobs;base=qualified.__bases__[0]
        expected=p['trainer_runtime_files']
        class LocalTrainerJobs(qualified):
            def __init__(self,config,controller):
                if config.get('optimizer_state_lifecycle')!='trainer-local-only-v1':return super().__init__(config,controller)
                base.__init__(self,config,controller)
                full={n:h for n,h in p['trainer_source']['files'].items()if n.startswith('subnet/')and n.endswith('.py')and '/'not in n[len('subnet/'): ]}
                if self.metadata['source_files']!=full:raise ValueError('actual separately pinned trainer source inventory')
                self.metadata=dict(self.metadata,source_files=expected)
                if config.get('optimizer_cache_volume')=='trainer-local-memory-v1':
                    helper='ops/trainer_local_cache_volume.py'
                    code='import hashlib,pathlib,runpy,sys; p=pathlib.Path('+repr(self.code+'/'+helper)+'); assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(p['trainer_source']['files'][helper])+'; sys.argv=[str(p),"--workspace",'+repr(self.workspace)+',"--authority",'+repr(old['authority'])+']; sys.path.insert(0,'+repr(self.code)+'); runpy.run_path(str(p),run_name="__main__")'
                    result=json.loads(self.command(shlex.quote(self.python)+' -I -B -c '+shlex.quote(code),timeout=900))
                    if result.get('ready')is not True or result.get('optimizer_bytes_uploaded')!=0:
                        raise ValueError('acknowledged local-only trainer working volume')
            def run(self,label,role,manifest,*args,**kwargs):
                if role=='train' and self.config.get('optimizer_state_lifecycle')=='trainer-local-only-v1':
                    peer=peers.get(manifest['source_bundle']['sha256'])
                    if peer is None:raise ValueError('approved trainer peer source required')
                    self.config['learner_selection_cpu_peer']=peer
                return super().run(label,role,manifest,*args,**kwargs)
        remote_backend.RemoteJobs=LocalTrainerJobs
        if args.check:
            print(json.dumps(dict(checked=True,optimizer_uploads=False,inference_contract_unchanged=True,jobs_dispatched=False)))
            return
        service._native_lifecycle_execute=True
        service.run(cfg)


if __name__=='__main__':main()
