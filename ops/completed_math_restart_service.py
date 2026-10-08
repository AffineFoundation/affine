"""ROOT-pinned, prospective completed-answer deployment and explicit base reset.

Historical source admissions are additive. This operator changes neither the
sampler nor optimizer mathematics, and never reissues historical jobs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

VERSION = 'completed-math-explicit-base-restart-service-v1'


def inventory(root):
    root = Path(root)
    if root.resolve() != root or root.is_symlink():
        raise ValueError('ordinary pinned deployment root')
    result = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('deployment symlink')
        if path.is_file():
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def additive(before, after, source):
    if source in before or set(after) != set(before) | {source} or any(after[k] != v for k, v in before.items()):
        raise ValueError('historical admissions must remain exact')


def validate(p, guards):
    if p['version'] != VERSION or p['authority'] != guards.AUTHORITY or p['execute_allowed'] is not True:
        raise ValueError('explicit ROOT completed-math execution policy')
    if guards.file_hash(__file__) != p['runner_sha256']:
        raise ValueError('restart operator drift')
    for row in p['pins']:
        if guards.file_hash(row['path']) != row['sha256']:
            raise ValueError('restart evidence drift')
    cfg = guards.read(p['config']['path'])
    if guards.file_hash(p['config']['path']) != p['config']['sha256']:
        raise ValueError('restart config drift')
    source = p['source_sha256']
    if inventory(p['source_root']) != p['source_files']:
        raise ValueError('complete new scientific deployment')
    old = inventory(p['previous_source_root'])
    allowed = {'subnet/environments.py', 'subnet/committed_training_inputs.py',
               'ops/native_training_outcome_filter.py', 'tests/test_committed_training_inputs.py',
               'tests/test_miner_task_errors.py'}
    additions = {'subnet/math_completion.py', 'tests/test_completed_math_answers.py',
                 'tests/test_native_training_outcome_filter.py', 'docs/COMPLETED_MATH_RESTART.md'}
    if set(p['source_files']) != set(old) | additions or any(old[n] != p['source_files'][n] for n in old if n not in allowed):
        raise ValueError('grading-only source delta; sampler and training objective unchanged')
    guards.validate_queue(p['queue']) if p['kind'] in ('API', 'auditor') else None
    if p['kind'] == 'learner':
        previous=guards.read(p['previous_config']['path'])
        editable={'source_bundle','environments','initial_checkpoint','initial_checkpoint_path','remote',
                  'evaluation_experiment_id','persistent_training_admission','persistent_training_qualification_translation',
                  'learner_blacklist_selection_authorization','continuous_reward_activation_document'}
        if set(cfg)!=set(previous) or any(cfg[k]!=previous[k] for k in cfg if k not in editable):
            raise ValueError('restart cannot change unrelated training, sampling or budgets')
        oldenv=json.loads(json.dumps(previous['environments']));newenv=json.loads(json.dumps(cfg['environments']))
        for row in newenv:
            row['spec']['version']=oldenv[0]['spec']['version']
            row['spec']['source_hash']=oldenv[0]['spec']['source_hash']
            row['spec']['config'].pop('math_outcome_policy',None)
        if newenv!=oldenv:
            raise ValueError('same math dataset, task indices and harness')
        for field in ('cpu_overlay', 'trainer_source'):
            if inventory(p[field]['root']) != p[field]['files']:
                raise ValueError('complete pinned '+field)
        if cfg['source_bundle']['sha256'] != source or cfg['initial_checkpoint']['id'] != p['base_checkpoint']:
            raise ValueError('approved new source and original base checkpoint')
        if cfg['model_id'] != 'Qwen/Qwen2.5-Math-7B-Instruct' or cfg['model_revision'] != 'ef9926d75ab1d54532f6a30dd5e760355eb9aa4d':
            raise ValueError('original pinned 7B base')
        if (cfg['samples_per_batch'] != 8 or cfg['K'] != 4 or cfg['L'] != 4 or cfg['max_batches'] != 3
                or cfg['sampling_policy']['max_attempts'] != 1000
                or cfg['training_input_policy'] != 'committed-unaudited-training-v1'
                or cfg['remote']['roles']['train'].get('optimizer_state_lifecycle') != 'trainer-local-only-v1'):
            raise ValueError('existing quota, draw and training contracts')
        spec = cfg['environments'][0]['spec']
        if (len(cfg['environments']) != 1 or spec['config']['math_outcome_policy'] != 'completed-boxed-math-outcome-v1'
                or spec['version'] != 'prime-v1-2-completed-math'):
            raise ValueError('prospective completed-answer taskset')
        state = guards.read(Path(cfg['state'])/'controller.json')
        if state['round'] < p['first_round']:
            raise ValueError('explicit reset boundary must precede startup')
        if state.get('trainer_state') is None:
            if (state['checkpoint']['id'] != p['base_checkpoint'] or state['round'] != p['first_round']
                    or state.get('persistent_state_committed') or state.get('training_steps') != 0):
                raise ValueError('fresh optimizer genesis only at explicit base boundary')
        else:
            if state['trainer_state']['genesis_sha256'] != cfg['persistent_training_admission']['genesis_sha256']:
                raise ValueError('new run optimizer lineage')
    else:
        oldcfg = guards.read(p['previous_config']['path'])
        if p['kind'] == 'API':
            oldad = guards.signed(guards.read(p['previous_admission']['path']))
            newad = guards.signed(guards.read(p['admission']['path']))
            additive(oldad['sources'], newad['sources'], source)
            row = newad['sources'][source]
            if row['runtime_files'] != p['runtime_files'] or row['sampling_versions'] != ['forced-inverse-cdf-prefill-miner-bound-v5']:
                raise ValueError('new scientific source and unchanged sampler')
            if cfg != oldcfg:
                raise ValueError('API admission-only change')
        else:
            before=oldcfg['continuous_audit_service']['source_admission']['payload']
            after=guards.signed(cfg['continuous_audit_service']['source_admission'])
            for key in ('approved_sources','job_metadata'):
                additive(before[key],after[key],source)
            additive(before['execution_evidence_policy']['sources'], after['execution_evidence_policy']['sources'],source)
            if after['approved_sources'][source] != p['runtime_files']:
                raise ValueError('new audit source closure')
            expected = json.loads(json.dumps(cfg))
            expected['continuous_audit_service']['source_admission'] = oldcfg['continuous_audit_service']['source_admission']
            if expected != oldcfg:
                raise ValueError('audit admission-only change')
    return cfg


def learner(p, guards, baseline):
    for name in list(sys.modules):
        if name == 'subnet' or name.startswith('subnet.'):
            del sys.modules[name]
    root = Path(p['cpu_overlay']['root']); sys.path.insert(0,str(root))
    from subnet import gpu_service, remote_backend
    if Path(gpu_service.__file__).resolve() != root/'subnet/gpu_service.py':
        raise ValueError('new CPU overlay origin')
    baseline.install_native_constructor(gpu_service,p)
    baseline.install_calibration_opening_quota()
    # An already-issued hourly assessment retains its original writer pin.
    # Admit the two exact ROOT-reviewed policies during the hourly handover;
    # every individual opening still binds one original signed assessment.
    from subnet import learner_blacklist_selection as selection
    original_opening=selection.prepare_opening
    def prepare_opening(controller,config,status,contract):
        import subprocess
        refresh=p['training_assessment_refresh']
        if guards.file_hash(refresh['script'])!=refresh['script_sha256']:
            raise ValueError('pinned read-only training assessment refresh')
        subprocess.run([sys.executable,'-I','-B',refresh['script'],'--runtime',refresh['runtime'],
                        '--writer-policy',refresh['writer_policy'],'--authority-seed',refresh['authority_seed'],
                        '--output',refresh['output']],check=True,timeout=900)
        approval=guards.signed(config[selection.AUTHORIZATION_FIELD])
        assessment=guards.signed(guards.read(approval['assessment_path']))
        writer=assessment.get('writer_policy_sha256')
        if writer not in p['approved_writer_policy_hashes']:
            raise ValueError('unapproved writer assessment at restart handover')
        selected=dict(config)
        selected[selection.AUTHORIZATION_FIELD]=controller.signed(dict(approval,writer_policy_sha256=writer))
        return original_opening(controller,selected,status,contract)
    selection.prepare_opening=prepare_opening
    original=remote_backend.RemoteJobs
    class PinnedJobs(original):
        def command(self,command,timeout=1800):
            if not getattr(self,'_bounded_ssh',False):
                self.ssh[1:1]=['-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes',
                              '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3']
                self._bounded_ssh=True
            return super().command(command,timeout=timeout)
        def __init__(self,config,controller):
            super().__init__(config,controller)
            trainer=config.get('optimizer_state_lifecycle')=='trainer-local-only-v1'
            observed=p['trainer_observed_runtime'] if trainer else p['observed_runtime']
            target=p['trainer_runtime_files'] if trainer else p['runtime_files']
            # Old verifiers retain historical source trees; their prospective
            # tree is independently staged and passed in this run's config.
            if self.metadata['source_files'] != observed or self.metadata['runtime_versions'] != p['runtime_versions']:
                raise ValueError('actual remote source/runtime changed')
            self.metadata=dict(source_files=target,runtime_versions=p['runtime_versions'])
        def run(self,label,role,manifest,cache=None,dispatch_only=False,**fields):
            if role=='mine': cache=None
            if role=='train':self.config['learner_selection_cpu_peer']=p['peer']
            return super().run(label,role,manifest,cache,dispatch_only=dispatch_only,**fields)
    remote_backend.RemoteJobs=PinnedJobs
    gpu_service._native_lifecycle_execute=True
    return gpu_service


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',required=True);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    document=json.loads(Path(args.policy).read_bytes());raw=document['payload']
    sys.path.insert(0,raw['baseline_runtime'])
    from ops import durable_audit_services as guards
    p=guards.signed(document);cfg=validate(p,guards)
    with guards.singleton(p['singleton_lock']):
        guards.no_predecessors(p)
        if p['kind']=='learner':
            from ops import durable_learner_service as baseline
            service=learner(p,guards,baseline)
        else:
            old=guards.signed(guards.read(p['baseline_policy']['path']))
            service=guards.prepare_runtime(old)
            if p['kind']=='API':
                from subnet import source_sampling_admission as gate
                admission=gate.SamplingAdmission(guards.read(p['admission']['path']),p['authority'],p['source_trees'])
                service.Coordinator=gate.guarded_coordinator(service.Coordinator.__bases__[0],admission)
        if args.check:
            print(json.dumps(dict(checked=True,kind=p['kind'],source=p['source_sha256'],jobs_dispatched=False)));return
        if p['kind']=='learner':service.run(cfg)
        elif p['kind']=='API':service.main(['--config',p['config']['path'],'--authority-seed',p['authority_seed']['path'],'--expected-authority',p['authority']])
        else:service.main(['--config',p['config']['path']])


if __name__=='__main__':main()
