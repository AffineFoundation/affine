"""Data-only projection of authenticated public MiMo epochs.

Never exports an open-epoch submission, token trace, transport capability, local
path or raw error. Historical incentive data and the existing page stay intact.
The explicit ROOT-signed selector enables this projection at the run handover.
"""
import json
import time
from pathlib import Path

from .run_projection import AUTHORITY, authenticated
from subnet.backend_jobs import signed
from subnet.mimo_release_policy import ReleasePolicy, digest, need


def read(path, limit=256*1024**2):
    need(path.resolve() == path.absolute() and path.is_file() and not path.is_symlink()
         and 0 < path.stat().st_size <= limit, 'ordinary bounded projection evidence')
    return json.loads(path.read_bytes())


def selection(path):
    config = authenticated(Path(path))
    need(set(config) == {'version','state_root','evaluation_directory','release_policy',
                        'first_round','started_at','trainer_workers','evaluator_workers'}
         and config['version'] == 'mimo-public-dashboard-selector-v1', 'explicit dashboard selector')
    need(type(config['first_round']) is int and config['first_round'] > 0
         and type(config['started_at']) in (int,float), 'public run boundary')
    for field in ('state_root', 'evaluation_directory'):
        p=Path(config[field]); need(p.is_absolute() and p.resolve() == p, 'ordinary explicit evidence root')
    for field in ('trainer_workers','evaluator_workers'):
        need(type(config[field]) is list and config[field] and
             all(type(w) is str and len(w) == 64 for w in config[field]), 'explicit role signing keys')
    policy = ReleasePolicy(config['release_policy'], AUTHORITY, allow_public=True)
    need(policy.mode == 'public', 'private diagnostics never replace the public learning run')
    return config, policy


def project(snapshot, config, policy, *, now=None):
    from subnet.mimo_audit_bridge import authenticate_population
    from subnet.mimo_worker import validate_report
    now = time.time() if now is None else now
    authority = policy.authority
    need(policy.mode == 'public', 'public projection requires public release')
    root=Path(config['state_root']); epochs=[]; openings={}; checkpoints={}
    for path in sorted((root/'epochs').glob('*/coordinator/OPENING.json'), key=lambda p:int(p.parent.parent.name)):
        opening=read(path,32*1024**2);m=signed(opening,authority);policy.validate_manifest(m)
        directory=path.parent.parent;start=m['start']
        need(start >= config['started_at'], 'opening belongs to new run boundary')
        number=len(epochs);final_path=directory/'FINALIZED.json'
        row=dict(id=m['epoch'],start=start,deadline=m['deadline'],checkpoint=m['checkpoint']['id'],
            source='live-reward-math',mechanism='mimo-fixed-group',training_run_id=m['run_id'],display_epoch=number+1,
            phase='collecting' if now < m['deadline'] else 'training',finalized=False,
            mode='live',reward_eligible=True,payable=True,k=m['group_contract']['min_success'],
            l=m['group_contract']['min_completed_failure'],group_size=m['group_contract']['group_size'],
            participants=len(m['capabilities']),submissions=0,batches=None,batches_available=False,
            accepted=0,rejected=0,unchecked=0,points=0,grid=[0]*256,grid_outcomes=[],miners={},
            unassigned_batches=0,training=None,audit_breakdown_available=False)
        if final_path.exists():
            final=signed(read(final_path),authority)
            dispatch_envelope=read(path.parent/'DISPATCH.json');dispatch=signed(dispatch_envelope,authority)
            need(final['policy_sha256'] == policy.policy_sha256 and final['epoch'] == m['epoch']
                 and final['opening_sha256'] == digest(opening)
                 and final['dispatch_sha256'] == digest(dispatch_envelope)
                 and final['input_checkpoint'] == m['checkpoint']['id']
                 and final['round'] == config['first_round']+number, 'original finalized epoch evidence')
            pop=authenticate_population(read(directory/'AUDIT-POPULATION.json'),authority,
                                        manifest_guard=policy.validate_manifest)
            need(pop['manifest_document'] == opening and pop['dispatch_document'] == dispatch_envelope,
                 'original public closed population')
            regs=signed(pop['registrations_document'],authority)['registrations']
            uids={v['public_key']:v['uid'] for v in regs.values()}
            grid=[0]*256
            for record in pop['records']:
                uid=uids.get(record['miner'])
                need(type(uid) is int and 0 <= uid < 256, 'original registered contribution UID')
                grid[uid]+=1
            count=len(pop['records']);eligible=len(pop['eligible_evidence_ids'])
            row.update(finalized=True,phase='trained' if final['result']['status']=='updated' else 'no-update',
                batches=count,batches_available=True,unchecked=count,grid=grid,
                submissions=len({r['miner'] for r in pop['records']}),learner_eligible=eligible,
                learner_excluded=count-eligible,learner_input_assurance='sampling-unaudited',
                batch_count_source='authenticated-mimo-frozen-population',
                learner_provenance_sha256=digest(pop),completion_sha256=digest(final))
            outcome=final['result'];cp=outcome['checkpoint']['id']
            training=dict(steps=int(outcome['status']=='updated'),weights_changed=outcome['status']=='updated',
                checkpoint=cp,objective='mimo-reference-grpo-v1',losses=None,input_pairs=None,
                input_groups=dispatch['selected_task_count'],input_rollouts=dispatch['selected_rollout_count'])
            if outcome['status']=='updated':
                job=signed(read(directory/'TRAINING-JOB.json'),authority)
                receipt=read(directory/'TRAINING-RESULT.json')
                need(receipt['signer'] == job['trainer_worker'] and receipt['signer'] in config['trainer_workers'],
                     'original admitted training worker')
                request=signed(receipt,receipt['signer']);report=request['report']
                need(request['action']=='report' and request['job_id']==job['job_id'] and job['manifest']==opening,
                     'training report belongs to this opening')
                validate_report(report,job,m,authority,digest(job),now)
                t=report['training']
                need(t['output_checkpoint']==cp and t['optimizer_step']==outcome['optimizer_step'],
                     'actual trained checkpoint handover')
                training.update(losses=[t['objective_loss']],gradient_norm=t['preclip_gradient_norm'],
                    learning_rate=t['optimizer']['lr'],accumulation_dtype=t['optimizer']['gradient_accumulation_dtype'])
            row['training']=training
            if outcome['status']=='updated':checkpoints.setdefault(cp,row['id'])
        openings[row['id']]=opening;epochs.append(row)
    evaluations=[]
    for path in sorted(Path(config['evaluation_directory']).glob('*.receipt.json')):
        receipt=signed(read(path,32*1024**2),authority)
        need(receipt['version']=='mimo-evaluation-receipt-v1'
             and receipt['policy_sha256']==policy.policy_sha256, 'qualified evaluation receipt')
        job=signed(receipt['job'],authority);opening=job['manifest'];m=signed(opening,authority)
        policy.validate_manifest(m);response=receipt['worker_report']
        need(response['signer'] in config['evaluator_workers'], 'admitted evaluator report')
        request=signed(response,response['signer'])
        need(request['action']=='report' and request['job_id']==job['job_id'], 'original evaluation job')
        report=request['report'];validate_report(report,job,m,authority,digest(job),now)
        e=report['evaluation'];cp=e['checkpoint']
        need(m['epoch'] in openings and openings[m['epoch']]==opening, 'evaluation opening in selected run')
        baseline=cp==policy.scope['initial_checkpoint']['id']
        evaluations.append(dict(run_id=job['job_id'],epoch_id=checkpoints.get(cp,m['epoch']),baseline=baseline,
            checkpoint=cp,model=m['model_id'],env_id=e['suite']['env_id'],dataset_id=e['dataset_id'],
            seed=e['suite']['seed'],count=e['count'],successes=e['successes'],mean_reward=e['mean_reward'],
            timestamp=report['completed_at'],status=e['status'],requested_count=e['requested_count'],
            completed_count=e['count'],error_count=e['error_count'],output_token_budget=e['harness']['max_output_tokens'],
            training_run_id=m['run_id'],model_runtime_revision=m['model_runtime_revision'],
            sampling_policy=e['draw_policy'],policy_kind='independent-mimo-heldout',
            harness=e['harness'],environment_version=next(d['spec']['version'] for d in m['environments'] if d['env_id']==e['suite']['env_id']),
            native_graded=True,proof_verification_performed=False,original_report_sha256=digest(receipt),
            fixed_task_ids=e['suite']['indices'],output_tokens=sum(x['output_length']for x in e['rows']),
            unresolved=e['unresolved'],completed_wrong=e['completed_wrong']))
    boundary=dict(version='dashboard-training-run-boundary-v1',run_id=policy.run_id,
        started_at=config['started_at'],first_round=config['first_round'],
        base_checkpoint=policy.scope['initial_checkpoint']['id'],
        model=policy.scope['immutable_manifest']['model_id'],
        source_sha256=policy.scope['immutable_manifest']['source_bundle']['sha256'])
    return dict(snapshot,training_run=boundary,epochs=sorted(epochs,key=lambda r:r['start'],reverse=True),
        evaluations=sorted(evaluations,key=lambda r:r['timestamp']),training_evaluations=[],
        summary=dict(updated_at=now,network='Finney',netuid=120,current_source='live-reward-math',mechanism='mimo-fixed-group',
            epochs=len(epochs),accepted=0,rejected=0,unchecked=sum(r['unchecked']for r in epochs),
            training_steps=sum((r['training']or{}).get('steps',0)for r in epochs)))


def from_selector(snapshot, path):
    config, policy=selection(path)
    return project(snapshot,config,policy)
