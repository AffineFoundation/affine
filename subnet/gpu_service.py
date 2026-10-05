"""Sustained synchronous nonpayable GPU epochs, with real read-only identities."""
import argparse
import datetime
import hashlib
import json
import logging
import time
from pathlib import Path
from .backend_jobs import FIXED_POLICY as FULL_POLICY
from .training_policy import epoch_policy
from .backend_profiles import resolve,for_config
from .remote_backend import RemoteController,save
from .storage import Bucket,Gateway,canonical
from .chain import ChainAdapter
from .service import definitions
from .harness import normalize
from .publication import publish_history
from .evaluation import wilson
from .epoch_timing import transition as transition_phase,completion as epoch_completion

log=logging.getLogger('affine-gpu')

def registration_policy(config):
    """Choose operator filtering after ChainAdapter authenticates membership."""
    policy=config.get('registration_policy','allowlist')
    if policy not in ('allowlist','all_activated_subnet'):
        raise ValueError('unknown registration admission policy')
    if policy=='all_activated_subnet' and 'registration_allowlist' in config:
        raise ValueError('open subnet admission must omit registration_allowlist')
    return policy

def admitted_registrations(config,registrations):
    """Use a fresh authenticated subnet snapshot at each epoch opening.

    ChainAdapter already requires current SN120 UID ownership and a valid
    Ed25519 activation signature. Open admission adds no operator key filter.
    Existing signed epochs keep their saved participant snapshots.
    """
    if registration_policy(config)=='all_activated_subnet':
        return dict(registrations)
    allowed=config.get('registration_allowlist')
    if (not isinstance(allowed,list) or any(not isinstance(k,str) for k in allowed)
            or len(set(allowed))!=len(allowed)):
        raise ValueError('explicit registration allowlist required')
    keys=set(allowed)
    return {hotkey:row for hotkey,row in registrations.items() if row['public_key'] in keys}

def owned_dispatch_allowed(config,manifest):
    """Permit external-only trials without changing admission or empty policy."""
    from .empty_epoch_policy import dispatch_allowed
    configured=config.get('owned_miner_dispatch',True)
    if type(configured)is not bool:raise ValueError('owned miner dispatch must be boolean')
    allowed=dispatch_allowed(manifest)
    return configured and allowed

def owned_mining_job_fields(config,manifest,round_number):
    """Rotate the operator's bounded search while external miners keep the full pool."""
    subset=config.get('owned_mining_subset')
    schedule=config.get('owned_mining_schedule')
    if schedule is not None:
        if subset is not None or not isinstance(schedule,list) or not 1<=len(schedule)<=1024:
            raise ValueError('one bounded owned mining schedule or static subset')
        if type(round_number)is not int or round_number<0:raise ValueError('owned mining round')
        subset=schedule[round_number%len(schedule)]
    if subset is None:return {}
    from .backend_jobs import mining_definitions
    fields={'mining_subset':subset}
    mining_definitions(manifest,fields)
    return fields

def owned_dispatch_identities(config,manifest,identities):
    """Select operator jobs without restricting the published participant snapshot.

    Paths are remote miner-only references; the coordinator never reads keys.
    Legacy transports retain their historical dispatch behavior.
    """
    if manifest.get('submission_transport_policy') is None:return list(identities)
    from .commitment_transport import VERSION
    if manifest['submission_transport_policy']!=VERSION:raise ValueError('owned commitment transport')
    paths=config.get('owned_miner_identity_files',{})
    if not isinstance(paths,dict):raise ValueError('owned miner scoped identity files')
    for key,path in paths.items():
        if (not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key)
                or not isinstance(path,str) or not path.startswith('/') or '\x00' in path):
            raise ValueError('owned miner scoped identity files')
    return [miner for miner in identities if miner in paths]

def contract(config,round_number):
    revision,profile,policy=for_config(config)
    rows=definitions(config)
    training_ids={r['spec']['id'] for r in rows if not r.get('evaluation_only',False)}
    groups=config.get('training_groups') or [[r['spec']['id'] for r in rows if r['spec']['id'] in training_ids]]
    selected=set(groups[round_number%len(groups)])
    if not selected or not selected<={r['spec']['id'] for r in rows}:raise ValueError('GPU training group')
    if not selected<=training_ids:raise ValueError('evaluation-only environment in training group')
    per_epoch=config.get('indices_per_environment_per_epoch')
    if per_epoch is not None and (type(per_epoch) is not int or not 1<=per_epoch<=32):
        raise ValueError('prospective index rotation budget')
    definitions_all=[]
    for row in rows:
        indices=row['indices'] if row['spec']['id'] in selected else []
        if indices and per_epoch is not None:
            # Rotate only within the operator-approved training set; fixed
            # held-out indices remain excluded by heldout() across all rounds.
            cycle=round_number//len(groups)
            start=(cycle*per_epoch)%len(indices)
            indices=[indices[(start+i)%len(indices)] for i in range(min(per_epoch,len(indices)))]
        from .sample_harness import project
        definitions_all.append(dict(row,indices=indices,harness=project(row['harness'],indices,row['indices'])))
    registry={row['spec']['id']:dict(indices=row['indices'],harness=row['harness']) for row in rows}
    result=dict(sample_harness_registry=registry,heldout_indices={r['env_id']:r['indices'] for r in config['heldout']},duration=config.get('duration',300),environments=definitions_all,audit_policy=config.get('audit_policy',{'mode':'full','version':1}),
        training_policy=epoch_policy(config),source_bundle=config['source_bundle'],model_runtime_revision=revision,numerical_policy=policy,
        backend_profile=profile,model_id=config.get('model_id','HuggingFaceTB/SmolLM2-1.7B-Instruct'))
    if config.get('training_input_policy') is not None:
        if config['training_input_policy'] not in ('authenticated-verifier-receipts-v1','authenticated-verifier-compact-inputs-v2'):
            raise ValueError('unapproved training input policy')
        from .training_receipts import POLICIES
        if epoch_policy(config) not in POLICIES:raise ValueError('receipt input requires covered/persistent objective')
        result['training_input_policy']=config['training_input_policy']
    if config.get('artifact_policy') is not None:result['artifact_policy']=config['artifact_policy']
    if config.get('artifact_compression_policy') is not None:
        from .batches import compression_policy
        result['artifact_compression_policy']=compression_policy(config['artifact_compression_policy'])
    if config.get('task_assets') is not None:result['task_assets']=config['task_assets']
    if config.get('sampling_policy') is not None:result['sampling_policy']=config['sampling_policy']
    if config.get('persistent_publication_policy') is not None:
        from .persistent_publication import validate_policy
        result['persistent_publication_policy']=validate_policy(config['persistent_publication_policy'])
    if config.get('optimizer_state_export_policy') is not None:
        from .persistent_publication import export_policy
        export_policy(config)
        result['optimizer_state_export_policy']=config['optimizer_state_export_policy']
    if config.get('optimizer_state_transport') is not None:
        from .persistent_training_state import transport_concurrency
        transport_concurrency(config)
        result['optimizer_state_transport']=dict(config['optimizer_state_transport'])
    if config.get('live_reward_anchor_document') is not None:result['live_reward_anchor_document']=config['live_reward_anchor_document']
    from .empty_epoch_policy import selected
    policy=selected(config,round_number)
    if policy is not None:result['operator_test_policy']=policy
    if config.get('submission_transport_policy') is not None:result['submission_transport_policy']=config['submission_transport_policy']
    if config.get('hourly_execution_policy')is not None:result['hourly_execution_policy']=config['hourly_execution_policy']
    if config.get('temporary_exclusion_policy')is not None:result['temporary_exclusion_policy']=config['temporary_exclusion_policy']
    if config.get('reward_publication_policy')is not None:
        from .reward_publication import validate_policy
        result['reward_publication_policy']=validate_policy(config['reward_publication_policy'])
    return result

def heldout(config,manifest):
    rows=[];all_training={r['spec']['id']:set(r['indices']) for r in definitions(config)}
    for suite in config['heldout']:
        definition=next(e for e in manifest['environments'] if e['env_id']==suite['env_id'])
        indices=suite['indices'];harness=normalize(suite['harness'])
        if set(indices)&all_training[suite['env_id']] or any(type(i) is not int or not 0<=i<definition['spec']['num_samples'] for i in indices):raise ValueError('fixed heldout binding')
        if harness['max_output_tokens']>definition['spec']['max_output_tokens']:raise ValueError('heldout model budget')
        rows.append(dict(env_id=suite['env_id'],indices=indices,seeds=[suite.get('seed',20260930)+i*1000 for i in indices],harness=harness))
    return rows

def evaluate(controller,manifest,cache,phase,steps,config):
    revision,profile,policy=resolve(manifest)
    label=manifest['epoch']+'-eval-'+phase
    rows=heldout(config,manifest);report=controller.jobs.run(label,'evaluate',manifest,cache,heldout=rows)
    records=[]
    for suite in rows:
        definition=next(e for e in manifest['environments'] if e['env_id']==suite['env_id']);values=[v for v in report['heldout'] if v['env_id']==suite['env_id']]
        failures=[v for v in report.get('heldout_failures',[]) if v['env_id']==suite['env_id']]
        expected=sorted(zip(suite['indices'],suite['seeds']))
        observed=sorted((v['index'],v['seed']) for v in values+failures)
        if observed!=expected or any(v.get('verified') is not True or not isinstance(v.get('task_hash'),str) or len(v['task_hash'])!=64 for v in values):raise ValueError('remote heldout exact plan/hash completeness')
        frozen=dict(env_id=suite['env_id'],environment=definition['spec'],harness=suite['harness'],indices=suite['indices'],seeds=suite['seeds'],model_runtime_revision=revision,backend_profile=profile,runtime_versions=report['runtime_versions'],harness_source_hash=manifest['harness_source_hash'],source_files={n:report['source_files'][n] for n in ('subnet/model.py','subnet/gpu_runtime.py','subnet/environments.py','subnet/harness.py','subnet/proofs.py')})
        dataset=hashlib.sha256(canonical(frozen)).hexdigest();successes=sum(v['classification']=='positive' for v in values)
        run_id=label+'-'+suite['env_id'];stamp=report['completed_at']
        record=dict(run_id=run_id,experiment_id=config.get('evaluation_experiment_id','gpu-continuous-fixed128'),epoch_id=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],model=config.get('model_id','HuggingFaceTB/SmolLM2-1.7B-Instruct'),
            env_id=suite['env_id'],environment_version=definition['spec']['version'],model_runtime_revision=revision,
            harness=suite['harness']['version']+':autoregressive',harness_version=suite['harness']['version'],harness_config=suite['harness'],policy_kind='autoregressive',
            dataset_id=dataset,taskset_hash=dataset,seed=config.get('evaluation_seed',20260930),heldout_indices=suite['indices'],fixed_task_ids=[v['task_hash'] for v in values],
            count=len(values),completed_count=len(values),requested_count=len(suite['indices']),attempted_count=len(suite['indices']),successes=successes,
            mean_reward=sum(v['reward'] for v in values)/len(values) if values and not failures else None,status='complete' if not failures else 'error',evaluation_failures=failures,uncertainty=wilson(successes,len(values)) if not failures else None,
            training_steps=steps,timestamp=stamp,timestamp_iso=datetime.datetime.fromtimestamp(stamp,datetime.timezone.utc).isoformat(),
            payable=False,weight_submission=False,backend_profile=profile,remote_job_id=report['job_id'],task_hashes=[v['task_hash'] for v in values],runtime_profile=dict(report['runtime_versions'],**profile))
        save(Path(config.get('evaluation_state','state/evaluations'))/(run_id+'.json'),record);records.append(record)
    return records

def initial_manifest(config,checkpoint):
    chosen=contract(config,0)
    revision,profile,policy=for_config(config)
    from .harness import source_hash
    result=dict(sample_harness_registry=chosen['sample_harness_registry'],epoch=config['epoch_prefix']+'-initial',payable=False,training_policy=chosen['training_policy'],checkpoint=checkpoint,environments=[dict(env_id=r['spec']['id'],**r) for r in chosen['environments']],K=1,L=1,max_batches=config.get('max_batches',3),audit_policy=config.get('audit_policy',{'mode':'full','version':1}),harness_source_hash=source_hash(),model_runtime_revision=revision,numerical_policy=policy,backend_profile=profile,model_id=chosen['model_id'],transport_policy='direct-r2-v1')
    if 'training_input_policy' in chosen:result['training_input_policy']=chosen['training_input_policy']
    if 'artifact_policy' in chosen:result['artifact_policy']=chosen['artifact_policy']
    if 'task_assets' in chosen:result['task_assets']=chosen['task_assets']
    return result

def run(config,once=False):
    for field in ('preparation_only','activation_allowed'):
        if field in config and type(config[field]) is not bool:
            raise ValueError(field+' must be boolean')
    if config.get('preparation_only',False):raise ValueError('preparation-only config cannot run')
    if not config.get('activation_allowed',True):raise ValueError('config activation is not allowed')
    prefix=config.get('epoch_prefix','nonpayable-gpu-continuous')
    if not prefix.startswith('nonpayable-') or config.get('payable_epochs',False):raise ValueError('GPU loop is permanently nonpayable')
    anchor_document=config.get('live_reward_anchor_document')
    if anchor_document is not None:
        if prefix!='nonpayable-live-reward-math-v1-' or prefix!=anchor_document.get('payload',{}).get('compute_epoch_prefix'):
            raise ValueError('dedicated prospective compute-only reward prefix')
    if type(config.get('owned_miner_dispatch',True))is not bool:raise ValueError('owned miner dispatch must be boolean')
    registration_policy(config)
    from .checkpoint_evaluator import evaluation_mode
    independent_evaluation=evaluation_mode(config)=='independent-checkpoints-v1'
    epoch_policy(config)
    if config.get('audit_policy',{}).get('version')=='bounded-random-v1':
        from .audit_policy import validate
        validate(config['audit_policy'])
        if 'submission_counts' in config['audit_policy']:raise ValueError('audit allocation must be generated after freeze')
        if config.get('balanced_replay'):raise ValueError('sampled historical replay requires separate admission')
    if not 60<=config.get('duration',300)<=3600 or type(config.get('max_batches',3)) is not int or not 1<=config.get('max_batches',3)<=256:raise ValueError('epoch budget')
    state=Path(config['state']);state.mkdir(parents=True,exist_ok=True);state.chmod(0o700)
    bucket=Bucket(config['bucket']);gateway=Gateway(bucket,state_path=state/'gateway.json',public_url='http://unused-gpu-operator.invalid',direct_r2=True)
    controller=RemoteController(bucket,gateway,state,config['remote']);chain=ChainAdapter(state/'chain')
    statuspath=state/'controller.json';ledgerpath=state/'finalized-reports.json'
    status=json.loads(statuspath.read_text()) if statuspath.exists() else dict(active=None,round=0,training_steps=0,checkpoint=config['initial_checkpoint'],checkpoint_path=config['initial_checkpoint_path'],initial_published=False)
    save(statuspath,status)
    while True:
        try:
            if not status.get('initial_published'):
                initial=initial_manifest(config,status['checkpoint']);status['checkpoint']=controller.publish_remote_checkpoint(initial,status['checkpoint_path']);status['initial_published']=True;save(statuspath,status)
            if not status['active']:
                registrations=admitted_registrations(config,chain.registrations())
                if not registrations:
                    save(state/'health.json',dict(status='waiting_for_activated_subnet_identity' if registration_policy(config)=='all_activated_subnet' else 'waiting_for_owned_registered_identity',time=time.time()));time.sleep(30);continue
                identities={r['public_key']:k for k,r in registrations.items()}
                if len(identities)!=len(registrations):raise ValueError('duplicate registered identity')
                epoch=prefix+'-'+str(int(time.time()))+'-'+str(status['round'])
                status['active']=dict(epoch=epoch,registrations=registrations,identities=identities,phase='opening',started_at=time.time(),phase_started_at=time.time());save(statuspath,status)
                save(state/(epoch+'-registrations.json'),registrations)
            active=status['active'];epoch=active['epoch'];manifestpath=state/(epoch+'-manifest.json')
            save(state/'health.json',dict(status=active['phase'],epoch=epoch,checkpoint=status['checkpoint']['id'],time=time.time(),chain_transactions=False))
            if active['phase']=='opening':
                if manifestpath.exists():manifest=json.loads(manifestpath.read_text())
                elif epoch in gateway.epochs:
                    gateway.freeze(epoch);save(state/(epoch+'-opening-aborted.json'),dict(epoch=epoch,payable=False,reason='interrupted before published manifest'));status['active']=None;status['round']+=1;save(statuspath,status);continue
                else:
                    opening_contract=contract(config,status['round'])
                    from .persistent_cpu_adamw import POLICY as PERSISTENT_POLICY
                    if opening_contract['training_policy']==PERSISTENT_POLICY:
                        from .persistent_training_protocol import opening_binding
                        journal=state/'latest-trainer-state.json'
                        latest=json.loads(journal.read_text())if journal.exists()else None
                        if latest!=status.get('trainer_state'):raise ValueError('controller latest committed trainer-state journal mismatch')
                        opening_contract['trainer_state_binding']=opening_binding(config,status,epoch)
                    if anchor_document is not None:opening_contract['live_reward_registration_snapshot']=active['registrations']
                    manifest=controller.open(epoch,status['checkpoint'],active['identities'],max_batches=config.get('max_batches',3),**opening_contract)
                if manifest['max_batches']!=config.get('max_batches',3):raise ValueError('immutable epoch quota/config mismatch')
                if manifest.get('live_reward_contract') is not None:
                    # Resume publishes the SAME final manifest before recovering sidecars.
                    # Missing expired opening attestations fail closed, never backdate.
                    bucket.json('public/'+epoch+'/manifest.json',controller.signed(manifest))
                    from .live_reward_bridge import emit_opening_documents
                    emit_opening_documents(controller,manifest,active['registrations'])
                ledger=json.loads(ledgerpath.read_text()) if ledgerpath.exists() else []
                key='public/streams/'+prefix+'/current.json';pointer=dict(epoch=epoch,manifest='public/'+epoch+'/manifest.json',manifest_url=bucket.presign('public/'+epoch+'/manifest.json'),current_url=bucket.presign(key),current_url_expires_at=time.time()+604800,transport_policy='direct-r2-v1',history_url=publish_history(controller,prefix,ledger,config['source_bundle']))
                bucket.json(key,controller.signed(pointer));save(state/'direct-discovery.json',dict(current_url=bucket.presign(key),authority=controller.authority.id,expires_at=time.time()+604800))
                transition_phase(active,'mine');save(statuspath,status)
            manifest=json.loads(manifestpath.read_text())
            if active['phase']=='mine':
                if owned_dispatch_allowed(config,manifest) and time.time()<manifest['deadline']:
                    for miner in owned_dispatch_identities(config,manifest,active['identities']):
                        capability=dict(put_url=bucket.presign('private/'+epoch+'/staging/'+miner+'.zip','put_object',max(1,manifest['deadline']-int(time.time()))),headers={'Content-Type':'application/octet-stream'})
                        owned_fields=owned_mining_job_fields(config,manifest,status['round'])
                        attempts=min(config.get('search_budget',64),manifest['sampling_contract']['max_attempts']) if manifest.get('sampling_contract') else config.get('search_budget',64)
                        seed_start=0 if manifest.get('sampling_contract') else 100+status['round']*1000
                        if manifest.get('submission_transport_policy') is not None:
                            from .commitment_transport import VERSION
                            if manifest['submission_transport_policy']!=VERSION:raise ValueError('owned commitment transport')
                            capability['put_url']=bucket.presign('private/'+epoch+'/commitments/'+miner+'.json','put_object',max(1,manifest['deadline']-int(time.time())))
                            capability['batch_put_urls']=[bucket.presign('private/'+epoch+'/staging/'+miner+'/'+str(i)+'.zip','put_object',max(1,manifest['deadline']-int(time.time())))for i in range(manifest['max_batches'])]
                            paths=config.get('owned_miner_identity_files',{})
                            if not isinstance(paths,dict) or miner not in paths:raise ValueError('owned miner scoped identity file required')
                            owned_fields=dict(owned_fields,miner_identity_file=paths[miner])
                        dispatch_fields=dict(owned_fields)
                        if manifest.get('hourly_execution_policy')is not None:dispatch_fields['dispatch_only']=True
                        from .remote_backend import RemoteMinerReserved
                        try:
                            observation=controller.jobs.run(epoch+'-mine-'+miner[:8],'mine',manifest,None,miner_id=miner,capability=capability,search_budget=attempts,seed_start=seed_start,**dispatch_fields)
                        except RemoteMinerReserved as exc:
                            observation=dict(dispatch_only=True,blocked_by_original_job=exc.job_id,new_job_started=False,terminal_observed=False)
                        if manifest.get('hourly_execution_policy')is not None:
                            active.setdefault('owned_miner_dispatches',{})[miner]=observation
                            save(statuspath,status)
                    status['checkpoint_path']=(controller.jobs.checkpoint_path('mine',status['checkpoint']['id']) if hasattr(controller.jobs,'checkpoint_path') else config['remote']['workspace']+'/checkpoints/'+status['checkpoint']['id'])
                transition_phase(active,'collect');save(statuspath,status)
            if active['phase']=='collect':
                if time.time()<manifest['deadline']:
                    save(state/'health.json',dict(status='collecting',epoch=epoch,deadline=manifest['deadline'],time=time.time()));time.sleep(min(10,max(1,manifest['deadline']-time.time())));continue
                from .capture_status import InfrastructureSkipped,close_epoch
                try:result,reports=controller.finalize(manifest,status['checkpoint_path'])
                except InfrastructureSkipped:
                    close_epoch(controller,manifest,status,statuspath,prefix)
                    if once:return
                    continue
                save(state/(epoch+'-verified.json'),reports)
                from .empty_epoch_policy import validate_empty_completion
                validate_empty_completion(manifest,result,reports)
                ledger=json.loads(ledgerpath.read_text()) if ledgerpath.exists() else []
                if not any(r['epoch_id']==epoch for r in ledger):ledger.append(dict(result,points={active['identities'][m]:p for m,p in result['points'].items()}))
                save(ledgerpath,ledger);transition_phase(active,'before');save(statuspath,status)
            reports=json.loads((state/(epoch+'-verified.json')).read_text())
            if active['phase']=='before':
                if independent_evaluation:
                    from .checkpoint_evaluator import enqueue
                    enqueue(controller,manifest,status['checkpoint_path'],'before',status['training_steps'],config,public_optimizer_steps=status.get('trainer_state',{}).get('optimizer_steps',manifest.get('trainer_state_binding',{}).get('global_step_before')))
                else:
                    evaluate(controller,manifest,status['checkpoint_path'],'before',status['training_steps'],config)
                transition_phase(active,'train');save(statuspath,status)
            if active['phase']=='train':
                if any(r['accepted'] for r in reports.values()):
                    replay=None;steps=config.get('training_steps',1)
                    if config.get('balanced_replay'):
                        from .replay_pool_preparation import prepare
                        from .replay_training import admitted
                        folder=state/(epoch+'-replay-pool')
                        if not folder.exists():prepare(config,state,epoch,folder,max_pairs=16)
                        request=folder/'job-replay.json'
                        if request.exists():replay=json.loads(request.read_bytes())
                        else:
                            replay={'manifest':json.loads((folder/'current-manifest.json').read_bytes()),'pool':json.loads((folder/'pool.json').read_bytes()),'reuse_counts':json.loads((state/'replay-reuse-ledger.json').read_bytes())['counts'] if (state/'replay-reuse-ledger.json').exists() else {}}
                            save(request,replay)
                        _,selection=admitted(manifest,replay,controller.authority.id)
                        families={r['environment_id'] for r in selection['selected']}|{b['env_id'] for r in reports.values() for b in r['accepted']}
                        steps=len(families)
                    cp,metrics=controller.train(manifest,reports,status['checkpoint_path'],steps=steps,replay=replay)
                    if replay is not None:
                        from .replay_commit import commit
                        commit(state/'replay-reuse-ledger.json',epoch,metrics)
                    active['next_checkpoint']=cp;active['next_path']=metrics['checkpoint_path'];active['next_steps']=status['training_steps']+metrics['steps']
                    if metrics.get('trainer_state')is not None:
                        active['next_trainer_state']=metrics['trainer_state']
                else:
                    save(state/(epoch+'-empty-closed.json'),dict(epoch=epoch,status='closed_no_accepted_batches',payable=False,checkpoint=status['checkpoint']['id']))
                    bucket.json('public/'+epoch+'/training.json',controller.signed(dict(status='closed_no_accepted_batches',checkpoint=status['checkpoint']['id'])))
                    active['next_checkpoint']=status['checkpoint'];active['next_path']=status['checkpoint_path'];active['next_steps']=status['training_steps']
                    if status.get('trainer_state')is not None:active['next_trainer_state']=status['trainer_state']
                transition_phase(active,'after');save(statuspath,status)
            if active['phase']=='after':
                nextmanifest=dict(manifest,checkpoint=active['next_checkpoint'])
                if independent_evaluation:
                    from .checkpoint_evaluator import enqueue
                    enqueue(controller,nextmanifest,active['next_path'],'after',active['next_steps'],config,public_optimizer_steps=active.get('next_trainer_state',{}).get('optimizer_steps'))
                else:
                    evaluate(controller,nextmanifest,active['next_path'],'after',active['next_steps'],config)
                save(state/(epoch+'-proposed-weights.json'),dict(epoch_id=epoch,payable=False,weights=json.loads((state/(epoch+'-scores.json')).read_text())['weights'],chain_transactions=False))
                publish_history(controller,prefix,json.loads(ledgerpath.read_text()),config['source_bundle'])
                # A failed history publication must remain in the after phase:
                # resume reuses the completed evaluation and never retrains.
                if active.get('next_trainer_state')is not None:
                    committed=json.loads((state/'latest-trainer-state.json').read_text())
                    if committed!=active['next_trainer_state']:
                        raise ValueError('next committed trainer state journal mismatch')
                    if committed['inference_checkpoint']!=active['next_checkpoint']['id']:
                        raise ValueError('next committed trainer state checkpoint mismatch')
                    status.update(trainer_state=committed,persistent_state_committed=True,public_optimizer_steps=committed['optimizer_steps'])
                from .reward_publication import emit
                emit(controller,manifest)
                timing=epoch_completion(active,manifest,status['training_steps'])
                bucket.json('public/'+epoch+'/controller-timing.json',controller.signed(timing))
                save(state/(epoch+'-controller-timing.json'),timing)
                status.update(checkpoint=active['next_checkpoint'],checkpoint_path=active['next_path'],training_steps=active['next_steps'],active=None,round=status['round']+1);save(statuspath,status)
                if once:return
        except Exception as error:
            log.exception('GPU epoch paused for retry');save(state/'health.json',dict(status='error_retry',error_type=type(error).__name__,time=time.time(),epoch=status.get('active',{}).get('epoch') if status.get('active') else None))
            if once:raise
            time.sleep(30)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.once)
if __name__=='__main__':logging.basicConfig(level=logging.INFO);main()
