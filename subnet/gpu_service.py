"""Sustained synchronous nonpayable GPU epochs, with real read-only identities."""
import argparse
import datetime
import hashlib
import json
import logging
import time
from pathlib import Path
from .backend_jobs import REVISION,NUMERICAL_POLICY,BACKEND_PROFILE
from .remote_backend import RemoteController,save
from .storage import Bucket,Gateway,canonical
from .chain import ChainAdapter
from .service import definitions
from .harness import normalize
from .publication import publish_history
from .evaluation import wilson

log=logging.getLogger('affine-gpu')

def contract(config,round_number):
    rows=definitions(config);groups=config.get('training_groups') or [[r['spec']['id'] for r in rows]]
    selected=set(groups[round_number%len(groups)])
    if not selected or not selected<={r['spec']['id'] for r in rows}:raise ValueError('GPU training group')
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
        definitions_all.append(dict(row,indices=indices))
    return dict(duration=config.get('duration',300),environments=definitions_all,audit_policy={'mode':'full','version':1},
        source_bundle=config['source_bundle'],model_runtime_revision=REVISION,numerical_policy=NUMERICAL_POLICY,
        backend_profile=BACKEND_PROFILE,model_id=config.get('model_id','HuggingFaceTB/SmolLM2-1.7B-Instruct'))

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
    label=manifest['epoch']+'-eval-'+phase
    rows=heldout(config,manifest);report=controller.jobs.run(label,'evaluate',manifest,cache,heldout=rows)
    records=[]
    for suite in rows:
        definition=next(e for e in manifest['environments'] if e['env_id']==suite['env_id']);values=[v for v in report['heldout'] if v['env_id']==suite['env_id']]
        failures=[v for v in report.get('heldout_failures',[]) if v['env_id']==suite['env_id']]
        expected=sorted(zip(suite['indices'],suite['seeds']))
        observed=sorted((v['index'],v['seed']) for v in values+failures)
        if observed!=expected or any(v.get('verified') is not True or not isinstance(v.get('task_hash'),str) or len(v['task_hash'])!=64 for v in values):raise ValueError('remote heldout exact plan/hash completeness')
        frozen=dict(env_id=suite['env_id'],environment=definition['spec'],harness=suite['harness'],indices=suite['indices'],seeds=suite['seeds'],model_runtime_revision=REVISION,backend_profile=BACKEND_PROFILE,runtime_versions=report['runtime_versions'],harness_source_hash=manifest['harness_source_hash'],source_files={n:report['source_files'][n] for n in ('subnet/model.py','subnet/gpu_runtime.py','subnet/environments.py','subnet/harness.py','subnet/proofs.py')})
        dataset=hashlib.sha256(canonical(frozen)).hexdigest();successes=sum(v['classification']=='positive' for v in values)
        run_id=label+'-'+suite['env_id'];stamp=report['completed_at']
        record=dict(run_id=run_id,experiment_id=config.get('evaluation_experiment_id','gpu-continuous-fixed128'),epoch_id=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],model=config.get('model_id','HuggingFaceTB/SmolLM2-1.7B-Instruct'),
            env_id=suite['env_id'],environment_version=definition['spec']['version'],model_runtime_revision=REVISION,
            harness=suite['harness']['version']+':autoregressive',harness_version=suite['harness']['version'],harness_config=suite['harness'],policy_kind='autoregressive',
            dataset_id=dataset,taskset_hash=dataset,seed=config.get('evaluation_seed',20260930),heldout_indices=suite['indices'],fixed_task_ids=[v['task_hash'] for v in values],
            count=len(values),completed_count=len(values),requested_count=len(suite['indices']),attempted_count=len(suite['indices']),successes=successes,
            mean_reward=sum(v['reward'] for v in values)/len(values) if values and not failures else None,status='complete' if not failures else 'error',evaluation_failures=failures,uncertainty=wilson(successes,len(values)) if not failures else None,
            training_steps=steps,timestamp=stamp,timestamp_iso=datetime.datetime.fromtimestamp(stamp,datetime.timezone.utc).isoformat(),
            payable=False,weight_submission=False,backend_profile=BACKEND_PROFILE,remote_job_id=report['job_id'],task_hashes=[v['task_hash'] for v in values],runtime_profile=dict(report['runtime_versions'],**BACKEND_PROFILE))
        save(Path(config.get('evaluation_state','state/evaluations'))/(run_id+'.json'),record);records.append(record)
    return records

def initial_manifest(config,checkpoint):
    chosen=contract(config,0)
    from .harness import source_hash
    return dict(epoch=config['epoch_prefix']+'-initial',payable=False,checkpoint=checkpoint,environments=[dict(env_id=r['spec']['id'],**r) for r in chosen['environments']],K=1,L=1,max_batches=config.get('max_batches',3),audit_policy={'mode':'full','version':1},harness_source_hash=source_hash(),model_runtime_revision=REVISION,numerical_policy=NUMERICAL_POLICY,backend_profile=BACKEND_PROFILE,model_id=chosen['model_id'],transport_policy='direct-r2-v1')

def run(config,once=False):
    prefix=config.get('epoch_prefix','nonpayable-gpu-continuous')
    if not prefix.startswith('nonpayable-') or config.get('payable_epochs',False):raise ValueError('GPU loop is permanently nonpayable')
    if not 60<=config.get('duration',300)<=3600 or not 1<=config.get('max_batches',3)<=3:raise ValueError('epoch budget')
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
                registrations=chain.registrations();allowed=set(config['registration_allowlist']);registrations={k:r for k,r in registrations.items() if r['public_key'] in allowed}
                if not registrations:
                    save(state/'health.json',dict(status='waiting_for_owned_registered_identity',time=time.time()));time.sleep(30);continue
                identities={r['public_key']:k for k,r in registrations.items()}
                if len(identities)!=len(registrations):raise ValueError('duplicate registered identity')
                epoch=prefix+'-'+str(int(time.time()))+'-'+str(status['round'])
                status['active']=dict(epoch=epoch,registrations=registrations,identities=identities,phase='opening');save(statuspath,status)
                save(state/(epoch+'-registrations.json'),registrations)
            active=status['active'];epoch=active['epoch'];manifestpath=state/(epoch+'-manifest.json')
            save(state/'health.json',dict(status=active['phase'],epoch=epoch,checkpoint=status['checkpoint']['id'],time=time.time(),chain_transactions=False))
            if active['phase']=='opening':
                if manifestpath.exists():manifest=json.loads(manifestpath.read_text())
                elif epoch in gateway.epochs:
                    gateway.freeze(epoch);save(state/(epoch+'-opening-aborted.json'),dict(epoch=epoch,payable=False,reason='interrupted before published manifest'));status['active']=None;status['round']+=1;save(statuspath,status);continue
                else:manifest=controller.open(epoch,status['checkpoint'],active['identities'],max_batches=config.get('max_batches',3),**contract(config,status['round']))
                if manifest['max_batches']!=config.get('max_batches',3):raise ValueError('immutable epoch quota/config mismatch')
                ledger=json.loads(ledgerpath.read_text()) if ledgerpath.exists() else []
                key='public/streams/'+prefix+'/current.json';pointer=dict(epoch=epoch,manifest='public/'+epoch+'/manifest.json',manifest_url=bucket.presign('public/'+epoch+'/manifest.json'),current_url=bucket.presign(key),current_url_expires_at=time.time()+604800,transport_policy='direct-r2-v1',history_url=publish_history(controller,prefix,ledger,config['source_bundle']))
                bucket.json(key,controller.signed(pointer));save(state/'direct-discovery.json',dict(current_url=bucket.presign(key),authority=controller.authority.id,expires_at=time.time()+604800))
                active['phase']='mine';save(statuspath,status)
            manifest=json.loads(manifestpath.read_text())
            if active['phase']=='mine':
                if time.time()<manifest['deadline']:
                    for miner in active['identities']:
                        capability=dict(put_url=bucket.presign('private/'+epoch+'/staging/'+miner+'.zip','put_object',max(1,manifest['deadline']-int(time.time()))),headers={'Content-Type':'application/octet-stream'})
                        controller.jobs.run(epoch+'-mine-'+miner[:8],'mine',manifest,None,miner_id=miner,capability=capability,search_budget=config.get('search_budget',64),seed_start=100+status['round']*1000)
                    status['checkpoint_path']=config['remote']['workspace']+'/checkpoints/'+status['checkpoint']['id']
                active['phase']='collect';save(statuspath,status)
            if active['phase']=='collect':
                if time.time()<manifest['deadline']:
                    save(state/'health.json',dict(status='collecting',epoch=epoch,deadline=manifest['deadline'],time=time.time()));time.sleep(min(10,max(1,manifest['deadline']-time.time())));continue
                result,reports=controller.finalize(manifest,status['checkpoint_path']);save(state/(epoch+'-verified.json'),reports)
                ledger=json.loads(ledgerpath.read_text()) if ledgerpath.exists() else []
                if not any(r['epoch_id']==epoch for r in ledger):ledger.append(dict(result,points={active['identities'][m]:p for m,p in result['points'].items()}))
                save(ledgerpath,ledger);active['phase']='before';save(statuspath,status)
            reports=json.loads((state/(epoch+'-verified.json')).read_text())
            if active['phase']=='before':
                evaluate(controller,manifest,status['checkpoint_path'],'before',status['training_steps'],config)
                active['phase']='train';save(statuspath,status)
            if active['phase']=='train':
                if any(r['accepted'] for r in reports.values()):
                    cp,metrics=controller.train(manifest,reports,status['checkpoint_path'],steps=config.get('training_steps',1))
                    active['next_checkpoint']=cp;active['next_path']=metrics['checkpoint_path'];active['next_steps']=status['training_steps']+metrics['steps']
                else:
                    save(state/(epoch+'-empty-closed.json'),dict(epoch=epoch,status='closed_no_accepted_batches',payable=False,checkpoint=status['checkpoint']['id']))
                    bucket.json('public/'+epoch+'/training.json',controller.signed(dict(status='closed_no_accepted_batches',checkpoint=status['checkpoint']['id'])))
                    active['next_checkpoint']=status['checkpoint'];active['next_path']=status['checkpoint_path'];active['next_steps']=status['training_steps']
                active['phase']='after';save(statuspath,status)
            if active['phase']=='after':
                nextmanifest=dict(manifest,checkpoint=active['next_checkpoint']);evaluate(controller,nextmanifest,active['next_path'],'after',active['next_steps'],config)
                save(state/(epoch+'-proposed-weights.json'),dict(epoch_id=epoch,payable=False,weights=json.loads((state/(epoch+'-scores.json')).read_text())['weights'],chain_transactions=False))
                status.update(checkpoint=active['next_checkpoint'],checkpoint_path=active['next_path'],training_steps=active['next_steps'],active=None,round=status['round']+1);save(statuspath,status)
                publish_history(controller,prefix,json.loads(ledgerpath.read_text()),config['source_bundle'])
                if once:return
        except Exception as error:
            log.exception('GPU epoch paused for retry');save(state/'health.json',dict(status='error_retry',error_type=type(error).__name__,time=time.time(),epoch=status.get('active',{}).get('epoch') if status.get('active') else None))
            if once:raise
            time.sleep(30)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.once)
if __name__=='__main__':logging.basicConfig(level=logging.INFO);main()
