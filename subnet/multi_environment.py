"""Retained remote miner, nonpayable synchronous epochs and fixed held-out eval.

No chain mutation APIs are imported. Existing UID is verified read-only; wallet
seed stays on operator host. The remote host receives only a scoped upload URL.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from .storage import Bucket, Gateway, Identity, canonical
from .controller import Controller,CheckpointCapacityError
from .model import model_files,check_runtime_profile
from .chain import ChainAdapter,hourly_points
from .environments import build_spec,legacy_spec,legacy_harness
from .evaluation import evaluate

PROFILE={'MKL_CBWR':'COMPATIBLE','ATEN_CPU_CAPABILITY':'default','ONEDNN_MAX_CPU_ISA':'SSE41','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='state/multi-environment/config.json');p.add_argument('--epochs',type=int,default=3);p.add_argument('--interval',type=int,default=300);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());state=Path(config.get('state','state/multi-environment'));state.mkdir(parents=True,exist_ok=True);state.chmod(0o700)
    status_path=state/'status.json'
    def status(phase,**extra):
        record=dict(run_id=config['run_id'],phase=phase,timestamp=time.time(),payable=False,weight_submission=False,**extra)
        status_path.write_bytes(canonical(record));print(json.dumps(record),flush=True)
    for k,v in PROFILE.items():
        if os.environ.get(k)!=v:raise ValueError('start with exact compatible CPU profile')
    wallet=Path(config['hotkey_file']);secret=json.loads(wallet.read_text())['privateKey'];secret=bytes.fromhex(secret.removeprefix('0x'));identity=Identity(secret[:32])
    if identity.id!=config['public_key']:raise ValueError('owned Ed25519 identity mismatch')
    chain=ChainAdapter(state/'chain');block=int(chain.chain.block);uid=chain.query('Uids',[120,config['hotkey']],block)
    if uid is None or chain.query('Keys',[120,int(uid)],block)!=config['hotkey']:raise ValueError('miner ownership not currently registered')
    registration=dict(block=block,uid=int(uid),hotkey=config['hotkey'],public_key=identity.id,ownership_verified=True,query_only=True)
    (state/'registrations.json').write_bytes(canonical({'miners':[registration], 'by_identity':{identity.id:registration}}))
    bucket=Bucket(json.loads(Path(config['bucket_config']).read_text()));gateway=Gateway(bucket,port=config.get('port',8792),state_path=state/'gateway.json',public_url=config['public_url']);controller=Controller(bucket,gateway,state)
    remote=config['remote'];ssh=['ssh','-o','BatchMode=yes','-o',f"UserKnownHostsFile={remote['known_hosts']}",'-p',str(remote['port']),f"{remote['user']}@{remote['host']}"]
    scp=['scp','-o','BatchMode=yes','-o',f"UserKnownHostsFile={remote['known_hosts']}",'-P',str(remote['port'])]
    def remote_command(command,timeout=1200):
        return subprocess.run(ssh+[command],check=True,timeout=timeout,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    archive=state/'miner-source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        tar.add('subnet',arcname='subnet',filter=lambda info:None if '__pycache__' in info.name or info.name.endswith('.pyc') else info)
        tar.add('prototype/vendor/mastermind',arcname='prototype/vendor/mastermind',filter=lambda info:None if '__pycache__' in info.name else info)
        for schedule in config['environments']:
            reference=schedule.get('config',{}).get('task_snapshot')
            if reference:
                path=Path(reference)
                if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(Path('state/original-task-snapshots').resolve()):
                    raise ValueError('task snapshot outside trusted operator store')
                tar.add(path,arcname=str(path),recursive=False)

    subprocess.run(scp+[str(archive),f"{remote['user']}@{remote['host']}:/root/multi-miner-source.tar.gz"],check=True,timeout=180)
    remote_command('mkdir -p /root/affine-miner && tar -xzf /root/multi-miner-source.tar.gz -C /root/affine-miner')
    if config.get('remote_pip'):
        command='/root/miner-venv/bin/pip install '+ ' '.join(shlex.quote(x) for x in config['remote_pip'])
        remote_command(command)
    current_path=Path(config['checkpoint_path']);checkpoint=controller.publish_checkpoint(current_path)
    total_training_steps=0;history=[];counter=0
    resume=state/'progress.json'
    if resume.exists():
        saved=json.loads(resume.read_text());current_path=Path(saved['checkpoint_path']);checkpoint=saved['checkpoint'];counter=saved['counter'];total_training_steps=saved['training_steps'];history=saved['history']
    active_stages={json.loads(p.read_text())['epoch_id'] for p in state.glob('epoch-stage-*.json')}
    abandoned=[]
    for previous in list(gateway.epochs):
        if previous.startswith('nonpayable-multi-'+config['run_id']) and previous not in active_stages and not gateway.epochs[previous].get('closed',False):
            gateway.freeze(previous);abandoned.append(previous)
    if abandoned:(state/'abandoned-epochs.json').write_bytes(canonical(dict(epochs=abandoned,closed_at=time.time(),payable=False)))
    status('ready',uid=int(uid),authority=controller.authority.id,gateway=config['public_url'],checkpoint=checkpoint['id'],counter=counter)
    while a.epochs==0 or counter<a.epochs:
        config=json.loads(Path(a.config).read_text())
        schedule=config['environments'][counter%len(config['environments'])]
        if schedule['source']=='mastermind':spec=legacy_spec(schedule['config']);harness=dict(legacy_harness(schedule['config']),**schedule.get('harness',{}))
        else:
            spec=build_spec(schedule['source'],schedule['config'],num_samples=schedule['num_samples'],max_turns=schedule['max_turns'],max_output_tokens=schedule.get('max_output_tokens',96));harness=schedule['harness']
        definition=dict(spec=spec.to_dict(),harness=harness,indices=schedule['training_indices'])
        stage_path=state/f'epoch-stage-{counter}.json'
        if stage_path.exists():
            stage=json.loads(stage_path.read_text());manifest=stage['manifest'];epoch=manifest['epoch']
            if stage['checkpoint']!=checkpoint or manifest['runtime_profile']!=PROFILE:
                raise ValueError('resume checkpoint/runtime drift')
        else:
            epoch=f"nonpayable-multi-{config['run_id']}-{counter}-{int(time.time())}"
            manifest=controller.open(epoch,checkpoint,[identity.id],duration=1800,environments=[definition],runtime_profile=PROFILE,audit_policy=config.get('audit_policy',{'mode':'full','version':1}),evaluation=dict(indices=schedule['heldout_indices'],seed=config.get('evaluation_seed',20260930),repeats=config.get('evaluation_repeats',2),harness=schedule.get('heldout_harness',harness)))
            stage=dict(epoch_id=epoch,manifest=manifest,checkpoint=checkpoint,checkpoint_path=str(current_path),phase='opened',opened_at=time.time())
            stage_path.write_bytes(canonical(stage));stage_path.chmod(0o600)
        frozen_definition=manifest['environments'][0]
        evaluation_definition=dict(frozen_definition)
        if manifest.get('evaluation'):
            evaluation_definition['harness']=manifest['evaluation']['harness']
        heldout=dict(run_id=config['run_id'],indices=schedule['heldout_indices'],seed=config.get('evaluation_seed',20260930),repeats=config.get('evaluation_repeats',2),training_steps=total_training_steps)
        if manifest.get('evaluation'):
            heldout.update({k:manifest['evaluation'][k] for k in ('indices','seed','repeats')})
        (state/f'{epoch}-heldout.json').write_bytes(canonical(heldout))
        status('heldout-before',epoch_id=epoch,env_id=spec.id,checkpoint=checkpoint['id'],training_steps=total_training_steps)
        heldout['experiment_id']=config['run_id'];heldout['run_id']=epoch+'-before'
        before_path=Path('state/evaluations')/f'{epoch}-before.json'
        before=json.loads(before_path.read_text()) if before_path.exists() else evaluate(manifest,current_path,evaluation_definition,heldout,before_path)
        if (state/f'{epoch}-scores.json').exists():
            result,reports=controller.finalize(manifest,current_path)
        else:
            cap=identity.decrypt(manifest['capabilities'][identity.id]);delegation=dict(identity=identity.id,epoch=epoch,put_url=cap['put_url'])
            delegated_path=state/f'{epoch}-capability.json';delegated_path.write_bytes(canonical(delegation));delegated_path.chmod(0o600)
            subprocess.run(scp+[str(delegated_path),f"{remote['user']}@{remote['host']}:/root/multi-epoch-capability.json"],check=True,timeout=120)
            remote_command('chmod 600 /root/multi-epoch-capability.json')
            cmd=['/root/miner-venv/bin/python','-m','subnet.cli','--gateway',config['public_url'],'--authority',controller.authority.id,'--manifest-url',f"{config['public_url']}/public/{epoch}/manifest.json",'--cap-file','/root/multi-epoch-capability.json','--state','/root/multi-miner-state','--once','--max-batches',str(config.get('max_batches',1))]
            shell='cd /root/affine-miner && env '+' '.join(f'{k}={shlex.quote(v)}' for k,v in PROFILE.items())+' '+shlex.join(cmd)
            status('remote-miner',epoch_id=epoch,env_id=spec.id,harness=harness['version'],checkpoint=checkpoint['id'])
            start=time.time()
            try:
                completed=remote_command(shell,timeout=1500)
                (state/f'{epoch}-remote.log').write_text(completed.stdout)
            except subprocess.CalledProcessError as exc:
                (state/f'{epoch}-remote.log').write_text(exc.stdout or '')
                status('remote-error',epoch_id=epoch,error='remote miner failed; inspect private local log');raise
            status('freeze-and-independent-audit',epoch_id=epoch,remote_seconds=time.time()-start)
            result,reports=controller.finalize(manifest,current_path)
            if result['payable'] or hourly_points([result],(int(result['finalized_at'])//3600+1)*3600):raise ValueError('nonpayable guard')
        proposed=dict(result,proposed_only=True,weight_submission=False)
        (state/f'{epoch}-proposed-weights.json').write_bytes(canonical(proposed))
        accepted=sum(len(r['accepted']) for r in reports.values())
        if not accepted:
            history.append(dict(epoch_id=epoch,env_id=spec.id,status='no-accepted-training-pairs',accepted_batches=0,points=result['points'],payable=False,weight_submission=False))
            counter+=1
            resume.write_bytes(canonical(dict(counter=counter,checkpoint_path=str(current_path),checkpoint=checkpoint,training_steps=total_training_steps,history=history)))
            status('no-accepted-training-pairs',epoch_id=epoch,points=result['points']);time.sleep(a.interval);continue
        destination=state/f'checkpoint-{counter+1}'
        status('real-training',epoch_id=epoch,accepted_batches=accepted)
        training_path=state/f'{epoch}-training-metrics.json'
        if training_path.exists() and model_files(destination):
            training=json.loads(training_path.read_text());checkpoint=controller.publish_checkpoint(destination)
            training.update(source_epoch=epoch,input_pairs=sum(len(r['accepted']) for r in reports.values()),checkpoint=checkpoint['id'])
            if not training.get('weights_changed'):raise ValueError('invalid cached training result')
        else:
            try:
                checkpoint,training=controller.train(manifest,reports,current_path,destination,steps=config.get('training_steps_per_epoch',1),min_free_bytes=config.get('training_min_free_bytes',2*1024**3))
            except CheckpointCapacityError as exc:
                status('capacity-hold-before-training',epoch_id=epoch,error=str(exc))
                time.sleep(60);continue
        total_training_steps+=training['steps'];current_path=destination
        post_manifest=dict(manifest,checkpoint=checkpoint)
        heldout['training_steps']=total_training_steps;heldout['run_id']=epoch+'-after'
        status('heldout-after',epoch_id=epoch,env_id=spec.id,checkpoint=checkpoint['id'])
        after=evaluate(post_manifest,current_path,evaluation_definition,heldout,Path('state/evaluations')/f'{epoch}-after.json')
        if before['dataset_id']!=after['dataset_id'] or before['task_hashes']!=after['task_hashes']:raise ValueError('held-out task drift')
        history.append(dict(epoch_id=epoch,env_id=spec.id,harness=harness['version'],accepted_batches=accepted,points=result['points'],proposed_weights=result['weights'],training=training,heldout_before=before,heldout_after=after,weight_submission=False,payable=False))
        counter+=1
        resume.write_bytes(canonical(dict(counter=counter,checkpoint_path=str(current_path),checkpoint=checkpoint,training_steps=total_training_steps,history=history)))
        report=dict(run_id=config['run_id'],complete=sum(h.get('accepted_batches',0)>0 for h in history)>=3,epochs=counter,history=history,uid=int(uid),hotkey=config['hotkey'],remote={k:remote[k] for k in ('host','port','user')},training_steps=total_training_steps,checkpoint=checkpoint,payable=False,chain_weight_submissions=0,authority=controller.authority.id)
        (state/'report.json').write_bytes(canonical(report));bucket.json(f"public/nonpayable-multi-{config['run_id']}/experiment.json",controller.signed(report))
        status('epoch-complete',epoch_id=epoch,epochs_completed=counter,accepted_batches=accepted,next_checkpoint=checkpoint['id'],training_steps=total_training_steps)
        if a.epochs==0 or counter<a.epochs:time.sleep(config.get("continuous_interval",300) if counter>=3 else a.interval)
    status('completed-epochs-gateway-retained',epochs_completed=counter,training_steps=total_training_steps)
    while True:time.sleep(30)

if __name__=='__main__':main()
