"""Persistent synchronous controller; blockchain writes belong to a separate guard.

Operator-owned environment/harness definitions and a signed numerical profile are
shared with the research runner. Default epochs are nonpayable. --once permits
early freeze only for an explicitly nonpayable trial, never a payable challenge.
"""
import argparse
import json
import logging
import time
from pathlib import Path
from .storage import Bucket,Gateway,canonical
from .controller import Controller
from .chain import ChainAdapter
from .environments import EnvironmentSpec,build_spec,legacy_spec,legacy_harness
from .harness import normalize
from .model import check_runtime_profile,model_files
from .evaluation import evaluate
from .publication import publish_history,publish_source_bundle

log=logging.getLogger('affine')
DEFAULT_PROFILE={'MKL_CBWR':'COMPATIBLE','ATEN_CPU_CAPABILITY':'default','ONEDNN_MAX_CPU_ISA':'SSE41','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_bytes(canonical(value));temporary.chmod(0o600);temporary.replace(path)


def definitions(config):
    """Compile trusted operator config; submitted batch data never chooses code."""
    result=[]
    for row in config.get('environments') or [dict(spec=config.get('environment') or {})]:
        evaluation_only=row.get('evaluation_only',False)
        if type(evaluation_only)is not bool:raise ValueError('evaluation-only flag must be boolean')
        if evaluation_only and row.get('indices')!=[]:
            raise ValueError('evaluation-only environment requires explicit empty mining indices')
        if row.get('source'):
            spec=legacy_spec(row.get('config')) if row['source']=='mastermind' else build_spec(row['source'],row.get('config',{}),num_samples=row.get('num_samples',4),max_turns=row.get('max_turns',4),max_output_tokens=row.get('max_output_tokens',96))
        else:
            raw=row['spec'];spec=EnvironmentSpec.from_dict(raw) if 'id' in raw else legacy_spec(raw or None)
        from .sample_harness import VERSION as INDEXED_HARNESS,validate as validate_sample_harness
        indices=row.get('indices',row.get('training_indices',list(range(spec.num_samples))))
        harness=validate_sample_harness(row.get('harness') or (legacy_harness(spec.config) if spec.adapter=='legacy_mastermind' else None),indices)
        # validate_sample_harness already checks exact index coverage and
        # normalizes every choice. Re-resolving for each index would repeatedly
        # validate the entire approved population, with quadratic work.
        if isinstance(harness,dict) and harness.get('version')==INDEXED_HARNESS:
            choices=harness['by_index'].values()
        elif harness is not None and (indices or evaluation_only):
            choices=(harness,)
        elif indices:
            choices=(normalize(legacy_harness(spec.config) if spec.adapter=='legacy_mastermind' else None),)
        else:
            choices=()
        if any(choice['max_output_tokens']>spec.max_output_tokens for choice in choices):raise ValueError('harness exceeds environment budget')
        if (not isinstance(indices,list) or (not indices and not evaluation_only) or
                len(set(indices))!=len(indices) or any(type(i) is not int or not 0<=i<spec.num_samples for i in indices)):raise ValueError('challenge indices')
        if any(r['spec']['id']==spec.id for r in result):raise ValueError('duplicate environment id')
        definition=dict(spec=spec.to_dict(),harness=harness,indices=indices)
        if evaluation_only:definition['evaluation_only']=True
        result.append(definition)
    return result


def evaluation_contract(config,rows):
    raw=config.get('evaluation')
    if not raw:return None
    suites=raw.get('suites',[raw]);by_id={r['spec']['id']:r for r in rows};checked=[]
    for suite in suites:
        env_id=suite.get('env_id',rows[0]['spec']['id']);row=by_id.get(env_id)
        if row is None:raise ValueError('evaluation selects unknown trusted environment')
        indices=suite.get('indices',[])
        if not indices or len(set(indices))!=len(indices) or any(type(i) is not int or not 0<=i<row['spec']['num_samples'] for i in indices) or set(indices)&set(row['indices']):raise ValueError('held-out indices invalid or overlap training')
        repeats=suite.get('repeats',raw.get('repeats',1));seed=suite.get('seed',raw.get('seed',20260930))
        if type(repeats) is not int or not 1<=repeats<=16 or type(seed) is not int or seed<0:raise ValueError('held-out sampling policy')
        harness=normalize(suite.get('harness',row['harness']))
        if harness['max_output_tokens']>row['spec']['max_output_tokens']:raise ValueError('held-out output budget')
        checked.append(dict(env_id=env_id,indices=indices,repeats=repeats,seed=seed,harness=harness))
    return dict(suites=checked)


def open_contract(config):
    rows=definitions(config);profile=config.get('runtime_profile',DEFAULT_PROFILE)
    check_runtime_profile({'runtime_profile':profile})
    audit=config.get('audit_policy',{'mode':'full','version':1})
    if not isinstance(audit,dict) or audit.get('mode','full') not in ('full','sampled'):raise ValueError('audit policy')
    duration=config.get('duration',600)
    if type(duration) is not int or not 1<=duration<=86400:raise ValueError('epoch duration')
    return dict(duration=duration,environments=rows,runtime_profile=profile,audit_policy=audit,evaluation=evaluation_contract(config,rows),source_bundle=config.get('source_bundle'))


def evaluate_epoch(manifest,path,phase,state,steps):
    for suite in manifest.get('evaluation',{}).get('suites',[]):
        definition=next(r for r in manifest['environments'] if r['env_id']==suite['env_id'])
        definition=dict(definition,harness=suite['harness'])
        heldout=dict(suite,run_id=manifest['epoch']+'-'+phase+'-'+suite['env_id'],experiment_id='configured-service',training_steps=steps)
        destination=Path('state/evaluations')/(heldout['run_id']+'.json')
        if not destination.exists():evaluate(manifest,path,definition,heldout,destination)


def epoch_prefix(config,once=False):
    prefix=config.get('epoch_prefix','nonpayable-service')
    nonpayable=prefix.startswith(('nonpayable-','test-','mock-'))
    if not nonpayable and not config.get('payable_epochs',False):raise ValueError('payable epoch requires explicit opt-in')
    if once and not nonpayable:raise ValueError('early freeze --once is restricted to nonpayable trial epochs')
    return prefix


def trained_checkpoint(controller,manifest,reports,path,destination,steps,min_free_bytes=2*1024**3):
    metrics_path=controller.state/(manifest['epoch']+'-training-metrics.json')
    if metrics_path.exists() and destination.is_dir() and model_files(destination):
        metrics=json.loads(metrics_path.read_text())
        if not metrics.get('weights_changed'):raise ValueError('cached training did not change weights')
        checkpoint=controller.publish_checkpoint(destination)
        if checkpoint['id']==manifest['checkpoint']['id']:raise ValueError('unchanged cached checkpoint')
        if metrics.get('checkpoint')!=checkpoint['id']:raise ValueError('cached trained checkpoint binding missing or changed')
        return checkpoint,metrics
    return controller.train(manifest,reports,path,destination,steps,min_free_bytes=min_free_bytes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    config=json.loads(Path(a.config).read_text());contract=open_contract(config);prefix=epoch_prefix(config,a.once)
    state=Path(config['state']);state.mkdir(parents=True,exist_ok=True);state.chmod(0o700)
    bucket=Bucket(config['bucket']);gateway=Gateway(bucket,host=config.get('host','127.0.0.1'),port=config.get('port',8788),state_path=state/'gateway.json',public_url=config['public_url'],direct_r2=config.get('direct_r2',False))
    controller=Controller(bucket,gateway,state);chain=ChainAdapter(state/'chain')
    log.info('authority=%s gateway=%s',controller.authority.id,gateway.url)
    livepath=state/'controller.json'
    if livepath.exists():status=json.loads(livepath.read_text())
    else:
        checkpoint=controller.publish_checkpoint(config['checkpoint']);status=dict(checkpoint=checkpoint,checkpoint_path=config['checkpoint'],round=0,training_steps=0,active=None);save(livepath,status)
    while True:
        try:
            if not status['active']:
                registrations=chain.registrations()
                allowed=config.get('registration_allowlist')
                if allowed is not None:
                    if not isinstance(allowed,list) or any(not isinstance(k,str) or len(k)!=64 for k in allowed):raise ValueError('registration allowlist')
                    registrations={hotkey:r for hotkey,r in registrations.items() if r['public_key'] in allowed}
                identities={r['public_key']:hotkey for hotkey,r in registrations.items()}
                if len(identities)!=len(registrations):raise ValueError('duplicate registration identity')
                if not identities:
                    save(state/'health.json',dict(status='waiting_for_registered_miners',time=time.time()));time.sleep(30);continue
                epoch=prefix+'-'+str(int(time.time()))+'-'+str(status['round'])
                status['active']=dict(epoch=epoch,registrations=registrations,identities=identities,phase='opening');save(livepath,status)
                save(state/f'{epoch}-registrations.json',registrations)
            active=status['active'];manifest_path=state/f"{active['epoch']}-manifest.json"
            if active['phase']=='opening':
                if manifest_path.exists():
                    manifest=json.loads(manifest_path.read_text())
                    bucket.json(f"public/{manifest['epoch']}/manifest.json",controller.signed(manifest))
                elif active['epoch'] in gateway.epochs:
                    # Capability creation committed but no local manifest exists:
                    # no public capability was published. Freeze the orphan and
                    # journal this aborted attempt before choosing a fresh epoch.
                    gateway.freeze(active['epoch'])
                    save(state/f"{active['epoch']}-opening-aborted.json",dict(epoch=active['epoch'],reason='interrupted before manifest commit',payable=False,time=time.time()))
                    status['active']=None;status['round']+=1;save(livepath,status);continue
                else:manifest=controller.open(active['epoch'],status['checkpoint'],active['identities'],**contract)
                stream_path=f'public/streams/{prefix}/current.json'
                pointer=dict(epoch=manifest['epoch'],manifest=f"public/{manifest['epoch']}/manifest.json")
                if gateway.direct_r2:pointer.update(manifest_url=bucket.presign(pointer['manifest']),current_url=bucket.presign(stream_path),current_url_expires_at=time.time()+604800,transport_policy='direct-r2-v1')
                if gateway.direct_r2:
                    ledger=state/'finalized-reports.json'
                    pointer['history_url']=publish_history(controller,prefix,json.loads(ledger.read_text()) if ledger.exists() else [],config.get('historical_source_bundle',config.get('source_bundle')),config.get('source_reconstructions',[]))
                bucket.json(stream_path,controller.signed(pointer))
                if gateway.direct_r2:save(state/'direct-discovery.json',dict(current_url=bucket.presign(stream_path),authority=controller.authority.id,expires_at=time.time()+604800))
                active['phase']='collect';save(livepath,status)
            manifest=json.loads(manifest_path.read_text())
            if manifest['payable'] and not config.get('payable_epochs',False):raise ValueError('resumed payable challenge needs explicit opt-in')
            if active['phase']=='collect':
                evaluate_epoch(manifest,status['checkpoint_path'],'before',state,status.get('training_steps',0))
                if not a.once and time.time()<manifest['deadline']:
                    save(state/'health.json',dict(status='collecting',epoch=manifest['epoch'],deadline=manifest['deadline'],time=time.time()));time.sleep(min(10,max(1,manifest['deadline']-time.time())));continue
                if a.once and manifest['payable']:raise ValueError('refuse early freeze of resumed payable epoch')
                result,reports=controller.finalize(manifest,status['checkpoint_path']);result['points']={active['identities'][identity]:points for identity,points in result['points'].items()}
                ledger=state/'finalized-reports.json';history=json.loads(ledger.read_text()) if ledger.exists() else []
                if not any(r['epoch_id']==result['epoch_id'] for r in history):history.append(result)
                save(ledger,history);save(state/'epoch-registrations.json',active['registrations']);save(state/f"{manifest['epoch']}-registrations.json",active['registrations']);save(state/f"{manifest['epoch']}-verified.json",reports)
                active['phase']='train';save(livepath,status)
            reports=json.loads((state/f"{manifest['epoch']}-verified.json").read_text())
            if any(r['accepted'] for r in reports.values()):
                destination=state/'checkpoints'/manifest['epoch'];new,metrics=trained_checkpoint(controller,manifest,reports,status['checkpoint_path'],destination,config.get('training_steps',1),config.get('training_min_free_bytes',2*1024**3))
                next_manifest=dict(manifest,checkpoint=new);evaluate_epoch(next_manifest,destination,'after',state,status.get('training_steps',0)+metrics['steps'])
                status['checkpoint'],status['checkpoint_path']=new,str(destination);status['training_steps']=status.get('training_steps',0)+metrics['steps']
            else:
                bucket.json(f"public/{manifest['epoch']}/training.json",controller.signed(dict(status='paused_no_verified_pairs')));save(state/'health.json',dict(status='paused_no_verified_training_data',epoch=manifest['epoch'],time=time.time()))
                if a.once:return
                if not manifest['payable'] and not reports:
                    save(state/f"{manifest['epoch']}-empty-closed.json",dict(epoch=manifest['epoch'],status='closed_without_submissions',checkpoint=manifest['checkpoint']['id'],payable=False,time=time.time()))
                    status['active']=None;status['round']+=1;save(livepath,status);continue
                time.sleep(60);continue
            status['active']=None;status['round']+=1;save(livepath,status)
            if a.once:return
        except Exception as e:
            log.exception('epoch paused for retry');save(state/'health.json',dict(status='error_retry',reason=str(e),time=time.time()))
            if a.once:raise
            time.sleep(30)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO);main()
