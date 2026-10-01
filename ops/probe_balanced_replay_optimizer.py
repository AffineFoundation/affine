"""Operator-approved current-model replay qualification, never a payout job.

Historical FULL audits authenticate chosen actions. Current probability rows,
proofs and immutable preference references are recomputed before optimization.
"""
import argparse,copy,hashlib,json,os,time,shutil,subprocess
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from subnet import verified_replay_pool as replay

def approve(plan_envelope,manifest_envelope,pool_envelope,authority):
    plan=replay.authenticated(plan_envelope,authority)
    manifest=replay.authenticated(manifest_envelope,authority)
    pool=replay.authenticated(pool_envelope,authority)
    if plan.get('revision')!='balanced-current-reference-qualification-v1' or plan.get('payable') is not False or plan.get('chain_transactions') is not False:
        raise ValueError('nonpayable qualification scope')
    if plan.get('current_manifest_sha256')!=replay.digest(manifest_envelope) or pool.get('current_manifest_sha256')!=replay.digest(manifest_envelope):
        raise ValueError('exact current manifest lineage')
    replay.heldout_registry(manifest,replay.definitions(manifest))
    selected=replay.select_pool(pool_envelope,authority,{})['selected']
    if not selected or plan.get('pool_sha256')!=pool['pool_sha256'] or plan.get('target_sha256')!=[e['target_sha256'] for e in selected]:
        raise ValueError('exact balanced selection')
    if type(plan.get('steps')) is not int or not len(selected)<=plan['steps']<=32:
        raise ValueError('every selected family must receive an update')
    for e in selected:
        if e.get('version')!=replay.VERSION or e.get('current_manifest_sha256')!=replay.digest(manifest_envelope) or not replay.exact(e.get('current_checkpoint'),manifest['checkpoint']):
            raise ValueError('current approved pair geometry')
        definition=replay.definitions(manifest)[e['environment_id']]
        if e['environment_index'] not in definition['indices'] or e['environment_index'] in manifest['heldout_indices'][e['environment_id']]:
            raise ValueError('replay mining/heldout exclusion')
        for label in ('positive','negative'):
            r=e[label]
            if r.get('classification')!=label or r.get('env_id')!=e['environment_id'] or r.get('index')!=e['environment_index'] or r.get('task_hash')!=e['task_hash']:
                raise ValueError('authenticated replay task and outcome')
    return plan,manifest,selected

def execute(plan,manifest,selected,out):
    from subnet.gpu_runtime import GPURuntime
    from subnet.epoch_optimizer import train_epoch
    from subnet.model import model_files
    out=Path(out)
    if out.exists():raise ValueError('immutable new qualification destination')
    out.mkdir(parents=True);out.chmod(0o700)
    definitions=replay.definitions(manifest);first=definitions[selected[0]['environment_id']]
    if shutil.disk_usage(out).free<40*1024**3:raise ValueError('nine-checkpoint history reserve')
    wait_until=time.time()+1800
    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])<19000:
        if time.time()>=wait_until:raise ValueError('CUDA admission capacity timeout; no optimizer launched')
        time.sleep(5)
    runtime=GPURuntime(plan['checkpoint_path'],manifest['checkpoint']['files'],first['spec'],first['harness'])
    pairs=[];checks=[]
    for entry in selected:
        definition=definitions[entry['environment_id']];runtime.configure(definition['spec'],definition['harness']);pair=[]
        for label in ('positive','negative'):
            original=entry[label];current=copy.deepcopy(original);arrays=[]
            for turn in current['turns']:
                acts,probabilities=runtime.compute(turn['prompt'],turn['output'])
                turn['proofs']=runtime.build_proofs(acts,decode_batching_size=16,topk=128);arrays.append(probabilities)
            if not runtime.verify(current,arrays):raise ValueError('current numerical and original native replay')
            pair.append(original)
        pairs.append((definition,*pair));checks.append({'env_id':entry['environment_id'],'index':entry['environment_index'],'target_sha256':entry['target_sha256'],'current_probability_and_proof_recomputed':True,'native_replay_verified':True})
        (out/'pretraining-checks.json').write_bytes(replay.canonical(checks))
    destination,updates=train_epoch(runtime,pairs,out,steps=plan['steps'])
    files=model_files(destination)
    report={'revision':plan['revision'],'input_checkpoint':manifest['checkpoint']['id'],'pool_sha256':plan['pool_sha256'],'checks':checks,'updates':updates,'output_checkpoint':{'id':replay.digest(files),'files':files,'path':str(destination)},'historical_probabilities_used_as_reference':False,'reference_recomputed_before_all_updates':True,'common_epoch_completed':False,'heldout_improvement_claimed':False,'payable':False,'chain_transactions':False,'completed_at':time.time()}
    (out/'report.json').write_bytes(replay.canonical(report));return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','manifest','pool','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--authority',required=True);a=p.parse_args()
    plan,manifest,selected=approve(json.loads(a.plan.read_bytes()),json.loads(a.manifest.read_bytes()),json.loads(a.pool.read_bytes()),a.authority)
    expected={str(f) for f in Path('subnet').glob('*.py')}
    if set(plan.get('source_files',{}))!=expected:raise ValueError('exact worker source membership')
    for name,sha in plan['source_files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('approved worker source bytes')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan.get('probe_sha256'):raise ValueError('approved probe source bytes')
    print(json.dumps(execute(plan,manifest,selected,a.out)))
if __name__=='__main__':main()
