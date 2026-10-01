"""Original NQueens common-session equal-token candidate proof search; no score, optimizer or chain writes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,base64,gc,hashlib,json,subprocess,time
from pathlib import Path
from nacl.signing import VerifyKey
from subnet.storage import canonical

def candidates(public):
    from subnet.native_prolog_public_policy import candidates as public_commands
    return [json.dumps({'tool_call':{'name':'bash','arguments':{'command':command}}},separators=(',',':')) for command in public_commands(public)]


def source_membership(files):
    expected={str(path) for path in Path('subnet').glob('*.py')}
    if not expected or not isinstance(files,dict) or set(files)!=expected:raise ValueError('exact source membership')


def validate_contract(plan):
    if plan.get('schema')!=1 or plan.get('experiment')!='original-nqueens-common-model-search-v1' or plan.get('payable') is not False or plan.get('chain_transactions') is not False:raise ValueError('nonpayable scoped probe')
    if type(plan.get('search_budget')) is not int or not 1<=plan['search_budget']<=32:raise ValueError('bounded search')
    if plan.get('indices')!=[0] or any(type(i) is not int for i in plan['indices']):raise ValueError('qualified original fixture index0 only')
    environment=plan.get('environment',{});config=environment.get('config',{})
    if environment.get('id')!='affine_prolog' or environment.get('adapter')!='prime_v1' or type(environment.get('num_samples'))is not int or environment.get('num_samples')!=3 or type(environment.get('success_reward'))not in (int,float) or environment.get('success_reward')!=1.:raise ValueError('original NQueens scope')
    if type(environment.get('max_turns'))is not int or environment['max_turns']!=2 or type(environment.get('max_output_tokens'))is not int or environment['max_output_tokens']<512:raise ValueError('two-turn complete output budget')
    if plan.get('artifact_policy')!={'max_compressed_bytes':100000000,'max_uncompressed_bytes':500000000,'array_dtype':'float32','max_turn_tokens':512,'max_vocab_size':200000}:raise ValueError('bounded full-probability artifact policy')
    if config.get('taskset')!={'tasks':['prolog-nqueens-0005','prolog-nqueens-0014','prolog-nqueens-0023'],'num_examples':32,'difficulty':'medium'}:raise ValueError('original selected fixtures')
    if config.get('prolog_session_revision')!='original-nqueens-common-session-v1':raise ValueError('common session revision')
    public=plan.get('public_task',{})
    from subnet.native_prolog_actor import PublicActor,isolation_command
    isolation_command('affine-prolog-native-'+'a'*16,config['prolog_runtime'])
    PublicActor(config['prolog_runtime'],public)
    expected={'version':'text-tools-v1','policy':'candidates','max_output_tokens':512,'temperature':4.0,'top_p':1.0,'candidates':candidates(public),'turn_overrides':{'1':{'policy':'candidates','candidates':['Done']}}}
    if canonical(plan.get('harness'))!=canonical(expected):raise ValueError('exact public-program candidate contract')
    return plan


def approved(document,authority):
    if document.get('signer')!=authority:raise ValueError('approval authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),base64.b64decode(document['signature'],validate=True))
    plan=document['payload']
    helper=Path('subnet/native_prolog_public_policy.py')
    if helper.is_symlink() or hashlib.sha256(helper.read_bytes()).hexdigest()!=plan.get('public_policy_sha256'):raise ValueError('approved public Prolog policy helper')
    validate_contract(plan)
    from subnet.backend_jobs import BACKEND_PROFILE,NUMERICAL_POLICY
    if plan.get('backend_profile')!=BACKEND_PROFILE or plan.get('numerical_policy')!=NUMERICAL_POLICY:raise ValueError('unchanged numerical policy')
    source_membership(plan.get('source_files'))
    for name,digest in plan['source_files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('source pin')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['probe_sha256']:raise ValueError('probe source pin')
    return plan


def gpu_wait():
    deadline=time.time()+1800
    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])<16384:
        if time.time()>=deadline:raise ValueError('CUDA capacity timeout before model loading')
        time.sleep(5)


def hydrate_checkpoint(plan):
    """Use only signed, per-object GET capabilities in an isolated checkpoint directory."""
    import shutil
    from subnet.backend_jobs import file_map, get_object, r2_url
    download=plan.get('checkpoint_downloads')
    if download is None:return
    files=plan['checkpoint']['files'];target=Path(plan['checkpoint_path'])
    file_map(files)
    if hashlib.sha256(canonical(files)).hexdigest()!=plan['checkpoint']['id']:raise ValueError('checkpoint identity')
    if not files or set(download)!=set(files):raise ValueError('checkpoint download membership')
    for name,digest in files.items():
        if Path(name).name!=name or name in ('.','..') or len(digest)!=64:raise ValueError('checkpoint filename/digest')
        row=download[name]
        if type(row.get('size')) is not int or not 0<row['size']<=4_000_000_000:raise ValueError('checkpoint size bound')
        r2_url(row['url'],'GET')
    if target.is_symlink():raise ValueError('checkpoint directory symlink')
    target.mkdir(parents=True,exist_ok=True)
    if any(p.name not in files for p in target.iterdir()):raise ValueError('unexpected checkpoint file')
    missing=[]
    for name,digest in files.items():
        path=target/name
        if path.is_symlink():raise ValueError('checkpoint file symlink')
        if path.exists():
            from subnet.model import file_hash
            if not path.is_file() or path.stat().st_size!=download[name]['size'] or file_hash(path)!=digest:raise ValueError('existing checkpoint mismatch')
        else:missing.append(name)
    if shutil.disk_usage(target).free<sum(download[n]['size'] for n in missing)+1_000_000_000:raise ValueError('checkpoint download disk budget')
    for name in missing:
        get_object(download[name]['url'],files[name],target/name,download[name]['size'])
        if (target/name).stat().st_size!=download[name]['size']:raise ValueError('checkpoint download size mismatch')


def validated_batch(plan,row,batches):
    index=row.get('index')
    if type(index)is not int or index not in plan['indices'] or len(batches)!=1:raise ValueError('approved artifact index/batch population')
    batch,arrays=batches[0]
    if type(batch.get('index'))is not int or batch['index']!=index or batch.get('env_id')!=plan['environment']['id'] or batch.get('checkpoint')!=plan['checkpoint']['id']:raise ValueError('frozen batch approval binding')
    rollouts=batch.get('rollouts',[])
    if not 1<=len(rollouts)<=2 or len(arrays)!=len(rollouts):raise ValueError('retained rollout count')
    for rollout in rollouts:
        if type(rollout.get('sample_index'))is not int or rollout['sample_index']!=index or rollout.get('env_id')!=plan['environment']['id'] or type(rollout.get('index'))is not int or rollout['index']!=index or rollout.get('environment_version')!=plan['environment']['version']:raise ValueError('rollout approval binding')
    classes=[r.get('classification') for r in rollouts]
    if any(c not in ('positive','negative')for c in classes) or len(set(classes))!=len(classes):raise ValueError('unique retained classes')
    positive=classes.count('positive');negative=classes.count('negative')
    if type(row.get('positive'))is not int or type(row.get('negative'))is not int or (positive,negative)!=(row['positive'],row['negative']) or type(row.get('qualifying_K1L1'))is not bool or row['qualifying_K1L1']!=(positive==negative==1):raise ValueError('actual K/L metadata binding')
    return batch,arrays,positive,negative


def execute(plan,out,verify=False):
    import torch
    from subnet.gpu_runtime import GPURuntime
    from subnet.batches import pack,unpack
    from subnet.environments import create_session
    session=create_session(plan['environment'])
    try:
        initial=session.reset(0,100)
        if canonical(session.public)!=canonical(plan['public_task']):raise ValueError('original public task binding')
    finally:session.close()
    hydrate_checkpoint(plan)
    gpu_wait();out.mkdir(parents=True,exist_ok=True)
    runtime=GPURuntime(plan['checkpoint_path'],plan['checkpoint']['files'],plan['environment'],plan['harness'])
    if len(runtime.tokenizer.encode(plan['harness']['candidates'][0],add_special_tokens=False))!=len(runtime.tokenizer.encode(plan['harness']['candidates'][1],add_special_tokens=False)):raise ValueError('equal-token candidate contract')
    if verify:
        records=json.loads((out/'search.json').read_text());results=[]
        if records.get('checkpoint')!=plan['checkpoint']['id'] or canonical(records.get('harness'))!=canonical(plan['harness']):raise ValueError('frozen search approval binding')
        if [r.get('index')for r in records.get('rows',[])]!=plan['indices']:raise ValueError('frozen search index population')
        for row in records['rows']:
            name='index-'+str(row['index'])+'.zip'
            if row.get('artifact')!=name:raise ValueError('frozen artifact filename')
            data=(out/name).read_bytes()
            if type(row.get('artifact_size'))is not int or len(data)!=row['artifact_size'] or hashlib.sha256(data).hexdigest()!=row.get('artifact_sha256'):raise ValueError('frozen probe bytes')
            batch,arrays,positive,negative=validated_batch(plan,row,unpack(data))
            for rollout,tensors in zip(batch['rollouts'],arrays):
                if not runtime.verify(rollout,tensors):raise ValueError('fresh verification')
            results.append(dict(index=row['index'],artifact=name,artifact_sha256=row['artifact_sha256'],rollouts_verified=len(batch['rollouts']),positive=positive,negative=negative,qualifying_K1L1=positive==negative==1))
        report=dict(checkpoint=plan['checkpoint']['id'],rows=results,independent_model_reload=True,full_logits_verified=True,toploc_verified=True,original_native_replay=True,optimizer_ran=False,chain_transactions=False,payable=False,completed_at=time.time())
        (out/'fresh-verification.json').write_bytes(canonical(report));print(json.dumps(report));return
    rows=[]
    for index in plan['indices']:
        found={};attempts=[]
        for attempt in range(plan['search_budget']):
            seed=100+index*1000+attempt
            rollout,arrays=runtime.rollout(index,seed);label=rollout['classification'];attempts.append(dict(seed=seed,reward=rollout['reward'],classification=label,output_tokens=sum(len(t['output']) for t in rollout['turns']),rollout_sha256=hashlib.sha256(canonical(rollout)).hexdigest()))
            if label in ('positive','negative') and label not in found:found[label]=(rollout,arrays)
            if len(found)==2:break
        chosen=[found[k] for k in ('positive','negative') if k in found]
        raw_probability_bytes=sum(t.nbytes for _,turns in chosen for t in turns)
        if raw_probability_bytes>=plan['artifact_policy']['max_uncompressed_bytes']-2000000:raise ValueError('full probability artifact raw budget')
        batch=dict(env_id=runtime.spec.id,index=index,checkpoint=plan['checkpoint']['id'],rollouts=[v[0] for v in chosen]);data=pack([(batch,[v[1] for v in chosen])]);name='index-'+str(index)+'.zip';(out/name).write_bytes(data)
        row=dict(index=index,attempts=attempts,positive=int('positive' in found),negative=int('negative' in found),qualifying_K1L1=len(found)==2,artifact=name,artifact_sha256=hashlib.sha256(data).hexdigest(),artifact_size=len(data),probability_array_raw_bytes=raw_probability_bytes,probability_shapes=[[list(t.shape) for t in turns]for _,turns in chosen]);rows.append(row)
        (out/'search.json').write_bytes(canonical(dict(checkpoint=plan['checkpoint']['id'],rows=rows,search_budget=plan['search_budget'],policy_kind=plan['harness']['policy'],harness=plan['harness'],training=False,public_starter_control_not_autonomous_search=True,chain_transactions=False,payable=False,completed_at=time.time())))
    print(json.dumps([dict(index=v['index'],attempts=len(v['attempts']),positive=v['positive'],negative=v['negative']) for v in rows]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True,type=Path);p.add_argument('--authority',required=True);p.add_argument('--out',required=True,type=Path);p.add_argument('--verify',action='store_true');a=p.parse_args();execute(approved(json.loads(a.plan.read_text()),a.authority),a.out,a.verify)
if __name__=='__main__':main()
