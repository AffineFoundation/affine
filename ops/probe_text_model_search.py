"""Controlled genuine text-source K/L search; no score, optimizer or chain writes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,base64,gc,hashlib,json,subprocess,time
from pathlib import Path
from nacl.signing import VerifyKey
from subnet.storage import canonical

def source_membership(files):
    expected={str(path) for path in Path('subnet').glob('*.py')}
    if not expected or not isinstance(files,dict) or set(files)!=expected:raise ValueError('exact source membership')


def approved(document,authority):
    if document.get('signer')!=authority:raise ValueError('approval authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),base64.b64decode(document['signature'],validate=True))
    plan=document['payload']
    if plan.get('schema')!=1 or plan.get('payable') is not False or plan.get('chain_transactions') is not False:raise ValueError('nonpayable probe scope')
    if type(plan.get('search_budget')) is not int or not 1<=plan['search_budget']<=64:raise ValueError('bounded search')
    if not isinstance(plan.get('indices'),list) or not plan['indices'] or len(plan['indices'])>16 or any(type(i) is not int or not 0<=i<16 for i in plan['indices']):raise ValueError('training-only indices')
    if len(set(plan['indices']))!=len(plan['indices']):raise ValueError('duplicate probe indices')
    if plan.get('environment',{}).get('id') not in ('affine_trivia','affine_popqa_abstain') or plan.get('environment',{}).get('adapter')!='prime_v1' or plan.get('harness',{}).get('policy')!='autoregressive':raise ValueError('native text source and unrestricted sampling scope')
    from subnet.backend_jobs import BACKEND_PROFILE,NUMERICAL_POLICY
    if plan.get('backend_profile')!=BACKEND_PROFILE or plan.get('numerical_policy')!=NUMERICAL_POLICY:raise ValueError('unchanged numerical policy')
    source_membership(plan.get('source_files'))
    for name,digest in plan['source_files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('source pin')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['probe_sha256']:raise ValueError('probe source pin')
    return plan


def gpu_wait():
    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])<8192:time.sleep(5)


def execute(plan,out,verify=False):
    import torch
    from subnet.gpu_runtime import GPURuntime
    from subnet.batches import pack,unpack
    gpu_wait();out.mkdir(parents=True,exist_ok=True)
    runtime=GPURuntime(plan['checkpoint_path'],plan['checkpoint']['files'],plan['environment'],plan['harness'])
    if verify:
        records=json.loads((out/'search.json').read_text());results=[]
        for row in records['rows']:
            data=(out/row['artifact']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=row['artifact_sha256']:raise ValueError('frozen probe bytes')
            batches=unpack(data)
            for batch,arrays in batches:
                for rollout,tensors in zip(batch['rollouts'],arrays):
                    if not runtime.verify(rollout,tensors):raise ValueError('fresh verification')
            results.append(dict(index=row['index'],artifact=row['artifact'],artifact_sha256=row['artifact_sha256'],rollouts_verified=len(batches[0][0]['rollouts']),positive=row['positive'],negative=row['negative']))
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
        batch=dict(env_id=runtime.spec.id,index=index,checkpoint=plan['checkpoint']['id'],rollouts=[v[0] for v in chosen]);data=pack([(batch,[v[1] for v in chosen])]);name='index-'+str(index)+'.zip';(out/name).write_bytes(data)
        row=dict(index=index,attempts=attempts,positive=int('positive' in found),negative=int('negative' in found),qualifying_K1L1=len(found)==2,artifact=name,artifact_sha256=hashlib.sha256(data).hexdigest(),artifact_size=len(data));rows.append(row)
        (out/'search.json').write_bytes(canonical(dict(checkpoint=plan['checkpoint']['id'],rows=rows,search_budget=plan['search_budget'],policy_kind='autoregressive',harness=plan['harness'],training=False,chain_transactions=False,payable=False,completed_at=time.time())))
    print(json.dumps([dict(index=v['index'],attempts=len(v['attempts']),positive=v['positive'],negative=v['negative']) for v in rows]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True,type=Path);p.add_argument('--authority',required=True);p.add_argument('--out',required=True,type=Path);p.add_argument('--verify',action='store_true');a=p.parse_args();execute(approved(json.loads(a.plan.read_text()),a.authority),a.out,a.verify)
if __name__=='__main__':main()
