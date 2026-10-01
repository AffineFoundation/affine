"""Signed standalone GPU qualification of fixed-reference epoch optimizer."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,base64,gc,hashlib,json,time,subprocess
from pathlib import Path
from nacl.signing import VerifyKey
from subnet.storage import canonical

def run(document,authority,out):
    if document['signer']!=authority:raise ValueError('operator authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),base64.b64decode(document['signature'],validate=True))
    plan=document['payload']
    if plan.get('payable') is not False or plan.get('chain_transactions') is not False:raise ValueError('nonpayable control')
    expected={str(p) for p in Path('subnet').glob('*.py')}
    if set(plan['source_files'])!=expected:raise ValueError('exact source closure')
    for name,sha in plan['source_files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('source bytes')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan['probe_sha256']:raise ValueError('probe source')
    from subnet.epoch_optimizer import POLICY,train_epoch
    if plan['training_policy']!=POLICY or plan['steps']!=3:raise ValueError('optimizer qualification policy')
    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])<8192:time.sleep(5)
    import torch
    from subnet.gpu_runtime import GPURuntime
    from subnet.batches import unpack
    from subnet.model import model_files
    data=Path(plan['artifact_path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=plan['artifact_sha256']:raise ValueError('approved audited bytes')
    batch,arrays=unpack(data)[0];env=plan['environment'];harness=plan['harness']
    runtime=GPURuntime(plan['checkpoint_path'],plan['checkpoint']['files'],env,harness)
    positives=[];negatives=[]
    for rollout,tensors in zip(batch['rollouts'],arrays):
        if not runtime.verify(rollout,tensors):raise ValueError('fresh full inference/native audit')
        (positives if rollout['classification']=='positive' else negatives).append(rollout)
    if len(positives)!=1 or len(negatives)!=1:raise ValueError('exact K1L1 qualification')
    out.mkdir(parents=True,exist_ok=True);definition=dict(env_id=env['id'],spec=env,harness=harness)
    destination,updates=train_epoch(runtime,[(definition,positives[0],negatives[0])],out,steps=3)
    files=model_files(destination);cp=hashlib.sha256(canonical(files)).hexdigest()
    if files.get('model.safetensors')==plan['checkpoint']['files'].get('model.safetensors'):raise ValueError('unchanged model')
    del runtime;gc.collect();torch.cuda.empty_cache()
    fresh=GPURuntime(destination,files,env,harness);rollout,tensors=fresh.rollout(0,9001)
    if not fresh.verify(rollout,tensors):raise ValueError('updated model full proof/native replay')
    report=dict(training_policy=POLICY,updates=updates,new_checkpoint=dict(id=cp,files=files,path=str(destination)),input_checkpoint=plan['checkpoint']['id'],fully_audited_pair=True,optimizer_steps=3,reference_fixed_across_steps=True,optimizer_persistent_across_steps=True,fresh_updated_model_proof_replay=True,updated_rollout_reward=rollout['reward'],updated_rollout_sha256=hashlib.sha256(canonical(rollout)).hexdigest(),quality_improvement_verified=False,common_epoch_complete=False,payable=False,chain_transactions=False,completed_at=time.time())
    (out/'report.json').write_bytes(canonical(report));print(json.dumps(dict(checkpoint=cp,losses=[u['losses'][0] for u in updates],state_steps=[u['optimizer_state_steps'] for u in updates],fresh_updated_model_proof_replay=True)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path);p.add_argument('--authority',required=True);p.add_argument('--out',required=True,type=Path);a=p.parse_args();run(json.loads(a.plan.read_text()),a.authority,a.out)
if __name__=='__main__':main()
