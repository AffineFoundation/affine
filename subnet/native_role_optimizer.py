"""Separate operator-authorized full CPU optimizer for audited native agent pairs.

Auxiliary model outputs remain proof evidence; no auxiliary output is a loss
label. This controlled job never imports network wallet or GPU service code.
"""
import argparse,base64,fcntl,hashlib,json,math,os,pathlib,resource,shutil,sys,time
from nacl.signing import SigningKey,VerifyKey

ROOT=pathlib.Path(__file__).resolve().parent.parent
AUTHORITY='d54a3a345d0de3e2c7898f30c0942d78f931f8c4b8036ffdc6adffcd2525062f'
REVISION='cpu-fp32-full-native-reference-preference-v1'
OPTIMIZER={'version':REVISION,'steps':1,'lr':1e-5,'beta':.1,'weight_decay':0.,'max_grad_norm':1.,'gradient_checkpointing':True,'use_cache':False}
SOURCE_FILES=tuple('subnet/'+name+'.py' for name in ('native_role_optimizer','native_role_batch','native_auxiliary_roles','native_tau2_curated','native_tau2_public_policy','native_tau2_model','native_tau2_replay','native_tau2_attestation','native_tau2_probe','model','harness','proofs'))

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def source_map():return {name:file_sha(ROOT/name) for name in SOURCE_FILES}
def runtime_environment():
    from importlib.metadata import version
    return {'python_version':sys.version,'interpreter_sha256':file_sha(pathlib.Path(sys.executable).resolve()),'package_versions':{name:version(name) for name in ('torch','transformers','toploc','tau2','litellm','verifiers')}}
def signed(x,key):return {'payload':x,'signer':key.verify_key.encode().hex(),'signature':base64.b64encode(key.sign(canonical(x)).signature).decode()}
def authenticate(x,authority):
    if x.get('signer')!=authority:raise ValueError('native optimizer authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(x['payload']),base64.b64decode(x['signature'],validate=True));return x['payload']
def bounded_path(value,prefix):
    path=pathlib.Path(value).resolve();base=(ROOT/prefix).resolve()
    if path==base or base not in path.parents:raise ValueError('native optimizer path scope')
    return path

def validate_job(envelope,authority):
    # Signature first: no job-specified paths, artifacts or model code read yet.
    job=authenticate(envelope,authority)
    if authority!=AUTHORITY or job.get('role')!='native-agent-full-optimizer' or job.get('revision')!=REVISION or job.get('optimizer')!=OPTIMIZER or job.get('payable') is not False or job.get('chain_transactions') is not False:raise ValueError('native optimizer role/policy')
    if job.get('source_files')!=source_map():raise ValueError('native optimizer source closure')
    if job.get('runtime_environment')!=runtime_environment():raise ValueError('native optimizer interpreter/package closure')
    if job.get('budget')!={'max_parameters':200000000,'max_context':8192,'max_peak_rss_bytes':64*1024**3,'min_disk_free_bytes':2*1024**3}:raise ValueError('native optimizer budget')
    for name in ('positive','negative'):bounded_path(job[name]['path'],'state/native-tau2-probe')
    bounded_path(job['checkpoint']['path'],'state/service-conformance/checkpoints')
    bounded_path(job['destination'],'state/native-tau2-training')
    return job

def score(model,prompt,output):
    import torch
    logits=model(torch.tensor([prompt+output]),use_cache=False).logits[0,len(prompt)-1:len(prompt)+len(output)-1]
    lp=torch.log_softmax(logits.float(),-1)
    return lp.gather(1,torch.tensor(output)[:,None]).mean()

def reference_score(out,authority,view):
    import numpy as np
    from .native_tau2_model import authenticate as native_authenticate
    receipts=json.loads((pathlib.Path(out)/'receipts.json').read_text())
    record=next(native_authenticate(r,authority) for r in receipts if r['payload']['role']=='agent')
    if record['prompt']!=view['prompt'] or record['output']!=view['output']:raise ValueError('reference agent target binding')
    name=record['probabilities_file']
    if pathlib.Path(name).name!=name or file_sha(pathlib.Path(out)/name)!=record['probabilities_sha256']:raise ValueError('reference LP integrity')
    lp=np.load(pathlib.Path(out)/name,allow_pickle=False)
    if lp.dtype!=np.float32 or lp.shape[0]!=len(view['output']) or not np.isfinite(lp).all():raise ValueError('reference LP framing')
    import torch
    return float(torch.from_numpy(lp).gather(1,torch.tensor(view['output'])[:,None]).mean())

def _execute_claimed(job,authority,seed_path):
    from .native_tau2_model import profile,make_runtime,authenticate as native_authenticate
    profile()
    from .native_role_batch import describe_sample,describe_batch,preference_pair
    positive=describe_sample(job['positive']['path'],authority,job['environment_index']);negative=describe_sample(job['negative']['path'],authority,job['environment_index'])
    if digest(positive)!=job['positive']['descriptor_hash'] or digest(negative)!=job['negative']['descriptor_hash']:raise ValueError('native training artifact descriptor')
    batch=describe_batch([positive,negative],1,1);pair=preference_pair(positive,negative)
    plan=native_authenticate(json.loads((pathlib.Path(job['positive']['path'])/'plan.json').read_text()),authority)
    if pair['checkpoint']!=job['checkpoint']['id'] or job['checkpoint']['files']!=plan['checkpoint']['files']:raise ValueError('native optimizer approved checkpoint')
    destination=pathlib.Path(job['destination']);out=destination.parent
    if destination.exists():raise ValueError('native optimizer namespace already exists')
    out.mkdir(parents=True,exist_ok=True);out.chmod(0o700)
    model_bytes=sum((pathlib.Path(job['checkpoint']['path'])/name).stat().st_size for name in job['checkpoint']['files'])
    if shutil.disk_usage(out).free<max(job['budget']['min_disk_free_bytes'],2*model_bytes+512*1024**2):raise ValueError('native checkpoint disk headroom')
    key=SigningKey(bytes.fromhex(pathlib.Path(seed_path).read_text()))
    if key.verify_key.encode().hex()!=authority:raise ValueError('native result signer')
    import torch
    torch.set_default_device('cpu')
    runtime=make_runtime(job['checkpoint']['path'],plan);model=runtime.model
    count=sum(p.numel() for p in model.parameters())
    if count>job['budget']['max_parameters'] or len(pair['prompt'])+max(len(pair['chosen']),len(pair['rejected']))>job['budget']['max_context']:raise ValueError('native optimizer model/context budget')
    if any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('full CPU float32 model required')
    for p in model.parameters():p.requires_grad_(True)
    torch.manual_seed(job['seed']);torch.use_deterministic_algorithms(True)
    old_cache=model.config.use_cache;model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if getattr(model.config,'attention_dropout',0.)!=0. or any(isinstance(m,torch.nn.Dropout) and m.p!=0 for m in model.modules()):raise ValueError('native deterministic zero-dropout training required')
    model.train()
    if not model.is_gradient_checkpointing or not model.training:raise ValueError('native gradient checkpointing not active')
    chosen_view=next(r for r in positive['training_view'] if r['role']=='agent');rejected_view=next(r for r in negative['training_view'] if r['role']=='agent')
    ref_chosen=reference_score(job['positive']['path'],authority,chosen_view);ref_rejected=reference_score(job['negative']['path'],authority,rejected_view)
    optimizer=torch.optim.AdamW(model.parameters(),lr=OPTIMIZER['lr'],weight_decay=OPTIMIZER['weight_decay']);optimizer.zero_grad(set_to_none=True);started=time.time()
    chosen=score(model,pair['prompt'],pair['chosen']);rejected=score(model,pair['prompt'],pair['rejected'])
    if abs(float(chosen.detach())-ref_chosen)>1e-5 or abs(float(rejected.detach())-ref_rejected)>1e-5:raise ValueError('reference computation mismatch')
    margin=(chosen-ref_chosen)-(rejected-ref_rejected);loss=torch.nn.functional.softplus(-OPTIMIZER['beta']*margin)
    if not torch.isfinite(loss):raise ValueError('nonfinite native preference loss')
    loss.backward();gradients=sum(p.grad is not None for p in model.parameters());norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),OPTIMIZER['max_grad_norm']))
    if not math.isfinite(norm) or norm<=0 or gradients!=len(list(model.parameters())):raise ValueError('full model gradient coverage')
    optimizer.step();optimizer.zero_grad(set_to_none=True);model.gradient_checkpointing_disable();model.config.use_cache=old_cache;model.eval()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError('native updated parameters nonfinite')
    from safetensors import safe_open
    changed=0;elements=0;delta_sq=0.;max_abs=0.
    with safe_open(str(pathlib.Path(job['checkpoint']['path'])/'model.safetensors'),framework='pt',device='cpu') as original:
        for name,p in model.named_parameters():
            before=original.get_tensor(name);delta=p.detach()-before;n=int(torch.count_nonzero(delta));elements+=n;changed+=int(n>0);delta_sq+=float(delta.double().square().sum());max_abs=max(max_abs,float(delta.abs().max()))
    if not changed or not elements:raise ValueError('native optimizer weights unchanged')
    with torch.no_grad():after_chosen=float(score(model,pair['prompt'],pair['chosen']));after_rejected=float(score(model,pair['prompt'],pair['rejected']))
    if not math.isfinite(after_chosen) or not math.isfinite(after_rejected):raise ValueError('native post-update probability nonfinite')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if peak>job['budget']['max_peak_rss_bytes']:raise ValueError('native optimizer peak memory budget exceeded')
    if source_map()!=job['source_files'] or runtime_environment()!=job['runtime_environment']:raise ValueError('native optimizer source/runtime changed during job')
    destination.mkdir();model.save_pretrained(destination,safe_serialization=True);runtime.tokenizer.save_pretrained(destination)
    from .model import model_files
    files=model_files(destination)
    if set(files)!=set(job['checkpoint']['files']):raise ValueError('native exported checkpoint file closure')
    checkpoint={'id':digest(files),'files':files}
    if checkpoint['id']==job['checkpoint']['id']:raise ValueError('native checkpoint did not change')
    report={'revision':REVISION,'job_hash':digest(job),'source_files':job['source_files'],'runtime_environment':job['runtime_environment'],'objective':'agent-only-native-outcome-reference-preference-v1','optimizer':OPTIMIZER,'parameters':count,'gradient_tensors':gradients,'changed_parameter_tensors':changed,'changed_parameter_elements':elements,'parameter_delta_l2':math.sqrt(delta_sq),'parameter_delta_max_abs':max_abs,'loss':float(loss.detach()),'gradient_norm':norm,'reference_chosen_mean_logprob':ref_chosen,'reference_rejected_mean_logprob':ref_rejected,'after_chosen_mean_logprob':after_chosen,'after_rejected_mean_logprob':after_rejected,'preference_margin_change':(after_chosen-after_rejected)-(ref_chosen-ref_rejected),'pair_hash':digest(pair),'native_batch_hash':digest(batch),'auxiliary_tokens_in_loss':False,'full_model_finetune':True,'gradient_checkpointing_activated':True,'steps':1,'peak_rss_bytes':peak,'started_at':started,'completed_at':time.time(),'previous_checkpoint':job['checkpoint']['id'],'checkpoint':checkpoint,'training_performed':True,'quality_improvement_claimed':False,'payable':False,'chain_transactions':False,'gpu_used':False}
    (out/'training-receipt.json').write_text(json.dumps(signed(report,key),indent=2)+'\n');(out/'checkpoint-manifest.json').write_text(json.dumps({'checkpoint':checkpoint},indent=2)+'\n');print(json.dumps(report,indent=2),flush=True);return report

def execute(envelope,authority,seed_path):
    job=validate_job(envelope,authority)
    destination=bounded_path(job['destination'],'state/native-tau2-training');out=destination.parent
    out.mkdir(parents=True,exist_ok=True);out.chmod(0o700)
    lock=(out/'optimizer.lock').open('a+')
    try:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('native optimizer job already active')
        status=out/'optimizer-status.json'
        if status.exists() or destination.exists():raise ValueError('native optimizer namespace already attempted; no retry')
        start=pathlib.Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]
        marker={'status':'running','job_hash':digest(job),'pid':os.getpid(),'pid_start':start,'started_at':time.time()}
        status.write_text(json.dumps(marker)+'\n')
        try:
            result=_execute_claimed(job,authority,seed_path)
        except Exception as e:
            marker.update(status='failed',error_type=type(e).__name__,completed_at=time.time());status.write_text(json.dumps(marker)+'\n');raise
        marker.update(status='complete',checkpoint=result['checkpoint']['id'],completed_at=time.time());status.write_text(json.dumps(marker)+'\n');return result
    finally:lock.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--authority',required=True);p.add_argument('--seed-file',required=True);a=p.parse_args();execute(json.loads(pathlib.Path(a.job).read_text()),a.authority,a.seed_file)
if __name__=='__main__':main()
