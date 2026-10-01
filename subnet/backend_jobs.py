"""Operator-authorized GPU jobs. No wallet, bucket credentials or chain writer."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import re
import time
import sys
import importlib.abc
import importlib.util
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from nacl.signing import VerifyKey

REVISION = 'cuda-bf16-eager-sm86-v1'
NUMERICAL_POLICY = dict(logprob_atol=1e-5, logprob_rtol=0, toploc_exp_mismatches=0,
                        toploc_mant_err_mean=0, toploc_mant_err_median=0)
BACKEND_PROFILE = dict(device='cuda', dtype='bfloat16', attention='eager', sm=[8,6],
    tf32=False, deterministic_algorithms=True, cublas_workspace_config=':4096:8',
    native_toploc_threads=2, torch_threads=2)
SOURCE_FILES = tuple('subnet/'+n+'.py' for n in
    ('backend_jobs','gpu_runtime','model','harness','environments','proofs','batches','protocol'))
ROLES = {'mine','verify','train','evaluate','upload'}
HEAD_POLICY='frozen-feature-head-adamw-v1'
FULL_POLICY='bf16-full-adamw-checkpointed-v1'

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def signed(value, authority):
    if value.get('signer')!=authority:raise ValueError('job authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(value['payload']),base64.b64decode(value['signature'],validate=True))
    return value['payload']

def r2_url(url, operation):
    p=urlparse(url);q=parse_qs(p.query)
    if p.scheme!='https' or not (p.hostname or '').endswith('.r2.cloudflarestorage.com') or p.username or p.password or p.port not in (None,443) or p.fragment:
        raise ValueError('direct R2 capability required')
    if q.get('X-Amz-Algorithm')!=['AWS4-HMAC-SHA256'] or not q.get('X-Amz-Signature'):
        raise ValueError('signed R2 capability required')
    if operation not in ('GET','PUT'):raise ValueError('capability operation')
    return url

def file_map(files):
    if not isinstance(files,dict) or not files or len(files)>32:raise ValueError('checkpoint files')
    for name,sha in files.items():
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',name) or name.startswith('.') or Path(name).suffix not in {'.json','.safetensors','.txt','.model','.jinja','.tiktoken'}:
            raise ValueError('checkpoint safe file allowlist')
        if not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{64}',sha):raise ValueError('checkpoint SHA')
    if 'config.json' not in files or not any(n.endswith('.safetensors') for n in files):raise ValueError('safe model checkpoint required')
    return hashlib.sha256(canonical(files)).hexdigest()

def mining_window(manifest, now=None):
    now=time.time() if now is None else now
    if any(type(manifest.get(k)) not in (int,float) for k in ('start','deadline')) or not manifest['start']<=now<manifest['deadline']:
        raise ValueError('signed mining epoch window closed')

def validate(envelope, authority, now=None):
    """Pure authorization/policy check; never opens an artifact or imports runtime."""
    job=signed(envelope,authority);now=time.time() if now is None else now
    if job.get('schema')!=1 or job.get('role') not in ROLES:raise ValueError('job role/schema')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',job.get('job_id','')):raise ValueError('job ID')
    if any(type(job.get(k)) not in (int,float) for k in ('created_at','expires_at')) or not job['created_at']<=now<job['expires_at'] or job['expires_at']-job['created_at']>86400:raise ValueError('job expired/time budget')
    manifest=signed(job['manifest'],authority)
    if manifest.get('model_runtime_revision')!=REVISION or manifest.get('numerical_policy')!=NUMERICAL_POLICY or manifest.get('backend_profile')!=BACKEND_PROFILE:raise ValueError('GPU profile or numerical policy')
    cp=manifest['checkpoint'];identifier=file_map(cp['files'])
    if cp.get('id')!=identifier:raise ValueError('checkpoint identity')
    urls=cp.get('read_urls',{})
    if urls and set(urls)!=set(cp['files']):raise ValueError('checkpoint capability file binding')
    for url in urls.values():r2_url(url,'GET')
    if not set(SOURCE_FILES)<=set(job.get('source_files',{})):raise ValueError('missing worker source pins')
    for name,sha in job['source_files'].items():
        if not name.startswith('subnet/') or '..' in name or Path(name).suffix!='.py' or not re.fullmatch('[0-9a-f]{64}',sha):raise ValueError('worker source pin')
    if set(job.get('runtime_versions',{}))!={'torch','transformers','toploc'}:raise ValueError('runtime version pins')
    for obj in job.get('submissions',[]):
        r2_url(obj['url'],'GET')
        if not re.fullmatch('[0-9a-f]{64}',obj['sha256']):raise ValueError('submission digest')
    if len(job.get('submissions',[]))>256:raise ValueError('submission job budget')
    if job['role'] in ('mine','verify','train'):
        if job['role']!='mine' and not job.get('submissions'):raise ValueError('no submissions')
        if manifest.get('audit_policy',{}).get('mode')!='full':raise ValueError('GPU training jobs require full audit')
        if any(type(manifest.get(k)) is not int or not 1<=manifest[k]<=16 for k in ('K','L')):raise ValueError('class quota')
    if job['role']=='mine':
        mining_window(manifest,now)
        if not re.fullmatch('[0-9a-f]{64}',job.get('miner_id','')):raise ValueError('owned miner identity')
        if type(job.get('search_budget')) is not int or not 1<=job['search_budget']<=128 or type(job.get('seed_start')) is not int or job['seed_start']<0:raise ValueError('mining search budget')
        r2_url(job['capability']['put_url'],'PUT')
        if job['capability'].get('headers')!={'Content-Type':'application/octet-stream'}:raise ValueError('signed upload headers')
    if job['role']=='train' and job.get('training_policy',HEAD_POLICY) not in (HEAD_POLICY,FULL_POLICY):raise ValueError('unapproved training objective')
    if job['role']=='train' and (type(job.get('steps')) is not int or not 1<=job['steps']<=32):raise ValueError('training step budget')
    if job['role']=='evaluate':
        if not job.get('heldout') or len(job['heldout'])>64:raise ValueError('heldout budget')
        for row in job['heldout']:
            if len(row['indices'])!=len(row['seeds']) or not 1<=len(row['indices'])<=32 or any(type(i) is not int or i<0 for i in row['indices']+row['seeds']):raise ValueError('heldout index/seed budget')
            if row['harness'].get('policy')!='autoregressive' or row['harness'].get('turn_overrides'):raise ValueError('heldout must use free autoregressive policy')
    if job['role']=='upload':
        if set(job.get('put_urls',{}))!=set(cp['files']):raise ValueError('upload capability file binding')
        for url in job['put_urls'].values():r2_url(url,'PUT')
    return job,manifest

def get_object(url, expected, destination, limit):
    import requests
    temporary=destination.with_suffix(destination.suffix+'.partial');h=hashlib.sha256();size=0
    try:
        with requests.get(r2_url(url,'GET'),stream=True,timeout=180,allow_redirects=False) as response:
            if response.status_code!=200:raise ValueError('R2 GET status '+str(response.status_code))
            with temporary.open('wb') as f:
                for part in response.iter_content(1024*1024):
                    size+=len(part)
                    if size>limit:raise ValueError('artifact size budget')
                    h.update(part);f.write(part)
        if h.hexdigest()!=expected:raise ValueError('artifact digest')
        temporary.replace(destination)
    finally:temporary.unlink(missing_ok=True)

def checkpoint(manifest, workspace, cache=None):
    cp=manifest['checkpoint'];target=Path(cache) if cache else workspace/'checkpoints'/cp['id']
    target.mkdir(parents=True,exist_ok=True)
    for name,sha in cp['files'].items():
        path=target/name
        if path.is_symlink():raise ValueError('checkpoint symlink')
        if path.is_file() and digest(path)==sha:continue
        if cache:raise ValueError('approved cached checkpoint mismatch')
        get_object(cp['read_urls'][name],sha,path,20_000_000_000)
    from .model import model_files
    if model_files(target)!=cp['files']:raise ValueError('checkpoint exact allowlist')
    return target

def audit(data, manifest, runtime):
    from .batches import unpack
    from .protocol import entries, entry, classification, sample_key
    definitions=entries(manifest);records=unpack(data)
    if len(records)>manifest.get('max_batches',4):raise ValueError('batch quota')
    outcomes=[];accepted=[];pairs=[];seen=set()
    for number,(batch,arrays) in enumerate(records):
        try:
            definition=entry(manifest,batch.get('env_id'));index=batch['index'];key=sample_key(batch)
            if batch.get('schema')!=2 or batch['epoch']!=manifest['epoch'] or batch['checkpoint']!=manifest['checkpoint']['id'] or batch.get('sample_index')!=index or type(index) is not int or index not in definition['indices'] or key in seen:raise ValueError('batch binding')
            selected=runtime.for_environment(definition['spec'],definition['harness'])
            if batch.get('environment_version')!=selected.spec.version:raise ValueError('environment version')
            seen.add(key);rolls=batch['rollouts'];tokens=set()
            if len(rolls)!=manifest['K']+manifest['L'] or len(arrays)!=len(rolls):raise ValueError('sample count')
            for rollout,probs in zip(rolls,arrays):
                signature=tuple(tuple(t['output']) for t in rollout['turns'])
                if signature in tokens or rollout['index']!=index or rollout.get('env_id')!=definition['env_id']:raise ValueError('sample binding/duplicate')
                tokens.add(signature)
                if not selected.verify(rollout,probs):raise ValueError('inference or replay')
            pos=[r for r in rolls if classification(r)=='positive'];neg=[r for r in rolls if classification(r)=='negative']
            if len(pos)!=manifest['K'] or len(neg)!=manifest['L']:raise ValueError('positive/negative quota')
            accepted.append(batch);pairs.extend((definition,p,n) for p,n in zip(pos,neg))
            outcomes.append(dict(batch=number,env_id=definition['env_id'],index=index,valid=True,fully_audited=True))
        except (ValueError,KeyError,TypeError,IndexError) as error:
            outcomes.append(dict(batch=number,valid=False,reason=type(error).__name__+': '+str(error)[:300]))
    return dict(epoch=manifest['epoch'],submission_sha256=hashlib.sha256(data).hexdigest(),outcomes=outcomes,accepted=accepted,training_eligibility='fully-audited-only'),pairs

def full_parameter_train(runtime, pairs, destination, steps=1):
    """Measured, separately selected full BF16 AdamW; not the head-only control."""
    import gc
    import torch
    model=runtime.model
    if any(isinstance(m,torch.nn.Dropout) and m.p>0 for m in model.modules()) or getattr(model.config,'attention_dropout',0)!=0:
        raise ValueError('full training profile requires dropout-free model')
    parameters=list(model.parameters());count=sum(p.numel() for p in parameters)
    free,total=torch.cuda.mem_get_info()
    # Four BF16 buffers (parameter, gradient, two moments), plus bounded reserve.
    # Parameters are already loaded, so require the three remaining buffers.
    required=count*6+3*1024**3
    if free<required:raise ValueError('full optimizer GPU memory reserve')
    def sequence(rollout):
        total_lp=0;tokens=0
        for turn in rollout['turns']:
            prompt,output=turn['prompt'],turn['output']
            logits=model(torch.tensor([prompt+output],device='cuda'),use_cache=False).logits[0,len(prompt)-1:len(prompt)+len(output)-1]
            lp=torch.log_softmax(logits.float(),-1)
            total_lp=total_lp+lp.gather(1,torch.tensor(output,device='cuda')[:,None]).sum();tokens+=len(output)
        return total_lp/tokens
    with torch.no_grad():references=[float(sequence(p)-sequence(n)) for p,n in pairs]
    for param in parameters:param.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    optimizer=torch.optim.AdamW(parameters,lr=1e-5,foreach=False);losses=[]
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(steps):
            pos,neg=pairs[step%len(pairs)];optimizer.zero_grad(set_to_none=True)
            loss=-torch.nn.functional.logsigmoid(.1*(sequence(pos)-sequence(neg)-references[step%len(pairs)]))
            if not torch.isfinite(loss):raise ValueError('nonfinite full training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1);optimizer.step();losses.append(float(loss.detach()))
        state_dtypes=sorted({str(value.dtype) for row in optimizer.state.values() for name,value in row.items() if name!='step' and hasattr(value,'dtype')})
        destination=Path(destination)
        if destination.exists():raise ValueError('refuse checkpoint overwrite')
        destination.mkdir(parents=True);model.save_pretrained(destination,safe_serialization=True);runtime.tokenizer.save_pretrained(destination)
        return dict(steps=steps,losses=losses,training_policy=FULL_POLICY,objective='reference-relative full-model sequence preference',
            full_model_finetune=True,trainable_parameters=count,learning_rate=1e-5,gradient_checkpointing=True,parameter_dtype='torch.bfloat16',optimizer_state_dtypes=state_dtypes,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_free_before_bytes=free,gpu_required_additional_bytes=required)
    finally:
        optimizer.zero_grad(set_to_none=True);del optimizer;model.eval();model.gradient_checkpointing_disable();gc.collect();torch.cuda.empty_cache()

class FreshSourceFinder(importlib.abc.MetaPathFinder):
    """Never accept a cached bytecode file as evidence of pinned Python source."""
    def __init__(self, root):self.root=root
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith('subnet.'):return None
        location=self.root.joinpath(*fullname.split('.')).with_suffix('.py')
        if not location.is_file():return None
        class Loader(importlib.abc.Loader):
            def create_module(self,spec):return None
            def exec_module(self,module):
                module.__file__=str(location)
                exec(compile(location.read_bytes(),str(location),'exec'),module.__dict__)
        return importlib.util.spec_from_file_location(fullname,location,loader=Loader())

def install_source_loader(root):
    for name in SOURCE_FILES:
        module_name=name[:-3].replace('/','.')
        if module_name in sys.modules and module_name!='subnet.backend_jobs':
            raise ValueError('GPU worker requires fresh process before runtime imports')
    sys.meta_path.insert(0,FreshSourceFinder(root))

def execute(envelope, authority, workspace, cache=None, runtime_factory=None):
    job,manifest=validate(envelope,authority)
    root=Path(__file__).resolve().parent.parent
    for name,expected in job['source_files'].items():
        if (root/name).is_symlink() or digest(root/name)!=expected:raise ValueError('worker source mismatch')
    for name,expected in job['runtime_versions'].items():
        if version(name)!=expected:raise ValueError('runtime package mismatch')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('CUDA environment profile')
    install_source_loader(root)
    workspace=Path(workspace);out=workspace/'jobs'/job['job_id']
    out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
    approved=checkpoint(manifest,workspace,cache)
    report=dict(schema=1,job_id=job['job_id'],role=job['role'],operator=authority,
        job_sha256=hashlib.sha256(canonical(job)).hexdigest(),checkpoint=manifest['checkpoint']['id'],
        epoch=manifest['epoch'],backend_profile=BACKEND_PROFILE,numerical_policy=NUMERICAL_POLICY,
        source_files=job['source_files'],runtime_versions=job['runtime_versions'],
        chain_transactions=False,full_model_finetune=False,execution_resources_enforced=False)
    if job['role']=='upload':
        import requests
        for name,url in job['put_urls'].items():
            with (approved/name).open('rb') as body:
                response=requests.put(url,data=body,headers={'Content-Type':'application/octet-stream'},timeout=600,allow_redirects=False)
            if response.status_code not in (200,201,204):raise ValueError('R2 PUT status '+str(response.status_code))
        report['uploaded_files']=manifest['checkpoint']['files']
    else:
        from .protocol import entries,entry
        from .gpu_runtime import GPURuntime
        factory=runtime_factory or GPURuntime;first=entries(manifest)[0]
        runtime=factory(approved,manifest['checkpoint']['files'],first['spec'],first['harness'])
        if job['role']=='mine':
            from .batches import pack
            import requests
            batches=[];search=[]
            for definition in entries(manifest):
                selected=runtime.for_environment(definition['spec'],definition['harness'])
                for index in definition['indices']:
                    mining_window(manifest)
                    classes={'positive':[],'negative':[]};fingerprints=set()
                    for attempt in range(job['search_budget']):
                        mining_window(manifest)
                        rollout,arrays=selected.rollout(index,job['seed_start']+attempt)
                        label=rollout['classification'];signature=tuple(tuple(t['output']) for t in rollout['turns'])
                        quota=manifest['K'] if label=='positive' else manifest['L']
                        if label in classes and len(classes[label])<quota and signature not in fingerprints:
                            classes[label].append((rollout,arrays));fingerprints.add(signature)
                        if len(classes['positive'])==manifest['K'] and len(classes['negative'])==manifest['L']:break
                    search.append(dict(env_id=definition['env_id'],index=index,attempts=attempt+1,positive=len(classes['positive']),negative=len(classes['negative'])))
                    if len(classes['positive'])==manifest['K'] and len(classes['negative'])==manifest['L']:
                        found=classes['positive']+classes['negative']
                        batch=dict(schema=2,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],env_id=definition['env_id'],environment_version=selected.spec.version,index=index,sample_index=index,rollouts=[r for r,a in found])
                        batches.append((batch,[a for r,a in found]))
                    if len(batches)>=manifest.get('max_batches',4):break
            if not batches:raise ValueError('GPU bounded search found no complete batch')
            data=pack(batches);artifact=out/'submission.zip';artifact.write_bytes(data)
            mining_window(manifest)
            response=requests.put(job['capability']['put_url'],data=data,headers=job['capability']['headers'],timeout=180,allow_redirects=False)
            if response.status_code not in (200,201,204):raise ValueError('R2 PUT status '+str(response.status_code))
            report.update(miner_id=job['miner_id'],submission_sha256=hashlib.sha256(data).hexdigest(),submission_size=len(data),batches=len(batches),search=search,operator_authorized_experiment=True)
        elif job['role'] in ('verify','train'):
            reports=[];pairs=[]
            for i,obj in enumerate(job['submissions']):
                path=out/('submission-'+str(i)+'.zip');get_object(obj['url'],obj['sha256'],path,100_000_000)
                result,verified=audit(path.read_bytes(),manifest,runtime);reports.append(result);pairs.extend(verified)
            report['audits']=reports
            if job['role']=='train':
                if not pairs:raise ValueError('no verified training pairs')
                metrics=[];destination=None
                # Each update uses its batch's exact approved environment/harness.
                for step in range(job['steps']):
                    definition,pos,neg=pairs[step%len(pairs)]
                    runtime.configure(definition['spec'],definition['harness'])
                    destination=out/('checkpoint-step-'+str(step+1))
                    policy=job.get('training_policy',HEAD_POLICY)
                    update=full_parameter_train(runtime,[(pos,neg)],destination,steps=1) if policy==FULL_POLICY else runtime.train([(pos,neg)],destination,steps=1)
                    update['training_policy']=policy;metrics.append(update)
                from .model import model_files
                files=model_files(destination)
                if files.get('model.safetensors')==manifest['checkpoint']['files'].get('model.safetensors'):raise ValueError('training did not change checkpoint weights')
                report['training']=dict(steps=job['steps'],updates=metrics,training_policy=job.get('training_policy',HEAD_POLICY),full_model_finetune=job.get('training_policy',HEAD_POLICY)==FULL_POLICY)
                report['full_model_finetune']=report['training']['full_model_finetune']
                report['new_checkpoint']=dict(id=file_map(files),files=files,path=str(destination))
        else:
            values=[]
            for row in job['heldout']:
                definition=entry(manifest,row['env_id'])
                if set(row['indices'])&set(definition['indices']):raise ValueError('heldout/training overlap')
                selected=runtime.for_environment(definition['spec'],row['harness'])
                for index,seed in zip(row['indices'],row['seeds']):
                    doc,arrays=selected.rollout(index,seed)
                    if not selected.verify(doc,arrays):raise ValueError('heldout audit')
                    values.append(dict(env_id=row['env_id'],index=index,seed=seed,reward=doc['reward'],classification=doc['classification'],verified=True))
            report['heldout']=values
    report['success']=True;report['completed_at']=time.time()
    (out/'report.json').write_bytes(canonical(report));return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('job');parser.add_argument('--authority',required=True);parser.add_argument('--workspace',required=True);parser.add_argument('--checkpoint-cache')
    args=parser.parse_args();data=Path(args.job).read_bytes()
    if len(data)>4_000_000:raise ValueError('job envelope size budget')
    report=execute(json.loads(data),args.authority,args.workspace,args.checkpoint_cache)
    print(json.dumps(dict(job_id=report['job_id'],role=report['role'],success=True,checkpoint=report.get('new_checkpoint',{}).get('id',report['checkpoint']))))
if __name__=='__main__':main()
