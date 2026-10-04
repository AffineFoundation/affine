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
    ('audit_policy','auditing','backend_jobs','backend_profiles','artifact_budget','task_assets','math_corpus_provider','math_corpus_assets','math_corpus','source_bootstrap','gpu_runtime','model','harness','environments','proofs','batches','protocol','forced_sampling'))
ROLES = {'mine','verify','train','evaluate','upload'}
HEAD_POLICY='frozen-feature-head-adamw-v1'
FULL_POLICY='bf16-full-adamw-checkpointed-v1'
FIXED_POLICY='bf16-full-adamw-fixed-epoch-reference-v2'
COVERED_POLICY='bf16-full-adamw-covered-fixed-reference-v3'
PERSISTENT_POLICY='bf16-cpu-fp32-master-task-normalized-persistent-v4'
TRAINING_ATTRIBUTION='verified-pair-v1'

def validate_single_put_sizes(checkpoint, files):
    """Reject unsupported objects before uploading any checkpoint member.

    A single tensor can exceed the requested export shard size, so the
    explicit 4 GB export setting alone is insufficient for arbitrary models.
    Larger objects require the separate multipart transport.
    """
    if any((Path(checkpoint)/name).stat().st_size > 5*1024**3 for name in files):
        raise ValueError('checkpoint object exceeds R2 single PUT limit; multipart required')

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def checkpoint_weights_changed(before, after):
    """Compare every safe weight file, including sharded checkpoints."""
    old={name:sha for name,sha in before.items() if name.endswith('.safetensors')}
    new={name:sha for name,sha in after.items() if name.endswith('.safetensors')}
    if not old or not new:raise ValueError('safe model checkpoint weights required')
    return old!=new

def parameter_value_digest(model):
    """Hash actual parameter values independently of export shard layout."""
    import torch
    h=hashlib.sha256()
    for name,value in sorted(model.named_parameters()):
        h.update(canonical(dict(name=name,dtype=str(value.dtype),shape=list(value.shape))))
        h.update(b'\0')
        raw=value.detach().to('cpu').contiguous().reshape(-1).view(torch.uint8).numpy()
        h.update(memoryview(raw).cast('B'))
        del raw
    return h.hexdigest()

def pair_attribution(definition,positive,negative,step):
    if positive['env_id']!=definition['env_id'] or negative['env_id']!=definition['env_id'] or positive['index']!=negative['index']:
        raise ValueError('training pair environment/index binding')
    result=dict(attribution_revision=TRAINING_ATTRIBUTION,optimizer_step=step+1,
        env_id=definition['env_id'],index=positive['index'],
        positive_rollout_sha256=hashlib.sha256(canonical(positive)).hexdigest(),
        negative_rollout_sha256=hashlib.sha256(canonical(negative)).hexdigest())
    from .sample_harness import VERSION as INDEXED_VERSION
    if isinstance(definition.get('harness'),dict)and definition['harness'].get('version')==INDEXED_VERSION:
        from .protocol import harness_for
        result['resolved_harness_sha256']=hashlib.sha256(canonical(harness_for(definition,positive['index']))).hexdigest()
    return result

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

def mining_definitions(manifest, job):
    """An owned job may narrow its search without narrowing the public challenge."""
    from .protocol import entries
    definitions = entries(manifest)
    subset = job.get('mining_subset')
    if subset is None:return definitions
    known = {row['env_id']: row for row in definitions}
    if not isinstance(subset,dict) or not subset or not set(subset)<=set(known):
        raise ValueError('owned mining subset environment')
    result = []
    for env_id, indices in subset.items():
        if (not isinstance(indices,list) or not 1<=len(indices)<=128 or
                any(type(i)is not int for i in indices) or
                len(indices)!=len(set(indices)) or not set(indices)<=set(known[env_id]['indices'])):
            raise ValueError('owned mining subset outside authorized training indices')
        definition = dict(known[env_id],indices=list(indices))
        from .sample_harness import VERSION,project
        if isinstance(definition.get('harness'),dict) and definition['harness'].get('version')==VERSION:
            definition['harness']=project(definition['harness'],indices,known[env_id]['indices'])
        result.append(definition)
    return result


def mine_cumulative(runtime,manifest,job,upload,clock=None,allow_empty=False):
    """Publish each complete private batch before searching the next task.

    The last acknowledged snapshot is authoritative if subsequent search runs
    out of time. A ten-second reserve avoids initiating overwrites at expiry;
    upload errors remain failures rather than pretending a PUT succeeded.
    """
    from .batches import pack,UploadBudgetExceeded
    from .protocol import entries,harness_for
    clock=clock or time.time
    mining_window(manifest,clock())
    if manifest.get('sampling_contract') is not None:
        limit=manifest['sampling_contract']['max_attempts']
        if type(job['seed_start'])is not int or type(job['search_budget'])is not int or not 0<=job['seed_start']<limit or not 1<=job['search_budget']<=limit-job['seed_start']:
            raise ValueError('owned miner outside signed attempt budget')
    batches=[];search=[];data=None;uploads=0;stopped=False;capacity_reached=False
    def available():
        now=clock()
        return manifest['start']<=now<manifest['deadline']-10
    for definition in mining_definitions(manifest,job):
        if len(batches)>=manifest.get('max_batches',4) or not available():break
        for index in definition['indices']:
            selected=runtime.for_environment(definition['spec'],harness_for(definition,index))
            if not available():stopped=True;break
            classes={'positive':[],'negative':[]};fingerprints=set();attempts=0;observed={'positive':0,'negative':0}
            for attempt in range(job['search_budget']):
                if not available():stopped=True;break
                rollout,arrays=selected.rollout(index,job['seed_start']+attempt);attempts+=1
                label=rollout['classification']
                if label not in classes:raise ValueError('rollout classification')
                observed[label]+=1
                signature=tuple(tuple(t['output']) for t in rollout['turns'])
                quota=manifest['K'] if label=='positive' else manifest['L']
                if label in classes and len(classes[label])<quota and signature not in fingerprints:
                    classes[label].append((rollout,arrays));fingerprints.add(signature)
                if len(classes['positive'])==manifest['K'] and len(classes['negative'])==manifest['L']:break
            search.append(dict(env_id=definition['env_id'],index=index,attempts=attempts,positive=len(classes['positive']),negative=len(classes['negative']),observed_positive=observed['positive'],observed_negative=observed['negative']))
            if len(classes['positive'])==manifest['K'] and len(classes['negative'])==manifest['L']:
                # A rollout can finish across the deadline. Keep the previously
                # uploaded snapshot instead of replacing it with a late object.
                if not available():stopped=True;break
                found=classes['positive']+classes['negative']
                batch=dict(schema=2,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],env_id=definition['env_id'],environment_version=selected.spec.version,index=index,sample_index=index,rollouts=[r for r,a in found])
                candidate=batches+[(batch,[a for r,a in found])]
                from .artifact_budget import for_manifest
                try:
                    candidate_data=(pack(candidate,budget=for_manifest(manifest))
                        if manifest.get('artifact_policy') is not None else pack(candidate))
                except UploadBudgetExceeded:
                    search[-1]['submission_status']='exceeds_cumulative_upload_budget'
                    if batches:capacity_reached=True;break
                    continue
                if not available():stopped=True;break
                upload(candidate_data,min(180,manifest['deadline']-clock()-1))
                batches=candidate;data=candidate_data;uploads+=1
            if stopped or len(batches)>=manifest.get('max_batches',4):break
        if stopped or capacity_reached:break
    if data is None and not allow_empty:raise ValueError('GPU bounded search found no complete batch before epoch window closed')
    return data,dict(batches=len(batches),search=search,cumulative_uploads=uploads,search_stopped_at_deadline=stopped,search_stopped_at_capacity=capacity_reached,mining_status='complete_batches_uploaded' if data is not None else 'no_complete_KL_batch')

def validate(envelope, authority, now=None):
    """Strict authorization/policy admission, including signed subset semantics."""
    return _validate(envelope,authority,now,resolve_source=True)

def _validate(envelope, authority, now=None, *, resolve_source, required_source_files=None):
    """Workers defer source-dependent semantics until pinned imports are installed."""
    job=signed(envelope,authority);now=time.time() if now is None else now
    if job.get('schema')!=1 or job.get('role') not in ROLES:raise ValueError('job role/schema')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',job.get('job_id','')):raise ValueError('job ID')
    if any(type(job.get(k)) not in (int,float) for k in ('created_at','expires_at')) or not job['created_at']<=now<job['expires_at'] or job['expires_at']-job['created_at']>86400:raise ValueError('job expired/time budget')
    manifest=signed(job['manifest'],authority)
    from .backend_profiles import resolve
    resolve(manifest)
    from .artifact_budget import for_manifest
    for_manifest(manifest)
    cp=manifest['checkpoint'];identifier=file_map(cp['files'])
    if cp.get('id')!=identifier:raise ValueError('checkpoint identity')
    urls=cp.get('read_urls',{})
    if urls and set(urls)!=set(cp['files']):raise ValueError('checkpoint capability file binding')
    for url in urls.values():r2_url(url,'GET')
    if manifest.get('training_input_policy') not in (None,'authenticated-verifier-receipts-v1','authenticated-verifier-compact-inputs-v2'):
        raise ValueError('unapproved training input policy')
    if (manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2'):
        if manifest.get('training_policy') not in (COVERED_POLICY,PERSISTENT_POLICY):
            raise ValueError('compact input requires covered/persistent objective')
        if not {'subnet/compact_training_inputs.py','subnet/training_receipts.py'} <= set(job.get('source_files',{})):
            raise ValueError('compact policy source pins required for every role')
    required=SOURCE_FILES if required_source_files is None else required_source_files
    if not required or not set(required)<=set(job.get('source_files',{})):raise ValueError('missing worker source pins')
    for name,sha in job['source_files'].items():
        if not name.startswith('subnet/') or '..' in name or Path(name).suffix!='.py' or not re.fullmatch('[0-9a-f]{64}',sha):raise ValueError('worker source pin')
    if set(job.get('runtime_versions',{}))!={'torch','transformers','toploc'}:raise ValueError('runtime version pins')
    for obj in job.get('submissions',[]):
        r2_url(obj['url'],'GET')
        if not re.fullmatch('[0-9a-f]{64}',obj['sha256']):raise ValueError('submission digest')
    if len(job.get('submissions',[]))>256:raise ValueError('submission job budget')
    if job['role'] in ('mine','verify','train'):
        if type(manifest.get('max_batches',4)) is not int or not 1<=manifest.get('max_batches',4)<=256:raise ValueError('signed per UID batch quota')
        if job['role']!='mine' and not job.get('submissions'):raise ValueError('no submissions')
        audit_policy=manifest.get('audit_policy',{})
        if audit_policy.get('mode')!='full':
            from .audit_policy import validate
            validate({k:v for k,v in audit_policy.items() if k!='count'})
            if job['role']!='mine':
                if not audit_policy.get('submission_counts') and 'count' not in audit_policy:raise ValueError('missing signed audit allocation')
        if any(type(manifest.get(k)) is not int or not 1<=manifest[k]<=16 for k in ('K','L')):raise ValueError('class quota')
    if job['role']=='mine':
        mining_window(manifest,now)
        if not re.fullmatch('[0-9a-f]{64}',job.get('miner_id','')):raise ValueError('owned miner identity')
        if type(job.get('search_budget')) is not int or not 1<=job['search_budget']<=128 or type(job.get('seed_start')) is not int or job['seed_start']<0:raise ValueError('mining search budget')
        r2_url(job['capability']['put_url'],'PUT')
        if job['capability'].get('headers')!={'Content-Type':'application/octet-stream'}:raise ValueError('signed upload headers')
        if resolve_source and job.get('mining_subset') is not None:mining_definitions(manifest,job)
    elif 'mining_subset' in job:raise ValueError('mining subset only in signed mining jobs')
    if job['role']=='train' and job.get('training_policy',HEAD_POLICY) not in (HEAD_POLICY,FULL_POLICY,FIXED_POLICY,COVERED_POLICY,PERSISTENT_POLICY):raise ValueError('unapproved training objective')
    if job['role']=='train' and (type(job.get('steps')) is not int or not 1<=job['steps']<=32):raise ValueError('training step budget')
    if job.get('training_policy')==FIXED_POLICY and manifest.get('training_policy')!=FIXED_POLICY:raise ValueError('signed fixed-reference training policy')
    if job['role']=='train' and (manifest.get('training_policy')==COVERED_POLICY or job.get('training_policy')==COVERED_POLICY):
        if manifest.get('training_policy')!=job.get('training_policy'):raise ValueError('signed covered training policy')
        if not {'subnet/training_policy.py','subnet/covered_epoch_optimizer.py','subnet/epoch_optimizer.py'}<=set(job['source_files']):raise ValueError('covered training execution source pins')
        from .training_policy import validate_coverage
        coverage_inputs=job.get('submissions')
        if manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2':
            from .compact_training_inputs import original_submissions
            coverage_inputs=original_submissions(coverage_inputs,manifest,authority)
        validate_coverage(manifest,coverage_inputs)
    if job['role']=='train' and (manifest.get('training_policy')==PERSISTENT_POLICY or job.get('training_policy')==PERSISTENT_POLICY):
        if manifest.get('training_policy')!=job.get('training_policy'):raise ValueError('signed persistent training policy')
        from .training_policy import validate_coverage
        from .persistent_training_protocol import validate_job
        coverage_inputs=job.get('submissions')
        if manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2':
            from .compact_training_inputs import original_submissions
            coverage_inputs=original_submissions(coverage_inputs,manifest,authority)
        validate_coverage(manifest,coverage_inputs)
        validate_job(job,manifest,authority)
    if job['role']=='train' and job.get('training_policy') in (COVERED_POLICY,PERSISTENT_POLICY):
        if (manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2'):
            from .compact_training_inputs import validate_job as validate_training_receipts
        else:
            from .training_receipts import validate_job as validate_training_receipts
        validate_training_receipts(job,manifest,authority)
    if job.get('replay') is not None:
        if job['role']!='train' or job.get('training_policy')!=FIXED_POLICY:raise ValueError('replay only in signed fixed optimizer job')
        if resolve_source:
            from .replay_training import admitted
            admitted(manifest,job['replay'],authority)
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
    from .forced_sampling import assurance as sampling_assurance
    from .batches import submission_records,SubmissionRejected
    from .protocol import entries, entry, classification, sample_key,harness_for
    from .artifact_budget import for_manifest
    definitions=entries(manifest);budget=for_manifest(manifest)
    try:records=submission_records(data,budget=budget,max_batches=manifest.get('max_batches',4))
    except SubmissionRejected as error:
        return dict(epoch=manifest['epoch'],submission_sha256=hashlib.sha256(data).hexdigest(),
            submission_rejected=True,rejection_stage='transport',
            sampling_assurance=sampling_assurance(manifest),
            outcomes=[dict(batch=None,valid=False,reason=str(error),rejection_stage='transport')],
            accepted=[],training_eligibility='fully-audited-only'),[]
    from .auditing import select,assurance
    policy=manifest.get('audit_policy',{'mode':'full'})
    selected_indices=set(select(len(records),policy,manifest.get('audit_seed'),hashlib.sha256(data).hexdigest()))
    outcomes=[];accepted=[];pairs=[];seen=set()
    for number,(batch,arrays) in enumerate(records):
        try:
            if not isinstance(batch,dict):raise ValueError('batch object')
            definition=entry(manifest,batch.get('env_id'));index=batch['index'];key=sample_key(batch)
            if batch.get('schema')!=2 or batch['epoch']!=manifest['epoch'] or batch['checkpoint']!=manifest['checkpoint']['id'] or batch.get('sample_index')!=index or type(index) is not int or index not in definition['indices'] or key in seen:raise ValueError('batch binding')
        except (ValueError,KeyError,TypeError,IndexError) as error:
            outcomes.append(dict(batch=number,valid=False,failure_kind='structural_invalid',fully_audited=False,reason=type(error).__name__+': '+str(error)[:300]));continue
        # The selected runtime/harness is operator-approved, not miner input.
        # Configuration or infrastructure refusal must fail the worker honestly.
        selected=runtime.for_environment(definition['spec'],harness_for(definition,index))
        from .audit_policy import InvalidSample
        confirmed_invalid=False
        try:
            if batch.get('environment_version')!=selected.spec.version:raise ValueError('environment version')
            seen.add(key);rolls=batch['rollouts'];tokens=set()
            if len(rolls)!=manifest['K']+manifest['L'] or len(arrays)!=len(rolls):raise ValueError('sample count')
            for rollout,probs in zip(rolls,arrays):
                signature=tuple(tuple(t['output']) for t in rollout['turns'])
                if signature in tokens or rollout['index']!=index or rollout.get('env_id')!=definition['env_id']:raise ValueError('sample binding/duplicate')
                tokens.add(signature)
                if number in selected_indices:
                    try:verified=selected.verify(rollout,probs)
                    except InvalidSample:raise
                    except Exception as error:
                        if policy.get('version')=='bounded-random-v1':
                            raise RuntimeError('audit execution failed; retry without miner penalty') from error
                        raise
                    if verified is not True:
                        confirmed_invalid=True
                        raise ValueError('inference or replay')
            pos=[r for r in rolls if classification(r)=='positive'];neg=[r for r in rolls if classification(r)=='negative']
            if len(pos)!=manifest['K'] or len(neg)!=manifest['L']:raise ValueError('positive/negative quota')
            if number in selected_indices:
                accepted.append(batch);pairs.extend((definition,p,n) for p,n in zip(pos,neg))
            outcomes.append(dict(batch=number,env_id=definition['env_id'],index=index,structural_valid=True,valid=True if number in selected_indices else None,fully_audited=number in selected_indices))
        except (ValueError,KeyError,TypeError,IndexError) as error:
            outcomes.append(dict(batch=number,valid=False,fully_audited=number in selected_indices,failure_kind='confirmed_invalid' if confirmed_invalid or isinstance(error,InvalidSample) else 'verification_error',reason=type(error).__name__+': '+str(error)[:300]))
    return dict(epoch=manifest['epoch'],submission_sha256=hashlib.sha256(data).hexdigest(),policy=policy,selected_batches=sorted(selected_indices),assurance=assurance(len(records),len(selected_indices)),sampling_assurance=sampling_assurance(manifest),outcomes=outcomes,accepted=accepted,training_eligibility='fully-audited-only'),pairs

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
        destination.mkdir(parents=True);model.save_pretrained(destination,safe_serialization=True,max_shard_size='4GB');runtime.tokenizer.save_pretrained(destination)
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

def install_source_loader(root,additional_files=()):
    # Pure admission helpers are used before workspace/artifact access. Their
    # pinned bytes have now been checked; discard bootstrap imports so compute
    # admission reloads them through the authenticated fresh-source finder.
    for module_name in ('subnet.backend_profiles','subnet.artifact_budget','subnet.audit_policy','subnet.auditing','subnet.training_policy',
            'subnet.persistent_cpu_adamw','subnet.persistent_training_state','subnet.persistent_training_protocol','subnet.training_receipts'):
        sys.modules.pop(module_name,None)
    if 'subnet/compact_training_inputs.py' in additional_files:
        sys.modules.pop('subnet.compact_training_inputs',None)
    for name in set(SOURCE_FILES)|set(additional_files):
        module_name=name[:-3].replace('/','.')
        if module_name in sys.modules and module_name!='subnet.backend_jobs':
            raise ValueError('GPU worker requires fresh process before runtime imports')
    sys.meta_path.insert(0,FreshSourceFinder(root))

def initial_configuration(manifest,job):
    from .protocol import entries,entry,harness_for
    definitions=entries(manifest)
    if job['role']=='evaluate':
        suite=job['heldout'][0]
        return entry(manifest,suite['env_id']),suite['harness']
    first=next((row for row in definitions if row['indices']),None)
    if first is None:raise ValueError('no authorized mining samples for model role')
    return first,harness_for(first,first['indices'][0])

def execute(envelope, authority, workspace, cache=None, runtime_factory=None):
    job,manifest=_validate(envelope,authority,resolve_source=False)
    root=Path(__file__).resolve().parent.parent
    for name,expected in job['source_files'].items():
        if (root/name).is_symlink() or digest(root/name)!=expected:raise ValueError('worker source mismatch')
    for name,expected in job['runtime_versions'].items():
        if version(name)!=expected:raise ValueError('runtime package mismatch')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('CUDA environment profile')
    compact_files=('subnet/compact_training_inputs.py',) if (manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2') else ()
    if job.get('training_policy')==PERSISTENT_POLICY:
        from .persistent_training_protocol import EXECUTION_FILES
        install_source_loader(root,(*EXECUTION_FILES,'subnet/training_receipts.py',*compact_files))
    elif job.get('training_policy')==COVERED_POLICY:
        install_source_loader(root,('subnet/training_receipts.py',*compact_files))
    else:install_source_loader(root,compact_files)
    from .backend_profiles import resolve
    revision,backend_profile,numerical_policy=resolve(manifest)
    from .artifact_budget import for_manifest
    for_manifest(manifest)
    from .task_assets import hydrate_manifest
    asset_root=Path(workspace)/'task-assets'
    if manifest.get('task_assets'):
        asset_root.mkdir(parents=True,exist_ok=True);asset_root.chmod(0o700)
        os.environ['AFFINE_MATH_CORPUS_ASSET_ROOT']=str(asset_root.resolve())
    hydrate_manifest(asset_root,manifest)
    # Resolve against authenticated fresh source before any artifact, workspace,
    # checkpoint or model is opened. Public validate() remains fully strict.
    if job.get('mining_subset') is not None:mining_definitions(manifest,job)
    if job.get('replay') is not None:
        from .replay_training import admitted
        admitted(manifest,job['replay'],authority)
    workspace=Path(workspace);out=workspace/'jobs'/job['job_id']
    out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
    approved=checkpoint(manifest,workspace,cache)
    report=dict(schema=1,job_id=job['job_id'],role=job['role'],operator=authority,
        job_sha256=hashlib.sha256(canonical(job)).hexdigest(),checkpoint=manifest['checkpoint']['id'],
        epoch=manifest['epoch'],backend_profile=backend_profile,numerical_policy=numerical_policy,
        source_files=job['source_files'],runtime_versions=job['runtime_versions'],
        chain_transactions=False,full_model_finetune=False,execution_resources_enforced=False)
    if job['role']=='upload':
        import requests
        validate_single_put_sizes(approved,manifest['checkpoint']['files'])
        for name,url in job['put_urls'].items():
            with (approved/name).open('rb') as body:
                response=requests.put(url,data=body,headers={'Content-Type':'application/octet-stream'},timeout=600,allow_redirects=False)
            if response.status_code not in (200,201,204):raise ValueError('R2 PUT status '+str(response.status_code))
        report['uploaded_files']=manifest['checkpoint']['files']
    else:
        from .protocol import entries,entry,harness_for
        from .gpu_runtime import GPURuntime
        factory=runtime_factory or GPURuntime;definitions=entries(manifest)
        first,initial_harness=initial_configuration(manifest,job)
        if runtime_factory is None:
            runtime=factory(approved,manifest['checkpoint']['files'],first['spec'],initial_harness,
                runtime_revision=revision)
        else:
            runtime=factory(approved,manifest['checkpoint']['files'],first['spec'],initial_harness)
        if job['role'] != 'evaluate':
            from .forced_sampling import bind_runtime
            bind_runtime(runtime,manifest)
        elif manifest.get('sampling_contract') is not None:
            runtime.sampling_context=None
            report['sampling_scope']='heldout-diagnostic-not-mining-evidence'
        if job['role']=='mine':
            from .batches import pack
            import requests
            def upload(data,timeout):
                response=requests.put(job['capability']['put_url'],data=data,headers=job['capability']['headers'],timeout=timeout,allow_redirects=False)
                if response.status_code not in (200,201,204):raise ValueError('R2 PUT status '+str(response.status_code))
            data,mining=mine_cumulative(runtime,manifest,job,upload,allow_empty=True)
            if data is not None:(out/'submission.zip').write_bytes(data)
            report.update(miner_id=job['miner_id'],submission_sha256=hashlib.sha256(data).hexdigest() if data is not None else None,submission_size=len(data) if data is not None else 0,operator_authorized_experiment=True,**mining)
        elif job['role'] in ('verify','train'):
            reports=[];pairs=[]
            for i,obj in enumerate(job['submissions']):
                from .artifact_budget import for_manifest
                compact_input=job['role']=='train' and (manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2')
                path=out/('submission-'+str(i)+('.json' if compact_input else '.zip'))
                limit=obj['size'] if compact_input else for_manifest(manifest)['compressed_bytes']
                get_object(obj['url'],obj['sha256'],path,limit)
                if job['role']=='train' and job.get('training_policy') in (COVERED_POLICY,PERSISTENT_POLICY):
                    if compact_input:
                        from .compact_training_inputs import admitted_submission
                    else:
                        from .training_receipts import admitted_submission
                    result,verified=admitted_submission(path,obj,manifest,authority,
                        retire=job.get('training_policy')==PERSISTENT_POLICY)
                else:result,verified=audit(path.read_bytes(),manifest,runtime)
                reports.append(result);pairs.extend(verified)
            receipt_training=job['role']=='train' and job.get('training_policy') in (COVERED_POLICY,PERSISTENT_POLICY)
            report['audits']=[] if receipt_training else reports
            if receipt_training:report['training_admissions']=reports
            if job['role']=='train':
                if not pairs:raise ValueError('no verified training pairs')
                values_before=parameter_value_digest(runtime.model)
                if job.get('replay') is not None:
                    from .replay_training import verified_pairs,merge_pairs
                    historical,replay_report=verified_pairs(runtime,manifest,job['replay'],authority)
                    pairs,targets=merge_pairs(pairs,historical,job['replay']['reuse_counts'])
                    replay_report['checks']=[row for row in replay_report['checks'] if row['target_sha256'] in targets]
                    replay_report['proposed_reuse_increments']={target:1 for target in targets}
                    report['replay_training']=replay_report
                    if job['steps']<len(pairs):raise ValueError('each fresh/replay family needs an optimizer step')
                metrics=[];destination=None
                if job.get('training_policy')==PERSISTENT_POLICY:
                    from .persistent_training_worker import train,report_updates
                    destination,persistent_diagnostics,persistent_state=train(runtime,pairs,out,manifest,job,authority,approved_checkpoint=approved)
                    metrics,persistent_diagnostics=report_updates(persistent_diagnostics,job,manifest)
                    report['persistent_training_state']=persistent_state
                elif job.get('training_policy')==FIXED_POLICY:
                    from .epoch_optimizer import train_epoch
                    destination,metrics=train_epoch(runtime,pairs,out,steps=job['steps'])
                elif job.get('training_policy')==COVERED_POLICY:
                    from .covered_epoch_optimizer import train_epoch,distinct_verified_pairs
                    original_pairs=len(pairs);pairs=distinct_verified_pairs(pairs)
                    report['covered_training_inputs']=dict(verified_pairs_before_deduplication=original_pairs,unique_verified_pairs=len(pairs),exact_duplicate_pairs_removed=original_pairs-len(pairs))
                    destination,metrics=train_epoch(runtime,pairs,out,seed=manifest['training_coverage']['seed'],steps=job['steps'])
                else:
                    for step in range(job['steps']):
                        definition,pos,neg=pairs[step%len(pairs)]
                        runtime.configure(definition['spec'],harness_for(definition,pos['index']))
                        destination=out/('checkpoint-step-'+str(step+1))
                        policy=job.get('training_policy',HEAD_POLICY)
                        update=full_parameter_train(runtime,[(pos,neg)],destination,steps=1) if policy==FULL_POLICY else runtime.train([(pos,neg)],destination,steps=1)
                        update['training_policy']=policy
                        update.update(pair_attribution(definition,pos,neg,step));metrics.append(update)
                from .model import model_files
                files=model_files(destination)
                persistent=job.get('training_policy')==PERSISTENT_POLICY
                if not persistent and not checkpoint_weights_changed(manifest['checkpoint']['files'],files):raise ValueError('training did not change checkpoint weights')
                values_after=parameter_value_digest(runtime.model)
                if not persistent and values_before==values_after:raise ValueError('optimizer did not change parameter values')
                report['training']=dict(steps=job['steps'],updates=metrics,training_policy=job.get('training_policy',HEAD_POLICY),full_model_finetune=job.get('training_policy',HEAD_POLICY) in (FULL_POLICY,FIXED_POLICY,COVERED_POLICY,PERSISTENT_POLICY),weights_changed=values_before!=values_after,parameter_values_sha256_before=values_before,parameter_values_sha256_after=values_after)
                if job.get('training_policy')==COVERED_POLICY:report['training']['training_coverage']=manifest['training_coverage']
                if receipt_training:
                    if (manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2'):
                        from .compact_training_inputs import VERSION as INPUT_POLICY
                    else:
                        from .training_receipts import VERSION as INPUT_POLICY
                    report['training'].update(training_input_policy=INPUT_POLICY,
                        trainer_verification_performed=False,all_pairs_authenticated_verifier_receipts=True)
                if persistent:
                    report['training'].update(training_coverage=manifest['training_coverage'],
                        state_updated=True,
                        persistent_diagnostics=persistent_diagnostics,
                        global_step_before=manifest['trainer_state_binding']['global_step_before'],
                        global_step_after=job['persistent_training']['global_step_after'])
                report['full_model_finetune']=report['training']['full_model_finetune']
                report['new_checkpoint']=dict(id=file_map(files),files=files,path=str(destination))
        else:
            values=[];failures=[]
            for row in job['heldout']:
                definition=entry(manifest,row['env_id'])
                if set(row['indices'])&set(definition['indices']):raise ValueError('heldout/training overlap')
                selected=runtime.for_environment(definition['spec'],row['harness'])
                for index,seed in zip(row['indices'],row['seeds']):
                    try:
                        doc,arrays=selected.rollout(index,seed)
                        if selected.verify(doc,arrays) is not True:raise ValueError('heldout audit')
                        values.append(dict(env_id=row['env_id'],index=index,seed=seed,reward=doc['reward'],classification=doc['classification'],task_hash=doc['task_hash'],verified=True))
                    except (ValueError,RuntimeError,KeyError) as error:
                        failures.append(dict(env_id=row['env_id'],index=index,seed=seed,error_type=type(error).__name__,error=str(error)[:300]))
            report['heldout']=values;report['heldout_failures']=failures
    report['success']=True;report['completed_at']=time.time()
    (out/'report.json').write_bytes(canonical(report));return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('job');parser.add_argument('--authority',required=True);parser.add_argument('--workspace',required=True);parser.add_argument('--checkpoint-cache')
    args=parser.parse_args();data=Path(args.job).read_bytes()
    if len(data)>4_000_000:raise ValueError('job envelope size budget')
    report=execute(json.loads(data),args.authority,args.workspace,args.checkpoint_cache)
    print(json.dumps(dict(job_id=report['job_id'],role=report['role'],success=True,checkpoint=report.get('new_checkpoint',{}).get('id',report['checkpoint']))))
if __name__=='__main__':main()
