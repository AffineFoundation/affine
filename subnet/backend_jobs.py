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
import tempfile
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
    ('trusted_native_evaluation','owned_cached_evaluation','cached_sampling','successor_calibration','training_documents','probability_artifacts','fast_prefill_audit','continuous_audit_policy','selected_proof_copy','commitment_transport','hourly_policy','audit_exclusion','audit_policy','auditing','backend_jobs','backend_profiles','artifact_budget','task_assets','math_corpus_provider','math_corpus_assets','math_corpus','source_bootstrap','gpu_runtime','model','harness','environments','proofs','batches','protocol','forced_sampling','sampling_uniqueness'))
ROLES = {'mine','verify','train','evaluate','upload'}
HEAD_POLICY='frozen-feature-head-adamw-v1'
FULL_POLICY='bf16-full-adamw-checkpointed-v1'
FIXED_POLICY='bf16-full-adamw-fixed-epoch-reference-v2'
COVERED_POLICY='bf16-full-adamw-covered-fixed-reference-v3'
PERSISTENT_POLICY='bf16-cpu-fp32-master-task-normalized-persistent-v4'
TRAINING_ATTRIBUTION='verified-pair-v1'

def write_private_report(path, report):
    """Publish canonical metadata atomically with private permissions."""
    path = Path(path)
    raw = canonical(report)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='report-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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


def owned_miner_identity(job):
    """Cheap local signer admission before checkpoint/model allocation."""
    from .storage import Identity
    import stat
    path=Path(job['miner_identity_file'])
    if not path.is_absolute() or path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_mode & 0o077:raise ValueError('private local miner identity file')
    seed=bytes.fromhex(path.read_text().strip())
    if len(seed)!=32:raise ValueError('miner signing seed size')
    identity=Identity(seed)
    if identity.id!=job['miner_id']:raise ValueError('local miner signer binding')
    return identity


def owned_commitment_upload(job,manifest,progress_path=None):
    """Use only the miner's local scoped signing seed; never authority material."""
    from .storage import Identity
    from .batches import unpack,pack
    from .commitment_transport import make,VERSION,VERSION2,VERSION3,canonical,pair_artifact,UploadJournal
    import stat,requests
    identity=owned_miner_identity(job)
    journal=UploadJournal(manifest,progress_path)
    def check_prepared(packed):
        from .commitment_transport import check_prepared_cumulative
        return check_prepared_cumulative(packed,manifest,len(job['capability']['batch_put_urls']))
    def upload_pairs(packed,timeout):
        check_prepared(packed);start=time.monotonic()
        def put(url,data):
            remaining=min(timeout-(time.monotonic()-start),manifest['deadline']-time.time()-1)
            if remaining<=0:raise TimeoutError('owned upload deadline; previous commitment retained')
            response=requests.put(url,data=data,headers=job['capability']['headers'],timeout=remaining,allow_redirects=False)
            if response.status_code not in (200,201,204):raise ValueError('R2 PUT status '+str(response.status_code))
        for slot,(url,(_,artifact))in enumerate(zip(job['capability']['batch_put_urls'],packed)):
            if journal.known(slot,artifact):continue
            put(url,artifact);journal.acknowledge(slot,artifact)
        if manifest.get('submission_transport_policy',VERSION)in (VERSION2,VERSION3):
            from .training_documents import document
            for slot,(batch,_)in enumerate(packed):
                body=document(batch,manifest,identity.id,slot)
                if journal.known('training-'+str(slot),body):continue
                put(job['capability']['training_put_urls'][slot],body);journal.acknowledge('training-'+str(slot),body)
        commitment=canonical(make(identity,manifest,packed));put(job['capability']['put_url'],commitment)
        return commitment
    def upload(data,timeout):
        from .artifact_budget import for_manifest
        if 'token_artifact_policy'in manifest:
            from .token_only_protocol import unpack as token_unpack
            rows=token_unpack(data,budget=for_manifest(manifest),max_batches=manifest['max_batches'])
        else:rows=unpack(data,budget=for_manifest(manifest))
        return upload_pairs([(batch,pair_artifact(batch,arrays,manifest))for batch,arrays in rows],timeout)
    upload.prepare_pair=lambda batch,arrays:pair_artifact(batch,arrays,manifest)
    upload.check_prepared=check_prepared
    upload.upload_pairs=upload_pairs
    return upload

def mine_cumulative(runtime,manifest,job,upload,clock=None,allow_empty=False,progress=None):
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
    batches=[];prepared=[];search=[];data=None;uploads=0;stopped=False;capacity_reached=False
    def available():
        now=clock()
        return manifest['start']<=now<manifest['deadline']-10
    # Keep a bounded frontier, rather than exhausting one task's attempt budget
    # while later tasks never get a chance to produce a success/failure pair.
    tasks=iter((definition,index) for definition in mining_definitions(manifest,job)
               for index in definition['indices'])
    search_policy=job.get('owned_search_policy',dict(version='bounded-round-robin-v1',active_tasks=8,dwell_attempts=2))
    if (not isinstance(search_policy,dict) or set(search_policy)!={'version','active_tasks','dwell_attempts'}
            or search_policy['version']!='bounded-round-robin-v1'
            or type(search_policy['active_tasks'])is not int or not 1<=search_policy['active_tasks']<=8
            or type(search_policy['dwell_attempts'])is not int or not 1<=search_policy['dwell_attempts']<=2):
        raise ValueError('bounded owned search policy')
    frontier=[]
    def refill():
        while len(frontier)<search_policy['active_tasks']:
            try:definition,index=next(tasks)
            except StopIteration:break
            frontier.append(dict(definition=definition,index=index,classes={'positive':[],'negative':[]},
                fingerprints=set(),attempts=0,observed={'positive':0,'negative':0},selected=None,row=None))
    def emit(state,phase,**extra):
        if progress is not None:
            progress(dict(phase=phase,env_id=state['definition']['env_id'],index=state['index'],
                attempts=state['attempts'],observed_positive=state['observed']['positive'],
                observed_negative=state['observed']['negative'],completed_batches=len(batches),
                cumulative_uploads=uploads,at=clock(),**extra))
    refill()
    while frontier and not stopped and not capacity_reached and len(batches)<manifest.get('max_batches',4):
        for state in list(frontier):
            if not available():stopped=True;break
            definition,index=state['definition'],state['index']
            if state['selected'] is None:
                state['selected']=runtime.for_environment(definition['spec'],harness_for(definition,index))
            selected=state['selected'];classes=state['classes'];fingerprints=state['fingerprints'];observed=state['observed']
            if state['row'] is None:
                state['row']=dict(env_id=definition['env_id'],index=index);search.append(state['row'])
            for _ in range(min(search_policy['dwell_attempts'],job['search_budget']-state['attempts'])):
                if not available():stopped=True;break
                seed=job['seed_start']+state['attempts']
                selected.mining_progress=lambda phase,**metrics:emit(state,phase,seed=seed,**metrics)
                emit(state,'attempt_started',seed=seed)
                started=clock();rollout,arrays=selected.rollout(index,seed);state['attempts']+=1
                label=rollout['classification']
                if label not in classes:
                    # Under the completed-answer math contract, an exhausted
                    # or otherwise incomplete response is unresolved. It is
                    # an ordinary search miss: it must not fill either quota,
                    # enter a batch, or abort the miner job.
                    from .math_completion import enabled as completed_math
                    if label in ('neutral', 'unresolved') and completed_math(definition['spec']):
                        emit(state, 'attempt_unresolved', seed=seed,
                             elapsed_seconds=clock()-started)
                        continue
                    raise ValueError('rollout classification')
                observed[label]+=1
                signature=tuple(tuple(t['output']) for t in rollout['turns'])
                quota=manifest['K'] if label=='positive' else manifest['L']
                if len(classes[label])<quota and signature not in fingerprints:
                    classes[label].append((rollout,arrays));fingerprints.add(signature)
                state['row'].update(attempts=state['attempts'],positive=len(classes['positive']),negative=len(classes['negative']),observed_positive=observed['positive'],observed_negative=observed['negative'])
                emit(state,'attempt_completed',seed=seed,classification=label,
                     output_tokens=sum(len(t['output']) for t in rollout['turns']),elapsed_seconds=clock()-started)
                if len(classes['positive'])==manifest['K'] and len(classes['negative'])==manifest['L']:break
            complete=len(classes['positive'])==manifest['K'] and len(classes['negative'])==manifest['L']
            if complete:
                if not available():stopped=True;break
                found=classes['positive']+classes['negative']
                batch=dict(schema=2,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],env_id=definition['env_id'],environment_version=selected.spec.version,index=index,sample_index=index,rollouts=[r for r,a in found])
                from .sampling_uniqueness import validate_batch
                validate_batch(batch,manifest,job.get('miner_id'))
                candidate=batches+[(batch,[a for r,a in found])]
                from .artifact_budget import for_manifest
                structured=callable(getattr(upload,'prepare_pair',None))
                pack_started=clock();emit(state,'artifact_pack_started')
                try:
                    if structured:
                        candidate_prepared=prepared+[(batch,upload.prepare_pair(batch,[a for r,a in found]))]
                        upload.check_prepared(candidate_prepared);candidate_data=None
                    else:
                        candidate_data=(pack(candidate,budget=for_manifest(manifest))
                            if manifest.get('artifact_policy') is not None else pack(candidate))
                except UploadBudgetExceeded:
                    state['row']['submission_status']='exceeds_cumulative_upload_budget'
                    frontier.remove(state)
                    if batches:capacity_reached=True;break
                    continue
                emit(state,'artifact_pack_completed',elapsed_seconds=clock()-pack_started)
                if not available():stopped=True;break
                put_started=clock();emit(state,'cumulative_upload_started')
                timeout=min(180,manifest['deadline']-clock()-1)
                if structured:
                    candidate_data=upload.upload_pairs(candidate_prepared,timeout);prepared=candidate_prepared
                else:upload(candidate_data,timeout)
                batches=([(b,None)for b,a in candidate] if structured else candidate);data=candidate_data;uploads+=1
                emit(state,'cumulative_upload_completed',elapsed_seconds=clock()-put_started)
                emit(state,'cumulative_upload_acknowledged',submission_bytes=len(data))
            if complete or state['attempts']>=job['search_budget']:frontier.remove(state)
            if stopped or len(batches)>=manifest.get('max_batches',4):break
        if not stopped and not capacity_reached:refill()
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
    from .unaudited_training_execution import admission as execution_admission
    execution_admission(envelope,authority,now=now)
    if resolve_source and ('token_artifact_policy'in manifest or manifest.get('submission_transport_policy')=='small-commitment-token-pairs-v3'):
        from .token_only_protocol import for_manifest as token_policy
        token_policy(manifest)
    from .backend_profiles import resolve
    resolve(manifest)
    from .backend_profiles import execution_profile
    execution_profile(manifest,job['role'])
    from .artifact_budget import for_manifest
    for_manifest(manifest)
    cp=manifest['checkpoint'];identifier=file_map(cp['files'])
    if cp.get('id')!=identifier:raise ValueError('checkpoint identity')
    urls=cp.get('read_urls',{})
    if urls and set(urls)!=set(cp['files']):raise ValueError('checkpoint capability file binding')
    for url in urls.values():r2_url(url,'GET')
    if manifest.get('training_input_policy') not in (None,'authenticated-verifier-receipts-v1','authenticated-verifier-compact-inputs-v2','committed-unaudited-training-v1'):
        raise ValueError('unapproved training input policy')
    if (manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2'):
        if manifest.get('training_policy') not in (COVERED_POLICY,PERSISTENT_POLICY):
            raise ValueError('compact input requires covered/persistent objective')
        if not {'subnet/compact_training_inputs.py','subnet/training_receipts.py'} <= set(job.get('source_files',{})):
            raise ValueError('compact policy source pins required for every role')
    if manifest.get('training_input_policy')=='committed-unaudited-training-v1' and 'subnet/committed_training_inputs.py'not in job.get('source_files',{}):
        raise ValueError('unaudited learner source pin required')
    if manifest.get('training_input_policy')=='committed-unaudited-training-v1'and job.get('role')=='train':
        native_math_prompt_enabled(job,manifest)
    # Prospective native MATH contracts pin the imported grader transport helper.
    # Historical signed epochs without this marker keep their original pin set.
    definitions = manifest.get('environments', [])
    if definitions is None:definitions = []
    native_specs = [row.get('spec', {}) for row in definitions if isinstance(row, dict)]
    if isinstance(manifest.get('environment'), dict):native_specs.append(manifest['environment'])
    if any(spec.get('id') == 'affine_math' and
           'native-math-grader-pinned-indeterminate-v1' in spec.get('config', {}).get('dependency_versions', {})
           for spec in native_specs):
        if 'subnet/native_math_grader.py' not in job.get('source_files', {}):
            raise ValueError('prospective native MATH grader source pin required')
    if manifest.get('training_startup_recovery')is not None:
        if job.get('role')!='train' or 'subnet/training_startup_recovery.py'not in job.get('source_files',{}):raise ValueError('explicit startup recovery source pin required')
    required=SOURCE_FILES if required_source_files is None else required_source_files
    if not required or not set(required)<=set(job.get('source_files',{})):raise ValueError('missing worker source pins')
    for name,sha in job['source_files'].items():
        if not name.startswith('subnet/') or '..' in name or Path(name).suffix!='.py' or not re.fullmatch('[0-9a-f]{64}',sha):raise ValueError('worker source pin')
    if set(job.get('runtime_versions',{}))!={'torch','transformers','toploc'}:raise ValueError('runtime version pins')
    for obj in job.get('submissions',[]):
        r2_url(obj['url'],'GET')
        if not re.fullmatch('[0-9a-f]{64}',obj['sha256']):raise ValueError('submission digest')
    from .committed_training_inputs import training_document_cap
    submission_cap=training_document_cap(manifest)if job['role']=='train'else 256
    if len(job.get('submissions',[]))>submission_cap:raise ValueError('submission job budget')
    if job['role'] in ('mine','verify','train'):
        if type(manifest.get('max_batches',4)) is not int or not 1<=manifest.get('max_batches',4)<=256:raise ValueError('signed per UID batch quota')
        if job['role']!='mine' and not job.get('submissions'):raise ValueError('no submissions')
        audit_policy=manifest.get('audit_policy',{})
        if audit_policy.get('mode')!='full':
            from .audit_policy import validate
            validate({k:v for k,v in audit_policy.items() if k!='count'})
            if job['role']!='mine':
                if not (job['role']=='train'and manifest.get('training_input_policy')=='committed-unaudited-training-v1')and not audit_policy.get('submission_counts') and 'count'not in audit_policy:raise ValueError('missing signed audit allocation')
        if any(type(manifest.get(k)) is not int or not 1<=manifest[k]<=16 for k in ('K','L')):raise ValueError('class quota')
    if job['role']=='mine':
        mining_window(manifest,now)
        if not re.fullmatch('[0-9a-f]{64}',job.get('miner_id','')):raise ValueError('owned miner identity')
        # Admission runs before authenticated fresh runtime imports.
        miner_bound=manifest.get('sampling_contract',{}).get('version')=='forced-inverse-cdf-prefill-miner-bound-v5'
        maximum_search_budget=1000 if miner_bound else 128
        if type(job.get('search_budget')) is not int or not 1<=job['search_budget']<=maximum_search_budget or type(job.get('seed_start')) is not int or job['seed_start']<0:raise ValueError('mining search budget')
        if miner_bound and (job['seed_start']>=1000 or job['search_budget']>1000-job['seed_start']):raise ValueError('miner-bound nonce search range')
        r2_url(job['capability']['put_url'],'PUT')
        if job['capability'].get('headers')!={'Content-Type':'application/octet-stream'}:raise ValueError('signed upload headers')
        if manifest.get('submission_transport_policy') is not None:
            from .commitment_transport import VERSION,VERSIONS,VERSION2,VERSION3
            if manifest['submission_transport_policy']not in VERSIONS or not isinstance(job.get('miner_identity_file'),str) or not Path(job['miner_identity_file']).is_absolute():raise ValueError('owned commitment miner identity path')
            urls=job['capability'].get('batch_put_urls')
            if not isinstance(urls,list) or len(urls)!=manifest['max_batches']:raise ValueError('owned commitment capability slots')
            for url in urls:r2_url(url,'PUT')
            if manifest.get('submission_transport_policy',VERSION)in (VERSION2,VERSION3):
                tokens=job['capability'].get('training_put_urls')
                if type(tokens)is not list or len(tokens)!=manifest['max_batches']:raise ValueError('owned token document capability slots')
                for url in tokens:r2_url(url,'PUT')
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
        if manifest.get('training_input_policy')!='committed-unaudited-training-v1':validate_coverage(manifest,coverage_inputs)
    if job['role']=='train' and (manifest.get('training_policy')==PERSISTENT_POLICY or job.get('training_policy')==PERSISTENT_POLICY):
        if manifest.get('training_policy')!=job.get('training_policy'):raise ValueError('signed persistent training policy')
        from .training_policy import validate_coverage
        from .persistent_training_protocol import validate_job
        coverage_inputs=job.get('submissions')
        if manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2':
            from .compact_training_inputs import original_submissions
            coverage_inputs=original_submissions(coverage_inputs,manifest,authority)
        if manifest.get('training_input_policy')!='committed-unaudited-training-v1':validate_coverage(manifest,coverage_inputs)
        validate_job(job,manifest,authority)
    if resolve_source and job['role']=='train' and job.get('training_policy') in (COVERED_POLICY,PERSISTENT_POLICY):
        if manifest.get('training_input_policy')=='committed-unaudited-training-v1':
            from .committed_training_inputs import validate_job as validate_training_receipts
        elif (manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2'):
            from .compact_training_inputs import validate_job as validate_training_receipts
        else:
            from .training_receipts import validate_job as validate_training_receipts
        validate_training_receipts(job,manifest,authority)
    if job.get('replay') is not None:
        if job['role']!='train' or job.get('training_policy')!=FIXED_POLICY:raise ValueError('replay only in signed fixed optimizer job')
        if resolve_source:
            from .replay_training import admitted
            admitted(manifest,job['replay'],authority)
    trusted=job.get('trusted_evaluation_policy')
    if 'trusted_evaluation_policy' in job:
        expected={'version':'trusted-native-generation-evaluation-v1','trust_scope':'operator-owned-process-native-grader','proof_reverification':False,'sampling_policy':'unchanged-signed-runtime'}
        if(job['role']!='evaluate'or job.get('owned_evaluation_policy')is not None or job.get('successor_calibration')is not None or not isinstance(trusted,dict)or set(trusted)!=set(expected)or any(type(trusted[k])is not type(v)or trusted[k]!=v for k,v in expected.items())):raise ValueError('explicit trusted native evaluation role/policy')
        if 'subnet/trusted_native_evaluation.py'not in job['source_files']:raise ValueError('trusted native evaluator source pin')
    owned=job.get('owned_evaluation_policy')
    if owned is not None:
        expected={'version':'owned-cached-native-evaluation-v1','trust_scope':'operator-owned-process-native-grader','proof_reverification':False}
        if (job['role']!='evaluate' or job.get('successor_calibration')is not None or not isinstance(owned,dict)or set(owned)!=set(expected)or any(type(owned[k])is not type(v)or owned[k]!=v for k,v in expected.items())):raise ValueError('explicit owned cached evaluation role/policy')
        if not {'subnet/owned_cached_evaluation.py','subnet/cached_sampling.py'}.issubset(job['source_files']):raise ValueError('owned evaluator execution source pins')
        for suite in job.get('heldout',[]):
            if suite['harness'].get('version')!='text-tools-long-kv-v3':raise ValueError('explicit cached diagnostic harness required')
    if job.get('successor_calibration') is not None and job['role']!='evaluate':raise ValueError('calibration evaluate role only')
    if job['role']=='evaluate' and job.get('successor_calibration') is not None:
        from .successor_calibration import request
        r=request(job['successor_calibration'])
        from .successor_calibration import draw_context,MINER_CALIBRATION_VERSION
        if resolve_source and r['version']==MINER_CALIBRATION_VERSION:draw_context(manifest,r)
        row=next((v for v in manifest.get('environments',[])if v.get('env_id')==r['env_id']),None)
        if row is None or not set(r['task_indices'])<=set(row.get('indices',[])):raise ValueError('calibration approved task scope')
        if set(r['task_indices'])&set(manifest.get('heldout_indices',{}).get(r['env_id'],[])):raise ValueError('calibration heldout leakage')
        if job.get('heldout') is not None:raise ValueError('calibration is not heldout evaluation')
    elif job['role']=='evaluate':
        if not job.get('heldout') or len(job['heldout'])>64:raise ValueError('heldout budget')
        for row in job['heldout']:
            if len(row['indices'])!=len(row['seeds']) or not 1<=len(row['indices'])<=32 or any(type(i) is not int or i<0 for i in row['indices']+row['seeds']):raise ValueError('heldout index/seed budget')
            if row['harness'].get('policy')!='autoregressive' or row['harness'].get('turn_overrides'):raise ValueError('heldout must use free autoregressive policy')
    if manifest.get('persistent_publication_policy') is not None:
        if resolve_source:
            from .persistent_publication import validate_policy
            validate_policy(manifest['persistent_publication_policy'])
        if 'subnet/persistent_publication.py' not in job['source_files']:
            raise ValueError('prospective publication execution source pin')
    if job['role']=='upload':
        if set(job.get('put_urls',{}))!=set(cp['files']):raise ValueError('upload capability file binding')
        for url in job['put_urls'].values():r2_url(url,'PUT')
    parallel_checkpoint_policy(manifest)
    return job,manifest

class ArtifactRejected(ValueError):
    '''Complete observed bytes violate the signed digest/size; not network failure.'''

def _get_object_once(url, expected, destination, limit, *, session=None, record_lifecycle=True):
    import requests
    temporary=destination.with_suffix(destination.suffix+'.partial');h=hashlib.sha256();size=0
    try:
        client=session if session is not None else requests
        with client.get(r2_url(url,'GET'),stream=True,timeout=180,allow_redirects=False) as response:
            if response.status_code!=200:raise ValueError('R2 GET status '+str(response.status_code))
            with temporary.open('wb') as f:
                for part in response.iter_content(1024*1024):
                    size+=len(part)
                    if size>limit:raise ArtifactRejected('artifact size budget')
                    h.update(part);f.write(part)
        if h.hexdigest()!=expected:raise ArtifactRejected('artifact digest')
        temporary.replace(destination)
        lifecycle_root=os.environ.get("AFFINE_CACHE_LIFECYCLE_ROOT")
        if record_lifecycle and lifecycle_root and destination.name.startswith("submission-"):
            from .cache_lifecycle import CacheLifecycle
            CacheLifecycle(lifecycle_root).record_download(destination,expected)
    finally:temporary.unlink(missing_ok=True)



_PARENT_READ_CONTEXT=None

class StateReadInfrastructureDeferred(RuntimeError):
    """Original bounded parent transport exhausted; no optimizer update."""

def _parent_read_binding(url,expected,destination,limit):
    context=_PARENT_READ_CONTEXT
    if context is None:return None
    job,manifest,authority,workspace=context
    if job.get('role')!='train'or job.get('training_policy')!='bf16-cpu-fp32-master-task-normalized-persistent-v4':return None
    urls=job.get('persistent_training',{}).get('parent_read_urls',{})
    matches=[name for name,cap in urls.items()if cap==url]
    if not matches:return None
    if len(matches)!=1:raise ValueError('one original parent capability')
    from .persistent_training_protocol import validate_job
    _,parent=validate_job(job,manifest,authority)
    row=next((r for r in parent['shards']if r['name']==matches[0]),None)if parent else None
    path=Path(destination).absolute();out=Path(workspace).absolute()/'jobs'/job['job_id']
    if (row is None or (expected,limit)!=(row['sha256'],row['size'])or path.name!=row['name']
        or path!=path.resolve()or not path.is_relative_to(out)or not path.parent.name.startswith('.fp32-state-transfer-')):raise ValueError('exact original parent read scope/path')
    return dict(row=row,expires_at=job['expires_at'],out=out)

def _retry_parent_object(url,binding,destination,*,max_attempts=3,sleep=time.sleep):
    import math,requests
    from .persistent_training_state import _hash_file
    row=binding['row'];expires_at=binding['expires_at'];path=Path(destination)
    if type(max_attempts)is not int or not 1<=max_attempts<=3 or type(expires_at)not in(int,float)or not math.isfinite(expires_at):raise ValueError('bounded original parent reads')
    evidence=dict(version='bounded-parent-state-read-retry-v1',name=row['name'],sha256=row['sha256'],size=row['size'],attempts=[],verified=False,reused_existing_bytes=False)
    def record():
        folder=Path(binding['out'])/'parent-state-read-retries';folder.mkdir(mode=0o700,exist_ok=True)
        target=folder/(row['name']+'.json');temporary=target.with_suffix('.tmp')
        temporary.write_bytes(canonical(evidence));temporary.chmod(0o600);temporary.replace(target)
    class RetryableStatus(requests.ConnectionError):pass
    class OneAttemptClient:
        def get(self,*args,**kwargs):
            response=requests.get(*args,**kwargs)
            if response.status_code in(429,500,502,503,504):
                status=response.status_code;response.close();error=RetryableStatus('transient parent GET status');error.status_code=status;raise error
            return response
    transient=(requests.Timeout,requests.ConnectionError,requests.exceptions.ChunkedEncodingError)
    for attempt in range(max_attempts):
        if time.time()>=expires_at:
            evidence['status']='infrastructure_deferred_original_expiry';record();raise StateReadInfrastructureDeferred('original parent read job expired; no extension')
        started=time.time()
        try:
            if path.exists():
                st=path.stat()
                if path.is_symlink()or path.absolute()!=path.resolve()or st.st_uid!=os.getuid()or st.st_nlink!=1 or not path.is_file():raise ValueError('owned single-link parent shard')
                if _hash_file(path)!=(row['sha256'],row['size']):raise ArtifactRejected('existing parent shard digest/size')
                evidence['reused_existing_bytes']=True
            else:_get_object_once(url,row['sha256'],path,row['size'],session=OneAttemptClient(),record_lifecycle=False)
            if path.stat().st_size!=row['size']:raise ArtifactRejected('parent shard exact size')
            evidence['attempts'].append(dict(attempt=attempt+1,started_at=started,completed_at=time.time(),status='verified_complete'))
            evidence.update(verified=True,status='complete')
            if time.time()>=expires_at:
                evidence['status']='infrastructure_deferred_original_expiry';record();raise StateReadInfrastructureDeferred('verified parent bytes arrived after original expiry; no update')
            record();return
        except transient as error:
            evidence['attempts'].append(dict(attempt=attempt+1,started_at=started,completed_at=time.time(),status='transient_transport_failure',error_type=type(error).__name__,HTTP_status=getattr(error,'status_code',None)))
            evidence['status']='retrying'if attempt+1<max_attempts else 'infrastructure_deferred_retry_exhausted';record()
            if attempt+1==max_attempts:raise StateReadInfrastructureDeferred('bounded original parent shard reads exhausted')from error
            delay=min(4,2**attempt)
            if time.time()+delay>=expires_at:
                evidence['status']='infrastructure_deferred_original_expiry';record();raise StateReadInfrastructureDeferred('original parent read expiry before retry')from error
            sleep(delay)

def get_object(url,expected,destination,limit,*,session=None,record_lifecycle=True):
    binding=_parent_read_binding(url,expected,destination,limit)
    if binding is not None:
        if session is not None:raise ValueError('parent reads use dedicated one-attempt client')
        return _retry_parent_object(url,binding,destination)
    return _get_object_once(url,expected,destination,limit,session=session,record_lifecycle=record_lifecycle)


def prefetched_training_submissions(submissions,out,timings,*,workers=4,session_factory=None):
    """Bounded four-ahead reads; admission and ownership receipts stay serial.

    Each HTTP session belongs to exactly one executor thread and is retained
    only for this job. No model/proof validation is performed by downloader
    threads. Preserve submission ordering and fail before training on errors.
    Caller must close this generator when eligibility admission fails.
    """
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    import threading
    import requests
    if type(workers)is not int or not 1<=workers<=4:raise ValueError('bounded training download concurrency')
    local=threading.local();sessions=[];lock=threading.Lock();pending=deque()
    factory=session_factory or requests.Session
    def fetch(i,obj):
        if type(obj.get('size'))is not int or not 0<obj['size']<=2_000_000:
            raise ArtifactRejected('bounded compact training artifact size')
        if not hasattr(local,'session'):
            local.session=factory()
            with lock:sessions.append(local.session)
        path=out/('submission-'+str(i)+'.json');started=time.monotonic()
        try:get_object(obj['url'],obj['sha256'],path,obj['size'],session=local.session,record_lifecycle=False)
        except requests.RequestException:raise ValueError('R2 training artifact transport failure')from None
        return i,obj,path,time.monotonic()-started
    pool=ThreadPoolExecutor(max_workers=workers);items=iter(enumerate(submissions));started=time.monotonic()
    def record_download(obj,path):
        lifecycle_root=os.environ.get('AFFINE_CACHE_LIFECYCLE_ROOT')
        if lifecycle_root:
            from .cache_lifecycle import CacheLifecycle
            CacheLifecycle(lifecycle_root).record_download(path,obj['sha256'])
    def submit_next():
        try:i,obj=next(items)
        except StopIteration:return
        pending.append(pool.submit(fetch,i,obj))
    try:
        for _ in range(workers):submit_next()
        while pending:
            i,obj,path,seconds=pending.popleft().result()
            # All receipt writes execute on this consumer thread. Concurrent
            # record_download updates for the same job would lose members.
            record_download(obj,path)
            row=timings.setdefault('submission_download_and_authentication',dict(seconds=0.0,calls=0))
            row['seconds']+=seconds;row['calls']+=1
            yield i,obj,path
            submit_next()
    finally:
        for future in pending:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        # Completed reads ahead of a failed admission are still owned,
        # authenticated downloads, not admitted training data. Record them
        # serially so a later terminal cleanup can retire these inputs too.
        try:
            for future in pending:
                if future.cancelled():continue
                try:i,obj,path,seconds=future.result()
                except Exception:continue
                record_download(obj,path)
        finally:
            for session in sessions:session.close()
        timings['submission_prefetch_pipeline_wall']=dict(seconds=time.monotonic()-started,calls=1,
            workers=workers,includes_ordered_eligibility_admission=True)

def parallel_checkpoint_policy(manifest):
    """An explicit signed transport option; numerical execution is unchanged."""
    p=manifest.get('checkpoint_download_policy')
    if p is None:return None
    if (type(p)is not dict or set(p)!={'version','workers','file_sizes','disk_floor_bytes'} or
            p['version']!='bounded-parallel-checkpoint-GET-v1' or
            type(p['workers'])is not int or not 2<=p['workers']<=4 or
            type(p['disk_floor_bytes'])is not int or not 2*1024**3<=p['disk_floor_bytes']<=20*1024**3 or
            type(p['file_sizes'])is not dict or set(p['file_sizes'])!=set(manifest['checkpoint']['files']) or
            any(type(s)is not int or not 0<s<=5*1024**3 for s in p['file_sizes'].values()) or
            sum(p['file_sizes'].values())>20*1024**3):
        raise ValueError('bounded signed checkpoint download policy')
    return p

def _parallel_checkpoint(cp,target,workspace,policy,lifecycle):
    import fcntl,stat,shutil,requests
    from concurrent.futures import ThreadPoolExecutor,as_completed
    # A separate same-checkpoint lock serializes concurrent candidate hydrations.
    # Existing outer checkpoint leases continue protecting files from retirement.
    if re.fullmatch('[0-9a-f]{64}',cp['id'])is None or target!=target.resolve():raise ValueError('owned checkpoint target')
    locks=Path(workspace)/'.checkpoint-download-locks';locks.mkdir(mode=0o700,exist_ok=True)
    if locks!=locks.resolve():raise ValueError('checkpoint lock symlink')
    fd=os.open(locks/(cp['id']+'.lock'),os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        st=os.fstat(fd)
        if not stat.S_ISREG(st.st_mode)or st.st_nlink!=1 or st.st_uid!=os.getuid():raise ValueError('owned checkpoint lock')
        fcntl.flock(fd,fcntl.LOCK_EX)
        missing=[]
        for name,sha in cp['files'].items():
            if Path(name).name!=name:raise ValueError('checkpoint member path')
            path=target/name
            if path.exists()or path.is_symlink():
                st=path.lstat()
                if path!=path.resolve()or not stat.S_ISREG(st.st_mode)or st.st_nlink!=1 or st.st_uid!=os.getuid():raise ValueError('owned checkpoint member')
                if st.st_size==policy['file_sizes'][name]and digest(path)==sha:
                    if lifecycle:lifecycle.record_checkpoint_member(cp['id'],name,cp['files'],sha)
                    continue
                raise ValueError('existing checkpoint digest/size mismatch')
            missing.append((name,sha,path))
        reserve=sum(policy['file_sizes'][n]for n,_,_ in missing)
        reserve+=sum(sorted((policy['file_sizes'][n]for n,_,_ in missing),reverse=True)[:policy['workers']])
        if shutil.disk_usage(target).free<reserve+policy['disk_floor_bytes']:raise ValueError('parallel checkpoint disk reserve')
        def fetch(row):
            name,sha,path=row
            if path.with_suffix(path.suffix+'.partial').exists()or path.with_suffix(path.suffix+'.partial').is_symlink():raise ValueError('preexisting checkpoint partial')
            # One independently closed session per member, never a shared pool.
            with requests.Session()as session:
                get_object(cp['read_urls'][name],sha,path,policy['file_sizes'][name],session=session)
            st=path.lstat()
            if (path!=path.resolve()or not stat.S_ISREG(st.st_mode)or st.st_nlink!=1 or st.st_uid!=os.getuid()or st.st_size!=policy['file_sizes'][name]):raise ArtifactRejected('checkpoint exact owned signed size')
            return name
        with ThreadPoolExecutor(max_workers=policy['workers'])as pool:
            futures=[pool.submit(fetch,row)for row in missing];failure=None
            for future in as_completed(futures):
                try:name=future.result()
                except BaseException as error:
                    if failure is None:failure=error
                    for pending in futures:pending.cancel()
                    continue
                # Serialize catalog writes, including successful in-flight reads
                # after a sibling failure. They remain owned, never admitted yet.
                if lifecycle:lifecycle.record_checkpoint_member(cp['id'],name,cp['files'],cp['files'][name])
            if failure is not None:raise failure
    finally:os.close(fd)

def checkpoint(manifest, workspace, cache=None):
    cp=manifest['checkpoint'];target=Path(cache) if cache else workspace/'checkpoints'/cp['id']
    target.mkdir(parents=True,exist_ok=True)
    lifecycle=None
    if os.environ.get('AFFINE_CACHE_LIFECYCLE_ROOT') and not cache:
        from .cache_lifecycle import CacheLifecycle
        lifecycle=CacheLifecycle(workspace)
    policy=parallel_checkpoint_policy(manifest)
    if policy is not None and not cache:
        _parallel_checkpoint(cp,target,workspace,policy,lifecycle)
        from .model import model_files
        if model_files(target)!=cp['files']:raise ValueError('checkpoint exact allowlist')
        return target
    for name,sha in cp['files'].items():
        path=target/name
        if path.is_symlink():raise ValueError('checkpoint symlink')
        if path.is_file() and digest(path)==sha:
            if lifecycle:lifecycle.record_checkpoint_member(cp["id"],name,cp["files"],sha)
            continue
        if cache:raise ValueError('approved cached checkpoint mismatch')
        get_object(cp['read_urls'][name],sha,path,20_000_000_000)
        if lifecycle:lifecycle.record_checkpoint_member(cp['id'],name,cp['files'],sha)
    from .model import model_files
    if model_files(target)!=cp['files']:raise ValueError('checkpoint exact allowlist')
    return target

def audit(data, manifest, runtime, *, commitment_miner=None):
    report,pairs=_audit(data,manifest,runtime,commitment_miner=commitment_miner)
    from .forced_sampling import MINER_VERSION
    if manifest.get('sampling_contract',{}).get('version')==MINER_VERSION:
        report['sampling_miner']=commitment_miner
    return report,pairs

def _audit(data, manifest, runtime, *, commitment_miner=None):
    from .sampling_uniqueness import validate_batch
    from .forced_sampling import MINER_VERSION, bind_runtime
    if manifest.get('sampling_contract',{}).get('version')==MINER_VERSION:
        bind_runtime(runtime,manifest,commitment_miner)
    from .forced_sampling import assurance as sampling_assurance
    from .fast_prefill_audit import NumericalAmbiguity,THREEWAY_VERSION
    threeway=manifest.get('sampling_contract',{}).get('version')==THREEWAY_VERSION
    from .batches import submission_records,SubmissionRejected
    from .protocol import entries, entry, classification, sample_key,harness_for
    from .artifact_budget import for_manifest
    definitions=entries(manifest);budget=for_manifest(manifest)
    token_only=False
    if 'token_artifact_policy'in manifest:
        from .token_only_protocol import for_manifest as token_policy
        token_only=token_policy(manifest) is not None
    try:
        if token_only:
            from .token_only_protocol import unpack
            try:records=unpack(data,budget=budget,max_batches=manifest.get('max_batches',4))
            except (ValueError,KeyError,TypeError,IndexError,AttributeError,EOFError,UnicodeError,__import__('zipfile').BadZipFile) as error:
                raise SubmissionRejected(str(error)) from error
        else:records=submission_records(data,budget=budget,max_batches=manifest.get('max_batches',4))
    except SubmissionRejected as error:
        return dict(epoch=manifest['epoch'],submission_sha256=hashlib.sha256(data).hexdigest(),
            submission_rejected=True,rejection_stage='transport',
            sampling_assurance=sampling_assurance(manifest),
            outcomes=[dict(batch=None,valid=False,reason=str(error),rejection_stage='transport')],
            accepted=[],training_eligibility='fully-audited-only'),[]
    if manifest.get('submission_transport_policy'):
        from .commitment_transport import VERSIONS
        if manifest['submission_transport_policy']not in VERSIONS:raise ValueError('unsupported commitment policy')
        digest=hashlib.sha256(data).hexdigest()
        root=manifest.get('audit_frozen_receipts',{}).get(commitment_miner,{})
        claims=[b for b in root.get('artifacts',[])if b['sha256']==digest]
        if len(claims)!=1 or len(records)!=1 or hashlib.sha256(canonical(records[0][0])).hexdigest()!=claims[0]['batch_sha256']:
            return dict(epoch=manifest['epoch'],submission_sha256=digest,policy=manifest['audit_policy'],sampling_assurance=sampling_assurance(manifest),outcomes=[dict(batch=0,valid=False,failure_kind='structural_invalid',fully_audited=False)],accepted=[],training_eligibility='fully-audited-only'),[]
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
        confirmed_invalid=False;uncertain=[]
        try:
            if batch.get('environment_version')!=selected.spec.version:raise ValueError('environment version')
            try:validate_batch(batch,manifest,commitment_miner)
            except (ValueError,KeyError,TypeError)as error:
                if manifest.get('sampling_contract',{}).get('version')==MINER_VERSION:raise InvalidSample('v5 structural attempt/content quota invalid')from error
                raise
            seen.add(key);rolls=batch['rollouts'];tokens=set()
            if len(rolls)!=manifest['K']+manifest['L'] or len(arrays)!=len(rolls):raise ValueError('sample count')
            for rollout_number,(rollout,probs) in enumerate(zip(rolls,arrays)):
                signature=tuple(tuple(t['output']) for t in rollout['turns'])
                if signature in tokens or rollout['index']!=index or rollout.get('env_id')!=definition['env_id']:raise ValueError('sample binding/duplicate')
                tokens.add(signature)
                if number in selected_indices:
                    try:
                        if token_only:
                            from .token_only_runtime import verify
                            verified=verify(selected,manifest,rollout,eligible_indices=definition['indices'])['valid']
                        else:verified=selected.verify(rollout,probs)
                    except NumericalAmbiguity as error:
                        if not threeway:raise
                        uncertain.append((rollout_number,error));continue
                    except InvalidSample:raise
                    except Exception as error:
                        if policy.get('version')=='bounded-random-v1':
                            raise RuntimeError('audit execution failed; retry without miner penalty') from error
                        raise
                    if verified is not True:
                        confirmed_invalid=True
                        raise ValueError('inference or replay')
            pos=[r for r in rolls if classification(r)=='positive'];neg=[r for r in rolls if classification(r)=='negative']
            if len(pos)!=manifest['K'] or len(neg)!=manifest['L']:
                raise (InvalidSample if threeway else ValueError)('positive/negative quota')
            if uncertain:
                error=uncertain[0][1]
                error.environment_verification_complete=all(getattr(e,'environment_verification_complete',False)for _,e in uncertain)
                error.uncertain_rollouts=[dict(rollout=i,turns=getattr(e,'uncertain_turns',[]),positions=getattr(e,'uncertain_positions',[]),count=getattr(e,'uncertain_position_count',0))for i,e in uncertain]
                error.uncertain_position_count=sum(row['count']for row in error.uncertain_rollouts)
                raise error
            if number in selected_indices:
                accepted.append(batch);pairs.extend((definition,p,n) for p,n in zip(pos,neg))
            outcomes.append(dict(batch=number,env_id=definition['env_id'],index=index,structural_valid=True,valid=True if number in selected_indices else None,fully_audited=number in selected_indices))
        except NumericalAmbiguity as error:
            outcome=dict(batch=number,valid=None,fully_audited=False,failure_kind='numerical_ambiguous',reason=str(error)[:300])
            from .fast_prefill_audit import THREEWAY_VERSION
            if manifest.get('sampling_contract',{}).get('version')==THREEWAY_VERSION:
                outcome.update(sampling_verification_complete=False,environment_verification_complete=getattr(error,'environment_verification_complete',False))
                outcome['uncertain_rollouts']=getattr(error,'uncertain_rollouts',[])
                if hasattr(error,'uncertain_positions'):
                    outcome.update(uncertain_token_positions=error.uncertain_positions,uncertain_token_position_count=error.uncertain_position_count)
            outcomes.append(outcome)
        except (ValueError,KeyError,TypeError,IndexError) as error:
            outcomes.append(dict(batch=number,valid=False,fully_audited=number in selected_indices,failure_kind='confirmed_invalid' if confirmed_invalid or isinstance(error,InvalidSample) else 'verification_error',reason=type(error).__name__+': '+str(error)[:300]))
    return dict(epoch=manifest['epoch'],submission_sha256=hashlib.sha256(data).hexdigest(),policy=policy,selected_batches=sorted(selected_indices),assurance=assurance(len(records),len(selected_indices)),sampling_assurance=sampling_assurance(manifest),outcomes=outcomes,accepted=accepted,training_eligibility='fully-audited-only',**({'sampling_miner':commitment_miner}if manifest.get('sampling_contract',{}).get('version')==MINER_VERSION else {})),pairs

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
    for module_name in ('subnet.successor_calibration','subnet.backend_profiles','subnet.artifact_budget','subnet.audit_policy','subnet.auditing','subnet.training_policy','subnet.commitment_transport',
            'subnet.persistent_cpu_adamw','subnet.persistent_training_state','subnet.persistent_training_protocol','subnet.training_receipts'):
        sys.modules.pop(module_name,None)
    if 'subnet/unaudited_training_execution.py'in additional_files:
        sys.modules.pop('subnet.unaudited_training_execution',None)
    if 'subnet/learning_rate_transition.py'in additional_files:
        sys.modules.pop('subnet.learning_rate_transition',None)
    if 'subnet/optimizer_state_cache.py'in additional_files:
        sys.modules.pop('subnet.optimizer_state_cache',None)
        sys.modules.pop('subnet.cache_lifecycle',None)
    if 'subnet/training_startup_recovery.py'in additional_files:
        sys.modules.pop('subnet.training_startup_recovery',None)
    if 'subnet/committed_training_inputs.py'in additional_files:
        sys.modules.pop('subnet.committed_training_inputs',None)
    if 'subnet/training_task_representatives.py'in additional_files:
        # Execution admission reads the representative policy before source
        # authentication. Reload this metadata helper through the pinned finder.
        sys.modules.pop('subnet.training_task_representatives',None)
    if 'subnet/native_math_prompt.py'in additional_files:
        sys.modules.pop('subnet.native_math_prompt',None)
    if 'subnet/compact_training_inputs.py' in additional_files:
        sys.modules.pop('subnet.compact_training_inputs',None)
    if 'subnet/persistent_publication.py' in additional_files:
        # _validate() uses this module's pure policy admission before execute()
        # authenticates every pinned source byte. Reload it through the finder,
        # just like the other bootstrap admission helpers; never retain its
        # pre-validation implementation for training or publication.
        sys.modules.pop('subnet.persistent_publication',None)
    if 'subnet/trainer_local_state.py' in additional_files:
        sys.modules.pop('subnet.trainer_local_state',None)
    for name in set(SOURCE_FILES)|set(additional_files):
        module_name=name[:-3].replace('/','.')
        if module_name in sys.modules and module_name!='subnet.backend_jobs':
            raise ValueError('GPU worker requires fresh process before runtime imports')
    sys.meta_path.insert(0,FreshSourceFinder(root))

def initial_configuration(manifest,job):
    from .protocol import entries,entry,harness_for
    definitions=entries(manifest)
    if job['role']=='evaluate' and job.get('successor_calibration') is not None:
        r=job['successor_calibration'];return entry(manifest,r['env_id']),r['harness']
    if job['role']=='evaluate':
        suite=job['heldout'][0]
        return entry(manifest,suite['env_id']),suite['harness']
    first=next((row for row in definitions if row['indices']),None)
    if first is None:raise ValueError('no authorized mining samples for model role')
    return first,harness_for(first,first['indices'][0])

def native_math_prompt_enabled(job,manifest):
    # Historical signed jobs retain the original session and original pin set.
    # A new authenticated job pin selects the eligibility-only implementation.
    marker=manifest.get('native_math_prompt_eligibility_policy')
    if marker not in (None,'authenticated-original-math-prompt-v1'):
        raise ValueError('native MATH prompt eligibility policy')
    pinned='subnet/native_math_prompt.py'in job.get('source_files',{})
    if marker is not None and not pinned:
        raise ValueError('native MATH prompt eligibility source pin required')
    return pinned

def measured_phase(timings,name,operation,*args,**kwargs):
    """Record completed original operations; never repeat or swallow failures."""
    started=time.monotonic()
    result=operation(*args,**kwargs)
    row=timings.setdefault(name,dict(seconds=0.0,calls=0))
    row['seconds']+=time.monotonic()-started;row['calls']+=1
    return result

def execute(envelope, authority, workspace, cache=None, runtime_factory=None):
    startup_timings={}
    job,manifest=measured_phase(startup_timings,'job_validation',_validate,envelope,authority,resolve_source=False)
    source_started=time.monotonic()
    root=Path(__file__).resolve().parent.parent
    if job.get('role')=='train' and job.get('training_policy')==PERSISTENT_POLICY and (root/'subnet/fp32_gradient_accumulation.py').exists() and 'unaudited_training_execution'not in job:
        raise ValueError('FP32 worker source requires explicit qualified execution declaration')
    for name,expected in job['source_files'].items():
        if (root/name).is_symlink() or digest(root/name)!=expected:raise ValueError('worker source mismatch')
    for name,expected in job['runtime_versions'].items():
        if version(name)!=expected:raise ValueError('runtime package mismatch')
    startup_timings['source_runtime_authentication']=dict(seconds=time.monotonic()-source_started,calls=1)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('CUDA environment profile')
    execution_files=('subnet/unaudited_training_execution.py','subnet/fp32_gradient_accumulation.py','subnet/learning_rate_transition.py') if 'unaudited_training_execution'in job else ()
    token_files=('subnet/token_only_protocol.py','subnet/token_only_runtime.py','subnet/threeway_prefill_research.py','subnet/native_session_validation.py')if 'token_artifact_policy'in manifest else ()
    publication_files=('subnet/persistent_publication.py',) if manifest.get('persistent_publication_policy') is not None else ()
    if manifest.get('optimizer_state_export_policy')=='trainer-local-only-v1':
        publication_files+=('subnet/trainer_local_state.py',)
    recovery_files=('subnet/training_startup_recovery.py',)if manifest.get('training_startup_recovery')is not None else ()
    learner_files=('subnet/committed_training_inputs.py',)if manifest.get('training_input_policy')=='committed-unaudited-training-v1'else ()
    if 'training_representative_policy' in manifest:
        if 'subnet/training_task_representatives.py' not in job['source_files']:raise ValueError('representative helper exact source pin required')
        learner_files+=('subnet/training_task_representatives.py',)
    if learner_files and job['role']=='train'and native_math_prompt_enabled(job,manifest):learner_files+=('subnet/native_math_prompt.py',)
    compact_files=('subnet/compact_training_inputs.py',) if (manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2') else ()
    if job.get('training_policy')==PERSISTENT_POLICY:
        from .persistent_training_protocol import EXECUTION_FILES,CACHE_EXECUTION_FILES
        cache_files=CACHE_EXECUTION_FILES if manifest.get('optimizer_state_local_cache')is not None else ()
        install_source_loader(root,(*EXECUTION_FILES,*cache_files,'subnet/training_receipts.py',*compact_files,*learner_files,*publication_files,*recovery_files,*token_files,*execution_files))
    elif job.get('training_policy')==COVERED_POLICY:
        install_source_loader(root,('subnet/training_receipts.py',*compact_files,*learner_files,*publication_files,*recovery_files,*token_files))
    else:install_source_loader(root,(*compact_files,*learner_files,*publication_files,*recovery_files,*token_files))
    # Receipt admission can import protocol/model definitions. Run it only
    # after the authenticated fresh-source loader, still before any artifacts,
    # model construction or optimizer mutation. Public validate remains strict.
    if job['role']=='train' and job.get('training_policy') in (COVERED_POLICY,PERSISTENT_POLICY):
        if manifest.get('training_input_policy')=='committed-unaudited-training-v1':
            from .committed_training_inputs import validate_job as validate_training_receipts
        elif manifest.get('training_input_policy')=='authenticated-verifier-compact-inputs-v2':
            from .compact_training_inputs import validate_job as validate_training_receipts
        else:
            from .training_receipts import validate_job as validate_training_receipts
        measured_phase(startup_timings,'authenticated_training_input_admission',validate_training_receipts,job,manifest,authority)
    if job['role']=='mine' and manifest.get('submission_transport_policy') is not None:
        owned_miner_identity(job)
    if publication_files:
        from .persistent_publication import validate_policy
        validate_policy(manifest['persistent_publication_policy'])
    if 'token_artifact_policy'in manifest or manifest.get('submission_transport_policy')=='small-commitment-token-pairs-v3':
        from .token_only_protocol import for_manifest as token_policy
        token_policy(manifest)
        required={'subnet/token_only_protocol.py','subnet/token_only_runtime.py','subnet/threeway_prefill_research.py'}
        if manifest.get('native_source_validation_policy')is not None:required.add('subnet/native_session_validation.py')
        if not required<=set(job['source_files']):raise ValueError('token-only complete source pins')
    global _PARENT_READ_CONTEXT
    _PARENT_READ_CONTEXT=(job,manifest,authority,str(Path(workspace).absolute()))
    from .backend_profiles import resolve
    from .backend_profiles import execution_profile
    revision,backend_profile,numerical_policy=execution_profile(manifest,job['role'])
    from .artifact_budget import for_manifest
    for_manifest(manifest)
    from .task_assets import hydrate_manifest
    asset_root=Path(workspace)/'task-assets'
    if manifest.get('task_assets'):
        asset_root.mkdir(parents=True,exist_ok=True);asset_root.chmod(0o700)
        os.environ['AFFINE_MATH_CORPUS_ASSET_ROOT']=str(asset_root.resolve())
    measured_phase(startup_timings,'task_asset_hydration',hydrate_manifest,asset_root,manifest)
    # Resolve against authenticated fresh source before any artifact, workspace,
    # checkpoint or model is opened. Public validate() remains fully strict.
    if job.get('mining_subset') is not None:mining_definitions(manifest,job)
    if job.get('replay') is not None:
        from .replay_training import admitted
        admitted(manifest,job['replay'],authority)
    workspace=Path(workspace);out=workspace/'jobs'/job['job_id']
    out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
    approved=measured_phase(startup_timings,'checkpoint_materialization_and_authentication',checkpoint,manifest,workspace,cache)
    report=dict(schema=1,job_id=job['job_id'],role=job['role'],operator=authority,
        job_sha256=hashlib.sha256(canonical(job)).hexdigest(),checkpoint=manifest['checkpoint']['id'],
        epoch=manifest['epoch'],backend_profile=backend_profile,numerical_policy=numerical_policy,
        source_files=job['source_files'],runtime_versions=job['runtime_versions'],
        chain_transactions=False,full_model_finetune=False,execution_resources_enforced=False,
        startup_timings=dict(version="original-role-phase-timings-v1",clock="monotonic",
            GPU_synchronized=False,phases=startup_timings))
    if 'unaudited_training_execution'in job:
        from .unaudited_training_execution import provenance
        report['unaudited_training_execution']=provenance(envelope,authority)
    if manifest.get('training_runtime')is not None:
        report['execution_runtime_revision']=revision
        report['generation_runtime_revision']=manifest['model_runtime_revision']
        report['training_runtime_sha256']=hashlib.sha256(canonical(manifest['training_runtime'])).hexdigest()
    if job['role']=='upload':
        import requests
        validate_single_put_sizes(approved,manifest['checkpoint']['files'])
        def upload(row):
            name,url=row
            from .persistent_training_worker import put_file
            # Bound stalled socket operations, not total streaming duration.
            # Transient retries rewind the same inode and immutable bytes.
            put_file(url,approved/name,timeout=(15,90))
        workers=1
        if manifest.get('persistent_publication_policy') is not None:
            from .persistent_publication import validate_policy
            workers=validate_policy(manifest['persistent_publication_policy'])['checkpoint_readback_workers']
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(upload,job['put_urls'].items()))
        report['uploaded_files']=manifest['checkpoint']['files']
    else:
        from .protocol import entries,entry,harness_for
        from .gpu_runtime import GPURuntime
        factory=runtime_factory or GPURuntime;definitions=entries(manifest)
        first,initial_harness=initial_configuration(manifest,job)
        if job['role']=='evaluate' and job.get('successor_calibration')is not None:
            from .successor_calibration import preflight_native_spec
            preflight_native_spec(first['spec'])
        native_validations={}
        if 'token_artifact_policy'in manifest:
            from .token_only_protocol import prepare_native_validations
            native_validations=prepare_native_validations(job,manifest,authority)
        elif manifest.get('native_source_validation_policy')is not None or job.get('native_source_validation_scopes')is not None:
            raise ValueError('explicit native validation candidate required')
        if runtime_factory is None:
            runtime=measured_phase(startup_timings,'runtime_model_construction',factory,approved,manifest['checkpoint']['files'],first['spec'],initial_harness,
                runtime_revision=revision)
        else:
            runtime=measured_phase(startup_timings,'runtime_model_construction',factory,approved,manifest['checkpoint']['files'],first['spec'],initial_harness)
        if native_validations:
            runtime.native_source_validations=native_validations
            runtime.native_source_validation=native_validations.get(runtime.spec.id)
        from .forced_sampling import MINER_VERSION
        if job['role']=='train' and manifest.get('sampling_contract',{}).get('version')==MINER_VERSION and manifest.get('training_input_policy')=='committed-unaudited-training-v1' and job.get('training_policy')in(COVERED_POLICY,PERSISTENT_POLICY):
            # Trainer uses signed cheap-admission documents from multiple miners;
            # it never generates or verifies their prescribed sampling attempts.
            runtime.sampling_context=None
            runtime.fast_sampling_calibration=None
            runtime.probability_artifact_policy=None
        elif job['role'] != 'evaluate':
            from .forced_sampling import bind_runtime, MINER_VERSION
            bind_runtime(runtime,manifest,job.get('miner_id') if job['role']=='mine' else (job.get('submissions')or[{}])[0].get('commitment_miner'))
            from .probability_artifacts import bind_runtime as bind_artifacts
            bind_artifacts(runtime,manifest)
            if 'token_artifact_policy'in manifest:
                from .token_only_protocol import bind_runtime as bind_tokens
                bind_tokens(runtime,manifest)
        elif manifest.get('sampling_contract') is not None:
            runtime.sampling_context=None
            report['sampling_scope']='heldout-diagnostic-not-mining-evidence'
        if job['role']=='mine':
            from .batches import pack
            import requests
            def upload(data,timeout):
                response=requests.put(job['capability']['put_url'],data=data,headers=job['capability']['headers'],timeout=timeout,allow_redirects=False)
                if response.status_code not in (200,201,204):raise ValueError('R2 PUT status '+str(response.status_code))
            if manifest.get('submission_transport_policy') is not None:upload=owned_commitment_upload(job,manifest,out/'commitment-upload-journal.json')
            def progress(value):
                path=out/'mining-progress.json';temp=out/'mining-progress.tmp'
                temp.write_bytes(canonical(value));temp.chmod(0o600);temp.replace(path)
                print(canonical(value).decode(),flush=True)
            data,mining=mine_cumulative(runtime,manifest,job,upload,allow_empty=True,progress=progress)
            if data is not None:
                name='submission-commitment.json' if manifest.get('submission_transport_policy') is not None else 'submission.zip'
                (out/name).write_bytes(data)
            report.update(miner_id=job['miner_id'],submission_sha256=hashlib.sha256(data).hexdigest() if data is not None else None,submission_size=len(data) if data is not None else 0,operator_authorized_experiment=True,**mining)
        elif job['role'] in ('verify','train'):
            reports=[];pairs=[]
            from contextlib import closing,nullcontext
            compact_training=(job['role']=='train' and job.get('training_policy')in (COVERED_POLICY,PERSISTENT_POLICY) and
                manifest.get('training_input_policy')in ('authenticated-verifier-compact-inputs-v2','committed-unaudited-training-v1'))
            prefetch=prefetched_training_submissions(job['submissions'],out,startup_timings)if compact_training else None
            with closing(prefetch)if prefetch is not None else nullcontext():
                for i,obj in enumerate(job['submissions']):
                    from .artifact_budget import for_manifest
                    compact_input=job['role']=='train' and manifest.get('training_input_policy')in ('authenticated-verifier-compact-inputs-v2','committed-unaudited-training-v1')
                    path=out/('submission-'+str(i)+('.json' if compact_input else '.zip'))
                    limit=obj['size'] if compact_input else for_manifest(manifest)['compressed_bytes']
                    try:
                        if prefetch is not None:
                            actual_i,actual_obj,actual_path=next(prefetch)
                            if actual_i!=i or actual_obj!=obj or actual_path!=path:raise ValueError('ordered training input binding')
                        else:measured_phase(startup_timings,'submission_download_and_authentication',get_object,obj['url'],obj['sha256'],path,limit)
                    except ArtifactRejected:
                        if job['role']!='verify'or not manifest.get('submission_transport_policy'):raise
                        from .forced_sampling import assurance
                        reports.append(dict(epoch=manifest['epoch'],submission_sha256=obj['sha256'],accepted=[],outcomes=[dict(batch=0,valid=False,fully_audited=False,failure_kind='structural_invalid')],sampling_assurance=assurance(manifest),training_eligibility='fully-audited-only',**({'sampling_miner':obj.get('commitment_miner')}if manifest.get('sampling_contract',{}).get('version')==MINER_VERSION else {})));continue
                    if job['role']=='train' and job.get('training_policy') in (COVERED_POLICY,PERSISTENT_POLICY):
                        if manifest.get('training_input_policy')=='committed-unaudited-training-v1':
                            from .committed_training_inputs import admitted_submission
                        elif compact_input:
                            from .compact_training_inputs import admitted_submission
                        else:
                            from .training_receipts import admitted_submission
                        result,verified=measured_phase(startup_timings,'submission_eligibility_admission',admitted_submission,path,obj,manifest,authority,
                            retire=job.get('training_policy')==PERSISTENT_POLICY)
                    else:result,verified=audit(path.read_bytes(),manifest,runtime,commitment_miner=obj.get('commitment_miner'))
                    reports.append(result);pairs.extend(verified)
            receipt_training=job['role']=='train' and job.get('training_policy') in (COVERED_POLICY,PERSISTENT_POLICY)
            report['audits']=[] if receipt_training else reports
            if receipt_training:report['training_admissions']=reports
            if job['role']=='train':
                if not pairs:raise ValueError('no admitted training pairs')
                if manifest.get('training_input_policy')=='committed-unaudited-training-v1':
                    from .committed_training_inputs import validate_native_prompt
                    measured_phase(startup_timings,'native_prompt_eligibility',validate_native_prompt,runtime,pairs,manifest,prompt_only=native_math_prompt_enabled(job,manifest))
                values_before=measured_phase(startup_timings,'parameter_digest_before',parameter_value_digest,runtime.model)
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
                    phases=persistent_diagnostics.get('transport_phase_seconds',{})
                    for name in ('parent_cache_validation_and_admission','parent_state_restore'):
                        if name in phases:startup_timings[name]=dict(seconds=phases[name],calls=1)
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
                values_after=measured_phase(startup_timings,'parameter_digest_after',parameter_value_digest,runtime.model)
                if not persistent and values_before==values_after:raise ValueError('optimizer did not change parameter values')
                report['training']=dict(steps=job['steps'],updates=metrics,training_policy=job.get('training_policy',HEAD_POLICY),full_model_finetune=job.get('training_policy',HEAD_POLICY) in (FULL_POLICY,FIXED_POLICY,COVERED_POLICY,PERSISTENT_POLICY),weights_changed=values_before!=values_after,parameter_values_sha256_before=values_before,parameter_values_sha256_after=values_after)
                if job.get('training_policy')==COVERED_POLICY:report['training']['training_coverage']=manifest['training_coverage']
                if receipt_training:
                    if manifest.get('training_input_policy')=='committed-unaudited-training-v1':
                        from .committed_training_inputs import VERSION as INPUT_POLICY
                    elif (manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2'):
                        from .compact_training_inputs import VERSION as INPUT_POLICY
                    else:
                        from .training_receipts import VERSION as INPUT_POLICY
                    report['training'].update(training_input_policy=INPUT_POLICY,
                        trainer_verification_performed=False,all_pairs_authenticated_verifier_receipts=INPUT_POLICY!='committed-unaudited-training-v1')
                    if INPUT_POLICY=='committed-unaudited-training-v1':report['training']['input_assurance']='unaudited'
                if persistent:
                    report['training'].update(training_coverage=manifest['training_coverage'],
                        state_updated=True,
                        persistent_diagnostics=persistent_diagnostics,
                        global_step_before=manifest['trainer_state_binding']['global_step_before'],
                        global_step_after=job['persistent_training']['global_step_after'])
                report['full_model_finetune']=report['training']['full_model_finetune']
                report['new_checkpoint']=dict(id=file_map(files),files=files,path=str(destination))
        elif job.get('trusted_evaluation_policy') is not None:
            from .trusted_native_evaluation import evaluate as evaluate_trusted
            from .environments import create_session
            def evaluation_progress(value):
                value.update(at=time.time(),job_id=job['job_id'],checkpoint=manifest['checkpoint']['id'],verified=False,proof_verification_performed=False)
                temporary=out/'trusted-evaluation-progress.tmp';temporary.write_bytes(canonical(value));temporary.chmod(0o600);temporary.replace(out/'trusted-evaluation-progress.json')
            report['heldout'],report['heldout_failures'],report['trusted_native_evaluation']=evaluate_trusted(runtime,manifest,job,create_session=create_session,progress=evaluation_progress)
        elif job.get('owned_evaluation_policy') is not None:
            from .owned_cached_evaluation import evaluate as evaluate_owned
            from .environments import create_session
            report['heldout'],report['heldout_failures'],report['owned_cached_evaluation']=evaluate_owned(runtime,manifest,job,create_session=create_session)
        elif job.get('successor_calibration') is not None:
            from .successor_calibration import execute as execute_calibration
            report['successor_calibration']=execute_calibration(runtime,manifest,job['successor_calibration'])
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
    write_private_report(out/'report.json',report);return report

JOB_ENVELOPE_MAX_BYTES=4_000_000
RECOVERY_JOB_ENVELOPE_MAX_BYTES=8_000_000
LARGE_RECOVERY_VERSIONS=frozenset(('terminal-parent-cache-ACK-pre-update-recovery-v1','terminal-parent-restore-pre-update-recovery-v2','terminal-parent-restore-pre-update-bootstrap-recovery-v3','terminal-post-update-uncommitted-recovery-v1'))

def prospective_transport_scope(manifest):
    """Pure byte allowance; full scientific/source validation still follows."""
    cap=manifest.get('training_task_capacity')
    return (type(cap)is dict and set(cap)=={'version','max_tasks'}
        and cap['version']=='signed-training-task-capacity-v1' and type(cap['max_tasks'])is int and cap['max_tasks']==512
        and type(manifest.get('max_batches'))is int and manifest['max_batches']==9
        and type(manifest.get('K'))is int and manifest['K']==4 and type(manifest.get('L'))is int and manifest['L']==4
        and manifest.get('training_policy')=='bf16-cpu-fp32-master-task-normalized-persistent-v4'
        and manifest.get('training_input_policy')=='committed-unaudited-training-v1'
        and ('samples_per_batch'not in manifest or type(manifest['samples_per_batch'])is int and manifest['samples_per_batch']==8)
        and 'training_startup_recovery'not in manifest)

def load_job_envelope(path,authority):
    """Bounded CPU parser; signed9/512 roles alone may exceed historical limits.

    Full source/protocol admission remains mandatory before model construction.
    Historical ordinary4MB and authenticated recovery/execution8MB rules remain.
    """
    CAPACITY_JOB_MAX_BYTES=32_000_000
    with Path(path).open('rb')as stream:data=stream.read(CAPACITY_JOB_MAX_BYTES+1)
    if len(data)>CAPACITY_JOB_MAX_BYTES:raise ValueError('job envelope absolute size budget')
    envelope=json.loads(data)
    if type(envelope)is not dict or ('payload'in envelope and type(envelope['payload'])is not dict):raise ValueError('job envelope object')
    if len(data)<=JOB_ENVELOPE_MAX_BYTES:return envelope
    def root_payload(document):
        if type(document)is not dict or set(document)!={'payload','signer','signature'}or type(document.get('payload'))is not dict:raise ValueError('authenticated recovery envelope object')
        return signed(document,authority)
    job=root_payload(envelope);manifest=root_payload(job.get('manifest'))
    if job.get('role')in ROLES and prospective_transport_scope(manifest):
        if job['role']=='train':
            from .unaudited_training_execution import validate as validate_execution
            validate_execution(envelope,authority)
        return envelope
    if len(data)>RECOVERY_JOB_ENVELOPE_MAX_BYTES:raise ValueError('signed prospective9/512 job envelope budget')
    if job.get('role')!='train'or job.get('training_policy')!=PERSISTENT_POLICY or job.get('training_input_policy')!='committed-unaudited-training-v1':raise ValueError('job envelope size budget')
    if 'unaudited_training_execution'in job:
        from .unaudited_training_execution import validate as validate_execution
        validate_execution(envelope,authority)
        return envelope
    declaration=root_payload(manifest.get('training_startup_recovery'))
    if declaration.get('version')not in LARGE_RECOVERY_VERSIONS or type(manifest.get('epoch'))is not str or not manifest['epoch']or declaration.get('epoch')!=manifest.get('epoch'):raise ValueError('explicit recovery envelope budget')
    original=root_payload(declaration.get('original_signed_job'));old=root_payload(original.get('manifest'))
    if (original.get('role')!='train'or original.get('training_policy')!=PERSISTENT_POLICY
        or old.get('epoch')!=manifest.get('epoch')or original.get('training_input_policy')!=job['training_input_policy']
        or hashlib.sha256(canonical(original)).hexdigest()!=declaration.get('original_job_sha256')
        or type(manifest.get('source_bundle'))is not dict or type(old.get('source_bundle'))is not dict
        or any(type(bundle.get('sha256'))is not str or re.fullmatch('[0-9a-f]{64}',bundle['sha256'])is None for bundle in (manifest['source_bundle'],old['source_bundle']))
        or declaration.get('original_input_source_sha256')!=old['source_bundle'].get('sha256')
        or declaration.get('replacement_execution_source_sha256')!=manifest['source_bundle'].get('sha256')):raise ValueError('explicit authenticated original recovery scope')
    return envelope

def main():
    parser=argparse.ArgumentParser();parser.add_argument('job');parser.add_argument('--authority',required=True);parser.add_argument('--workspace',required=True);parser.add_argument('--checkpoint-cache')
    args=parser.parse_args();envelope=load_job_envelope(args.job,args.authority)
    report=execute(envelope,args.authority,args.workspace,args.checkpoint_cache)
    print(json.dumps(dict(job_id=report['job_id'],role=report['role'],success=True,checkpoint=report.get('new_checkpoint',{}).get('id',report['checkpoint']))))
if __name__=='__main__':
    # -m executes this file as __main__; worker/protocol imports use the
    # canonical name. Share the admitted implementation and parent-read scope.
    module=sys.modules[__name__]
    if sys.modules.get('subnet.backend_jobs',module)is not module:
        raise ValueError('backend entrypoint requires fresh canonical module')
    sys.modules['subnet.backend_jobs']=module
    setattr(sys.modules['subnet'],'backend_jobs',module)
    main()
