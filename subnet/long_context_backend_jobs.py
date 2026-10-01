"""Isolated signed 32K GPU roles; no chain writer or account credentials."""
from __future__ import annotations
import argparse,base64,hashlib,json,os,re,time,sys,io,zipfile
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from importlib.metadata import version
from nacl.signing import VerifyKey

REVISION='cuda-bf16-sdpa-flash-sm86-selective-head-common-v1'
NUMERICAL_POLICY=dict(logprob_atol=1e-5,logprob_rtol=0,toploc_exp_mismatches=0,toploc_mant_err_mean=0,toploc_mant_err_median=0)
BACKEND_PROFILE=dict(device='cuda',dtype='bfloat16',attention='sdpa-flash-only',sm=[8,6],tf32=False,deterministic_algorithms=True,cublas_workspace_config=':4096:8',native_toploc_threads=2,torch_threads=2,max_context=32768,output_head='output-prediction-rows-full-vocabulary',candidate_score_reduction='numpy-float32-sum')
TRANSPORT_POLICY=dict(compressed_bytes=250_000_000,raw_bytes=500_000_000)
TRAINING_POLICY='bf16-full-sequential-agent-turn-dpo-mean-v1'
TRAINING_PARAMETERS={'revision':TRAINING_POLICY,'objective':'reference-relative-all-agent-token-mean-preference','parameters':'bfloat16','gradients':'bfloat16','adam_moments':'bfloat16','master_parameters':False,'gradient_checkpointing':True,'use_cache':False,'turn_gradient_accumulation':'sequential-before-single-optimizer-step','learning_rate':5e-5,'beta':.1,'weight_decay':0.,'eps':1e-8,'clip_norm':1.,'max_peak_allocated_bytes':8*1024**3,'max_parameters':600000000}
ROLES={'mine','verify','train','evaluate','upload'}
SOURCE_FILES=tuple('subnet/'+n+'.py' for n in ('long_context_backend_jobs','long_context_native_factory','native_eog_deployment','native_eog_adapter','native_eog_split','native_eog_isolation','native_eog_clock','backend_jobs','long_context_runtime','long_context_training','long_context_service_runtime','long_context_service_training','model','harness','environments','proofs','batches','protocol','storage'))

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def exact_policy(value,expected):
    """JSON type-exact equality; bool/int/float substitutions are not policy."""
    try:return canonical(value)==canonical(expected)
    except (TypeError,ValueError):return False

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def signed(value,authority):
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


def pack(batches):
    import numpy as np
    out = io.BytesIO()
    manifest = []
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for bi, (batch, arrays) in enumerate(batches):
            refs = []
            for ri, turns in enumerate(arrays):
                row = []
                for ti, tensor in enumerate(turns):
                    name = f'{bi}-{ri}-{ti}.npy'
                    buf = io.BytesIO(); np.save(buf, tensor, allow_pickle=False)
                    z.writestr(name, buf.getvalue()); row.append(name)
                refs.append(row)
            manifest.append(dict(batch=batch, arrays=refs))
        z.writestr('manifest.json', canonical(manifest))
    with zipfile.ZipFile(io.BytesIO(out.getvalue())) as check:
        if sum(e.file_size for e in check.infolist())>TRANSPORT_POLICY['raw_bytes']:raise ValueError('raw upload budget')
    if out.tell() > TRANSPORT_POLICY['compressed_bytes']:
        raise ValueError('upload exceeds budget')
    return out.getvalue()

def bounded_tensor(data):
    import numpy as np
    """Validate NPY framing before NumPy can allocate its declared shape."""
    stream=io.BytesIO(data)
    version=np.lib.format.read_magic(stream)
    if version==(1,0):reader=np.lib.format.read_array_header_1_0
    elif version==(2,0):reader=np.lib.format.read_array_header_2_0
    else:raise ValueError('unsupported tensor NPY version')
    shape,fortran_order,dtype=reader(stream,max_header_size=10000)
    if dtype!=np.dtype(np.float32) or len(shape)!=2 or any(type(n) is not int or n<=0 for n in shape) or shape[0]>512 or shape[1]>200000:
        raise ValueError('tensor header shape or dtype')
    expected=shape[0]*shape[1]*dtype.itemsize
    if len(data)-stream.tell()!=expected:
        raise ValueError('tensor payload length mismatch')
    stream.seek(0)
    value=np.load(stream,allow_pickle=False)
    if value.dtype!=dtype or value.shape!=shape:
        raise ValueError('tensor decoded shape mismatch')
    return value


def unpack(data):
    import numpy as np
    if len(data) > TRANSPORT_POLICY['compressed_bytes']:
        raise ValueError('compressed upload budget')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        entries = z.infolist()
        names = [e.filename for e in entries]
        if len(names) != len(set(names)) or len(names) > 4096 or sum(e.file_size for e in entries) > TRANSPORT_POLICY['raw_bytes']:
            raise ValueError('archive budget or duplicate entries')
        if any('/' in n or '..' in n for n in names):
            raise ValueError('archive paths')
        if z.getinfo('manifest.json').file_size > 2_000_000:
            raise ValueError('manifest budget')
        records = json.loads(z.read('manifest.json'))
        if not isinstance(records, list) or len(records) > 32:
            raise ValueError('batch budget')
        result, referenced = [], {'manifest.json'}
        for record in records:
            arrays = []
            for turns in record['arrays']:
                if len(turns) > 32:
                    raise ValueError('turn budget')
                row = []
                for name in turns:
                    if name in referenced or name not in names:
                        raise ValueError('tensor reference')
                    referenced.add(name)
                    value = bounded_tensor(z.read(name))
                    if value.dtype != np.float32 or value.ndim != 2 or value.shape[0] > 512 or value.shape[1] > 200000:
                        raise ValueError('tensor shape')
                    row.append(value)
                arrays.append(row)
            result.append((record['batch'], arrays))
        if referenced != set(names):
            raise ValueError('unexpected entries')
        return result

def mine_cumulative(runtime,manifest,job,upload,clock=None,allow_empty=False):
    """Publish each complete private batch before searching the next task.

    The last acknowledged snapshot is authoritative if subsequent search runs
    out of time. A ten-second reserve avoids initiating overwrites at expiry;
    upload errors remain failures rather than pretending a PUT succeeded.
    """
    from .protocol import entries
    clock=clock or time.time
    mining_window(manifest,clock())
    batches=[];search=[];data=None;uploads=0;stopped=False
    def available():
        now=clock()
        return manifest['start']<=now<manifest['deadline']-10
    for definition in entries(manifest):
        if len(batches)>=manifest.get('max_batches',4) or not available():break
        selected=runtime.for_environment(definition['spec'],definition['harness'])
        for index in definition['indices']:
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
                candidate=batches+[(batch,[a for r,a in found])];candidate_data=pack(candidate)
                if not available():stopped=True;break
                upload(candidate_data,min(180,manifest['deadline']-clock()-1))
                batches=candidate;data=candidate_data;uploads+=1
            if stopped or len(batches)>=manifest.get('max_batches',4):break
        if stopped:break
    if data is None and not allow_empty:raise ValueError('GPU bounded search found no complete batch before epoch window closed')
    return data,dict(batches=len(batches),search=search,cumulative_uploads=uploads,search_stopped_at_deadline=stopped,mining_status='complete_batches_uploaded' if data is not None else 'no_complete_KL_batch')


def validate(envelope, authority, now=None):
    """Pure authorization/policy check; never opens an artifact or imports runtime."""
    job=signed(envelope,authority);now=time.time() if now is None else now
    if job.get('schema')!=1 or job.get('role') not in ROLES:raise ValueError('job role/schema')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',job.get('job_id','')):raise ValueError('job ID')
    if any(type(job.get(k)) not in (int,float) for k in ('created_at','expires_at')) or not job['created_at']<=now<job['expires_at'] or job['expires_at']-job['created_at']>86400:raise ValueError('job expired/time budget')
    manifest=signed(job['manifest'],authority)
    if manifest.get('model_runtime_revision')!=REVISION or not exact_policy(manifest.get('numerical_policy'),NUMERICAL_POLICY) or not exact_policy(manifest.get('backend_profile'),BACKEND_PROFILE):raise ValueError('GPU profile or numerical policy')
    if manifest.get('transport_policy')!='direct-r2-v1' or not exact_policy(manifest.get('artifact_policy'),TRANSPORT_POLICY) or not exact_policy(job.get('artifact_policy'),TRANSPORT_POLICY):raise ValueError('signed long-context transport policy')
    if job.get('chain_transactions') is not False or job.get('payable') is not False:raise ValueError('controlled nonpayable role required')
    cp=manifest['checkpoint'];identifier=file_map(cp['files'])
    if cp.get('id')!=identifier:raise ValueError('checkpoint identity')
    urls=cp.get('read_urls',{})
    if urls and set(urls)!=set(cp['files']):raise ValueError('checkpoint capability file binding')
    for url in urls.values():r2_url(url,'GET')
    if not set(SOURCE_FILES)<=set(job.get('source_files',{})):raise ValueError('missing worker source pins')
    for name,sha in job['source_files'].items():
        if not name.startswith('subnet/') or '..' in name or Path(name).suffix!='.py' or not re.fullmatch('[0-9a-f]{64}',sha):raise ValueError('worker source pin')
    if not re.fullmatch('[0-9a-f]{64}',job.get('interpreter_sha256','')):raise ValueError('interpreter pin')
    if set(job.get('runtime_versions',{}))!={'torch','transformers','toploc','numpy'}:raise ValueError('runtime version pins')
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
    if job['role']=='train' and not exact_policy(job.get('training_parameters'),TRAINING_PARAMETERS):raise ValueError('signed complete sequential optimizer policy')
    if job['role']=='train' and job.get('training_policy')!=TRAINING_POLICY:raise ValueError('unapproved training objective')
    if job['role']=='train' and (type(job.get('steps')) is not int or job['steps']!=1):raise ValueError('only separately-qualified single training step admitted')
    if job['role']=='evaluate':
        if not job.get('heldout') or len(job['heldout'])>64:raise ValueError('heldout budget')
        heldout_seen=set()
        for row in job['heldout']:
            if len(row['indices'])!=len(row['seeds']) or not 1<=len(row['indices'])<=32 or any(type(i) is not int or i<0 for i in row['indices']+row['seeds']):raise ValueError('heldout index/seed budget')
            for index,seed in zip(row['indices'],row['seeds']):
                key=(row['env_id'],index,seed)
                if key in heldout_seen:raise ValueError('duplicate heldout request')
                heldout_seen.add(key)
            if row['harness'].get('policy')!='autoregressive' or row['harness'].get('turn_overrides'):raise ValueError('heldout must use free autoregressive policy')
    if job['role']=='upload':
        if set(job.get('put_urls',{}))!=set(cp['files']):raise ValueError('upload capability file binding')
        for url in job['put_urls'].values():r2_url(url,'PUT')
    expected={'min_free_vram_bytes':20*1024**3 if job['role']=='train' else 12*1024**3,'wait_seconds':1800}
    if job['role']!='upload' and not exact_policy(job.get('resource_policy'),expected):raise ValueError('signed GPU role resources')
    return job,manifest


def audit(data, manifest, runtime):
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


def execute(envelope, authority, workspace, cache=None, session_factory=None):
    # Authentication and every source/package/resource binding precede runtime,
    # factory import, checkpoint access, or submission download.
    job,manifest=validate(envelope,authority)
    root=Path(__file__).resolve().parent.parent
    for name,expected in job['source_files'].items():
        path=root/name
        if path.is_symlink() or digest(path)!=expected:raise ValueError('worker source mismatch')
    if digest(sys.executable)!=job['interpreter_sha256']:raise ValueError('worker interpreter mismatch')
    for name,expected in job['runtime_versions'].items():
        if version(name)!=expected:raise ValueError('runtime package mismatch')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        if os.environ.get(name)!='2':raise ValueError('thread environment profile')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('CUDA environment profile')
    from . import backend_jobs as base
    # Source loader covers all new modules too, rejecting preimported runtimes.
    for name in SOURCE_FILES:
        key=name[:-3].replace('/','.')
        if key in sys.modules and key not in ('subnet.long_context_backend_jobs','subnet.backend_jobs'):
            raise ValueError('long-context worker requires fresh process')
    sys.meta_path.insert(0,base.FreshSourceFinder(root))
    workspace=Path(workspace);out=workspace/'jobs'/job['job_id']
    out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
    approved=base.checkpoint(manifest,workspace,cache)
    report=dict(schema=1,job_id=job['job_id'],role=job['role'],operator=authority,
        job_sha256=hashlib.sha256(canonical(job)).hexdigest(),checkpoint=manifest['checkpoint']['id'],
        epoch=manifest['epoch'],backend_profile=BACKEND_PROFILE,numerical_policy=NUMERICAL_POLICY,
        artifact_policy=TRANSPORT_POLICY,source_files=job['source_files'],runtime_versions=job['runtime_versions'],
        chain_transactions=False,payable=False,full_model_finetune=False,execution_resources_enforced=False,resource_policy=job.get('resource_policy'))
    if job['role']=='upload':
        import requests
        for name,url in job['put_urls'].items():
            with (approved/name).open('rb') as body:
                response=requests.put(url,data=body,headers={'Content-Type':'application/octet-stream'},timeout=600,allow_redirects=False)
            if response.status_code not in (200,201,204):raise ValueError('R2 PUT status '+str(response.status_code))
        report['uploaded_files']=manifest['checkpoint']['files']
    else:
        from .protocol import entries,entry
        from .long_context_runtime import wait_vram
        from .long_context_service_runtime import LongContextServiceRuntime
        if session_factory is None:
            from .long_context_native_factory import create
            session_factory=create
        observed_free_mib=wait_vram(job['resource_policy']['min_free_vram_bytes']//(1024**2),job['resource_policy']['wait_seconds'])
        report['execution_resources_enforced']=True
        report['observed_free_vram_bytes_before_load']=observed_free_mib*1024**2
        first=entries(manifest)[0]
        runtime=LongContextServiceRuntime(approved,manifest['checkpoint']['files'],first['spec'],first['harness'],session_factory)
        if job['role']=='mine':
            import requests
            def upload(data,timeout):
                if len(data)>TRANSPORT_POLICY['compressed_bytes']:raise ValueError('compressed upload budget')
                response=requests.put(job['capability']['put_url'],data=data,headers=job['capability']['headers'],timeout=timeout,allow_redirects=False)
                if response.status_code not in (200,201,204):raise ValueError('R2 PUT status '+str(response.status_code))
            data,mining=mine_cumulative(runtime,manifest,job,upload,allow_empty=True)
            if data is not None:(out/'submission.zip').write_bytes(data)
            report.update(miner_id=job['miner_id'],submission_sha256=hashlib.sha256(data).hexdigest() if data is not None else None,submission_size=len(data) if data is not None else 0,operator_authorized_experiment=True,**mining)
        elif job['role'] in ('verify','train'):
            reports=[];pairs=[]
            for i,obj in enumerate(job['submissions']):
                path=out/('submission-'+str(i)+'.zip')
                base.get_object(obj['url'],obj['sha256'],path,TRANSPORT_POLICY['compressed_bytes'])
                result,verified=audit(path.read_bytes(),manifest,runtime);reports.append(result);pairs.extend(verified)
            report['audits']=reports
            if job['role']=='train':
                if not pairs:raise ValueError('no verified training pairs')
                from .long_context_service_training import POLICY
                if not exact_policy(job.get('training_parameters'),POLICY):raise ValueError('signed complete sequential optimizer policy')
                # Fail-closed authorization permits one optimizer/reference
                # lifetime only; remaining audited pairs are not claimed trained.
                definition,pos,neg=pairs[0]
                runtime.configure(definition['spec'],definition['harness'])
                update=runtime.full_parameter_train([(pos,neg)],steps=1)
                update.update(base.pair_attribution(definition,pos,neg,0));updates=[update]
                destination=out/'checkpoint';destination.mkdir()
                runtime.model.save_pretrained(destination,safe_serialization=True)
                runtime.tokenizer.save_pretrained(destination)
                from .model import model_files
                files=model_files(destination)
                if files.get('model.safetensors')==manifest['checkpoint']['files'].get('model.safetensors'):raise ValueError('training did not change weights')
                report['training']=dict(steps=job['steps'],updates=updates,training_policy=TRAINING_POLICY,full_model_finetune=True,execution_scope='qualified-single-pair-single-step-v1',audited_pairs=len(pairs),optimized_pairs=1)
                report['full_model_finetune']=True
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
                        if not selected.verify(doc,arrays):raise ValueError('heldout inference/native audit')
                        token=hashlib.sha256(canonical(doc)).hexdigest()
                        (out/(token+'.json')).write_bytes(canonical(doc))
                        import numpy as np
                        for turn,array in enumerate(arrays):np.save(out/(token+'-'+str(turn)+'.npy'),array,allow_pickle=False)
                        values.append(dict(env_id=row['env_id'],index=index,seed=seed,reward=doc['reward'],classification=doc['classification'],task_hash=doc['task_hash'],verified=True,rollout_sha256=token))
                    except (ValueError,RuntimeError,KeyError) as error:
                        failures.append(dict(env_id=row['env_id'],index=index,seed=seed,error_type=type(error).__name__,error=str(error)[:300]))
            report['heldout']=values;report['heldout_failures']=failures
    report['success']=True;report['completed_at']=time.time()
    report['artifact_files']={p.name:dict(sha256=digest(p),size=p.stat().st_size) for p in out.iterdir() if p.is_file()}
    (out/'report.json').write_bytes(canonical(report));return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('job');parser.add_argument('--authority',required=True);parser.add_argument('--workspace',required=True);parser.add_argument('--checkpoint-cache')
    args=parser.parse_args();data=Path(args.job).read_bytes()
    if len(data)>4_000_000:raise ValueError('job envelope size budget')
    report=execute(json.loads(data),args.authority,args.workspace,args.checkpoint_cache)
    print(json.dumps(dict(job_id=report['job_id'],role=report['role'],success=True,checkpoint=report.get('new_checkpoint',{}).get('id',report['checkpoint']))))
if __name__=='__main__':main()
