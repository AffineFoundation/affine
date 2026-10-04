"""Root-reviewed prospective GPU diagnostics; never a live verifier/job worker.

Runs only an authenticated, short-lived private request. Scientific source and
checkpoint are independently byte-pinned. The new sampler is a separate pinned
sidecar. No operator seed, master bucket key, coordinator, chain, live reports,
GPU rental, service stop or new registration is used. Outputs are unsigned
private diagnostic artifacts for operator readback/review, not admissions.

The diagnostic runtime is a distinct subclass: full hidden-state forward plus
full-row logsoftmax permits exact causal-row extraction. It intentionally does
not claim the original E9 prefix sampler ran. Original probabilities/TOPLOC/
native environment checks still run via the pinned Runtime.verify, with the
legacy sampling context absent; the prospective robust-CDF gate runs separately.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

DOMAIN='isolated-full-forward-cdf-gpu-qualification-v1'
COMPUTE_REVISION='full-hidden-forward-full-logsoftmax-sidecar-v1'
MAX_METADATA=8_000_000
MAX_ARTIFACT=100_000_000
MAX_OUTPUT=128


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(value):return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    path=Path(path)
    if not path.is_file()or path.is_symlink():raise ValueError('regular non-symlink diagnostic bytes')
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path,limit=MAX_METADATA):
    path=Path(path)
    if not path.is_file()or path.is_symlink()or not 0<path.stat().st_size<=limit:
        raise ValueError('bounded regular diagnostic metadata')
    return json.loads(path.read_bytes())


def authenticate(envelope,authority):
    from nacl.signing import VerifyKey
    if not isinstance(envelope,dict)or set(envelope)!={'payload','signer','signature'}or envelope['signer']!=authority:
        raise ValueError('exact diagnostic authority envelope')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(envelope['payload']),base64.b64decode(envelope['signature'],validate=True))
    return envelope['payload']


def write(path,data):
    path=Path(path)
    with path.open('xb')as stream:stream.write(data)
    path.chmod(0o600)


def validate_request(request,now):
    fields={'version','run_id','mode','created_at','expires_at','helper_sha256','candidate_path','candidate_sha256',
        'source_path','source_files','source_archive_path','source_archive_sha256',
        'checkpoint_path','checkpoint','runtime_versions','gpu','workspace','manifest',
        'task_index','reference_artifacts','reference_completion_path','reference_completion_sha256','task_asset_root','task_asset_files',
        'diagnostic_only','normal_queue_allowed','chain_transactions_allowed','live_admission_allowed'}
    if not isinstance(request,dict)or set(request)!=fields:raise ValueError('exact isolated diagnostic request')
    if (request['version']!=DOMAIN or request['mode']not in('generate-reference','verify-reference')
            or not isinstance(request['run_id'],str)or re.fullmatch('[A-Za-z0-9_-]{1,100}',request['run_id'])is None
            or request['diagnostic_only']is not True or any(request[k]is not False for k in
                ('normal_queue_allowed','chain_transactions_allowed','live_admission_allowed'))):
        raise ValueError('diagnostic scope cannot authorize live actions')
    created,expires=request['created_at'],request['expires_at']
    if (any(type(v)not in(int,float)or not math.isfinite(v)for v in(created,expires,now))
            or not created<=now<expires or not 0<expires-created<=3500):
        raise ValueError('diagnostic original request lifetime')
    if request['helper_sha256']!=file_sha(__file__):raise ValueError('diagnostic helper source pin')
    versions=request['runtime_versions']
    if (not isinstance(versions,dict)or set(versions)!={'torch','transformers','toploc','datasets','verifiers'}
            or any(not isinstance(v,str)or not v for v in versions.values())):
        raise ValueError('complete approved runtime/environment package pins')
    gpu=request['gpu']
    if (not isinstance(gpu,dict)or set(gpu)!={'index','name','sm','driver','idle_fence_sha256'}
            or type(gpu['index'])is not int or not 0<=gpu['index']<8
            or not isinstance(gpu['name'],str)or not gpu['name']or not isinstance(gpu['sm'],list)
            or len(gpu['sm'])!=2 or any(type(v)is not int for v in gpu['sm'])
            or not isinstance(gpu['driver'],str)or not gpu['driver']):
        raise ValueError('explicit actual GPU/index/profile binding')
    for digest in(request['helper_sha256'],request['candidate_sha256'],request['source_archive_sha256'],gpu['idle_fence_sha256']):
        if not isinstance(digest,str)or re.fullmatch('[0-9a-f]{64}',digest)is None:raise ValueError('diagnostic SHA256 pin')
    workspace=Path(request['workspace'])
    if (not workspace.is_absolute()or workspace.resolve()!=workspace or 'diagnostic'not in workspace.name
            or workspace==Path(request['source_path'])or workspace==Path(request['checkpoint_path'])):
        raise ValueError('distinct absolute private diagnostic workspace')
    manifest=request['manifest'];env=manifest['environment'];harness=manifest['harness']
    if (manifest.get('payable')is not False or not str(manifest.get('epoch','')).startswith('nonpayable-lab-')
            or manifest.get('checkpoint')!=request['checkpoint']or env.get('max_turns')!=1
            or harness.get('policy')!='autoregressive'or harness.get('top_p')!=1
            or type(harness.get('max_output_tokens'))is not int or not 1<=harness['max_output_tokens']<=MAX_OUTPUT
            or type(request['task_index'])is not int or request['task_index']not in manifest['indices']
            or type(manifest['sampling_contract'].get('max_attempts'))is not int
            or not 2<=manifest['sampling_contract']['max_attempts']<=8):
        raise ValueError('bounded separately authorized one-turn lab challenge')
    artifacts=request['reference_artifacts']
    if request['mode']=='generate-reference':
        if (artifacts or request['reference_completion_sha256']is not None or request['reference_completion_path']is not None
                or 'H200'not in gpu['name']or gpu['sm']!=[9,0]):
            raise ValueError('original reference requires separately fenced H200')
    elif (not isinstance(artifacts,dict)or set(artifacts)!={'honest','synthetic'}
            or not isinstance(request['reference_completion_sha256'],str)
            or re.fullmatch('[0-9a-f]{64}',request['reference_completion_sha256'])is None
            or not isinstance(request['reference_completion_path'],str)):
        raise ValueError('operator-reviewed exact reference artifacts required')
    return request


def verify_inventory(path,files):
    root=Path(path)
    if root.resolve()!=root or not root.is_dir()or not isinstance(files,dict)or not files or len(files)>20000:
        raise ValueError('explicit source/task inventory')
    actual={}
    for item in root.rglob('*'):
        if item.is_symlink():raise ValueError('diagnostic inventory symlink')
        if item.is_file():actual[str(item.relative_to(root))]=file_sha(item)
    if actual!=files:raise ValueError('complete diagnostic inventory readback')


def preflight(request):
    if file_sha(request['candidate_path'])!=request['candidate_sha256']:raise ValueError('candidate sidecar source pin')
    if file_sha(request['source_archive_path'])!=request['source_archive_sha256']:raise ValueError('original scientific source archive')
    verify_inventory(request['source_path'],request['source_files'])
    if request['task_asset_files']:verify_inventory(request['task_asset_root'],request['task_asset_files'])
    cp=Path(request['checkpoint_path']);files=request['checkpoint']['files']
    if (not isinstance(files,dict)or not files or any(Path(name).name!=name for name in files)
            or cp.resolve()!=cp or not cp.is_dir()or {p.name for p in cp.iterdir()}!=set(files)
            or any(not(cp/name).is_file()or(cp/name).is_symlink()or file_sha(cp/name)!=digest for name,digest in files.items())
            or sha(files)!=request['checkpoint']['id']):raise ValueError('independent exact checkpoint readback')
    # Root separately fences legacy CPU demand. The helper never stops services.
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU occupied; no new diagnostic execution')
    observed=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).splitlines()
    if observed[request['gpu']['index']].strip()!=request['gpu']['name']+', '+request['gpu']['driver']:
        raise ValueError('actual device/driver identity')
    from importlib.metadata import version
    if {n:version(n)for n in request['runtime_versions']}!=request['runtime_versions']:
        raise ValueError('actual installed runtime package pins')


def authenticate_reference(request,authority):
    """Root signs independently read-back ORIGINAL completion, not worker self-admission."""
    if request['mode']=='generate-reference':return None
    envelope=read(request['reference_completion_path'])
    if sha(envelope)!=request['reference_completion_sha256']:
        raise ValueError('original authority-reviewed reference completion bytes')
    reference=authenticate(envelope,authority)
    expected=dict(version=DOMAIN,mode='generate-reference',status='reference_ready',
        diagnostic_only=True,live_admission=False,normal_queue_used=False,chain_transactions=False,
        historical_execution_proven=False,source_archive_sha256=request['source_archive_sha256'],
        complete_source_files_sha256=sha(request['source_files']),helper_sha256=request['helper_sha256'],
        candidate_sha256=request['candidate_sha256'],checkpoint=request['checkpoint']['id'],
        lab_manifest_sha256=sha(request['manifest']),runtime_versions=request['runtime_versions'])
    if any(reference.get(key)!=value for key,value in expected.items()):
        raise ValueError('exact original reference scientific provenance')
    hardware=reference.get('hardware',{})
    if ('H200'not in hardware.get('gpu_name','')or hardware.get('sm')!=[9,0]
            or hardware.get('compute_revision')!=COMPUTE_REVISION):
        raise ValueError('actual H200 reference compute provenance')
    for name,status in [('honest','accepted'),('synthetic','sampler_mismatch')]:
        control=reference.get('controls',{}).get(name,{})
        if (control.get('status')!=status or control.get('probability_TOPLOC_environment_passed')is not True
                or control.get('old_prefix_sampler_checked')is not False or control.get('live_admission')is not False):
            raise ValueError('reference original honest/synthetic outcomes')
        original=control.get('artifact',{});item=request['reference_artifacts'][name]
        if set(item)!={'path','name','sha256','size','batch_sha256','rollout_sha256'}:
            raise ValueError('exact operator-reviewed reference artifact descriptor')
        if any(item.get(key)!=original.get(key)for key in('name','sha256','size','batch_sha256','rollout_sha256')):
            raise ValueError('reference artifact independently readback provenance')
    return reference


def load_runtime(request,candidate):
    """Explicit diagnostic subclass; never mutates original runtime/profile classes."""
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from subnet.gpu_runtime import GPURuntime
    from toploc import build_proofs_base64
    from subnet.proofs import verify_mapped_proofs
    from toploc.C.csrc.utils import get_fp_parts
    import toploc.poly as poly
    if (torch.cuda.device_count()!=1 or torch.cuda.get_device_name()!=request['gpu']['name']
            or list(torch.cuda.get_device_capability())!=request['gpu']['sm']or not torch.cuda.is_bf16_supported()):
        raise ValueError('actual diagnostic GPU family/context readiness')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    context=candidate.context_for_manifest(request['manifest'])
    class DiagnosticRuntime(GPURuntime):
        def __init__(self):
            self.tokenizer=AutoTokenizer.from_pretrained(request['checkpoint_path'],local_files_only=True,trust_remote_code=False)
            self.model=AutoModelForCausalLM.from_pretrained(request['checkpoint_path'],local_files_only=True,
                trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
            self.configure(request['manifest']['environment'],request['manifest']['harness'])
            self.sampling_context=None;self.scripted_output=None;self.last_forward=None
            self.build_proofs=build_proofs_base64;self.verify_proofs=verify_mapped_proofs;self.toploc_threads=2
        def compute(self,prompt,output):
            with torch.inference_mode():
                result=self.model(torch.tensor([prompt+output],device='cuda'),output_hidden_states=True,use_cache=False)
                hidden=result.hidden_states[-1][0].to(torch.bfloat16).cpu().contiguous()
                full=torch.log_softmax(result.logits[0].float(),-1).cpu().numpy()
            self.last_forward=dict(prompt=list(prompt),output=list(output),inputs=prompt+output,full=full)
            acts=[hidden[:len(prompt)]]+[hidden[i:i+1]for i in range(len(prompt),len(prompt)+len(output))]
            return acts,full[len(prompt)-1:len(prompt)+len(output)-1]
        def sampling_receipt(self,seed):
            return dict(sampling=dict(version=candidate.VERSION,binding_sha256=sha(context),attempt=seed))
        def sample_output(self,prompt,seed,messages,turn,index,task_hash):
            if self.scripted_output is not None:return list(self.scripted_output)
            output=[]
            for position in range(self.harness['max_output_tokens']):
                with torch.inference_mode():
                    result=self.model(torch.tensor([prompt+output],device='cuda'),use_cache=False)
                    row=torch.log_softmax(result.logits[0,-1].float(),-1).cpu().numpy()
                u=candidate.uniform(context,self.spec.id,task_hash,index,seed,turn,position)
                output.append(candidate.select_token(row,u,self.harness['temperature']))
                if output[-1]==self.tokenizer.eos_token_id:break
            return output
    poly.get_fp_parts=lambda tensor:get_fp_parts(tensor,num_threads=2)
    runtime=DiagnosticRuntime()
    return runtime,context,dict(gpu_name=torch.cuda.get_device_name(),sm=list(torch.cuda.get_device_capability()),
        torch_cuda=torch.version.cuda,tf32=False,deterministic_algorithms=True,
        bf16_reduced_precision_reduction=str(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        compute_revision=COMPUTE_REVISION)


def check(runtime,candidate,context,doc,arrays):
    """Use original independent prob/TOPLOC/native check, then the NEW CDF gate."""
    from subnet.audit_policy import InvalidSample
    if doc.get('sampling')!=runtime.sampling_receipt(doc.get('seed')).get('sampling'):
        raise ValueError('exact prospective diagnostic sampling receipt')
    try:valid=runtime.verify(doc,arrays)
    except InvalidSample as error:
        reason=str(error)
        if reason in('probabilities','TOPLOC'):status='cross_hardware_numerical_mismatch'
        else:status='source_or_environment_mismatch'
        return dict(status=status,probability_TOPLOC_environment_passed=False,reason=reason[:120],
            old_prefix_sampler_checked=False,live_admission=False,probability_comparison=probability_statistics(runtime,arrays))
    if valid is not True:raise ValueError('independent native diagnostic check')
    captured=runtime.last_forward;turn=doc['turns'][0]
    if captured['prompt']!=turn['prompt']or captured['output']!=turn['output']:raise ValueError('original forward context readback')
    if len(doc['turns'])!=1:raise ValueError('one-turn diagnostic trajectory')
    result=candidate.verify_turn(context,runtime.spec.id,doc['task_hash'],doc['index'],doc['seed'],0,
        prompt_tokens=captured['prompt'],output_tokens=captured['output'],forward_input_ids=captured['inputs'],
        full_forward_logprobs=captured['full'],temperature=runtime.harness['temperature'],
        max_output_tokens=runtime.harness['max_output_tokens'],eos_token_id=runtime.tokenizer.eos_token_id)
    return dict(result,probability_TOPLOC_environment_passed=True,old_prefix_sampler_checked=False,live_admission=False,
        probability_comparison=probability_statistics(runtime,arrays))


def probability_statistics(runtime,arrays):
    """Bounded numeric metadata only; no tokens, distributions, logits or model outputs."""
    import numpy as np
    captured=getattr(runtime,'last_forward',None)
    if not isinstance(captured,dict)or not arrays:return None
    claimed=arrays[0];start=len(captured['prompt'])-1
    recomputed=captured['full'][start:start+len(captured['output'])]
    if not isinstance(claimed,np.ndarray):return dict(shape_match=False)
    stats=dict(claimed_shape=list(claimed.shape),recomputed_shape=list(recomputed.shape),claimed_dtype=str(claimed.dtype),
        recomputed_dtype=str(recomputed.dtype),claimed_nonfinite=int(np.count_nonzero(~np.isfinite(claimed))),
        recomputed_nonfinite=int(np.count_nonzero(~np.isfinite(recomputed))),shape_match=claimed.shape==recomputed.shape,
        unchanged_logprob_atol=1e-5,unchanged_logprob_rtol=0)
    if stats['shape_match']:
        finite=np.isfinite(claimed)&np.isfinite(recomputed)
        differences=np.abs(claimed[finite].astype(np.float64)-recomputed[finite].astype(np.float64))
        stats.update(max_absolute_difference=float(differences.max())if differences.size else None,
            outside_existing_tolerance=int(np.count_nonzero(differences>1e-5)),finite_compared=int(differences.size))
    return stats


def artifact(workspace,name,request,doc,arrays):
    from subnet.batches import pack
    batch=dict(schema=2,diagnostic_only=True,epoch=request['manifest']['epoch'],checkpoint=request['checkpoint']['id'],
        env_id=doc['env_id'],environment_version=doc['environment_version'],index=doc['index'],sample_index=doc['index'],rollouts=[doc])
    raw=pack([(batch,[arrays])])
    if not 0<len(raw)<=MAX_ARTIFACT:raise ValueError('bounded reference artifact export')
    path=workspace/(name+'.zip');write(path,raw)
    return dict(name=path.name,sha256=hashlib.sha256(raw).hexdigest(),size=len(raw),batch_sha256=sha(batch),rollout_sha256=sha(doc))


def read_artifact(spec,request):
    from subnet.batches import submission_records
    from subnet.artifact_budget import LEGACY
    path=Path(spec['path'])
    if (not path.is_file()or path.is_symlink()or type(spec['size'])is not int or not 0<spec['size']<=MAX_ARTIFACT
            or path.stat().st_size!=spec['size']or file_sha(path)!=spec['sha256']):
        raise ValueError('exact operator-reviewed reference artifact bytes')
    records=submission_records(path.read_bytes(),budget=LEGACY,max_batches=1)
    if len(records)!=1:raise ValueError('one bounded diagnostic rollout')
    batch,arrays=records[0]
    if (batch.get('diagnostic_only')is not True or batch.get('epoch')!=request['manifest']['epoch']
            or batch.get('checkpoint')!=request['checkpoint']['id']or batch.get('index')!=request['task_index']
            or len(batch.get('rollouts',[]))!=1 or len(arrays)!=1
            or sha(batch)!=spec['batch_sha256']or sha(batch['rollouts'][0])!=spec['rollout_sha256']):
        raise ValueError('exact lab reference context/decoded artifact hashes')
    return batch['rollouts'][0],arrays[0]


def synthetic_output(output,vocabulary,eos,max_output):
    """Change the first choice but preserve independently checkable EOS/budget stopping."""
    copied=list(output)
    copied[0]=next(token for token in range(vocabulary)if token!=copied[0]and token!=eos)
    if len(copied)<max_output and copied[-1]!=eos:copied.append(eos)
    return copied


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--authority',required=True)
    args=parser.parse_args();envelope=read(args.request);request=validate_request(authenticate(envelope,args.authority),time.time())
    sys.dont_write_bytecode=True;os.umask(0o077)
    workspace=Path(request['workspace'])
    if workspace.exists():raise ValueError('fresh private diagnostic workspace required; no uncertain rerun')
    authenticate_reference(request,args.authority)
    preflight(request)
    os.environ['CUDA_VISIBLE_DEVICES']=str(request['gpu']['index']);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['AFFINE_MATH_CORPUS_ASSET_ROOT']=request['task_asset_root']
    sys.path.insert(0,request['source_path'])
    spec=importlib.util.spec_from_file_location('operator_pinned_cdf_sidecar',request['candidate_path'])
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    # Durable exclusive start marker blocks uncertain retry even if CUDA/model load fails.
    workspace.mkdir(parents=True,exist_ok=False);workspace.chmod(0o700);started=time.time()
    write(workspace/'original-start.json',canonical(dict(request_sha256=sha(envelope),started_at=started,
        run_id=request['run_id'],pid=os.getpid(),diagnostic_only=True)))
    context=candidate.context_for_manifest(request['manifest'])
    if request['mode']=='verify-reference':
        reference=authenticate_reference(request,args.authority)
        if reference.get('public_context_sha256')!=sha(context):raise ValueError('original public draw context')
    runtime,context,hardware=load_runtime(request,candidate)
    report=dict(version=DOMAIN,request_sha256=sha(envelope),run_id=request['run_id'],mode=request['mode'],
        source_archive_sha256=request['source_archive_sha256'],complete_source_files_sha256=sha(request['source_files']),
        helper_sha256=request['helper_sha256'],candidate_sha256=request['candidate_sha256'],checkpoint=request['checkpoint']['id'],
        lab_manifest_sha256=sha(request['manifest']),public_context_sha256=sha(context),runtime_versions=request['runtime_versions'],
        hardware=hardware,diagnostic_only=True,live_admission=False,normal_queue_used=False,chain_transactions=False,
        historical_execution_proven=False,reference_completion_sha256=request['reference_completion_sha256'],controls={})
    if request['mode']=='generate-reference':
        attempts=[];honest=None
        for attempt in range(context['contract']['max_attempts']):
            if time.time()>=request['expires_at']:raise ValueError('reference generation original request expired')
            doc,arrays=runtime.rollout(request['task_index'],attempt);checked=check(runtime,candidate,context,doc,arrays)
            attempts.append(dict(attempt=attempt,status=checked['status'],tokens=len(doc['turns'][0]['output'])))
            if checked['status']=='accepted':honest=(doc,arrays,checked);break
        report['attempts']=attempts
        if honest is None:
            report.update(status='no_robust_reference_within_attempt_budget',completed_at=time.time())
            write(workspace/'diagnostic-result.json',canonical(report));return
        doc,arrays,checked=honest;report['controls']['honest']=dict(checked,artifact=artifact(workspace,'honest',request,doc,arrays))
        copied=synthetic_output(doc['turns'][0]['output'],runtime.model.config.vocab_size,
            runtime.tokenizer.eos_token_id,runtime.harness['max_output_tokens'])
        runtime.scripted_output=copied
        fake,fake_arrays=runtime.rollout(request['task_index'],doc['seed'])
        synthetic=check(runtime,candidate,context,fake,fake_arrays)
        report['controls']['synthetic']=dict(synthetic,artifact=artifact(workspace,'synthetic',request,fake,fake_arrays),
            genuine_probabilities_and_TOPLOC_recomputed=True,construction='copied-trace-first-token-substitution')
        report['status']='reference_ready'if synthetic['status']=='sampler_mismatch'else'synthetic_control_failed'
    else:
        for name,item in request['reference_artifacts'].items():
            doc,arrays=read_artifact(item,request)
            result=check(runtime,candidate,context,doc,arrays)
            report['controls'][name]=dict(result,reference_artifact_sha256=item['sha256'])
        report['status']='honest_pass_synthetic_rejected'if(report['controls']['honest']['status']=='accepted'
            and report['controls']['synthetic']['status']=='sampler_mismatch'
            and report['controls']['synthetic']['probability_TOPLOC_environment_passed'])else'not_qualified'
    report.update(elapsed_seconds=time.time()-started,completed_at=time.time())
    write(workspace/'diagnostic-result.json',canonical(report))
    print(json.dumps(dict(run_id=request['run_id'],status=report['status'],diagnostic_only=True,live_admission=False)))


if __name__=='__main__':main()
