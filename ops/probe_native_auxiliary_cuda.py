#!/usr/bin/env python3
"""Bounded signed private native-auxiliary input controls; no native simulation."""
import argparse,gc,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from subnet.long_context_runtime import authenticate,AUTHORITY,file_sha,digest,canonical,wait_vram,runtime_environment

def validate_job(envelope,authority):
    from subnet.native_auxiliary_cuda import POLICY,CHECKPOINT,validate_descriptor
    job=authenticate(envelope,authority)
    if job.get('role')!='native-auxiliary-cuda-proof-probe' or canonical(job.get('policy'))!=canonical(POLICY) or job.get('payable') is not False or job.get('chain_transactions') is not False:raise ValueError('signed auxiliary role/policy')
    if job.get('min_free_vram_mib')!=12288 or job.get('wait_seconds')!=1800:raise ValueError('auxiliary capacity guard')
    if job.get('runtime_environment')!=runtime_environment():raise ValueError('auxiliary interpreter/packages')
    validate_descriptor(job['descriptor'])
    if job['checkpoint']['id']!=CHECKPOINT or job['checkpoint']['files']!=job['descriptor']['checkpoint']['files']:raise ValueError('fixed auxiliary job checkpoint')
    return job

def main():
    args=argparse.ArgumentParser();args.add_argument('--job',required=True);options=args.parse_args()
    job=validate_job(json.loads(pathlib.Path(options.job).read_text()),AUTHORITY)
    if job.get('experiment')!='native-auxiliary-smol-role-compute-v1' or job.get('probe_source_sha256')!=file_sha(__file__):raise ValueError('approved native-role experiment/source')
    for name,expected in job['source_files'].items():
        path=ROOT/name
        if path.is_symlink() or file_sha(path)!=expected:raise ValueError('native-role approved source closure')
    inputs=[]
    for row in job['inputs']:
        path=pathlib.Path(row['path'])
        if path.is_symlink() or file_sha(path)!=row['sha256'] or path.stat().st_size>2000000:raise ValueError('approved bounded private role input')
        inputs.append(json.loads(path.read_text()))
    if not 1<=len(inputs)<=2:raise ValueError('bounded native role inputs')
    from subnet.native_auxiliary_cuda import NativeAuxiliaryCUDARuntime
    import numpy as np,torch,os
    out=pathlib.Path(job['out']);out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
    report={'kind':'controlled-native-auxiliary-smol-role-proof-v1','job_sha256':digest(job),'completed':False,'records':[],'auxiliary_role_verified':False,'whole_native_trajectory_verified':False,'native_replay_performed':False,'originally_sampled':False,'chain_transactions':False,'payable':False}
    def write(name,value):
        path=out/name
        with path.open('w') as f:json.dump(value,f,indent=2);f.write('\n')
        path.chmod(0o600)
    def save():write('report.json',report)
    runtime=None
    def release():
        nonlocal runtime
        runtime=None;gc.collect();torch.cuda.empty_cache()
    try:
        wait_vram();runtime=NativeAuxiliaryCUDARuntime(job['checkpoint']['path'],job['descriptor']);report['runtime_profile']=runtime.profile();save()
        for index,value in enumerate(inputs):
            prompt=runtime.render(value['request']);output=runtime.tokenizer.encode(value['curated_output'],add_special_tokens=False)
            if runtime.tokenizer.decode(output,skip_special_tokens=True)!=value['curated_output'] or len(output)>job['descriptor']['max_output_tokens']:raise ValueError('native curated action complete token roundtrip')
            acts,lp=runtime.compute(prompt,output);artifact={'prompt':prompt,'output':output,'proofs':runtime.build_proofs(acts,decode_batching_size=16,topk=128),'profile':runtime.profile()}
            write(f'role-{index}.json',artifact);np.savez_compressed(out/f'role-{index}.npz',logprobs=lp);(out/f'role-{index}.npz').chmod(0o600)
            report['records'].append({'request_sha256':digest(value['request']),'curated_output_sha256':digest(value['curated_output']),'prompt_tokens':len(prompt),'output_tokens':len(output),'full_logprob_shape':list(lp.shape),'full_proof_verified':False});save()
        release();wait_vram();runtime=NativeAuxiliaryCUDARuntime(job['checkpoint']['path'],job['descriptor'])
        for index,value in enumerate(inputs):
            artifact=json.loads((out/f'role-{index}.json').read_text());lp=np.load(out/f'role-{index}.npz',allow_pickle=False)['logprobs']
            if artifact['prompt']!=runtime.render(value['request']) or artifact['output']!=runtime.tokenizer.encode(value['curated_output'],add_special_tokens=False):raise ValueError('fresh complete native context/action binding')
            report['records'][index]['full_proof_verified']=runtime.verify(artifact,lp)
        # Well-framed target-token and probability mutations must not pass.
        artifact=json.loads((out/'role-0.json').read_text());lp=np.load(out/'role-0.npz',allow_pickle=False)['logprobs'];bad=json.loads(json.dumps(artifact));bad['output'][0]=(bad['output'][0]+1)%49152
        tests={}
        for label,doc,values in [('changed_output',bad,lp),('changed_full_probabilities',artifact,lp+np.float32(.001))]:
            try:runtime.verify(doc,values)
            except ValueError:tests[label]=True
            else:raise ValueError('forged native role artifact accepted')
        report.update(completed=True,full_model_recompute=True,auxiliary_role_verified=True,tamper_rejections=tests)
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error)[:400]);raise
    finally:
        report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated();release();report['completed_at']=time.time();report['artifact_files']={p.name:{'sha256':file_sha(p),'size':p.stat().st_size} for p in out.iterdir() if p.is_file() and p.name!='report.json'};save();print(json.dumps({'completed':report['completed'],'records':report['records'],'peak_gpu_allocated_bytes':report['peak_gpu_allocated_bytes']}),flush=True)
if __name__=='__main__':main()
