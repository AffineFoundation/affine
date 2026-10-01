#!/usr/bin/env python3
"""Bounded isolated retained-GPU long-context proof conformance probe."""
import argparse,copy,gc,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from subnet.long_context_runtime import LongContextRuntime,validate_job,AUTHORITY,wait_vram,file_sha,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    job=validate_job(json.loads(pathlib.Path(a.job).read_text()),AUTHORITY)
    if job.get('probe_source_sha256')!=file_sha(__file__):raise ValueError('long-context probe source closure')
    out=pathlib.Path(job['out']);out.mkdir(parents=True,exist_ok=False)
    import torch,numpy as np
    result={'completed':False,'job_hash':digest(job),'checkpoint':job['checkpoint'],'policy':job['policy'],'payable':False,'chain_transactions':False,'quality_improvement_claimed':False,'stage':'waiting-vram','started_at':time.time()}
    def save():
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stage':result['stage']}),flush=True)
    runtime=None
    def release():
        nonlocal runtime
        runtime=None;gc.collect();torch.cuda.empty_cache()
    try:
        save();result['free_vram_before_load_mib']=wait_vram(job['min_free_vram_mib'],job['wait_seconds']);torch.cuda.reset_peak_memory_stats()
        runtime=LongContextRuntime(job['checkpoint']['path'],job['checkpoint']['files'])
        if 'prompt_file' in job:
            raw=pathlib.Path(job['prompt_file']).read_bytes()
            if __import__('hashlib').sha256(raw).hexdigest()!=job['prompt_file_sha256']:raise ValueError('long-context public input hash')
            request=json.loads(raw);prompt=runtime.tokenizer.apply_chat_template(request['messages'],tools=request.get('tools'),tokenize=True,add_generation_prompt=True)
        else:
            unit=runtime.tokenizer.encode('This is a public long-context computation control. ',add_special_tokens=False)
            prompt=(unit*2000)[:17485]
        if not 17000<=len(prompt)<=32000:raise ValueError('actual long-context input requirement')
        result['prompt_tokens']=len(prompt);result['stage']='genuine-autoregressive-generation';save()
        output=runtime.greedy(prompt,4);acts,lp=runtime.compute(prompt,output)
        proofs=runtime.build_proofs(acts,decode_batching_size=16,topk=128)
        artifact={'prompt':prompt,'output':output,'proofs':proofs,'profile':runtime.profile(),'generation':'genuine-target-model-greedy','checkpoint':job['checkpoint']}
        np.save(out/'logprobs.npy',lp,allow_pickle=False);(out/'artifact.json').write_text(json.dumps(artifact)+'\n')
        result['profile']=artifact['profile'];result['output_tokens']=len(output);result['full_vocabulary_logprob_shape']=list(lp.shape);result['proof_count']=len(proofs);result['stage']='fresh-independent-model-verification';save();release();wait_vram(job['min_free_vram_mib'],job['wait_seconds'])
        runtime=LongContextRuntime(job['checkpoint']['path'],job['checkpoint']['files']);result['honest_fresh_verified']=runtime.verify(artifact,np.load(out/'logprobs.npy',allow_pickle=False))
        controls=[]
        for kind in ('output-token','last-prefix-token','logprob','proof','profile'):
            forged=copy.deepcopy(artifact);values=lp.copy()
            if kind=='output-token':forged['output'][-1]=(forged['output'][-1]+1)%runtime.model.config.vocab_size
            if kind=='last-prefix-token':forged['prompt'][-1]=(forged['prompt'][-1]+1)%runtime.model.config.vocab_size
            if kind=='logprob':values[0,0]+=.1
            if kind=='proof':forged['proofs'][0]='not-base64'
            if kind=='profile':forged['profile']['policy']['max_context']=65536
            try:runtime.verify(forged,values)
            except (ValueError,__import__('binascii').Error) as e:controls.append({'kind':kind,'rejected':True,'reason':str(e)})
            else:raise ValueError('long-context tampering accepted: '+kind)
        result['tamper_controls']=controls;result['completed']=True;result['stage']='complete'
    except Exception as e:
        result.update(stage='failed',error_type=type(e).__name__,error=str(e)[:1000]);raise
    finally:
        result['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated();release();result['gpu_allocated_after_cleanup']=torch.cuda.memory_allocated();result['completed_at']=time.time();save()
if __name__=='__main__':main()
