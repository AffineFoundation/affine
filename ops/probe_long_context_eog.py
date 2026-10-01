#!/usr/bin/env python3
"""Model computation on complete public native EOG traces, no private grader."""
import argparse,copy,gc,hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from subnet.long_context_runtime import LongContextRuntime,validate_job,AUTHORITY,wait_vram,file_sha,digest,canonical

def trace_rows(public,harness):
    messages=copy.deepcopy(public['public']['messages']);tools=public['public']['tools'];rows=[]
    for i,event in enumerate(public['events']):
        action={'name':event['name'],'arguments':event['arguments']};text=canonical({'tool_call':action}).decode()
        rows.append({'turn_index':i,'messages':copy.deepcopy(messages),'text':text,
                     'messages_sha256':digest(messages),'action_sha256':digest(action),
                     'observation_sha256':hashlib.sha256(event['observation'].encode()).hexdigest()})
        messages=messages+[{'role':'assistant','content':text}]+harness.observations([{'role':'tool','content':event['observation']}],{'version':'text-tools-v1'})
    return tools,rows
def verify_expected(runtime,artifact,values,prompt,output):
    if artifact['prompt']!=prompt or artifact['output']!=output:raise ValueError('approved native public messages/action binding')
    return runtime.verify(artifact,values)

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();job=validate_job(json.loads(pathlib.Path(a.job).read_text()),AUTHORITY)
    if job.get('experiment')!='native-public-eog-v1' or job.get('probe_source_sha256')!=file_sha(__file__) or job.get('harness_source_sha256')!=file_sha(ROOT/'subnet/harness.py'):raise ValueError('EOG approved source/role')
    if job.get('max_turns')!=6 or job.get('max_output_tokens')!=512:raise ValueError('EOG signed trajectory budget')
    raw=pathlib.Path(job['public_trace_file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=job['public_trace_sha256']:raise ValueError('EOG approved public trace')
    public=json.loads(raw)
    if set(public)!={'schema','scope','public','events','claimed_reward','runtime','original_seed_sha256','source_files'}:raise ValueError('public-only trace schema')
    out=pathlib.Path(job['out']);out.mkdir(parents=True,exist_ok=False)
    import torch,numpy as np
    from subnet import harness
    tools,rows=trace_rows(public,harness)
    if len(rows)!=job['max_turns']:raise ValueError('EOG complete six-tool trace')
    report={'completed':False,'job_hash':digest(job),'public_trace_sha256':job['public_trace_sha256'],'records':[],
            'curated_target_model_computation':True,'originally_sampled':False,'native_grader_verified':False,
            'payable':False,'chain_transactions':False,'quality_improvement_claimed':False,'stage':'waiting-vram','started_at':time.time()};runtime=None
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def release():
        nonlocal runtime
        runtime=None;gc.collect();torch.cuda.empty_cache()
    try:
        save();wait_vram();torch.cuda.reset_peak_memory_stats();runtime=LongContextRuntime(job['checkpoint']['path'],job['checkpoint']['files'])
        report['runtime_profile']=runtime.profile();report['checkpoint']=digest(job['checkpoint']['files']);report['checkpoint_files']=job['checkpoint']['files'];report['harness_source_sha256']=job['harness_source_sha256']
        report['stage']='full-public-trajectory-computation';save()
        for row in rows:
            i=row['turn_index'];prompt=harness.render(runtime.tokenizer,row['messages'],tools,{'version':'text-tools-v1'});output=runtime.tokenizer.encode(row['text'],add_special_tokens=False)
            if len(output)>job['max_output_tokens']:raise ValueError('EOG emitted token budget')
            if runtime.tokenizer.decode(output,skip_special_tokens=True)!=row['text']:raise ValueError('native action token roundtrip')
            acts,lp=runtime.compute(prompt,output);artifact={'prompt':prompt,'output':output,'proofs':runtime.build_proofs(acts,decode_batching_size=16,topk=128),'profile':runtime.profile()}
            (out/f'turn-{i}.json').write_text(json.dumps(artifact)+'\n');np.savez_compressed(out/f'turn-{i}.npz',logprobs=lp)
            report['records'].append({k:row[k] for k in ('turn_index','messages_sha256','action_sha256','observation_sha256')}|{'prompt_tokens':len(prompt),'output_tokens':len(output),'logprobs_raw_bytes':lp.nbytes,'compressed_array_bytes':(out/f'turn-{i}.npz').stat().st_size,'probability_shape':list(lp.shape),'full_proof_verified':False});save()
        release();wait_vram();runtime=LongContextRuntime(job['checkpoint']['path'],job['checkpoint']['files']);report['stage']='fresh-independent-all-turn-verification';save()
        for row in rows:
            i=row['turn_index'];artifact=json.loads((out/f'turn-{i}.json').read_text());values=np.load(out/f'turn-{i}.npz',allow_pickle=False)['logprobs'];prompt=harness.render(runtime.tokenizer,row['messages'],tools,{'version':'text-tools-v1'});output=runtime.tokenizer.encode(row['text'],add_special_tokens=False)
            report['records'][i]['full_proof_verified']=verify_expected(runtime,artifact,values,prompt,output);save()
        # Complete recomputation on a forged observed tool response still cannot
        # replace the exact native public-history commitment.
        row=rows[-1];messages=copy.deepcopy(row['messages']);messages[-1]['content']+=' '
        prompt=harness.render(runtime.tokenizer,messages,tools,{'version':'text-tools-v1'});output=runtime.tokenizer.encode(row['text'],add_special_tokens=False);acts,lp=runtime.compute(prompt,output)
        forged={'prompt':prompt,'output':output,'proofs':runtime.build_proofs(acts,decode_batching_size=16,topk=128),'profile':runtime.profile()};report['forged_tool_response_numerically_self_consistent']=runtime.verify(forged,lp)
        (out/'forged-tool-response.json').write_text(json.dumps(forged)+'\n');np.savez_compressed(out/'forged-tool-response.npz',logprobs=lp)
        try:verify_expected(runtime,forged,lp,harness.render(runtime.tokenizer,row['messages'],tools,{'version':'text-tools-v1'}),output)
        except ValueError as e:report['forged_tool_response_rejected']={'rejected':True,'reason':str(e)}
        else:raise ValueError('forged native observation accepted')
        report['artifact_files']={p.name:{'sha256':file_sha(p),'size':p.stat().st_size} for p in out.iterdir() if p.is_file() and p.name!='report.json'}
        report['total_logprobs_raw_bytes']=sum(r['logprobs_raw_bytes'] for r in report['records']);report['total_compressed_arrays_bytes']=sum(r['compressed_array_bytes'] for r in report['records']);report['total_honest_artifact_bytes']=sum(p.stat().st_size for p in out.glob('turn-*'))
        report.update(completed=True,stage='complete',full_model_recompute=True)
    except Exception as e:report.update(stage='failed',error_type=type(e).__name__,error=str(e)[:1000]);raise
    finally:report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated();release();report['completed_at']=time.time();save();print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
