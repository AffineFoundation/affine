#!/usr/bin/env python3
"""New approved control: valid false proofs, wrong weights and input binding."""
import argparse,base64,copy,gc,hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from subnet.long_context_runtime import LongContextRuntime,validate_job,AUTHORITY,wait_vram,file_sha,digest

def verify_bound(runtime,artifact,lp,prompt_hash):
    if digest(artifact['prompt'])!=prompt_hash:raise ValueError('signed expected-input binding')
    return runtime.verify(artifact,lp)

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();job=validate_job(json.loads(pathlib.Path(a.job).read_text()),AUTHORITY)
    if job.get('experiment')!='adversarial-v2' or job.get('probe_source_sha256')!=file_sha(__file__):raise ValueError('adversarial approved source/role')
    for name,expected in job['input_files'].items():
        if file_sha(pathlib.Path(job['input_directory'])/name)!=expected:raise ValueError('approved initial-control inputs')
    out=pathlib.Path(job['out']);out.mkdir(parents=True,exist_ok=False)
    import torch,numpy as np
    from subnet.proofs import validate_framing
    original=json.loads((pathlib.Path(job['input_directory'])/'artifact.json').read_text());lp=np.load(pathlib.Path(job['input_directory'])/'logprobs.npy',allow_pickle=False)
    if digest(original['prompt'])!=job['expected_prompt_hash']:raise ValueError('approved input commitment')
    report={'completed':False,'job_hash':digest(job),'stage':'waiting-vram','controls':[],'payable':False,'chain_transactions':False,'quality_improvement_claimed':False,'started_at':time.time()};runtime=None
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def release():
        nonlocal runtime
        runtime=None;gc.collect();torch.cuda.empty_cache()
    def reject(kind,artifact,values):
        try:verify_bound(runtime,artifact,values,job['expected_prompt_hash'])
        except ValueError as e:report['controls'].append({'kind':kind,'rejected':True,'reason':str(e)})
        else:raise ValueError('adversarial false computation accepted: '+kind)
    try:
        save();wait_vram();torch.cuda.reset_peak_memory_stats();runtime=LongContextRuntime(job['checkpoint']['path'],job['checkpoint']['files'])
        parameter=runtime.model.model.layers[-1].mlp.down_proj.weight
        def tensor_sha():return hashlib.sha256(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        before=tensor_sha()
        with torch.no_grad():parameter.add_(.02)
        after=tensor_sha()
        if before==after:raise ValueError('changed-model control did not change weights')
        report['in_memory_changed_weight']={'parameter':'model.layers.23.mlp.down_proj.weight','before_sha256':before,'after_sha256':after,'checkpoint_files_modified':False}
        output=runtime.greedy(original['prompt'],4);acts,wrong_lp=runtime.compute(original['prompt'],output)
        wrong=copy.deepcopy(original);wrong['output']=output;wrong['proofs']=runtime.build_proofs(acts,decode_batching_size=16,topk=128)
        (out/'wrong-model-artifact.json').write_text(json.dumps(wrong)+'\n');np.save(out/'wrong-model-logprobs.npy',wrong_lp,allow_pickle=False)
        del parameter;release();wait_vram();runtime=LongContextRuntime(job['checkpoint']['path'],job['checkpoint']['files'])
        report['honest_fresh_verified']=verify_bound(runtime,original,lp,job['expected_prompt_hash']);reject('changed-model-output-logprobs-proofs',wrong,wrong_lp)
        forged=copy.deepcopy(original);raw=bytearray(base64.b64decode(forged['proofs'][0]));prime=int.from_bytes(raw[:2],'big');value=int.from_bytes(raw[-2:],'little');raw[-2:]=((value+1)%prime).to_bytes(2,'little');forged['proofs'][0]=base64.b64encode(raw).decode()
        validate_framing(forged['proofs'],len(original['proofs']))
        reject('well-formed-wrong-fingerprint',forged,lp)
        (out/'well-formed-wrong-proof.json').write_text(json.dumps(forged)+'\n')
        changed=copy.deepcopy(original);changed['prompt'][0]=(changed['prompt'][0]+1)%runtime.model.config.vocab_size
        acts,changed_lp=runtime.compute(changed['prompt'],changed['output']);changed['proofs']=runtime.build_proofs(acts,decode_batching_size=16,topk=128)
        # Self-consistent target-model computation on unauthorized inputs is not
        # legitimate task execution; reject the signed input binding explicitly.
        report['edited_input_numerically_self_consistent']=runtime.verify(changed,changed_lp)
        reject('recomputed-early-prefix-input-substitution',changed,changed_lp)
        (out/'edited-prefix-artifact.json').write_text(json.dumps(changed)+'\n');np.save(out/'edited-prefix-logprobs.npy',changed_lp,allow_pickle=False)
        report.update(completed=True,stage='complete')
    except Exception as e:report.update(stage='failed',error_type=type(e).__name__,error=str(e)[:1000]);raise
    finally:report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated();release();report['completed_at']=time.time();save();print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
