#!/usr/bin/env python3
"""Signed, isolated full BF16 optimizer qualification on admitted native pairs."""
import argparse,gc,json,math,pathlib,resource,sys,time
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from subnet.long_context_runtime import LongContextRuntime,validate_job,authenticate,AUTHORITY,file_sha,digest,wait_vram

OPTIMIZER={'steps':1,'lr':5e-5,'beta':.1,'weight_decay':0.,'eps':1e-8,'max_grad_norm':1.,
           'parameter_dtype':'bfloat16','gradient_dtype':'bfloat16','adam_moment_dtype':'bfloat16',
           'master_parameters':False,'objective':'mean-agent-reference-relative-preference-v1',
           'gradient_checkpointing':True,'use_cache':False,'foreach':False}
RESOURCES={'min_free_vram_mib':20480,'wait_seconds':1800,'max_peak_allocated_bytes':8*1024**3,'max_parameters':600000000}

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();job=validate_job(json.loads(pathlib.Path(a.job).read_text()),AUTHORITY)
    if job.get('experiment')!='full-gradient-optimizer-v1' or job.get('probe_source_sha256')!=file_sha(__file__) or job.get('training_helper_sha256')!=file_sha(ROOT/'subnet/long_context_training.py') or job.get('optimizer')!=OPTIMIZER or job.get('training_resources')!=RESOURCES:raise ValueError('full long-context signed optimizer/source/resources')
    out=pathlib.Path(job['out']);out.mkdir(parents=True,exist_ok=False)
    import torch,numpy as np
    from subnet.long_context_training import sequence_logprob,configure_full_training
    records=[];arrays=[]
    for side in ('positive','negative'):
        row=job['pair'][side];directory=pathlib.Path(row['directory']);audit=authenticate(row['model_audit'],AUTHORITY)
        if digest(audit['checkpoint_files'])!=job['checkpoint_id'] or audit['checkpoint_files']!=job['checkpoint']['files'] or audit['full_model_recompute'] is not True:raise ValueError('admitted exact model pair')
        i=row['turn_index'];name=f'turn-{i}.json';prob=f'turn-{i}.npz'
        if file_sha(directory/name)!=audit['artifact_files'][name]['sha256'] or file_sha(directory/prob)!=audit['artifact_files'][prob]['sha256']:raise ValueError('optimizer audited arrays/tokens')
        records.append(json.loads((directory/name).read_text()));arrays.append(np.load(directory/prob,allow_pickle=False)['logprobs'])
    if records[0]['prompt']!=records[1]['prompt'] or records[0]['output']==records[1]['output'] or job['native_admission']['positive']['original_reward']!=1 or job['native_admission']['negative']['original_reward']!=0:raise ValueError('native matched agent preference')
    refs=[float(torch.from_numpy(lp).gather(1,torch.tensor(record['output'])[:,None]).mean()) for record,lp in zip(records,arrays)]
    report={'completed':False,'job_hash':digest(job),'stage':'waiting-vram','optimizer':OPTIMIZER,'resources':RESOURCES,
            'parameters_dtype':'bfloat16','auxiliary_tokens_in_loss':False,'full_model_finetune':True,
            'payable':False,'chain_transactions':False,'quality_improvement_claimed':False,'started_at':time.time()};runtime=None
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'stage':report['stage']}),flush=True)
    def release():
        nonlocal runtime
        runtime=None;gc.collect();torch.cuda.empty_cache()
    try:
        save();wait_vram(**{'minimum':RESOURCES['min_free_vram_mib'],'seconds':RESOURCES['wait_seconds']})
        torch.cuda.set_per_process_memory_fraction(RESOURCES['max_peak_allocated_bytes']/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        runtime=LongContextRuntime(job['checkpoint']['path'],job['checkpoint']['files']);model=runtime.model;count=sum(p.numel() for p in model.parameters())
        if count>RESOURCES['max_parameters']:raise ValueError('optimizer model budget')
        configure_full_training(runtime);optimizer=torch.optim.AdamW(model.parameters(),lr=OPTIMIZER['lr'],weight_decay=0.,eps=OPTIMIZER['eps'],foreach=False);optimizer.zero_grad(set_to_none=True)
        report.update(stage='full-gradient-reference-forward',parameters=count,prompt_tokens=len(records[0]['prompt']),chosen_tokens=len(records[0]['output']),rejected_tokens=len(records[1]['output']),reference_mean_logprobs=refs);save()
        chosen=sequence_logprob(runtime,records[0]['prompt'],records[0]['output']);rejected=sequence_logprob(runtime,records[1]['prompt'],records[1]['output'])
        before=[float(chosen.detach()),float(rejected.detach())];report['training_forward_reference_errors']=[abs(x-y) for x,y in zip(before,refs)]
        if any(abs(x-y)>1e-5 for x,y in zip(before,refs)):raise ValueError('strict training forward reference mismatch')
        loss=torch.nn.functional.softplus(-OPTIMIZER['beta']*((chosen-refs[0])-(rejected-refs[1])))
        if not torch.isfinite(loss):raise ValueError('nonfinite full preference loss')
        report.update(stage='full-gradient-backward',loss=float(loss.detach()));save();loss.backward()
        gradients=sum(p.grad is not None for p in model.parameters());norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if gradients!=len(list(model.parameters())) or not math.isfinite(norm) or norm<=0:raise ValueError('full parameter gradient coverage')
        optimizer.step();moment_dtypes=sorted({str(v.dtype) for state in optimizer.state.values() for k,v in state.items() if k in ('exp_avg','exp_avg_sq')})
        if moment_dtypes!=['torch.bfloat16']:raise ValueError('optimizer state dtype policy')
        report.update(stage='measuring-changed-weights',gradient_tensors=gradients,gradient_norm=norm,adam_moment_dtypes=moment_dtypes);save()
        optimizer.zero_grad(set_to_none=True);model.gradient_checkpointing_disable();model.eval()
        from safetensors import safe_open
        changed=0;elements=0;squared=0.;maximum=0.
        with safe_open(str(pathlib.Path(job['checkpoint']['path'])/'model.safetensors'),framework='pt',device='cpu') as original:
            for name,parameter in model.named_parameters():
                if not bool(torch.isfinite(parameter).all()):raise ValueError('nonfinite changed weights')
                delta=parameter.detach().cpu().float()-original.get_tensor(name).float();n=int(torch.count_nonzero(delta));changed+=int(n>0);elements+=n;maximum=max(maximum,float(delta.abs().max()))
                for chunk in delta.flatten().split(4*1024*1024):squared+=float(chunk.double().square().sum())
        if not elements:raise ValueError('optimizer weights unchanged')
        with torch.no_grad():after=[float(sequence_logprob(runtime,r['prompt'],r['output'])) for r in records]
        report.update(changed_parameter_tensors=changed,changed_parameter_elements=elements,parameter_delta_l2=math.sqrt(squared),parameter_delta_max_abs=maximum,after_mean_logprobs=after,preference_margin_change=(after[0]-after[1])-(refs[0]-refs[1]))
        if torch.cuda.max_memory_allocated()>RESOURCES['max_peak_allocated_bytes']:raise ValueError('measured full optimizer allocator cap exceeded')
        destination=out/'checkpoint';destination.mkdir();model.save_pretrained(destination,safe_serialization=True);runtime.tokenizer.save_pretrained(destination)
        files={p.name:file_sha(p) for p in destination.iterdir() if p.is_file()};checkpoint={'id':digest(files),'files':files,'path':str(destination),'base_hf_revision':job['checkpoint']['revision'],'trained_from':job['checkpoint_id']}
        report.update(stage='fresh-changed-checkpoint-verification',checkpoint=checkpoint);save()
        del optimizer,model,chosen,rejected,loss,parameter,delta,chunk;release();wait_vram();runtime=LongContextRuntime(destination,files)
        acts,lp=runtime.compute(records[0]['prompt'],records[0]['output']);artifact={'prompt':records[0]['prompt'],'output':records[0]['output'],'proofs':runtime.build_proofs(acts,decode_batching_size=16,topk=128),'profile':runtime.profile()}
        (out/'changed-checkpoint-artifact.json').write_text(json.dumps(artifact)+'\n');np.savez_compressed(out/'changed-checkpoint-logprobs.npz',logprobs=lp)
        release();wait_vram();runtime=LongContextRuntime(destination,files);report['fresh_changed_checkpoint_verified']=runtime.verify(artifact,np.load(out/'changed-checkpoint-logprobs.npz',allow_pickle=False)['logprobs']);report.update(completed=True,stage='complete')
    except Exception as e:report.update(stage='failed',error_type=type(e).__name__,error=str(e)[:1000]);raise
    finally:report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated();release();report['peak_cpu_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;report['completed_at']=time.time();save()
if __name__=='__main__':main()
