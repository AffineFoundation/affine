#!/usr/bin/env python3
"""Signed ALL7 native-pair qualification of sequential full-model gradients."""
import argparse,gc,json,math,pathlib,resource,sys,time
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from subnet.long_context_runtime import LongContextRuntime,validate_job,authenticate,AUTHORITY,file_sha,digest,wait_vram
RESOURCES={'min_free_vram_mib':20480,'wait_seconds':1800,'max_peak_allocated_bytes':8*1024**3,'max_parameters':600000000}

def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();job=validate_job(json.loads(pathlib.Path(a.job).read_text()),AUTHORITY)
 from subnet.long_context_service_training import POLICY,full_parameter_train,rollout_mean
 if job.get('experiment')!='full-sequential-seven-agent-turns-v1' or job.get('probe_source_sha256')!=file_sha(__file__) or job.get('training_policy')!=POLICY or job.get('training_resources')!=RESOURCES:raise ValueError('signed ALL7 sequential policy/source/resource')
 for name,sha in job['training_source_files'].items():
  if pathlib.Path(name).is_absolute() or '..' in pathlib.Path(name).parts or file_sha(ROOT/name)!=sha:raise ValueError('sequential source closure')
 import torch,numpy as np
 rolls=[];refs=[]
 for side,reward in [('positive',1),('negative',0)]:
  row=job['pair'][side];audit=authenticate(row['model_audit'],AUTHORITY);native=job['native_admission'][side]
  if audit['kind']!='controlled-original-eog-model-audit-v3-terminal' or audit['checkpoint_files']!=job['checkpoint']['files'] or audit['checkpoint']!=digest(job['checkpoint']['files']) or audit['terminal_model_proof'] is not True or len(audit['records'])!=7 or not all(r['full_proof_verified'] for r in audit['records']):raise ValueError('strict complete ALL7 model audit')
  if native.get('passed') is not True or native.get('original_reward')!=reward or native.get('checkpoint')!=audit['checkpoint'] or native.get('model_audit_sha256')!=digest(row['model_audit']):raise ValueError('fresh original native terminal admission')
  directory=pathlib.Path(row['directory']);turns=[];summed=0;tokens=0
  for i in range(7):
   name=f'turn-{i}.json';prob=f'turn-{i}.npz'
   if file_sha(directory/name)!=audit['artifact_files'][name]['sha256'] or file_sha(directory/prob)!=audit['artifact_files'][prob]['sha256']:raise ValueError('audited all-turn raw data')
   record=json.loads((directory/name).read_text());lp=np.load(directory/prob,allow_pickle=False)['logprobs']
   if lp.dtype!=np.float32 or lp.shape!=(len(record['output']),151936) or not np.isfinite(lp).all():raise ValueError('full exact-vocabulary probability rows')
   summed+=float(torch.from_numpy(lp).gather(1,torch.tensor(record['output'])[:,None]).sum());tokens+=len(record['output']);turns.append({'prompt':record['prompt'],'output':record['output'],'model_role':'agent'})
  rolls.append({'turns':turns});refs.append(summed/tokens)
 if rolls[0]['turns'][3]['prompt']!=rolls[1]['turns'][3]['prompt'] or rolls[0]['turns'][3]['output']==rolls[1]['turns'][3]['output']:raise ValueError('public-prefix divergent candidate pair')
 out=pathlib.Path(job['out']);out.mkdir(parents=True,exist_ok=False);report={'completed':False,'job_hash':digest(job),'stage':'waiting-vram','training_policy':POLICY,'resources':RESOURCES,'full_model_finetune':True,'auxiliary_tokens_in_loss':False,'quality_improvement_claimed':False,'payable':False,'chain_transactions':False,'started_at':time.time()};runtime=None
 def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'stage':report['stage']}),flush=True)
 def release():
  nonlocal runtime
  runtime=None;gc.collect();torch.cuda.empty_cache()
 try:
  save();wait_vram(RESOURCES['min_free_vram_mib'],RESOURCES['wait_seconds']);torch.cuda.set_per_process_memory_fraction(RESOURCES['max_peak_allocated_bytes']/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats();runtime=LongContextRuntime(job['checkpoint']['path'],job['checkpoint']['files'])
  with torch.no_grad():actual=[float(rollout_mean(runtime,r)) for r in rolls]
  errors=[abs(x-y) for x,y in zip(actual,refs)]
  if any(e>1e-5 for e in errors):raise ValueError('strict all-agent reference computation')
  report.update(stage='sequential-full-all-agent-turn-gradients',reference_mean_logprobs=actual,training_forward_reference_errors=errors,agent_turns=[7,7],agent_output_tokens=[sum(len(t['output']) for t in r['turns']) for r in rolls]);save()
  report['training']=full_parameter_train(runtime,[(rolls[0],rolls[1])],steps=1)
  from safetensors import safe_open
  changed=0;elements=0;squared=0.;maximum=0.
  with safe_open(str(pathlib.Path(job['checkpoint']['path'])/'model.safetensors'),framework='pt',device='cpu') as original:
   for name,parameter in runtime.model.named_parameters():
    if not torch.isfinite(parameter).all():raise ValueError('nonfinite changed weights')
    delta=parameter.detach().cpu().float()-original.get_tensor(name).float();n=int(torch.count_nonzero(delta));changed+=int(n>0);elements+=n;maximum=max(maximum,float(delta.abs().max()))
    for chunk in delta.flatten().split(4*1024*1024):squared+=float(chunk.double().square().sum())
  if not elements:raise ValueError('full optimizer did not change parameters')
  report.update(changed_parameter_tensors=changed,changed_parameter_elements=elements,parameter_delta_l2=math.sqrt(squared),parameter_delta_max_abs=maximum)
  with torch.no_grad():report['after_mean_logprobs']=[float(rollout_mean(runtime,r)) for r in rolls]
  report['preference_margin_change']=(report['after_mean_logprobs'][0]-report['after_mean_logprobs'][1])-(actual[0]-actual[1])
  destination=out/'checkpoint';destination.mkdir();runtime.model.save_pretrained(destination,safe_serialization=True);runtime.tokenizer.save_pretrained(destination);files={p.name:file_sha(p) for p in destination.iterdir() if p.is_file()};report['checkpoint']={'id':digest(files),'files':files,'path':str(destination),'base_hf_revision':job['checkpoint']['revision'],'trained_from':digest(job['checkpoint']['files'])};report.update(stage='fresh-changed-checkpoint-proof');save();del parameter,delta,chunk;release();wait_vram();runtime=LongContextRuntime(destination,files)
  turn=rolls[0]['turns'][3];acts,lp=runtime.compute(turn['prompt'],turn['output']);artifact={'prompt':turn['prompt'],'output':turn['output'],'proofs':runtime.build_proofs(acts,decode_batching_size=16,topk=128),'profile':runtime.profile()};(out/'changed-checkpoint-artifact.json').write_text(json.dumps(artifact)+'\n');np.savez_compressed(out/'changed-checkpoint-logprobs.npz',logprobs=lp);release();wait_vram();runtime=LongContextRuntime(destination,files);report['fresh_changed_checkpoint_verified']=runtime.verify(artifact,np.load(out/'changed-checkpoint-logprobs.npz',allow_pickle=False)['logprobs']);report.update(completed=True,stage='complete',fresh_full_seven_turn_trace_verified=False)
 except Exception as e:report.update(stage='failed',error_type=type(e).__name__,error=str(e)[:1000]);raise
 finally:report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated();release();report['peak_cpu_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;report['completed_at']=time.time();save()
if __name__=='__main__':main()
