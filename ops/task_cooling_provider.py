"""Bounded finalized task exposure reader; no signing, network, or model calls.

ROOT-signed closure selects the endpoint. ROOT-signed jobs/admissions bind inputs;
the trainer's *unsigned* report attributes actual pairs/updates. This is not a
cryptographic proof of execution or an independent signature over report bytes.
Cache hits still hash every actual evidence byte and recheck the candidate set.
"""
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from collections import Counter, OrderedDict
from nacl.signing import VerifyKey

AUTHORITY='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
HISTORY=12
MAX_REPORT=64*1024**2
MAX_JOB=24*1024**2
MAX_MANIFEST=8*1024**2
MAX_ATTEMPTS=8
MAX_SECONDS=45
_CACHE=OrderedDict()
_SUMMARIES=OrderedDict()

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
class ExposureUnavailable(ValueError):
 def __init__(self,reason_code):self.reason_code=reason_code;super().__init__(reason_code)
def need(condition,code):
 if not condition:raise ExposureUnavailable(code)
def integer(v):return type(v)is int and v>=0
def hash_value(v):return isinstance(v,str)and re.fullmatch('[0-9a-f]{64}',v)is not None
def finite_time(v):return type(v)in(int,float)and math.isfinite(v)and v>=0
def safe_name(v):return isinstance(v,str)and re.fullmatch('[A-Za-z0-9_-]{1,220}',v)is not None

def signed(envelope,authority=AUTHORITY):
 try:
  need(type(envelope)is dict and set(envelope)=={'payload','signature','signer'},'signed_envelope_shape')
  need(authority is None or envelope['signer']==authority,'wrong_signer')
  VerifyKey(bytes.fromhex(envelope['signer'])).verify(canonical(envelope['payload']),base64.b64decode(envelope['signature'],validate=True))
  return envelope['payload']
 except ExposureUnavailable:raise
 except Exception as error:raise ExposureUnavailable('invalid_signature')from error

def _remember(cache,key,value,maximum):
 cache[key]=copy.deepcopy(value);cache.move_to_end(key)
 while len(cache)>maximum:cache.popitem(last=False)

def _read(path,root,limit,deadline):
 need(time.monotonic()<deadline,'history_time_budget')
 need(path.parent.resolve()in(root,root/'roles')and path.parent==path.parent.resolve(),'evidence_parent_path')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd)
  need(stat.S_ISREG(before.st_mode)and before.st_uid==os.getuid()and 0<before.st_size<=limit,'evidence_file_bounds')
  with os.fdopen(fd,'rb',closefd=False)as stream:raw=stream.read(limit+1)
  after=os.fstat(fd);current=path.lstat()
  fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
  need(fields(before)==fields(after)==fields(current)and len(raw)==before.st_size,'evidence_changed_during_read')
  return raw,{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
 finally:os.close(fd)

def _json(raw):
 try:return json.loads(raw)
 except Exception as error:raise ExposureUnavailable('invalid_evidence_json')from error

def _glob(directory,pattern,maximum):
 result=[]
 for p in directory.glob(pattern):
  result.append(p);need(len(result)<=maximum,'ambiguous_or_excess_evidence')
 return sorted(result)

def _validate_training(closure,original_envelope,job_envelope,report,authority):
 """Attribute each reported pair to exact ROOT/miner-bound admitted contents."""
 epoch=closure['epoch'];checkpoint=closure['checkpoint'];output=closure['next_checkpoint']
 original=signed(original_envelope,authority);job=signed(job_envelope,authority);manifest=signed(job['manifest'],authority)
 need(job['role']==report['role']=='train'and job['job_id']==report['job_id']and report['job_sha256']==digest(job),'training_job_identity')
 need(original['epoch']==manifest['epoch']==report['epoch']==epoch,'training_epoch_identity')
 need(original['checkpoint']['id']==manifest['checkpoint']['id']==report['checkpoint']==checkpoint and report['new_checkpoint']['id']==output,'training_checkpoint_identity')
 need(report['success']is True and finite_time(report['completed_at'])and report['completed_at']<=closure['completed_at'],'training_not_successfully_closed')
 # ROOT-authenticated execution amendments intentionally add native/local-state
 # fields to job.manifest. Bind the original opening explicitly, not by guessing
 # which overlay fields could safely be deleted.
 declaration_envelope=job['unaudited_training_execution'];declaration=signed(declaration_envelope,authority)
 proof=report['unaudited_training_execution']
 need(declaration['epoch']==epoch and declaration['job_id']==job['job_id']and declaration['original_public_manifest_sha256']==digest(original_envelope)and declaration['original_signed_manifest_sha256']==digest(job['manifest']),'original_manifest_binding')
 need(job['source_files']==report['source_files']==declaration['execution_source_files']and type(job['source_files'])is dict and 1<=len(job['source_files'])<=1024 and all(isinstance(k,str)and hash_value(v)for k,v in job['source_files'].items()),'execution_source_binding')
 need(proof['declaration_sha256']==digest(declaration_envelope)and proof['execution_source_files_sha256']==digest(job['source_files'])and proof['execution_source_bundle_sha256']==declaration['training_source_bundle']['sha256']and report['training_runtime_sha256']==declaration['training_runtime_sha256'],'reported_execution_binding')
 need(declaration['original_source_bundle_sha256']==original['source_bundle']['sha256']==manifest['source_bundle']['sha256'],'original_source_binding')
 need(original['environments']==manifest['environments'],'task_environment_binding')
 environments=original['environments'];need(type(environments)is list and len(environments)==1 and environments[0]['env_id']=='affine_math','unsupported_environment')
 allowed=environments[0]['indices'];need(type(allowed)is list and 0<len(allowed)<=10000 and all(integer(i)for i in allowed)and len(set(allowed))==len(allowed),'allowed_task_inventory');allowed=set(allowed)
 need(integer(original['K'])and integer(original['L'])and original['K']==original['L']and 1<=original['K']<=64,'unsupported_pair_quota')
 submissions=job['submissions'];admissions=report['training_admissions']
 need(type(submissions)is list and type(admissions)is list and 0<len(submissions)==len(admissions)<=4096,'admission_count')
 signed_inputs={}
 for obj in submissions:
  admission=signed(obj['learner_admission'],authority)
  need(admission['epoch']==epoch and admission['checkpoint']==checkpoint and admission['document_sha256']==obj['sha256']and admission['document_size']==obj['size'],'signed_admission_binding')
  commit=signed(admission['original_commitment'],None)
  need(commit['miner']==admission['miner_identity']==admission['original_commitment']['signer']and commit['epoch']==epoch and commit['checkpoint']==checkpoint,'miner_commitment_binding')
  batches=commit['batches'];need(type(batches)is list and 0<len(batches)<=128,'commitment_batch_bounds')
  matches=[b for b in batches if b['slot']==admission['slot']];need(len(matches)==1,'ambiguous_commitment_slot');batch=matches[0]
  need(batch['batch_sha256']==admission['batch_sha256']and batch['training_sha256']==obj['sha256']and admission['batch_sha256']not in signed_inputs,'committed_batch_binding')
  signed_inputs[admission['batch_sha256']]=(admission,batch,digest(obj['learner_admission']))
 pair_map={};task_map={};seen_indices=set()
 for item in admissions:
  batch=item['claimed_batch'];batch_hash=digest(batch)
  need(batch_hash in signed_inputs,'unbound_reported_batch');admission,committed,admission_hash=signed_inputs.pop(batch_hash)
  need(item['batch_sha256']==batch_hash and item['learner_admission_sha256']==admission_hash and item['document_sha256']==admission['document_sha256']and item['miner_identity']==admission['miner_identity'],'reported_admission_binding')
  index=batch['index'];need(integer(index)and index in allowed and index==committed['index']and batch['env_id']==committed['env_id']=='affine_math'and batch['epoch']==epoch and batch['checkpoint']==checkpoint and index not in seen_indices,'reported_task_binding');seen_indices.add(index)
  rollouts=batch['rollouts'];need(type(rollouts)is list and len(rollouts)==original['K']+original['L'],'rollout_count')
  hashes={r['task_hash']for r in rollouts};need(len(hashes)==1 and all(hash_value(h)for h in hashes)and all(r['index']==index and r['env_id']=='affine_math'for r in rollouts),'rollout_task_binding')
  task_hash=digest(dict(env_id='affine_math',index=index,task_hash=next(iter(hashes))));task_map[task_hash]=index
  positive=[r for r in rollouts if r['classification']=='positive'];negative=[r for r in rollouts if r['classification']=='negative'];need(len(positive)==original['K']and len(negative)==original['L'],'pair_classification')
  for pos,neg in zip(positive,negative):
   pair_hash=digest(dict(env_id='affine_math',index=index,positive=pos,negative=neg));need(pair_hash not in pair_map,'duplicate_reported_pair');pair_map[pair_hash]=(task_hash,index)
 need(not signed_inputs,'unattributed_signed_inputs')
 training=report['training'];updates=training['updates'];diag=training['persistent_diagnostics'];before=training['global_step_before'];after=training['global_step_after']
 need(type(updates)is list and 1<=len(updates)<=64 and integer(before)and integer(after)and after==before+len(updates)and training['steps']==job['steps']==declaration['steps']==len(updates),'optimizer_counter_binding')
 need(diag['epoch']==epoch and diag['input_checkpoint']==checkpoint and diag['global_optimizer_step_before']==before==declaration['optimizer_step_before']and diag['global_optimizer_step_after']==after and diag['optimizer_steps']==len(updates)and diag['task_count']==len(task_map)and diag['pair_count']==len(pair_map),'diagnostic_counter_binding')
 out=[];used=Counter()
 for n,update in enumerate(updates,1):
  need(update['global_optimizer_step']==before+n and update['epoch_optimizer_step']==n and update['input_checkpoint']==checkpoint,'update_identity')
  pairs=update['pairs'];need(type(pairs)is list and 0<len(pairs)<=262144,'update_pair_bounds');indices=set();local=set()
  for pair in pairs:
   pair_hash=pair['pair_sha256'];need(pair_hash in pair_map and pair_hash not in local and pair['task_sha256']==pair_map[pair_hash][0],'update_pair_attribution');local.add(pair_hash);used[pair_hash]+=1;indices.add(pair_map[pair_hash][1])
  need(update['gradient_pairs']==update['unique_verified_pairs']==len(pairs)and update['gradient_tasks']==update['unique_tasks']==len(indices),'reported_update_counts')
  out.append(dict(global_optimizer_step=before+n,indices=sorted(indices)))
 # Current public recipe partitions its admitted pairs once across the update
 # groups. A future repetition/SFT recipe requires its own explicit attribution.
 need(used==Counter({key:1 for key in pair_map}),'pair_inventory_not_once_through')
 return out

def evidence_provider(config,round_number,epoch,*,authority=AUTHORITY):
 """Return exactly the preceding twelve closed rounds or a stable refusal code.

The caller freezes either this input or a full-pool fallback before opening.
Only compact derived rows survive in-process; no disk cache is trusted.
"""
 deadline=time.monotonic()+MAX_SECONDS
 try:
  need(integer(round_number)and round_number>=HISTORY and safe_name(epoch)and epoch.endswith('-'+str(round_number)),'opening_identity')
  root=Path(config['state']);need(root.is_absolute()and root.resolve()==root and root.is_dir()and root.stat().st_uid==os.getuid(),'state_directory')
  roles=root/'roles';need(roles.resolve()==roles and roles.is_dir()and roles.stat().st_uid==os.getuid(),'roles_directory')
  # Names carry a timestamp between prefix and round. Match the opening stream,
  # not an unrelated controller's equal-numbered historical epoch.
  prefix=epoch.rsplit('-',2)[0]
  rows=[]
  for number in range(round_number-HISTORY,round_number):
   candidates=_glob(root,prefix+'-*-'+str(number)+'-signed-learner-completion.json',2)
   need(len(candidates)==1,'missing_or_ambiguous_closed_round');closure_raw,closure_ref=_read(candidates[0],root,64*1024,deadline);closure=signed(_json(closure_raw),authority)
   previous=closure['epoch'];need(safe_name(previous)and previous.rsplit('-',2)[0]==prefix and previous.endswith('-'+str(number))and closure['round']==number and finite_time(closure['completed_at'])and closure['completed_at']<=time.time()and hash_value(closure['checkpoint'])and hash_value(closure['next_checkpoint']),'closed_epoch_identity')
   need(candidates[0].name==previous+'-signed-learner-completion.json','closure_filename')
   original_raw,original_ref=_read(root/(previous+'-first-signed-manifest.json'),root,MAX_MANIFEST,deadline)
   reports=_glob(roles,previous+'-train-*-report.json',MAX_ATTEMPTS);need(reports,'missing_training_report')
   selected=[];inventory=[]
   for path in reports:
    raw,reference=_read(path,root,MAX_REPORT,deadline);inventory.append(reference)
    summary_key=(str(path),reference['sha256'])
    if summary_key in _SUMMARIES:summary=_SUMMARIES[summary_key]
    else:
     report=_json(raw);summary={k:report.get(k)for k in('success','job_id','epoch','checkpoint')};summary['output_checkpoint']=report.get('new_checkpoint',{}).get('id');_remember(_SUMMARIES,summary_key,summary,192);del report
    if summary['success']is True and summary['epoch']==previous and summary['checkpoint']==closure['checkpoint']and summary['output_checkpoint']==closure['next_checkpoint']:selected.append((raw,reference,summary))
   need(len(selected)==1,'missing_or_ambiguous_successful_recovery');report_raw,report_ref,summary=selected[0]
   job_id=summary['job_id'];need(safe_name(job_id)and job_id.startswith(previous+'-train-')and Path(report_ref['path']).name==job_id+'-report.json','job_filename')
   job_raw,job_ref=_read(roles/(job_id+'-job.json'),root,MAX_JOB,deadline)
   cache_key=(authority,digest([closure_ref,original_ref,job_ref,inventory]))
   if cache_key in _CACHE:row=copy.deepcopy(_CACHE[cache_key])
   else:
    updates=_validate_training(closure,_json(original_raw),_json(job_raw),_json(report_raw),authority)
    row=dict(round=number,epoch=previous,input_checkpoint=closure['checkpoint'],output_checkpoint=closure['next_checkpoint'],updates=updates,provenance=dict(closure=closure_ref,job=job_ref,report=report_ref));_remember(_CACHE,cache_key,row,48)
   rows.append(row);need(time.monotonic()<deadline,'history_time_budget')
  # No implicit run reset within this simple trailing-window provider. A real
  # reset/gap falls back to the original pool until twelve coherent closes exist.
  need(all(a['output_checkpoint']==b['input_checkpoint']and a['updates'][-1]['global_optimizer_step']+1==b['updates'][0]['global_optimizer_step']for a,b in zip(rows,rows[1:])),'history_lineage_discontinuity')
  return dict(version='finalized-trained-task-exposure-input-v1',assurance='signed-closure-and-job-linked-unsigned-trainer-report',records=rows)
 except ExposureUnavailable:raise
 except (OSError,KeyError,TypeError,ValueError,OverflowError)as error:raise ExposureUnavailable('missing_or_malformed_finalized_evidence')from error
