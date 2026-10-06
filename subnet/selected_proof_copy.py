"""Prospective bounded copies of post-commit audit selections, never a redraw."""
import time
from concurrent.futures import ThreadPoolExecutor
VERSION='selected-proof-copy-v1'

def validate_policy(value):
 if type(value)is not dict or value!={'version':VERSION,'workers':4}:raise ValueError('exact selected proof copy policy')
 return dict(value)

def freeze_metadata(gateway,epoch):
 from .commitment_transport import canonical,sha,FreezeMetadataIncomplete
 from .storage import SubmissionPolicyError
 state=gateway.epochs[epoch];validate_policy(state['commitment_binding']['proof_copy_policy'])
 cutoff=state['commitment_binding'].get('freeze_until');pending=state['commitment_pending'];receipts={}
 # Full population is authenticated before this pass. Each declared child HEAD
 # is captured before the signed boundary. Confirmed absent objects reject
 # only their miner; bucket/access/outage ambiguity cannot finalize a population.
 items=[(miner,b)for miner,p in sorted(pending.items())if miner not in state['rejections']for b in p['document']['payload']['batches']if str(b['slot'])not in p['artifact_plans']]
 def missing_child(error):
  from botocore.exceptions import ClientError
  return isinstance(error,ClientError)and str(error.response.get('Error',{}).get('Code'))in('NoSuchKey','NotFound','404')
 def head_one(item):
  miner,b=item
  try:
   if cutoff is not None and time.time()>=cutoff:raise TimeoutError('complete child metadata cutoff')
   key='private/'+epoch+'/staging/'+miner+'/'+str(b['slot'])+'.zip'
   meta=gateway.bucket.client.head_object(Bucket=gateway.bucket.name,Key=key)
   if cutoff is not None and time.time()>=cutoff:raise TimeoutError('completed child metadata after cutoff')
   if b['size']>state.get('upload_limit',100_000_000)or meta['ContentLength']!=b['size']or not state['start']<=meta['LastModified'].timestamp()<state['deadline']:raise SubmissionPolicyError('artifact size/time')
   return miner,b,dict(etag=meta['ETag'],received_at=meta['LastModified'].timestamp()),None
  except Exception as exc:
   if cutoff is not None and time.time()>=cutoff:exc=TimeoutError('child metadata boundary expired')
   return miner,b,None,exc
 failures=[]
 with ThreadPoolExecutor(max_workers=4)as pool:
  for miner,b,meta,error in pool.map(head_one,items):
   if missing_child(error):state['rejections'][miner]='missing completed declared artifact';gateway.persist()
   elif isinstance(error,SubmissionPolicyError):state['rejections'][miner]=str(error);gateway.persist()
   elif error is not None:failures.append(error)
   else:pending[miner]['artifact_plans'][str(b['slot'])]=meta;gateway.persist()
 if failures:
  state['commitment_metadata_incomplete']=dict(reason='child_HEAD_budget_incomplete'if cutoff is not None and time.time()>=cutoff else'child_HEAD_infrastructure_incomplete',at=time.time(),error_type=type(failures[0]).__name__);gateway.persist()
  raise FreezeMetadataIncomplete('complete child metadata unavailable')from failures[0]
 for miner,p in sorted(pending.items()):
  if miner in state['rejections']:continue
  data=canonical(p['document'])
  if sha(data)!=p['sha256']:raise ValueError('original canonical commitment integrity')
  gateway.bucket.put(p['root']+'/commitment.json',data)
  artifacts=[]
  for b in p['document']['payload']['batches']:
   ap=p['artifact_plans'][str(b['slot'])]
   artifacts.append(dict(b,key='private/'+epoch+'/staging/'+miner+'/'+str(b['slot'])+'.zip',frozen_key=p['root']+'/'+str(b['slot'])+'.zip',**ap))
  receipts[miner]=dict(sha256=p['sha256'],commitment_document=p['document'],commitment_key=p['root']+'/commitment.json',artifacts=artifacts,size=p['size'],received_at=p['received_at'],hash_assurance='declared-payload-hashes-until-selected-verifier',proof_copy_policy=VERSION,artifact_public_availability='only-successfully-copied-selected-proofs')
 if state['commitment_binding'].get('version')in ('small-commitment-pairs-v2','small-commitment-token-pairs-v3'):
  from .training_documents import attach
  attach(state,receipts)
 state['frozen_receipts']=receipts;state.pop('commitment_metadata_incomplete',None);gateway.persist();gateway.bucket.json('public/'+epoch+'/receipts.json',receipts)
 return receipts

def copy_selected(gateway,manifest,receipts,selected_slots,until):
 """Copies exact original ETags with four workers; failures retain the draw.

 Only the owner thread journals completions. Recovery reuses copied inventory;
 no public availability or fraud claim follows from an unselected/failed copy.
 """
 validate_policy(manifest['proof_copy_policy']);epoch=manifest['epoch'];state=gateway.epochs[epoch]
 if state.get('frozen_receipts')!=receipts:raise ValueError('exact original selected proof population')
 selection={miner:list(slots)for miner,slots in sorted(selected_slots.items())}
 original=state.get('selected_proof_selection')
 if original is not None and original!=selection:raise ValueError('original postcommit selection cannot redraw')
 journal=state.setdefault('selected_proof_copies',{});items=[]
 for miner,slots in selected_slots.items():
  if miner not in receipts or any(type(slot)is not int for slot in slots)or len(slots)!=len(set(slots)):raise ValueError('selected proof inventory')
  for slot in slots:
   matches=[b for b in receipts[miner]['artifacts']if b['slot']==slot]
   if len(matches)!=1:raise ValueError('selected original slot')
   b=matches[0];binding={k:b[k]for k in ('sha256','size','etag','key','frozen_key')}
   old=journal.get(miner,{}).get(str(slot))
   if old is not None:
    if old!=binding:raise ValueError('immutable selected copy journal')
   else:items.append((miner,slot,b,binding))
 if original is None:state['selected_proof_selection']=selection;gateway.persist()
 def copy_one(item):
  miner,slot,b,binding=item
  try:
   if until is not None and time.time()>=until:raise TimeoutError('selected copy audit cutoff')
   gateway.bucket.copy(b['key'],b['frozen_key'],expected_etag=b['etag'])
   return miner,slot,binding,None
  except Exception as exc:return miner,slot,binding,type(exc).__name__
 errors={}
 with ThreadPoolExecutor(max_workers=4)as pool:
  for miner,slot,binding,error in pool.map(copy_one,items):
   if error is not None:errors[miner]=error;continue
   journal.setdefault(miner,{})[str(slot)]=binding;gateway.persist()
 return errors

def signed_copy_inventory(gateway,epoch):
 """Capability generation follows actual copy completion, persisted for replay."""
 state=gateway.epochs[epoch];urls=state.setdefault('selected_proof_read_urls',{});result={}
 for miner,copies in state.get('selected_proof_copies',{}).items():
  for slot,binding in copies.items():
   if slot not in urls.setdefault(miner,{}):
    urls[miner][slot]=gateway.bucket.presign(binding['frozen_key']);gateway.persist()
   result.setdefault(miner,{})[slot]=dict(binding,read_url=urls[miner][slot])
 return result
