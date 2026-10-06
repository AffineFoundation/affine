"""Small immutable token metadata transport, explicitly unaudited at capture."""
import json,time
from .commitment_transport import canonical,sha,need,is_digest
VERSION='committed-training-documents-v1'
MAX_BYTES=2_000_000
TRANSPORT='small-commitment-pairs-v2'

def document(batch,manifest,miner,slot):
 value=dict(version=VERSION,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],miner=miner,slot=slot,batch=batch)
 data=canonical(value);need(0<len(data)<=MAX_BYTES,'bounded token document');return data

def validate(data,epoch,checkpoint,miner,entry):
 need(type(data)is bytes and 0<len(data)<=MAX_BYTES,'bounded token document')
 need(len(data)==entry['training_size']and sha(data)==entry['training_sha256'],'token document declared byte binding')
 value=json.loads(data);need(data==canonical(value),'canonical token document')
 need(type(value)is dict and set(value)=={'version','epoch','checkpoint','miner','slot','batch'}and value['version']==VERSION,'token document schema')
 need(value['epoch']==epoch and value['checkpoint']==checkpoint and value['miner']==miner and type(value['slot'])is int and value['slot']==entry['slot'],'token document scope')
 b=value['batch'];need(type(b)is dict and sha(canonical(b))==entry['batch_sha256']and b.get('env_id')==entry['env_id']and type(b.get('index'))is int and b['index']==entry['index'],'token batch commitment')
 return value

def capture(gateway,epoch):
 """Complete bounded GET + exact SHA before immutable token publication.

 GET and immutable publication share at most four bounded workers; durable
 journal updates remain serial and follow successful publication.
 No model/grader verification is claimed. Only previously authenticated tiny
 commitments participate; same captured bytes survive retries and restarts.
 """
 from concurrent.futures import ThreadPoolExecutor
 from collections import deque
 from .commitment_transport import FreezeMetadataIncomplete
 from .storage import SubmissionPolicyError
 state=gateway.epochs[epoch];need(state.get('commitment_capture_complete')is True,'complete signed commitment capture required')
 pending=state['commitment_pending'];journal=state.setdefault('training_document_snapshots',{});cutoff=state['commitment_binding'].get('freeze_until')
 def remaining():return [(miner,b)for miner,p in sorted(pending.items())if miner not in state['rejections']for b in p['document']['payload']['batches']if str(b['slot'])not in journal.get(miner,{})]
 client=gateway.bucket.commitment_read_client()if hasattr(gateway.bucket,'commitment_read_client')else gateway.bucket.client
 def fetch(item):
  miner,b=item;key='private/'+epoch+'/training/'+miner+'/'+str(b['slot'])+'.json'
  try:
   if cutoff is not None and time.time()>=cutoff:raise TimeoutError('token capture cutoff')
   response=client.get_object(Bucket=gateway.bucket.name,Key=key);body=response['Body']
   try:data=body.read(MAX_BYTES+1)
   finally:body.close()
   if not state['start']<=response['LastModified'].timestamp()<state['deadline']:raise SubmissionPolicyError('token document upload time')
   try:validate(data,epoch,state['commitment_binding']['checkpoint'],miner,b)
   except (ValueError,KeyError,TypeError,json.JSONDecodeError)as exc:raise SubmissionPolicyError('token document binding')from exc
   if cutoff is not None and time.time()>=cutoff:raise TimeoutError('token capture cutoff')
   frozen=pending[miner]['root']+'/training/'+str(b['slot'])+'.json'
   receipt=dict(slot=b['slot'],sha256=sha(data),size=len(data),frozen_key=frozen,captured_at=time.time(),assurance='unaudited')
   try:gateway.bucket.put(frozen,data)
   except Exception as exc:
    failure=RuntimeError('token snapshot publication infrastructure failure');failure.__cause__=exc
    return miner,b,None,None,failure
   return miner,b,None,receipt,None
  except Exception as exc:
   from botocore.exceptions import ClientError
   if isinstance(exc,ClientError)and str(exc.response.get('Error',{}).get('Code'))in('NoSuchKey','NotFound','404'):exc=SubmissionPolicyError('missing completed token document')
   return miner,b,None,None,exc
 failures=[]
 try:
  while remaining():
   items=remaining();wave_errors=[]
   with ThreadPoolExecutor(max_workers=4)as pool:
    waiting=deque();todo=iter(items)
    def fill():
     while len(waiting)<4:
      try:item=next(todo)
      except StopIteration:return
      waiting.append(pool.submit(fetch,item))
    fill()
    while waiting:
     miner,b,data,receipt,error=waiting.popleft().result()
     if isinstance(error,SubmissionPolicyError):state['rejections'][miner]=str(error);gateway.persist()
     elif error is not None:wave_errors.append(error)
     else:
      journal.setdefault(miner,{})[str(b['slot'])]=receipt;gateway.persist()
     fill()
   failures.extend(wave_errors)
   if not remaining():break
   if cutoff is None:raise FreezeMetadataIncomplete('bounded signed token capture cutoff required')
   if time.time()>=cutoff:break
   time.sleep(min(.25,max(0,cutoff-time.time())))
 finally:
  if client is not gateway.bucket.client:client.close()
 deferred=state.setdefault('training_document_deferred',{})
 for miner,b in remaining():deferred.setdefault(miner,{})[str(b['slot'])]=dict(reason='infrastructure_deferred',closed_at=time.time())
 state.pop('training_document_capture_incomplete',None);state['training_document_capture_complete']=True;gateway.persist()
 return journal

def attach(state,receipts):
 for miner,receipt in receipts.items():
  rows=state.get('training_document_snapshots',{}).get(miner,{})
  for b in receipt['artifacts']:
   if str(b['slot'])not in rows:continue
   row=rows[str(b['slot'])];need(set(row)=={'slot','sha256','size','frozen_key','captured_at','assurance'}and row['slot']==b['slot']and type(row['slot'])is int and row['sha256']==b['training_sha256']and row['size']==b['training_size']and type(row['size'])is int and row['assurance']=='unaudited','captured token journal integrity')
   need(row['frozen_key']==state['commitment_pending'][miner]['root']+'/training/'+str(b['slot'])+'.json','captured token immutable location')
  receipt['training_documents']=[dict(rows[str(b['slot'])])for b in receipt['artifacts']if str(b['slot'])in rows]
  receipt['training_document_deferred_slots']=[b['slot']for b in receipt['artifacts']if str(b['slot'])not in rows]
  receipt['training_document_capture_status']='captured'if not receipt['training_document_deferred_slots']else'infrastructure_deferred'
 return receipts


def freeze_receipts(gateway,epoch):
 """Declared proof population + actual captured token bytes; no proof I/O."""
 state=gateway.epochs[epoch];need(state.get('training_document_capture_complete')is True,'complete token capture')
 receipts={}
 for miner,p in sorted(state['commitment_pending'].items()):
  if miner in state['rejections']:continue
  data=canonical(p['document']);need(sha(data)==p['sha256'],'signed canonical parent digest')
  gateway.bucket.put(p['root']+'/commitment.json',data)
  children=[dict(b,key='private/'+epoch+'/staging/'+miner+'/'+str(b['slot'])+'.zip',frozen_key=p['root']+'/'+str(b['slot'])+'.zip',proof_capture_status='declared-not-captured')for b in p['document']['payload']['batches']]
  receipts[miner]=dict(sha256=p['sha256'],commitment_document=p['document'],commitment_key=p['root']+'/commitment.json',artifacts=children,size=p['size'],received_at=p['received_at'],hash_assurance='declared-proof-hashes-until-selected-verifier',artifact_public_availability='only-successfully-copied-selected-proofs')
 attach(state,receipts);state['frozen_receipts']=receipts;gateway.persist();gateway.bucket.json('public/'+epoch+'/receipts.json',receipts);return receipts
