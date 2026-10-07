"""Small immutable token metadata transport, explicitly unaudited at capture."""
import json,time
from .commitment_transport import canonical,sha,need,is_digest
VERSION='committed-training-documents-v1'
TOKEN_VERSION='committed-token-training-documents-v2'
MAX_BYTES=2_000_000
TRANSPORT='small-commitment-pairs-v2'
CAPTURE_VERSION='bounded-parallel-token-capture-v1'
DURABLE_CAPTURE_VERSION='bounded-parallel-token-capture-v2'

def capture_policy(value):
 keys={'version','workers','max_document_bytes','max_inflight_bytes','completion_order'}
 durable=type(value)is dict and value.get('version')==DURABLE_CAPTURE_VERSION
 if durable:keys|={'journal_version','state_checkpoint_documents'}
 need(type(value)is dict and set(value)==keys,'exact prospective token capture policy')
 if durable:
  need(value['journal_version']=='fsynced-per-epoch-capture-v1'and type(value['state_checkpoint_documents'])is int and 1<=value['state_checkpoint_documents']<=16,'bounded fsynced capture checkpoint')
 need(value['version']in(CAPTURE_VERSION,DURABLE_CAPTURE_VERSION) and type(value['workers'])is int and value['workers']in(4,8,16),'bounded token capture workers')
 need(type(value['max_document_bytes'])is int and value['max_document_bytes']==MAX_BYTES,'unchanged token document bound')
 need(type(value['max_inflight_bytes'])is int and value['max_inflight_bytes']==value['workers']*MAX_BYTES,'exact capture byte budget')
 need(value['completion_order']=='first-completed','bounded prospective completion policy')
 return dict(value)


def document(batch,manifest,miner,slot):
 version=VERSION
 if 'token_artifact_policy'in manifest:
  from .token_only_protocol import for_manifest,framing
  for_manifest(manifest);framing(batch,[[]for _ in batch['rollouts']]);version=TOKEN_VERSION
 value=dict(version=version,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],miner=miner,slot=slot,batch=batch)
 data=canonical(value);need(0<len(data)<=MAX_BYTES,'bounded token document');return data

def validate(data,epoch,checkpoint,miner,entry,*,transport=TRANSPORT):
 need(type(data)is bytes and 0<len(data)<=MAX_BYTES,'bounded token document')
 need(len(data)==entry['training_size']and sha(data)==entry['training_sha256'],'token document declared byte binding')
 version=TOKEN_VERSION if transport=='small-commitment-token-pairs-v3'else VERSION
 need(transport in (TRANSPORT,'small-commitment-token-pairs-v3'),'training document transport')
 value=json.loads(data);need(data==canonical(value),'canonical token document')
 need(type(value)is dict and set(value)=={'version','epoch','checkpoint','miner','slot','batch'}and value['version']==version,'token document schema')
 need(value['epoch']==epoch and value['checkpoint']==checkpoint and value['miner']==miner and type(value['slot'])is int and value['slot']==entry['slot'],'token document scope')
 b=value['batch'];need(type(b)is dict and sha(canonical(b))==entry['batch_sha256']and b.get('env_id')==entry['env_id']and type(b.get('index'))is int and b['index']==entry['index'],'token batch commitment')
 if version==TOKEN_VERSION:
  from .token_only_protocol import framing
  framing(b,[[]for _ in b['rollouts']])
 return value

def capture(gateway,epoch):
 """Complete bounded GET + exact SHA before immutable token publication.

 Legacy capture uses four FIFO workers; an explicit signed policy opts into
 bounded first-completed scheduling. Journal updates remain serial and follow
 successful publication. The byte limit covers raw documents, not decoded
 Python objects or HTTP buffers; each read may include one oversize sentinel byte.
 No model/grader verification is claimed. Only previously authenticated tiny
 commitments participate; same captured bytes survive retries and restarts.
 """
 from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
 from collections import deque
 from .commitment_transport import FreezeMetadataIncomplete
 from .storage import SubmissionPolicyError
 state=gateway.epochs[epoch];need(state.get('commitment_capture_complete')is True,'complete signed commitment capture required')
 pending=state['commitment_pending'];journal=state.setdefault('training_document_snapshots',{});cutoff=state['commitment_binding'].get('freeze_until')
 if state.get('capture_recovery_authorizations'):
  from .late_capture_recovery import cutoff as capture_cutoff
  cutoff=capture_cutoff(state,epoch)
 policy=state['commitment_binding'].get('learner_capture_policy')
 policy=capture_policy(policy)if policy is not None else None
 workers=policy['workers']if policy is not None else 4
 stats=dict(version=policy['version']if policy is not None else CAPTURE_VERSION,policy=policy,started_at=time.time(),GET_attempts=0,published_documents=0,transient_failures=0,structural_failures=0,maximum_inflight=0,raw_document_byte_limit=workers*MAX_BYTES,oversize_sentinel_byte_limit=workers,replayed_documents=0,reconciled_publications=0,global_state_checkpoints=0)if policy is not None else None
 def remaining():return [(miner,b)for miner,p in sorted(pending.items())if miner not in state['rejections']for b in p['document']['payload']['batches']if str(b['slot'])not in journal.get(miner,{})]
 client=(gateway.bucket.commitment_read_client(parallel_workers=workers)if policy is not None else gateway.bucket.commitment_read_client())if hasattr(gateway.bucket,'commitment_read_client')else gateway.bucket.client
 wal=None;checkpoint_rows=0
 if policy is not None and policy['version']==DURABLE_CAPTURE_VERSION:
  from .capture_journal import CaptureJournal,JournalDurabilityError
  try:
   before_replay=sum(len(v)for v in journal.values())
   wal=CaptureJournal(gateway,epoch);wal.replay()
   stats['replayed_documents']=sum(len(v)for v in journal.values())-before_replay
  except BaseException:
   if wal is not None:wal.close()
   if client is not gateway.bucket.client:client.close()
   raise
 def durable_checkpoint():
  if wal is None:gateway.persist()
  else:
   stats['global_state_checkpoints']+=1;wal.checkpoint(gateway)
 def published(miner,receipt):
  nonlocal checkpoint_rows
  if wal is not None:wal.commit(miner,receipt)
  journal.setdefault(miner,{})[str(receipt['slot'])]=receipt
  state.get('training_document_deferred',{}).get(miner,{}).pop(str(receipt['slot']),None)
  checkpoint_rows+=1
  if wal is None or checkpoint_rows>=policy['state_checkpoint_documents']:
   durable_checkpoint();checkpoint_rows=0

 def fetch(item):
  miner,b=item;key='private/'+epoch+'/training/'+miner+'/'+str(b['slot'])+'.json'
  try:
   if cutoff is not None and time.time()>=cutoff:raise TimeoutError('token capture cutoff')
   response=client.get_object(Bucket=gateway.bucket.name,Key=key);body=response['Body']
   try:data=body.read(MAX_BYTES+1)
   finally:body.close()
   if not state['start']<=response['LastModified'].timestamp()<state['deadline']:raise SubmissionPolicyError('token document upload time')
   try:validate(data,epoch,state['commitment_binding']['checkpoint'],miner,b,transport=state['commitment_binding'].get('version',TRANSPORT))
   except (ValueError,KeyError,TypeError,json.JSONDecodeError)as exc:raise SubmissionPolicyError('token document binding')from exc
   if cutoff is not None and time.time()>=cutoff:raise TimeoutError('token capture cutoff')
   frozen=pending[miner]['root']+'/training/'+str(b['slot'])+'.json'
   captured_at=time.time()
   if cutoff is not None and captured_at>=cutoff:raise TimeoutError('token capture cutoff')
   receipt=dict(slot=b['slot'],sha256=sha(data),size=len(data),frozen_key=frozen,captured_at=captured_at,assurance='unaudited')
   try:
    if wal is not None:wal.intent(miner,receipt)
    gateway.bucket.put(frozen,data)
   except Exception as exc:
    if wal is not None and isinstance(exc,JournalDurabilityError):raise
    failure=RuntimeError('token snapshot publication infrastructure failure');failure.__cause__=exc
    return miner,b,None,None,failure
   return miner,b,None,receipt,None
  except Exception as exc:
   if wal is not None and isinstance(exc,JournalDurabilityError):raise
   from botocore.exceptions import ClientError
   if isinstance(exc,ClientError)and str(exc.response.get('Error',{}).get('Code'))in('NoSuchKey','NotFound','404'):exc=SubmissionPolicyError('missing completed token document')
   return miner,b,None,None,exc
 failures=[]
 try:
  if wal is not None:
   from botocore.exceptions import ClientError
   for miner,receipt in wal.unresolved():
    try:
     response=client.get_object(Bucket=gateway.bucket.name,Key=receipt['frozen_key']);body=response['Body']
     try:data=body.read(MAX_BYTES+1)
     finally:body.close()
    except ClientError as exc:
     if str(exc.response.get('Error',{}).get('Code'))in('NoSuchKey','NotFound','404'):continue
     raise
    b=next(v for v in pending[miner]['document']['payload']['batches']if v['slot']==receipt['slot'])
    validate(data,epoch,state['commitment_binding']['checkpoint'],miner,b,transport=state['commitment_binding'].get('version',TRANSPORT))
    published(miner,receipt);stats['reconciled_publications']+=1
   wal.replay()
  while remaining():
   items=remaining();wave_errors=[]
   def record(result):
    miner,b,data,receipt,error=result
    if isinstance(error,SubmissionPolicyError):
     state['rejections'][miner]=str(error);gateway.persist()
     if stats is not None:stats['structural_failures']+=1
    elif error is not None:
     wave_errors.append(error)
     if stats is not None:stats['transient_failures']+=1
    else:
     published(miner,receipt)
     if stats is not None:stats['published_documents']+=1
   with ThreadPoolExecutor(max_workers=workers)as pool:
    if policy is None:
     # Original f213 FIFO semantics are preserved when no new signed policy exists.
     waiting=deque();todo=iter(items)
     def fill():
      while len(waiting)<4:
       try:item=next(todo)
       except StopIteration:return
       waiting.append(pool.submit(fetch,item))
     fill()
     while waiting:record(waiting.popleft().result());fill()
    else:
     waiting=set();todo=iter(items)
     def fill():
      while len(waiting)<workers and not(cutoff is not None and time.time()>=cutoff):
       try:item=next(todo)
       except StopIteration:return
       waiting.add(pool.submit(fetch,item));stats['GET_attempts']+=1
       stats['maximum_inflight']=max(stats['maximum_inflight'],len(waiting))
     fill()
     while waiting:
      completed,waiting=wait(waiting,return_when=FIRST_COMPLETED)
      for future in completed:record(future.result())
      fill()
   failures.extend(wave_errors)
   if not remaining():break
   if cutoff is None:raise FreezeMetadataIncomplete('bounded signed token capture cutoff required')
   if time.time()>=cutoff:break
   time.sleep(min(.25,max(0,cutoff-time.time())))
  deferred=state.setdefault('training_document_deferred',{})
  for miner,b in remaining():deferred.setdefault(miner,{})[str(b['slot'])]=dict(reason='infrastructure_deferred',closed_at=time.time())
  state.pop('training_document_capture_incomplete',None);state['training_document_capture_complete']=True
  if stats is not None:
   stats.update(finished_at=time.time(),deferred_slots=len(remaining()),cutoff=cutoff,unaudited=True)
   if wal is not None:stats['journal_records']=wal.sequence
   state.setdefault('training_capture_runs',[]).append(stats)
  durable_checkpoint()
  return journal
 finally:
  if client is not gateway.bucket.client:client.close()
  if wal is not None:wal.close()

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
