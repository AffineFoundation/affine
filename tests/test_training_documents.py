import copy,datetime,io,json,time,unittest
from types import SimpleNamespace
from unittest.mock import patch
from subnet.storage import Identity
from subnet.commitment_transport import make,validate,canonical,sha,VERSION,VERSION2,FreezeMetadataIncomplete
from subnet.training_documents import document,validate as validate_document,capture,attach,freeze_receipts

class DocumentTests(unittest.TestCase):
 def setUp(self):
  self.identity=Identity(b'k'*32);self.manifest={'epoch':'token-v2','checkpoint':{'id':'a'*64},'source_bundle':{'sha256':'b'*64},'submission_transport_policy':VERSION2}
  self.batch={'env_id':'math','index':4,'samples':[{'tokens':[1,2,3],'reward':True},{'tokens':[1,8],'reward':False}]};self.packed=[(self.batch,b'proof-zip')]
  self.env=make(self.identity,self.manifest,self.packed);self.entry=self.env['payload']['batches'][0];self.data=document(self.batch,self.manifest,self.identity.id,0)
 def test_exact_canonical_signed_v2_and_independent_documents(self):
  self.assertEqual(validate(canonical(self.env),'token-v2',self.identity.id),self.env)
  self.assertEqual(validate_document(self.data,'token-v2','a'*64,self.identity.id,self.entry)['batch'],self.batch)
  self.assertNotEqual(self.entry['training_sha256'],self.entry['sha256'])
 def test_legacy_framing_unchanged(self):
  old=make(self.identity,dict(self.manifest,submission_transport_policy=VERSION),self.packed)
  self.assertNotIn('training_sha256',old['payload']['batches'][0]);validate(canonical(old),'token-v2',self.identity.id)
 def test_token_hash_scope_size_and_bool_slot_fail(self):
  for key,value in [('epoch','foreign'),('checkpoint','c'*64),('miner','d'*64),('slot',True),('batch',dict(self.batch,index=5))]:
   bad=json.loads(self.data);bad[key]=value;data=canonical(bad);entry=dict(self.entry,training_sha256=sha(data),training_size=len(data))
   with self.assertRaises(ValueError):validate_document(data,'token-v2','a'*64,self.identity.id,entry)
  with self.assertRaises(ValueError):validate_document(self.data+b' ','token-v2','a'*64,self.identity.id,self.entry)
 def test_oversized_tokens_not_committable(self):
  with self.assertRaises(ValueError):document(dict(self.batch,x='x'*2_000_000),self.manifest,self.identity.id,0)
 def gateway(self,data=None):
  now=time.time();miner=self.identity.id;state=dict(start=now-10,deadline=now+10,commitment_binding={'checkpoint':'a'*64,'freeze_until':now+20},commitment_capture_complete=True,commitment_pending={miner:{'document':self.env,'root':'public/token/root','sha256':sha(canonical(self.env)),'size':len(canonical(self.env)),'received_at':now}},rejections={})
  puts={};calls=[]
  def get(**kwargs):calls.append(kwargs['Key']);return {'Body':io.BytesIO(self.data if data is None else data),'LastModified':datetime.datetime.fromtimestamp(now,datetime.timezone.utc)}
  bucket=SimpleNamespace(name='bucket',client=SimpleNamespace(get_object=get),put=lambda k,b:puts.setdefault(k,b),json=lambda k,b:None);gateway=SimpleNamespace(epochs={'token-v2':state},bucket=bucket,persist=lambda:None)
  return gateway,state,puts,calls
 def test_capture_actual_fullSHA_immutable_resume_without_network(self):
  gateway,state,puts,calls=self.gateway();capture(gateway,'token-v2');self.assertEqual(len(calls),1);self.assertEqual(list(puts.values()),[self.data]);capture(gateway,'token-v2');self.assertEqual(len(calls),1)
  receipts={self.identity.id:{'artifacts':[self.entry]}};attach(state,receipts);row=receipts[self.identity.id]['training_documents'][0];self.assertEqual(row['assurance'],'unaudited');self.assertEqual(row['sha256'],sha(self.data));self.assertNotIn('verified',row)
 def test_v2_receipts_never_fetch_or_claim_frozen_heavy_proofs(self):
  gateway,state,puts,calls=self.gateway();capture(gateway,'token-v2');receipts=freeze_receipts(gateway,'token-v2')
  proof=receipts[self.identity.id]['artifacts'][0];self.assertEqual(proof['proof_capture_status'],'declared-not-captured');self.assertNotIn('etag',proof);self.assertNotIn('read_url',proof);self.assertEqual(len(calls),1)
 def test_changed_persisted_snapshot_hash_is_rejected(self):
  gateway,state,puts,calls=self.gateway();capture(gateway,'token-v2');state['training_document_snapshots'][self.identity.id]['0']['sha256']='f'*64
  with self.assertRaises(ValueError):freeze_receipts(gateway,'token-v2')
 def test_missing_document_is_structural_not_population_deadlock(self):
  from botocore.exceptions import ClientError
  gateway,state,puts,calls=self.gateway();gateway.bucket.client.get_object=lambda **kw:(_ for _ in()).throw(ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject'))
  capture(gateway,'token-v2');self.assertIn(self.identity.id,state['rejections']);self.assertEqual(freeze_receipts(gateway,'token-v2'),{})
 def test_corrupt_document_excludes_only_miner_without_fraud_claim(self):
  gateway,state,puts,calls=self.gateway(b'corrupt');capture(gateway,'token-v2');self.assertIn(self.identity.id,state['rejections']);self.assertFalse(puts);self.assertNotIn('confirmed_invalid',str(state))
 def test_infrastructure_timeout_deferred_without_fraud_or_blocking_captured_sibling(self):
  gateway,state,puts,calls=self.gateway();state['commitment_binding']['freeze_until']=time.time()+.001
  gateway.bucket.client.get_object=lambda **kw:(_ for _ in()).throw(TimeoutError('infra'))
  capture(gateway,'token-v2');receipts=freeze_receipts(gateway,'token-v2')
  self.assertFalse(state['rejections']);self.assertEqual(receipts[self.identity.id]['training_documents'],[])
  self.assertEqual(receipts[self.identity.id]['training_document_deferred_slots'],[0]);self.assertEqual(receipts[self.identity.id]['training_document_capture_status'],'infrastructure_deferred')
 def test_transient_infra_retries_original_slot_and_then_captures(self):
  gateway,state,puts,calls=self.gateway();original=gateway.bucket.client.get_object;count=[0]
  def get(**kw):
   count[0]+=1
   if count[0]==1:raise TimeoutError('retry')
   return original(**kw)
  gateway.bucket.client.get_object=get
  with patch('subnet.training_documents.time.sleep'):capture(gateway,'token-v2')
  self.assertEqual(count[0],2);self.assertEqual(len(puts),1);self.assertFalse(state.get('training_document_deferred'))
 def test_bounded_parallel_publication_and_serial_durable_journal(self):
  import threading
  gateway,state,puts,calls=self.gateway();miner=self.identity.id;documents={};packed=[]
  for slot in range(9):
   batch=copy.deepcopy(self.batch);batch['index']+=slot;packed.append((batch,b'proof'+bytes([slot])))
   documents[slot]=document(batch,self.manifest,miner,slot)
  env=make(self.identity,self.manifest,packed);pending=state['commitment_pending'][miner]
  pending.update(document=env,sha256=sha(canonical(env)),size=len(canonical(env)))
  now=time.time();lock=threading.Lock();active=[0];peak=[0];first_four=threading.Event();threads=[]
  def get(**kw):
   slot=int(kw['Key'].rsplit('/',1)[1].split('.')[0]);calls.append(slot)
   return {'Body':io.BytesIO(documents[slot]),'LastModified':datetime.datetime.fromtimestamp(now,datetime.timezone.utc)}
  def put(key,body):
   with lock:
    active[0]+=1;peak[0]=max(peak[0],active[0])
    if active[0]==4:first_four.set()
   try:
    if not first_four.wait(2):raise AssertionError('publication was serialized')
    time.sleep(.005);puts[key]=body
   finally:
    with lock:active[0]-=1
  gateway.bucket.client.get_object=get;gateway.bucket.put=put
  gateway.persist=lambda:threads.append(threading.get_ident())
  capture(gateway,'token-v2')
  self.assertEqual(peak[0],4);self.assertEqual(len(puts),9)
  self.assertEqual(set(threads),{threading.get_ident()})
  self.assertEqual(len(state['training_document_snapshots'][miner]),9)
  capture(gateway,'token-v2');self.assertEqual(len(calls),9)
 def test_failed_publication_does_not_journal_before_retry(self):
  gateway,state,puts,calls=self.gateway();original=gateway.bucket.put;attempts=[0]
  def put(key,data):
   attempts[0]+=1
   if attempts[0]==1:
    from subnet.storage import SubmissionPolicyError
    self.assertFalse(state['training_document_snapshots']);raise SubmissionPolicyError('publication unavailable')
   return original(key,data)
  gateway.bucket.put=put
  with patch('subnet.training_documents.time.sleep'):capture(gateway,'token-v2')
  self.assertEqual(attempts[0],2);self.assertEqual(len(calls),2)
  self.assertEqual(len(puts),1);self.assertFalse(state['rejections'])
if __name__=='__main__':unittest.main()
