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
 def test_infrastructure_timeout_does_not_become_fraud_or_exclusion(self):
  gateway,state,puts,calls=self.gateway();gateway.bucket.client.get_object=lambda **kw:(_ for _ in()).throw(TimeoutError('infra'))
  with self.assertRaises(FreezeMetadataIncomplete):capture(gateway,'token-v2')
  self.assertFalse(state['rejections']);self.assertFalse(puts)
if __name__=='__main__':unittest.main()
