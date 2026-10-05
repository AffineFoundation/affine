"""Actual immutable prepared artifacts: resume, integrity and deadline controls."""
import copy,hashlib,json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from nacl.signing import SigningKey
from training_receipt_fixtures import transport_fixture
from subnet.batches import pack,unpack
from subnet.miner import Miner,EpochClosed
from subnet.storage import Identity
from subnet.commitment_transport import VERSION,validate,write_prepared_state,pair_artifact,read_prepared_state

class PreparedMiner(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.state=Path(self.tmp.name)/'epoch.zip';self.identity=Identity();f=transport_fixture(SigningKey.generate());self.rows=unpack(f['data']);self.m=dict(f['manifest'],source_bundle={'sha256':'b'*64},deadline=time.time()+120,submission_transport_policy=VERSION,max_batches=3);self.cap=dict(put_url='COMMIT',batch_put_urls=['PAIR0','PAIR1','PAIR2'],headers={});self.objects={};self.calls=[]
 def miner(self):
  with patch('subnet.miner.check_runtime_profile'):return Miner(self.identity,self.m,'unused',capability=self.cap,state_path=self.state)
 def put(self,url,**kw):
  self.calls.append(url);self.objects[url]=kw['data'];return SimpleNamespace(status_code=200,raise_for_status=lambda:None)
 def test_search_and_repeated_upload_compress_new_pair_only_once(self):
  batch,arrays=self.rows[0];index=batch['index'];fake=SimpleNamespace(spec=SimpleNamespace(version=batch['environment_version']),rollout=lambda i,seed:(copy.deepcopy(batch['rollouts'][seed%2]),arrays[seed%2]))
  first=self.miner()
  with patch('subnet.miner.make_runtime',return_value=fake),patch('subnet.miner.pack',side_effect=AssertionError('legacy cumulative pack')),patch('subnet.commitment_transport.pair_artifact',wraps=pair_artifact)as compress,patch('requests.put',side_effect=self.put):
   first.search(index,seed=0,max_attempts=2,env_id=batch['env_id']);first.upload();first.upload()
   self.assertEqual(compress.call_count,1)
  self.assertEqual(self.calls.count('PAIR0'),1);self.assertEqual(self.calls.count('COMMIT'),2)
 def test_restart_reuses_exact_bytes_without_any_compression_or_old_slot_PUT(self):
  first=self.miner();first.batches=self.rows
  with patch('requests.put',side_effect=self.put):first.upload()
  descriptor=json.loads(self.state.read_bytes());self.assertEqual(descriptor['version'],'prepared-miner-pairs-v1');original=self.objects['PAIR0'];self.calls=[]
  with patch('subnet.commitment_transport.pair_artifact',side_effect=AssertionError('recompression')),patch('subnet.miner.pack',side_effect=AssertionError('cumulativecompression')):
   second=self.miner()
   with patch('requests.put',side_effect=self.put):second.upload()
  self.assertEqual(self.calls,['COMMIT']);self.assertEqual(self.objects['PAIR0'],original)
  claim=validate(self.objects['COMMIT'],self.m['epoch'],self.identity.id,3)['payload']['batches'][0]
  self.assertEqual(claim['sha256'],hashlib.sha256(original).hexdigest())
 def test_partial_PUT_failure_resume_is_same_original_pairs_and_commit_last(self):
  first=self.miner();batch,arrays=self.rows[0];first.batches=[self.rows[0],(dict(batch,index=batch['index']+1,sample_index=batch['index']+1),arrays)]
  def fail(url,**kw):
   if url=='PAIR1':raise RuntimeError('temporaryfailure')
   return self.put(url,**kw)
  with patch('requests.put',side_effect=fail):
   with self.assertRaisesRegex(RuntimeError,'temporaryfailure'):first.upload()
  self.assertNotIn('COMMIT',self.objects);old=self.objects['PAIR0'];self.calls=[]
  second=self.miner()
  with patch('requests.put',side_effect=self.put):second.upload()
  self.assertEqual(self.calls,['PAIR1','COMMIT']);self.assertEqual(self.objects['PAIR0'],old)
 def test_corrupt_immutable_cached_artifact_refuses_before_any_upload(self):
  first=self.miner();first.batches=self.rows
  with patch('requests.put',side_effect=self.put):first.upload()
  d=json.loads(self.state.read_bytes());artifact=self.state.with_name(self.state.name+'.pairs')/(d['pairs'][0]['sha256']+'.zip');artifact.write_bytes(b'x'*artifact.stat().st_size)
  with patch('requests.put')as put:
   with self.assertRaisesRegex(ValueError,'full SHA256'):self.miner()
   put.assert_not_called()
 def test_stale_source_or_sampling_contract_cannot_resume(self):
  first=self.miner();first.batches=self.rows
  with patch('requests.put',side_effect=self.put):first.upload()
  for field,value in [('source_bundle',{'sha256':'c'*64}),('sampling_contract',{'nonce':'changed'})]:
   with patch.dict(self.m,{field:value}),self.assertRaisesRegex(ValueError,'stale local prepared'):read_prepared_state(self.state,self.m)
 def test_legacy_ZIP_state_migrates_once_then_reuses_original_artifacts(self):
  self.state.write_bytes(pack(self.rows));first=self.miner()
  with patch('requests.put',side_effect=self.put):first.upload()
  self.assertEqual(json.loads(self.state.read_bytes())['version'],'prepared-miner-pairs-v1');self.calls=[]
  second=self.miner()
  with patch('subnet.commitment_transport.pair_artifact',side_effect=AssertionError('recompression')),patch('requests.put',side_effect=self.put):second.upload()
  self.assertEqual(self.calls,['COMMIT'])
 def test_symlink_artifact_or_state_index_is_not_followed(self):
  first=self.miner();first.batches=self.rows
  with patch('requests.put',side_effect=self.put):first.upload()
  d=json.loads(self.state.read_bytes());artifact=self.state.with_name(self.state.name+'.pairs')/(d['pairs'][0]['sha256']+'.zip');other=artifact.with_suffix('.saved');artifact.rename(other);artifact.symlink_to(other)
  with self.assertRaisesRegex(ValueError,'size/path'):self.miner()
 def test_expiry_between_pair_ack_and_commit_preserves_original_commit(self):
  first=self.miner();first.batches=self.rows
  original_deadline=self.m['deadline'];old=b'previous-authoritative-commitment';self.objects['COMMIT']=old
  def expire(url,**kw):
   response=self.put(url,**kw);self.m['deadline']=time.time()-1;return response
  with patch('requests.put',side_effect=expire):
   with self.assertRaisesRegex(EpochClosed,'commitment upload deadline'):first.upload()
  self.assertEqual(self.calls,['PAIR0']);self.assertEqual(self.objects['COMMIT'],old)
 def test_resume_rejects_aggregate_compressed_overflow_before_tensor_decode(self):
  first=self.miner();batch,arrays=self.rows[0];first.batches=[self.rows[0],(dict(batch,index=batch['index']+1,sample_index=batch['index']+1),arrays)]
  with patch('requests.put',side_effect=self.put):first.upload()
  value=json.loads(self.state.read_bytes());limit=max(p['size']for p in value['pairs'])+1
  with patch('subnet.artifact_budget.for_manifest',return_value={'compressed_bytes':limit}),patch('subnet.batches.unpack',side_effect=AssertionError('tensor decode before aggregate admission')):
   with self.assertRaisesRegex(ValueError,'aggregate compressed cap'):read_prepared_state(self.state,self.m)
 def test_mutating_existing_batch_cannot_replace_prepared_slot(self):
  first=self.miner();first.batches=self.rows
  with patch('requests.put',side_effect=self.put):first.upload()
  first.batches[0][0]['index']+=1
  with patch('requests.put')as put:
   with self.assertRaisesRegex(ValueError,'batch replacement|prepared batch metadata'):first.upload()
   put.assert_not_called()
if __name__=='__main__':unittest.main()
