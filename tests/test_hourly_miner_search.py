"""Bounded task scheduling and exact canonical one-pass commitment controls."""
import base64,hashlib,io,json,tempfile,time,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from subnet.backend_jobs import mine_cumulative,owned_commitment_upload
from subnet.batches import pack,unpack,UploadBudgetExceeded
from subnet.commitment_transport import pair_artifact,validate
from subnet.storage import Identity,canonical

class Search(unittest.TestCase):
 def setUp(self):
  self.now=10;self.calls=[];self.uploads=[];self.progress=[]
  self.m=dict(start=0,deadline=100,epoch='nonpayable',checkpoint={'id':'a'*64},K=1,L=1,max_batches=1,sampling_contract={'max_attempts':128})
  self.job=dict(seed_start=0,search_budget=128)
  self.definitions=[dict(env_id='math',spec={},harness={},indices=[0,1])]
  self.runtime=SimpleNamespace(spec=SimpleNamespace(version='original'))
  self.runtime.for_environment=lambda *args:self.runtime
 def run_search(self,fn,**kwargs):
  def rollout(index,seed):
   self.calls.append((index,seed));self.now+=1
   label=fn(index,seed)
   return {'classification':label,'turns':[{'output':[index,seed]}]},[]
  self.runtime.rollout=rollout
  with patch('subnet.protocol.entries',return_value=self.definitions),patch('subnet.batches.pack',side_effect=lambda rows:canonical([b for b,a in rows])):
   return mine_cumulative(self.runtime,self.m,self.job,lambda data,timeout:self.uploads.append(data),clock=lambda:self.now,progress=self.progress.append,**kwargs)
 def test_one_label_first_task_cannot_starve_fast_later_pair(self):
  data,result=self.run_search(lambda i,s:'negative'if i==0 or s%2 else'positive')
  self.assertEqual(self.calls,[(0,0),(0,1),(1,0),(1,1)])
  self.assertEqual(json.loads(data)[0]['index'],1);self.assertEqual(result['batches'],1)
 def test_previous_class_pool_survives_next_sweep_without_reused_seed(self):
  self.definitions[0]['indices']=[0]
  data,result=self.run_search(lambda i,s:'negative'if s==3 else'positive')
  self.assertEqual(self.calls,[(0,0),(0,1),(0,2),(0,3)])
  self.assertEqual([r['turns'][0]['output'][1]for r in json.loads(data)[0]['rollouts']],[0,3])
 def test_each_task_attempt_namespace_and_signed_limit_unchanged(self):
  self.m['deadline']=1000
  data,result=self.run_search(lambda i,s:'positive',allow_empty=True)
  self.assertIsNone(data)
  for i in (0,1):self.assertEqual([s for j,s in self.calls if j==i],list(range(128)))
  self.assertEqual([r['attempts']for r in result['search']],[128,128])
 def test_late_finishing_rollout_never_uploads(self):
  def classify(i,s):
   if s==1:self.now=100
   return 'positive'if s==0 else'negative'
  data,result=self.run_search(classify,allow_empty=True)
  self.assertIsNone(data);self.assertEqual(self.uploads,[]);self.assertTrue(result['search_stopped_at_deadline'])
 def test_safe_progress_records_counts_not_tokens_arrays_or_secrets(self):
  self.run_search(lambda i,s:'positive'if s==0 else'negative')
  self.assertTrue(any(x['phase']=='attempt_started'for x in self.progress))
  self.assertTrue(any(x['phase']=='attempt_completed'for x in self.progress))
  self.assertTrue(any(x['phase']=='artifact_pack_completed'for x in self.progress))
  self.assertFalse(any(k in x for x in self.progress for k in ('output','prompt','text','arrays','logprobs','proofs','put_url','seed_material')))
 def test_owned_frontier_policy_cannot_exceed_eight_or_change_dwell_bound(self):
  for field,value in [('active_tasks',9),('dwell_attempts',3),('active_tasks',True)]:
   self.job['owned_search_policy']=dict(version='bounded-round-robin-v1',active_tasks=8,dwell_attempts=2);self.job['owned_search_policy'][field]=value
   with self.assertRaisesRegex(ValueError,'bounded owned search'):self.run_search(lambda i,s:'positive')
 def test_frontier_does_not_instantiate_unbounded_harnesses(self):
  self.definitions[0]['indices']=list(range(20));self.m['deadline']=10000;self.job['search_budget']=2
  data,result=self.run_search(lambda i,s:'positive',allow_empty=True)
  self.assertEqual([i for i,s in self.calls if s==0],list(range(20)))
  self.assertEqual(len(self.calls),40);self.assertIsNone(data)

class RuntimeTelemetry(unittest.TestCase):
 def test_phase_timing_does_not_change_rollout_tokens_or_proofs(self):
  from subnet.gpu_runtime import GPURuntime
  events=[];session=SimpleNamespace(reset=lambda *a:dict(messages=[],task_hash='task'),step=lambda action:dict(observations=[],done=True,reward=1,classification='positive'),close=lambda:None)
  runtime=SimpleNamespace(spec=SimpleNamespace(id='math',version='v1',config={},max_turns=1),harness={'max_output_tokens':8},model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=128)),prompt=lambda *a:[7],sample_output=lambda *a:[8,9],tokenizer=SimpleNamespace(decode=lambda *a,**kw:'safe'),compute=lambda *a:('acts','probs'),build_proofs=lambda *a,**kw:['proof'],sampling_receipt=lambda seed:{},mining_progress=lambda phase,**metrics:events.append((phase,metrics)))
  with patch('subnet.gpu_runtime.create_session',return_value=session),patch('subnet.gpu_runtime.policy.action',return_value='action'):
   rollout,arrays=GPURuntime.rollout(runtime,0,1)
  self.assertEqual(rollout['turns'][0]['output'],[8,9]);self.assertEqual(rollout['turns'][0]['proofs'],['proof']);self.assertEqual(arrays,['probs'])
  self.assertEqual([p for p,m in events],['generation_started','generation_completed','probabilities_completed','TOPLOC_completed','grading_completed'])
  self.assertFalse(any(k in m for p,m in events for k in ('output','prompt','text','proofs','arrays','logprobs')))

class Artifacts(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.identity=Identity();p=Path(self.temp.name)/'miner.seed';p.write_text(self.identity.key.encode().hex());p.chmod(0o600)
  self.m=dict(epoch='nonpayable',checkpoint={'id':'a'*64},source_bundle={'sha256':'b'*64},deadline=time.time()+60,max_batches=3)
  self.job=dict(miner_id=self.identity.id,miner_identity_file=str(p),capability=dict(put_url='COMMIT',batch_put_urls=['SLOT0','SLOT1','SLOT2'],headers={}))
  self.batch={'env_id':'math','index':3,'rollouts':[{'output':[1]},{'output':[2]}]}
  self.arrays=[[np.random.default_rng(1).normal(size=(16,32)).astype(np.float32)],[np.zeros((16,32),dtype=np.float32)]]
 def historical(self,batch,arrays):
  old=pack([(batch,arrays)]);out=io.BytesIO()
  with zipfile.ZipFile(io.BytesIO(old))as source,zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED)as target:
   for entry in source.infolist():
    info=zipfile.ZipInfo(entry.filename,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=0o600<<16;target.writestr(info,source.read(entry.filename))
  return out.getvalue()
 def test_structured_pair_cannot_bypass_original_tensor_admission(self):
  for bad in (np.zeros((2,3),dtype=np.float64),np.zeros((1,1,1),dtype=np.float32),np.zeros((513,3),dtype=np.float32)):
   with self.assertRaisesRegex(ValueError,'stable tensor'):pair_artifact(self.batch,[[bad]],self.m)
 def test_one_pass_bytes_equal_exact_historical_canonical_artifact(self):
  actual=pair_artifact(self.batch,self.arrays,self.m);self.assertEqual(actual,self.historical(self.batch,self.arrays))
  decoded,arrays=unpack(actual)[0];self.assertEqual(decoded,self.batch)
  for aa,bb in zip(arrays,self.arrays):
   for a,b in zip(aa,bb):np.testing.assert_array_equal(a,b)
 def test_prepared_path_never_unpacks_or_repacks_and_commits_actual_hash(self):
  upload=owned_commitment_upload(self.job,self.m);artifact=upload.prepare_pair(self.batch,self.arrays);objects={}
  with patch('subnet.batches.unpack',side_effect=AssertionError('unexpected unpack')),patch('subnet.batches.pack',side_effect=AssertionError('unexpected repack')),patch('requests.put',side_effect=lambda url,**kw:(objects.update({url:kw['data']})or SimpleNamespace(status_code=200))):
   data=upload.upload_pairs([(self.batch,artifact)],60)
  self.assertEqual(data,objects['COMMIT']);self.assertEqual(objects['SLOT0'],artifact)
  claim=validate(data,self.m['epoch'],self.identity.id,3)['payload']['batches'][0];self.assertEqual(claim['sha256'],hashlib.sha256(artifact).hexdigest())
 def test_cumulative_caps_preserved_even_when_individual_pairs_fit(self):
  upload=owned_commitment_upload(self.job,self.m);a=upload.prepare_pair(self.batch,self.arrays);b=dict(self.batch,index=4);bb=upload.prepare_pair(b,self.arrays)
  for limits in ({'compressed_bytes':len(a)+1,'raw_bytes':10**9},{'compressed_bytes':10**9,'raw_bytes':1}):
   with patch('subnet.artifact_budget.for_manifest',return_value=limits),patch('requests.put')as put:
    with self.assertRaises(UploadBudgetExceeded):upload.upload_pairs([(self.batch,a),(b,bb)],60)
    put.assert_not_called()
 def test_two_digit_slots_cannot_expand_historical_cumulative_raw_budget(self):
  job=dict(self.job,capability=dict(self.job['capability'],batch_put_urls=['SLOT'+str(i)for i in range(12)]))
  upload=owned_commitment_upload(job,self.m);rows=[(dict(self.batch,index=i),self.arrays)for i in range(12)]
  prepared=[(b,upload.prepare_pair(b,a))for b,a in rows]
  old=pack(rows)
  with zipfile.ZipFile(io.BytesIO(old))as archive:oldraw=sum(e.file_size for e in archive.infolist())
  with patch('subnet.artifact_budget.for_manifest',return_value=dict(raw_bytes=oldraw-1,compressed_bytes=10**9)),patch('requests.put')as put:
   with self.assertRaises(UploadBudgetExceeded):upload.upload_pairs(prepared,60)
   put.assert_not_called()
 def test_partial_pair_put_failure_never_commits_or_rewrites_acknowledged_slot(self):
  upload=owned_commitment_upload(self.job,self.m);a=upload.prepare_pair(self.batch,self.arrays);b=dict(self.batch,index=4);bb=upload.prepare_pair(b,self.arrays);calls=[]
  def put(url,**kw):
   calls.append(url)
   if url=='SLOT1':raise RuntimeError('transport interruption')
   return SimpleNamespace(status_code=200)
  with patch('requests.put',side_effect=put):
   with self.assertRaisesRegex(RuntimeError,'transport interruption'):upload.upload_pairs([(self.batch,a),(b,bb)],60)
  self.assertEqual(calls,['SLOT0','SLOT1'])
  with patch('requests.put',side_effect=lambda url,**kw:(calls.append(url)or SimpleNamespace(status_code=200))):upload.upload_pairs([(self.batch,a),(b,bb)],60)
  self.assertEqual(calls.count('SLOT0'),1);self.assertEqual(calls[-1],'COMMIT')
if __name__=='__main__':unittest.main()
