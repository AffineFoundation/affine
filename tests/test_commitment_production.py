import base64,copy,hashlib,io,json,tempfile,time,unittest
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from botocore.exceptions import ClientError
from subnet import commitment_transport as c
from subnet.storage import Identity,canonical
from subnet.remote_backend import RemoteController
from subnet.forced_sampling import assurance
class Bucket:
 def __init__(self):self.name='test';self.client=self;self.objects={};self.heavy_reads=0;self.fail_copy=None;self.copies=[]
 def put(self,k,b,content_type=None):self.objects[k]=(b,hashlib.sha256(b).hexdigest(),10)
 def json(self,k,v):self.put(k,canonical(v))
 def get_object(self,Bucket,Key):
  if Key not in self.objects:raise ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject')
  b,e,t=self.objects[Key]
  if Key.endswith('.zip'):self.heavy_reads+=1
  return dict(Body=io.BytesIO(b),ETag=e,ContentLength=len(b),LastModified=datetime.fromtimestamp(t,timezone.utc))
 def head_object(self,Bucket,Key):
  if Key not in self.objects:raise ClientError({'Error':{'Code':'NoSuchKey'}},'HeadObject')
  b,e,t=self.objects[Key];return dict(ETag=e,ContentLength=len(b),LastModified=datetime.fromtimestamp(t,timezone.utc))
 def copy(self,k,d,expected_etag=None):
  if self.fail_copy and self.fail_copy(k):raise RuntimeError('real transient copy refusal')
  if self.objects[k][1]!=expected_etag:raise RuntimeError('ETag changed')
  self.objects[d]=self.objects[k];self.copies.append((k,d))
 def presign(self,k,*a,**kw):return 'https://bucket.r2.cloudflarestorage.com/'+k
class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.state=Path(self.temp.name);self.ids=[Identity(),Identity(),Identity()];self.b=Bucket();self.g=SimpleNamespace(bucket=self.b,epochs={'e':dict(start=0,deadline=20,miners=set(i.id for i in self.ids),max_batches=3,commitment_binding=dict(checkpoint='a'*64,source='b'*64))},persist=lambda:None)
  self.m=dict(epoch='e',checkpoint={'id':'a'*64},source_bundle={'sha256':'b'*64},submission_transport_policy=c.VERSION,start=0,deadline=20,max_batches=3,payable=False,audit_policy=dict(mode='sampled',version='bounded-random-v1',epoch_budget=1,escalation_budget=0,minimum_per_miner=0,maximum_per_miner=3,penalties=dict(invalid_batch_multiplier=0,zero_epoch_after=1,penalize_structural=False)))
 def submit(self,i,index=1):
  batch={'env_id':'math','index':index,'sample_index':index};data=('opaque original proof ZIP '+str(i)).encode();env=c.make(self.ids[i],self.m,[(batch,data)]);self.b.put('private/e/commitments/'+self.ids[i].id+'.json',canonical(env));self.b.put('private/e/staging/'+self.ids[i].id+'/0.zip',data);return batch
 def test_malformed_sibling_never_blocks_honest_freeze(self):
  self.submit(0);self.b.put('private/e/commitments/'+self.ids[1].id+'.json',b'bad');r=c.freeze(self.g,'e');self.assertEqual(set(r),{self.ids[0].id});self.assertIn(self.ids[1].id,self.g.epochs['e']['rejections']);self.assertEqual(self.b.heavy_reads,0)
 def test_transient_copy_retains_original_and_completed_siblings(self):
  self.submit(0);self.submit(1,2);miner=self.ids[1].id;self.b.fail_copy=lambda k:miner in k
  with self.assertRaises(RuntimeError):c.freeze(self.g,'e')
  saved=copy.deepcopy(self.g.epochs['e']['commitment_pending']);self.assertNotIn('frozen_receipts',self.g.epochs['e']);self.b.fail_copy=None;r=c.freeze(self.g,'e');self.assertEqual(len(r),2);self.assertEqual(r[miner]['commitment_document'],saved[miner]['document'])
 def test_zero_allocations_no_job_and_no_heavy_download(self):
  batches={self.ids[i].id:self.submit(i,i+1)for i in range(3)};self.g.freeze=lambda epoch:c.freeze(self.g,epoch);controller=RemoteController.__new__(RemoteController);controller.gateway=self.g;controller.bucket=self.b;controller.state=self.state;controller.signed=lambda x:{'payload':x};calls=[]
  def run(label,role,manifest,cp,submissions):
   calls.append(submissions);audits=[]
   for obj in submissions:
    miner=obj['commitment_miner'];audit=dict(epoch='e',submission_sha256=obj['sha256'],accepted=[batches[miner]],outcomes=[dict(batch=0,index=batches[miner]['index'],env_id='math',valid=True,fully_audited=True)],sampling_assurance=assurance(manifest));audits.append(audit)
   return dict(audits=audits,job_id='ACTUAL-FAKE-QUEUE-JOB',backend_profile={},execution_resources_enforced=True)
  controller.jobs=SimpleNamespace(run=run,queue=object(),verifiers=[1,2]);result,reports=controller.finalize(self.m,'/unused');self.assertEqual(len(calls),1);self.assertEqual(len(calls[0]),1);self.assertEqual(self.b.heavy_reads,0);self.assertEqual(sum(result['points'].values()),1);self.assertEqual(sum('remote_job_id'in r for r in reports.values()),1);self.assertEqual(sum(r['commitment_status']=='not_selected'for r in reports.values()),2)
 def test_integer_fake_validity_refused(self):
  self.submit(0);receipt=c.freeze(self.g,'e')[self.ids[0].id];manifest=dict(self.m,audit_seed='c'*64);remote=dict(job_id='real',backend_profile={},execution_resources_enforced=True,audits=[dict(submission_sha256=receipt['artifacts'][0]['sha256'],accepted=[],outcomes=[{'valid':1}])])
  with self.assertRaisesRegex(ValueError,'boolean'):c.combine(manifest,receipt,remote)
 def test_claimed_model_swap_excluded_not_global_deadlock(self):
  self.submit(0);identity=self.ids[1];wrong=dict(self.m,checkpoint={'id':'c'*64});env=c.make(identity,wrong,[]);self.b.put('private/e/commitments/'+identity.id+'.json',canonical(env));self.assertEqual(set(c.freeze(self.g,'e')),{self.ids[0].id})
 def test_nested_original_receipt_still_admits_without_trainer_reverification(self):
  from nacl.signing import SigningKey
  from training_receipt_fixtures import transport_fixture,signed_receipt
  from subnet import training_receipts as r
  key=SigningKey.generate();fixture=transport_fixture(key);m=copy.deepcopy(fixture['manifest']);m['submission_transport_policy']=c.VERSION;frozen=fixture['frozen'];m['audit_frozen_receipts']={fixture['miner']:{'sha256':'a'*64,'artifacts':[frozen]}}
  receipt,audit,job,request=signed_receipt(key,m,fixture['miner'],frozen,fixture['batch']);obj=dict(fixture['submission'],verifier_receipt=receipt);path=self.state/'actual-original.zip';path.write_bytes(fixture['data'])
  with patch('subnet.model.Runtime.verify',side_effect=AssertionError('must not reverify')),patch('subnet.backend_jobs.audit',side_effect=AssertionError('must not reverify')):
   summary,pairs=r.admitted_submission(path,obj,m,key.verify_key.encode().hex())
  self.assertEqual(len(pairs),1);self.assertIs(summary['trainer_verification_performed'],False)
 def test_new_compact_population_can_include_two_original_artifacts_from_one_miner(self):
  from subnet import compact_training_inputs as compact
  with patch.object(compact,'validate_receipt',side_effect=lambda e,o,m,a:({}, {'miner_identity':'e'*64,'submission_sha256':o['sha256']})):
   job=dict(role='train',training_policy='bf16-cpu-fp32-master-task-normalized-persistent-v4',training_input_policy=compact.VERSION,source_files={'subnet/compact_training_inputs.py':'a'*64,'subnet/training_receipts.py':'b'*64},submissions=[{'sha256':'a'*64},{'sha256':'b'*64}]);m=dict(training_policy=job['training_policy'],training_input_policy=compact.VERSION,submission_transport_policy=c.VERSION)
   compact.validate_job(job,m,'unused')
   job['submissions'].append({'sha256':'a'*64})
   with self.assertRaises(ValueError):compact.validate_job(job,m,'unused')
 def test_real_miner_uploads_per_pair_then_signed_small_commitment(self):
  from nacl.signing import SigningKey
  from training_receipt_fixtures import transport_fixture
  from subnet.miner import Miner
  from subnet.batches import unpack
  fixture=transport_fixture(SigningKey.generate());m=Miner.__new__(Miner);m.identity=self.ids[0];m.manifest=dict(fixture['manifest'],source_bundle={'sha256':'b'*64},deadline=time.time()+100,submission_transport_policy=c.VERSION);m.state_path=None;m.batches=unpack(fixture['data']);m.cap=dict(put_url='COMMITMENT-ENDPOINT',batch_put_urls=['PAIR-ENDPOINT'],headers={});calls=[]
  def put(url,data,**kw):calls.append((url,data));return SimpleNamespace(status_code=200,raise_for_status=lambda:None)
  with patch('subnet.miner.requests.put',side_effect=put):self.assertEqual(m.upload(),200)
  self.assertEqual([url for url,data in calls],['PAIR-ENDPOINT','COMMITMENT-ENDPOINT']);doc=c.validate(calls[1][1],m.manifest['epoch'],m.identity.id,3);self.assertEqual(doc['payload']['batches'][0]['sha256'],hashlib.sha256(calls[0][1]).hexdigest());self.assertLess(len(calls[1][1]),c.MAX_BYTES)
 def test_unaudited_reader_refuses_any_credit_job_or_penalty_mutation(self):
  self.submit(0);receipt=c.freeze(self.g,'e')[self.ids[0].id];m=dict(self.m,audit_seed='c'*64);report=c.unchecked(m,receipt);self.assertTrue(c.validate_unchecked(m,receipt,report))
  for field,value in [('accepted',[{'env_id':'math','index':1}]),('remote_job_id','forged'),('outcomes',[{'batch':0,'valid':False,'fully_audited':True,'failure_kind':'confirmed_invalid'}])]:
   mutated=copy.deepcopy(report);mutated[field]=value
   with self.assertRaises(ValueError):c.validate_unchecked(m,receipt,mutated)
 def test_owned_miner_real_local_signer_and_per_pair_upload(self):
  from nacl.signing import SigningKey
  from training_receipt_fixtures import transport_fixture
  from subnet.backend_jobs import owned_commitment_upload
  f=transport_fixture(SigningKey.generate());path=self.state/'miner-only.seed';path.write_text(self.ids[0].key.encode().hex());path.chmod(0o600)
  manifest=dict(f['manifest'],source_bundle={'sha256':'b'*64},deadline=time.time()+100,submission_transport_policy=c.VERSION)
  job=dict(miner_id=self.ids[0].id,miner_identity_file=str(path),capability=dict(put_url='SMALL',batch_put_urls=['HEAVY'],headers={}))
  calls=[]
  with patch('requests.put',side_effect=lambda url,**kw:(calls.append((url,kw['data']))or SimpleNamespace(status_code=200))):owned_commitment_upload(job,manifest)(f['data'],60)
  self.assertEqual([x[0]for x in calls],['HEAVY','SMALL']);doc=c.validate(calls[1][1],manifest['epoch'],self.ids[0].id,3);self.assertEqual(doc['payload']['batches'][0]['sha256'],hashlib.sha256(calls[0][1]).hexdigest())
  job['miner_id']=self.ids[1].id
  with self.assertRaisesRegex(ValueError,'signer binding'):owned_commitment_upload(job,manifest)
 def test_actual_freeze_cutoff_preserves_completed_sibling_and_no_fraud(self):
  self.submit(0);self.submit(1,2);miner=self.ids[1].id;self.b.fail_copy=lambda k:miner in k
  with self.assertRaises(RuntimeError):c.freeze(self.g,'e')
  self.g.epochs['e']['commitment_binding']['freeze_until']=25
  with patch('subnet.commitment_transport.time.time',return_value=30):r=c.freeze(self.g,'e')
  self.assertEqual(set(r),{self.ids[0].id});self.assertIn(miner,self.g.epochs['e']['commitment_deferred']);self.assertNotIn(miner,self.g.epochs['e']['rejections']);self.assertEqual(self.b.heavy_reads,0)
 def test_deferred_cutoff_cannot_claim_credit_fraud_or_early_close(self):
  self.submit(0);receipt=c.freeze(self.g,'e')[self.ids[0].id]
  m=dict(self.m,audit_seed='c'*64,hourly_execution_policy=dict(version='bounded-hourly-phases-v1',mine_seconds=20,freeze_seconds=10,audit_seconds=10,train_publication_seconds=1200,weight_seconds=300,slack_seconds=600))
  with self.assertRaisesRegex(ValueError,'cutoff'):c.deferred(m,receipt,'budget_deferred',39)
  report=c.deferred(m,receipt,'budget_deferred',40);self.assertTrue(c.validate_deferred(m,receipt,report))
  for field,value in [('accepted',[{}]),('remote_job_id','fake'),('outcomes',[dict(valid=False,fully_audited=True,failure_kind='confirmed_invalid')])]:
   bad=dict(report);bad[field]=value
   with self.assertRaises(ValueError):c.validate_deferred(m,receipt,bad)
 def test_hourly_policy_rejects_oversubscribed_or_bool_budgets(self):
  from subnet.hourly_policy import validate
  p=dict(version='bounded-hourly-phases-v1',mine_seconds=600,freeze_seconds=300,audit_seconds=600,train_publication_seconds=1200,weight_seconds=300,slack_seconds=600)
  self.assertEqual(sum(v for v in validate(p,600).values()if type(v)is int),3600)
  for change in [dict(slack_seconds=601),dict(mine_seconds=True),dict(freeze_seconds=0)]:
   with self.assertRaises(ValueError):validate(dict(p,**change),600)
 def test_temporary_exclusion_only_repeated_confirmed_invalid_epochs(self):
  from subnet.audit_exclusion import snapshot
  p=dict(version='confirmed-invalid-temporary-exclusion-v1',threshold_epochs=2,lookback_epochs=10,exclusion_epochs=2)
  invalid=dict(outcomes=[dict(valid=False,fully_audited=True,failure_kind='confirmed_invalid')]);miner=self.ids[0].id
  h=dict(version='authenticated-confirmed-invalid-history-v1',epochs=[dict(epoch='one',reports={miner:invalid})]);self.assertEqual(snapshot(h,p),[])
  h['epochs'].append(dict(epoch='two',reports={miner:invalid}));self.assertEqual(snapshot(h,p),[miner]);h['epochs'].append(dict(epoch='three',reports={}));self.assertEqual(snapshot(h,p),[miner]);h['epochs'].append(dict(epoch='four',reports={}));self.assertEqual(snapshot(h,p),[])
  for kind in ['verification_error','budget_deferred','structural_invalid','not_selected']:
   bad=dict(version=h['version'],epochs=[dict(epoch='fake',reports={miner:dict(outcomes=[dict(valid=False,fully_audited=True,failure_kind=kind)])})])
   with self.assertRaises(ValueError):snapshot(bad,p)
 def test_original_queue_request_retained_when_audit_observation_closes(self):
  from unittest.mock import Mock
  from subnet.role_router import RoutedJobs
  router=RoutedJobs.__new__(RoutedJobs);router.state=self.state;router.config={};router.metadata={};router.history_prefix='history';router.controller=SimpleNamespace(signed=lambda x:{'payload':x},bucket=self.b);router.queue=Mock();router.queue.status.return_value={'status':'running'};router.verifiers=[Mock()]
  manifest=dict(epoch='e',payable=False,checkpoint={'id':'a'*64})
  # Original request is enqueued once, then observation closes with the lease live.
  with patch('subnet.role_router.time.time',side_effect=[30,30,30,41]):
   with self.assertRaisesRegex(TimeoutError,'original lease retained'):router.run('real-label','verify',manifest,submissions=[],observe_until=40)
  original=json.loads((self.state/'real-label.json').read_text());request=(self.state/(original['job_id']+'-job.json')).read_bytes();self.assertEqual(router.queue.enqueue.call_count,1)
  with patch('subnet.role_router.time.time',return_value=45):
   with self.assertRaisesRegex(TimeoutError,'original request retained'):router.run('real-label','verify',manifest,submissions=[],observe_until=40)
  self.assertEqual(json.loads((self.state/'real-label.json').read_text())['job_id'],original['job_id']);self.assertEqual((self.state/(original['job_id']+'-job.json')).read_bytes(),request);self.assertEqual(router.queue.enqueue.call_count,1)
 def test_failed_second_owned_commitment_preserves_exact_prior_pair_bytes(self):
  from nacl.signing import SigningKey
  from training_receipt_fixtures import transport_fixture
  from subnet.backend_jobs import owned_commitment_upload
  from subnet.batches import pack,unpack
  import requests
  f=transport_fixture(SigningKey.generate());rows=unpack(f['data']);batch,arrays=rows[0];other=copy.deepcopy(batch);other['index']+=1;other['sample_index']+=1
  path=self.state/'miner.seed';path.write_text(self.ids[0].key.encode().hex());path.chmod(0o600);journal=self.state/'actual-upload-progress.json'
  m=dict(f['manifest'],source_bundle={'sha256':'b'*64},deadline=time.time()+100,submission_transport_policy=c.VERSION)
  job=dict(miner_id=self.ids[0].id,miner_identity_file=str(path),capability=dict(put_url='SMALL',batch_put_urls=['SLOT0','SLOT1'],headers={}))
  objects={};calls=[];fail=[False]
  def put(url,**kw):
   calls.append(url)
   if fail[0]and url=='SMALL':raise requests.ConnectionError('actual failed next commitment upload')
   objects[url]=kw['data'];return SimpleNamespace(status_code=200)
  with patch('requests.put',side_effect=put):
   uploader=owned_commitment_upload(job,m,journal);uploader(pack([(batch,arrays)]),60);first_commit=objects['SMALL'];first_bytes=objects['SLOT0'];first_doc=c.validate(first_commit,m['epoch'],self.ids[0].id,3)
   fail[0]=True
   with self.assertRaises(requests.ConnectionError):uploader(pack([(batch,arrays),(other,arrays)]),60)
   self.assertEqual(objects['SMALL'],first_commit);self.assertEqual(objects['SLOT0'],first_bytes);self.assertEqual(first_doc['payload']['batches'][0]['sha256'],hashlib.sha256(objects['SLOT0']).hexdigest());self.assertEqual(calls.count('SLOT0'),1)
   # Restart reads genuine acknowledged hashes and still skips the old object.
   fail[0]=False;owned_commitment_upload(job,m,journal)(pack([(batch,arrays),(other,arrays)]),60);self.assertEqual(calls.count('SLOT0'),1);self.assertEqual(calls.count('SLOT1'),1)
 def test_stable_pair_framing_keeps_model_arrays_and_rejects_slot_replacement(self):
  from nacl.signing import SigningKey
  from training_receipt_fixtures import transport_fixture
  from subnet.batches import unpack
  import numpy as np
  f=transport_fixture(SigningKey.generate());batch,arrays=unpack(f['data'])[0];m=dict(f['manifest'],source_bundle={'sha256':'b'*64})
  with patch('zipfile.time.localtime',return_value=(2020,1,1,1,1,1,0,0,0)):first=c.pair_artifact(batch,arrays,m)
  with patch('zipfile.time.localtime',return_value=(2026,10,5,2,3,4,0,0,0)):second=c.pair_artifact(batch,arrays,m)
  self.assertEqual(first,second);actual,actual_arrays=unpack(first)[0];self.assertEqual(actual,batch)
  for a,b in zip(arrays,actual_arrays):
   for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
  j=c.UploadJournal(m);j.acknowledge(0,first)
  with self.assertRaisesRegex(ValueError,'append-only'):j.known(0,first+b'changed')
if __name__=='__main__':unittest.main()
