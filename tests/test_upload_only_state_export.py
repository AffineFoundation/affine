"""New signed transport only; actual tiny FP32 bytes and authority gate controls."""
import copy,json,unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace
from test_parallel_persistent_state import ParallelState
from test_parallel_persistent_publication import PublicationControls
from test_remote_state_commit import RemoteAdmission
from subnet.persistent_publication import export_policy,EXPORT_POLICY,complete
from subnet.persistent_training_state import export_state,restore_state

class UploadOnlyExport(unittest.TestCase):
 def fixture(self):
  f=ParallelState();f.setUp();self.addCleanup(f.doCleanups);return f
 def test_actual_upload_only_bytes_restore_but_never_claim_trainer_readback(self):
  f=self.fixture();objects={};staged=[]
  def put(name,path):objects[name]=path.read_bytes()
  def stage(descriptor):staged.append(descriptor);return dict(descriptor_sha256=__import__('subnet.persistent_cpu_adamw',fromlist=['sha']).sha(descriptor),durable_readback_verified=True,authority_committed=False)
  descriptor,evidence=export_state(f.optimizer,epoch='new-policy',inference_checkpoint='22'*32,workspace=f.root,publish_shard=put,readback_shard=Mock(side_effect=AssertionError('no trainer shard GET')),commit_descriptor=stage,resource_admission=f.admission,shard_bytes=f.cap,concurrency=4,readback_mode=EXPORT_POLICY)
  self.assertEqual(len(objects),len(descriptor['shards']));self.assertFalse(evidence['trainer_full_readback_performed']);self.assertTrue(evidence['independent_full_readback_required']);self.assertFalse(evidence['descriptor_committed_last'])
  for receipt in evidence['shards']:
   self.assertIs(receipt['durable_readback_verified'],False);self.assertIs(receipt['local_sha_verified'],True);self.assertIs(receipt['upload_completed'],True)
  def fetch(shard,path):path.write_bytes(objects[shard])
  restored,receipts=restore_state(descriptor,evidence['descriptor_sha256'],'22'*32,f.inventory,workspace=f.root,fetch_shard=fetch,resource_admission=f.admission,concurrency=4)
  self.assertEqual(len(receipts),len(objects));self.assertTrue(all(x['verified_materialization']for x in receipts))
 def test_policy_cannot_use_local_reader_or_default_without_explicit_contract(self):
  self.assertEqual(export_policy({}),'trainer-full')
  for bad in ({'optimizer_state_export_policy':EXPORT_POLICY},{'optimizer_state_export_policy':EXPORT_POLICY,'persistent_publication_policy':dict(version='parallel-persistent-publication-v1',state_readback='local-full',checkpoint_readback_workers=4)},{'optimizer_state_export_policy':'unknown'}):
   with self.assertRaises(ValueError):export_policy(bad)
 def publication_fixture(self):
  f=PublicationControls();f.setUp();self.addCleanup(f.doCleanups)
  f.manifest.update(optimizer_state_export_policy=EXPORT_POLICY);f.policy['state_readback']='qualified-remote-full'
  from subnet import remote_optimizer_readback as r
  f.job['manifest']=r.sign(f.manifest,f.key);f.envelope=r.sign(f.job,f.key);(f.root/'roles/original-job.json').write_bytes(r.canonical(f.envelope));return f
 def test_actual_reader_validation_precedes_any_checkpoint_state_authority(self):
  f=self.publication_fixture();events=[]
  f.controller.independent_state_reader=SimpleNamespace(prepare_original_readback=lambda *a:(events.append('actual-reader')or{}))
  f.controller.stage_remote_checkpoint=Mock(side_effect=lambda *a:(events.append('checkpoint-staging')or{}));f.controller.commit_remote_checkpoint=Mock(side_effect=lambda *a:(events.append('checkpoint-authority')or{'id':'checkpoint'}))
  def commit(*a,**kw):
   events.append('reader-validation'if kw.get('verify_only')else'state-authority')
   return {'optimizer_steps':4}
  with patch('subnet.persistent_training_protocol.validate_report'),patch('subnet.remote_state_commit.independently_commit_remote',side_effect=commit):
   _,pointer,timings=complete(f.controller,f.report,f.job,f.manifest,'/original')
  self.assertEqual(set(events[:2]),{'actual-reader','checkpoint-staging'});self.assertEqual(events[2:],['reader-validation','checkpoint-authority','state-authority']);self.assertTrue(timings['authority_checkpoint_signed_after_independent_state_readback'])
 def test_failed_reader_validation_has_no_checkpoint_or_state_authority(self):
  f=self.publication_fixture();f.controller.independent_state_reader=SimpleNamespace(prepare_original_readback=lambda *a:{});f.controller.stage_remote_checkpoint=Mock(return_value={});f.controller.commit_remote_checkpoint=Mock()
  with patch('subnet.persistent_training_protocol.validate_report'),patch('subnet.remote_state_commit.independently_commit_remote',side_effect=ValueError('corrupt actual GET'))as commit:
   with self.assertRaisesRegex(ValueError,'corrupt actual GET'):complete(f.controller,f.report,f.job,f.manifest,'/original')
  f.controller.publish_remote_checkpoint.assert_not_called();f.controller.commit_remote_checkpoint.assert_not_called();self.assertEqual(commit.call_count,1)
 def test_real_signed_reader_receipt_verify_only_cannot_publish(self):
  f=RemoteAdmission();f.setUp();self.addCleanup(f.doCleanups)
  from subnet.remote_state_commit import independently_commit_remote
  with patch('subnet.persistent_training_protocol.validate_report',return_value=f.descriptor),patch('subnet.persistent_training_protocol.read_json',return_value=f.descriptor),patch('subnet.persistent_training_protocol._publish_verified_descriptor')as publish:
   result=independently_commit_remote(f.controller,f.report,f.job_env,f.request_bytes,f.receipt,f.launch,f.terminal,qualified_reader=f.identity,reader_host=f.host,trainer_host=f.trainer,storage_binding=f.storage,original_child=f.child,now=152,verify_only=True)
   self.assertTrue(result['independent_full_readback_verified']);self.assertFalse(result['authority_publication_written']);publish.assert_not_called();f.bucket.json.assert_not_called()
   f.receipt['payload']['objects'][0]['sha256']='f'*64
   with self.assertRaises(Exception):independently_commit_remote(f.controller,f.report,f.job_env,f.request_bytes,f.receipt,f.launch,f.terminal,qualified_reader=f.identity,reader_host=f.host,trainer_host=f.trainer,storage_binding=f.storage,original_child=f.child,now=152,verify_only=True)
   publish.assert_not_called();f.bucket.json.assert_not_called()
 def test_production_report_requires_explicit_uploaded_not_readback_evidence(self):
  from test_persistent_training_integration import PersistentIntegrationTests
  from subnet.persistent_training_protocol import validate_report
  f=PersistentIntegrationTests();f.setUp();self.addCleanup(f.doCleanups);report,job=f.report();manifest=job['manifest']['payload']
  job['source_files']['subnet/persistent_publication.py']='b'*64
  manifest.update(optimizer_state_export_policy=EXPORT_POLICY,persistent_publication_policy=dict(version='parallel-persistent-publication-v1',state_readback='qualified-remote-full',checkpoint_readback_workers=4));job['manifest']=f.sign(manifest)
  from subnet.training_receipts import computation_binding,sha
  receipt=copy.deepcopy(job['submissions'][0]['verifier_receipt']['payload']);receipt['computation_binding_sha256']=sha(computation_binding(manifest));receipt['original_signed_manifest_sha256']=sha(job['manifest']);job['submissions'][0]['verifier_receipt']=f.sign(receipt);report['training_admissions'][0]['verifier_receipt_sha256']=sha(job['submissions'][0]['verifier_receipt'])
  state=report['persistent_training_state'];state['authority_committed']=False
  rows=[dict(name=s['name'],size=s['size'],sha256=s['sha256'],durable_readback_verified=False,local_sha_verified=True,upload_completed=True,export_verification='uploaded-local-sha-only',independent_full_readback_required=True)for s in state['descriptor']['shards']]
  state['publication_evidence']=dict(optimizer_state_export_policy=EXPORT_POLICY,trainer_full_readback_performed=False,independent_full_readback_required=True,descriptor_committed_last=False,authority_commit_required=True,shards=rows)
  validate_report(report,job,manifest)
  for name,value in [('durable_readback_verified',True),('durable_readback_verified',0),('upload_completed',1),('local_sha_verified',False)]:
   bad=copy.deepcopy(report);bad['persistent_training_state']['publication_evidence']['shards'][0][name]=value
   with self.assertRaises(ValueError):validate_report(bad,job,manifest)
  bad=copy.deepcopy(report);bad['persistent_training_state']['publication_evidence']['shards']=[]
  with self.assertRaises(ValueError):validate_report(bad,job,manifest)
  bad=copy.deepcopy(report);bad['persistent_training_state']['authority_committed']=True
  with self.assertRaises(ValueError):validate_report(bad,job,manifest)
 def test_actual_checkpoint_staging_hashes_bytes_without_authority_then_commits_exact_journal(self):
  import hashlib,tempfile
  from pathlib import Path
  from subnet.remote_backend import RemoteController
  from subnet.storage import Identity,canonical
  from botocore.exceptions import ClientError
  key=Identity();data=b'actual independent checkpoint bytes';cp=dict(id='a'*64,files={'model.safetensors':hashlib.sha256(data).hexdigest()});manifest=dict(epoch='fresh-stage',checkpoint=cp)
  response=Mock();response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False);response.status_code=200;response.headers={'Content-Length':str(len(data))};response.iter_content.return_value=[data]
  with tempfile.TemporaryDirectory()as d:
   ctl=RemoteController.__new__(RemoteController);ctl.state=Path(d);ctl.authority=key;ctl.bucket=Mock();ctl.bucket.presign.return_value='https://approved/object';ctl.bucket.get.side_effect=ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject');ctl.jobs=SimpleNamespace(capacity=lambda *a:{},run=lambda *a,**kw:{});ctl.signed=lambda value:dict(payload=value,signer=key.id,signature='test');ctl.checkpoint_with_reads=lambda value:value
   with patch('subnet.remote_backend.requests.get',return_value=response):staged=ctl.stage_remote_checkpoint(manifest,'/original')
   ctl.bucket.json.assert_not_called();self.assertEqual(staged['objects']['model.safetensors']['bytes'],len(data))
   tampered=copy.deepcopy(staged);tampered['objects']['model.safetensors']['sha256']='f'*64
   with self.assertRaises(ValueError):ctl.commit_remote_checkpoint(manifest,tampered)
   ctl.bucket.json.assert_not_called();ctl.commit_remote_checkpoint(manifest,staged);ctl.bucket.json.assert_called_once()
 def test_upload_only_cannot_use_local_full_commit_entrypoint(self):
  from subnet.persistent_training_protocol import independently_commit
  manifest=dict(optimizer_state_export_policy=EXPORT_POLICY,persistent_publication_policy=dict(version='parallel-persistent-publication-v1',state_readback='qualified-remote-full',checkpoint_readback_workers=4))
  with self.assertRaisesRegex(ValueError,'no local fallback'):independently_commit(None,None,None,manifest)
