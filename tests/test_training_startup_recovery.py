import base64,copy,tempfile,json,unittest
from pathlib import Path
from types import SimpleNamespace
from subnet.storage import Identity,canonical
from subnet.training_receipts import sha,original_computation_manifest
from subnet import training_startup_recovery as r

class StartupRecovery(unittest.TestCase):
 def setUp(self):
  self.key=Identity();self.authority=self.key.id
  self.old=dict(epoch='same-epoch',checkpoint=dict(id='c'*64,files={},read_urls={'old':'expired'}),source_bundle={'sha256':'a'*64},trainer_state_binding=dict(global_step_before=3,parent={'descriptor_sha256':'p'*64}),training_policy='bf16-cpu-fp32-master-task-normalized-persistent-v4',training_input_policy='authenticated-verifier-compact-inputs-v2',training_coverage={'groups':[[0]]},sampling_contract={'exact':'unchanged'})
  self.original=dict(job_id='old-train',role='train',created_at=1,expires_at=90,manifest=self.sign(self.old),training_policy=self.old['training_policy'],training_input_policy=self.old['training_input_policy'],steps=1,submissions=[dict(url='expired',sha256='b'*64,verifier_receipt={'signed':'unchanged'},accepted_batch_sha256=['d'*64])],source_files={'subnet/model.py':'e'*64},runtime_versions={'torch':'same'},persistent_training={'output_namespace':'private/original'})
  self.terminal=dict(phase='failed',job_id='old-train',exit_code=1,runner_pid=10,runner_pid_ticks='101',child_pid=11,child_pid_ticks='102',started_at=2,finished_at=3)
  self.witness=dict(version='operator-startup-failure-witness-v1',observed_at=4,exception='fresh-source-bootstrap-admission',execution_started=False,cuda_allocated=False,model_loaded=False,original_processes_absent=True,output_namespace_empty=True,physical_gpu_idle=True,evidence_sha256='f'*64)
  self.value=dict(version=r.VERSION,epoch='same-epoch',original_signed_job=self.sign(self.original),original_job_sha256=sha(self.original),original_terminal=self.terminal,startup_witness=self.witness,replacement_source_bundle={'sha256':'f'*64},replacement_job_label='same-epoch-train-startup-recovery',created_at=5,expires_at=100)
  self.manifest=dict(self.old,source_bundle=self.value['replacement_source_bundle'],training_startup_recovery=self.sign(self.value));self.job=dict(self.original,job_id=self.value['replacement_job_label']+'-new',created_at=6,manifest=self.sign(self.manifest),source_files=dict(self.original['source_files'],**{'subnet/training_startup_recovery.py':'1'*64}),persistent_training={'output_namespace':'private/distinct'})
 def sign(self,p):return dict(payload=copy.deepcopy(p),signer=self.key.id,signature=base64.b64encode(self.key.key.sign(canonical(p)).signature).decode())
 def update(self):self.manifest[r.FIELD]=self.sign(self.value);self.job['manifest']=self.sign(self.manifest)
 def test_original_science_receipts_parent_and_failure_remain_immutable(self):
  self.job['submissions']=copy.deepcopy(self.original['submissions']);self.job['submissions'][0]['url']='refreshed-GET-only'
  self.assertEqual(r.validate(self.job,self.manifest,self.authority),self.value);self.assertEqual(original_computation_manifest(self.manifest,self.authority),self.old)
 def test_no_live_ambiguous_completed_or_post_compute_recovery(self):
  for k,v in [('phase','running'),('exit_code',0),('job_id','different')]:
   with self.subTest(k=k):
    value=copy.deepcopy(self.value);value['original_terminal'][k]=v;m=dict(self.manifest,training_startup_recovery=self.sign(value))
    with self.assertRaises(ValueError):r.validate(self.job,m,self.authority)
  for k in ('execution_started','cuda_allocated','model_loaded','original_processes_absent','output_namespace_empty','physical_gpu_idle'):
   value=copy.deepcopy(self.value);value['startup_witness'][k]=not value['startup_witness'][k];m=dict(self.manifest,training_startup_recovery=self.sign(value))
   with self.subTest(k=k),self.assertRaises(ValueError):r.validate(self.job,m,self.authority)
 def test_changed_parent_sampler_steps_inputs_runtime_math_or_same_namespace_rejected(self):
  for kind in ('parent','sampling','steps','receipt','runtime','math','namespace','samejob','missingpin','expiry'):
   job=copy.deepcopy(self.job);m=copy.deepcopy(self.manifest)
   if kind=='parent':m['trainer_state_binding']['global_step_before']=0
   if kind=='sampling':m['sampling_contract']={'exact':'changed'}
   if kind=='steps':job['steps']=2
   if kind=='receipt':job['submissions'][0]['verifier_receipt']={'new':'fake'}
   if kind=='runtime':job['runtime_versions']={'torch':'changed'}
   if kind=='math':job['source_files']['subnet/model.py']='0'*64
   if kind=='namespace':job['persistent_training']['output_namespace']='private/original'
   if kind=='samejob':job['job_id']='old-train'
   if kind=='missingpin':del job['source_files']['subnet/training_startup_recovery.py']
   if kind=='expiry':job['expires_at']=101
   with self.subTest(kind=kind),self.assertRaises(ValueError):r.validate(job,m,self.authority)
 def test_original_signature_and_witness_lifetime_fail_closed(self):
  for kind in ('signature','late','originalsha','nested'):
   value=copy.deepcopy(self.value);job=copy.deepcopy(self.job)
   if kind=='signature':value['original_signed_job']['signature']='bad'
   if kind=='late':job['created_at']=100
   if kind=='originalsha':value['original_job_sha256']='0'*64
   if kind=='nested':value['original_signed_job']['payload']['manifest']['payload'][r.FIELD]={};value['original_signed_job']['payload']['manifest']=self.sign(value['original_signed_job']['payload']['manifest']['payload']);value['original_signed_job']=self.sign(value['original_signed_job']['payload']);value['original_job_sha256']=sha(value['original_signed_job']['payload'])
   with self.subTest(kind=kind),self.assertRaises(Exception):r.validate(job,dict(self.manifest,training_startup_recovery=self.sign(value)),self.authority)

class OperatorRecovery(StartupRecovery):
 def setUp(self):
  super().setUp();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.state=Path(self.tmp.name);(self.state/'roles').mkdir()
  self.document=self.state/'declaration.json';self.document.write_bytes(canonical(self.sign(self.value)))
  (self.state/'roles'/'same-epoch-train.json').write_bytes(canonical(dict(job_id=self.original['job_id'],job_sha256=sha(self.original))))
  (self.state/'roles'/'old-train-failure.json').write_bytes(canonical(self.terminal));(self.state/'latest-trainer-state.json').write_bytes(canonical(self.old['trainer_state_binding']['parent']))
  self.controller=SimpleNamespace(state=self.state,authority=self.key,training_startup_recovery_files={'same-epoch':str(self.document)},checkpoint_with_reads=lambda x:dict(x,read_urls={'new':'GET'}),bucket=SimpleNamespace(presign=lambda key:'fresh:'+key),signed=self.sign)
 def test_original_inputs_reused_no_new_receipt_and_one_immutable_declaration(self):
  from unittest.mock import patch
  with patch('subnet.training_startup_recovery.time.time',return_value=6):manifest,inputs=r.apply(self.controller,self.old,1)
  self.assertEqual(r.input_inventory(inputs),r.input_inventory(self.original['submissions']));self.assertEqual(manifest['trainer_state_binding']['global_step_before'],3);self.assertEqual(r.label(self.controller,'same-epoch'),self.value['replacement_job_label'])
  self.value['replacement_job_label']='attempt-two';self.document.write_bytes(canonical(self.sign(self.value)))
  with self.assertRaisesRegex(ValueError,'one immutable'):r.apply(self.controller,self.old,1)
 def test_original_completed_parent_changed_expired_or_failure_tampering_hold(self):
  from unittest.mock import patch
  for kind in ('report','parent','expired','failure'):
   self.setUp()
   if kind=='report':(self.state/'roles'/'old-train-report.json').write_text('{}')
   if kind=='parent':(self.state/'latest-trainer-state.json').write_text('{}')
   if kind=='failure':(self.state/'roles'/'old-train-failure.json').write_text('{}')
   with self.subTest(kind=kind),patch('subnet.training_startup_recovery.time.time',return_value=100 if kind=='expired'else 6),self.assertRaises(ValueError):r.apply(self.controller,self.old,1)
   self.assertFalse((self.state/'same-epoch-startup-recovery-reservation.json').exists())
 def test_same_replacement_resume_keeps_original_caps_and_id_even_after_expiry(self):
  from unittest.mock import patch
  with patch('subnet.training_startup_recovery.time.time',return_value=6):manifest,rows=r.apply(self.controller,self.old,1)
  job=copy.deepcopy(self.job);job['manifest']=self.sign(manifest);job['submissions']=rows
  path=self.state/'roles'/(self.value['replacement_job_label']+'.json');path.write_bytes(canonical(dict(job_id=job['job_id'],job_sha256=sha(job))))
  (self.state/'roles'/(job['job_id']+'-job.json')).write_bytes(canonical(self.sign(job)))
  with patch('subnet.training_startup_recovery.time.time',return_value=1000):adopted,inputs=r.apply(self.controller,self.old,1)
  self.assertEqual(adopted,manifest);self.assertEqual(inputs,rows);self.assertTrue((self.state/'roles'/'same-epoch-train.json').exists())

class ActualCompactLineage(unittest.TestCase):
 def test_signed_recovery_admits_original_compact_inputs_without_reverification(self):
  import test_compact_training_integration as fixtures
  from subnet import backend_jobs as backend
  from subnet.persistent_training_protocol import prepare_job,validate_job,validate_binding
  from unittest.mock import patch
  fx=fixtures.CompactPersistentIntegrationTests();fx.setUp();self.addCleanup(fx.doCleanups)
  original=fx.job(steps=1);original['source_files']['subnet/model.py']='e'*64
  terminal=dict(phase='failed',job_id=original['job_id'],exit_code=1,runner_pid=10,runner_pid_ticks='10',child_pid=11,child_pid_ticks='11',started_at=23,finished_at=24)
  witness=dict(version='operator-startup-failure-witness-v1',observed_at=30,exception='fresh-source-bootstrap-admission',execution_started=False,cuda_allocated=False,model_loaded=False,original_processes_absent=True,output_namespace_empty=True,physical_gpu_idle=True,evidence_sha256='f'*64)
  declaration=dict(version=r.VERSION,epoch=fx.manifest['epoch'],original_signed_job=fx.sign(original),original_job_sha256=sha(original),original_terminal=terminal,startup_witness=witness,replacement_source_bundle={'sha256':'f'*64},replacement_job_label='recovered-train',created_at=40,expires_at=100)
  manifest=dict(fx.manifest,source_bundle=declaration['replacement_source_bundle'],training_startup_recovery=fx.sign(declaration))
  job=dict(original,job_id='recovered-train-new',created_at=50,manifest=fx.sign(manifest),source_files=dict(original['source_files'],**{'subnet/training_startup_recovery.py':'1'*64}))
  job['persistent_training']=prepare_job(fx.controller,manifest,job['job_id'],1,100)
  with patch('subnet.backend_jobs.time.time',return_value=50):
   backend.validate(fx.sign(job),fx.authority,now=50)
   validate_job(job,manifest,fx.authority)
  self.assertEqual(validate_binding(manifest['trainer_state_binding'],manifest),fx.manifest['trainer_state_binding']);self.assertEqual(r.input_inventory(job['submissions']),r.input_inventory(original['submissions']))
  self.assertNotEqual(job['persistent_training']['output_namespace'],original['persistent_training']['output_namespace'])

class RecoveryPublication(unittest.TestCase):
 def test_real_compact_cpu_state_releases_replacement_reward_gate(self):
  import sqlite3
  import test_compact_training_integration as fixtures
  from training_receipt_fixtures import signed_receipt
  from subnet import compact_training_inputs as compact
  from subnet.persistent_training_protocol import prepare_job,independently_commit
  from subnet.reward_publication import VERSION,emit,require
  from subnet.remote_backend import save
  f=fixtures.CompactPersistentIntegrationTests();f.setUp();self.addCleanup(f.doCleanups)
  f.manifest['reward_publication_policy']=VERSION
  receipt,audit,verifyjob,request=signed_receipt(f.key,f.manifest,f.miner,f.receipts[f.miner],f.batch)
  f.legacy_submission=dict(f.legacy_submission,verifier_receipt=receipt);f.verifier_audit=dict(audit,remote_job_id='synthetic-verify');f.queue.workers={request['signer']:['verify']}
  with sqlite3.connect(f.queue.path)as db:db.execute('UPDATE jobs SET digest=?,envelope=?,worker=?,report=?,report_digest=?,report_request=? WHERE id=?',(sha(verifyjob['payload']),canonical(verifyjob).decode(),request['signer'],canonical(request['payload']['report']).decode(),sha(request['payload']['report']),canonical(request).decode(),'synthetic-verify'))
  f.submission=compact.prepare_submissions(f.controller,f.manifest,{f.miner:f.verifier_audit},f.receipts)[0];f.data=f.bucket.get('private/compact-training-inputs/'+f.submission['sha256']+'.json')
  oldmanifest=copy.deepcopy(f.manifest);original=f.job();epoch=oldmanifest['epoch']
  terminal=dict(phase='failed',job_id=original['job_id'],exit_code=1,runner_pid=10,runner_pid_ticks='10',child_pid=11,child_pid_ticks='11',started_at=23,finished_at=24)
  witness=dict(version='operator-startup-failure-witness-v1',observed_at=30,exception='fresh-source-bootstrap-admission',execution_started=False,cuda_allocated=False,model_loaded=False,original_processes_absent=True,output_namespace_empty=True,physical_gpu_idle=True,evidence_sha256='f'*64)
  value=dict(version=r.VERSION,epoch=epoch,original_signed_job=f.sign(original),original_job_sha256=sha(original),original_terminal=terminal,startup_witness=witness,replacement_source_bundle={'sha256':'f'*64},replacement_job_label='recovered-train',created_at=31,expires_at=100)
  document=f.sign(value);f.manifest=dict(oldmanifest,source_bundle=value['replacement_source_bundle'],training_startup_recovery=document)
  job=dict(original,job_id='recovered-train-new',created_at=32,manifest=f.sign(f.manifest),source_files=dict(original['source_files'],**{'subnet/training_startup_recovery.py':'1'*64}));job['persistent_training']=prepare_job(f.controller,f.manifest,job['job_id'],3,100)
  report,job=f.report(job);pointer=independently_commit(f.controller,report,job,f.manifest)
  cp=dict(f.cp,descriptor_key='public/checkpoint-authority.json');f.bucket.json(cp['descriptor_key'],f.sign(dict(id=cp['id'],files=cp['files'])))
  save(f.root/'roles'/(epoch+'-train.json'),dict(job_id=original['job_id'],job_sha256=sha(original)));save(f.root/'roles'/(original['job_id']+'-job.json'),f.sign(original));save(f.root/'roles'/(original['job_id']+'-failure.json'),terminal)
  save(f.root/(epoch+'-startup-recovery-reservation.json'),dict(declaration=document,declaration_sha256=sha(document),label=value['replacement_job_label'],original_job_sha256=sha(original)))
  save(f.root/'roles'/(value['replacement_job_label']+'.json'),dict(job_id=job['job_id'],job_sha256=sha(job)));save(f.root/'roles'/(job['job_id']+'-job.json'),f.sign(job));save(f.root/'roles'/(job['job_id']+'-report.json'),report)
  save(f.root/'latest-trainer-state.json',pointer);score=dict(epoch_id=epoch,points={f.miner:1},receipts=f.receipts,finalized_at=25);save(f.root/(epoch+'-scores.json'),score);save(f.root/(epoch+'-signed-compute-scores.json'),f.sign(score))
  save(f.root/(epoch+'-checkpoint-publication.json'),dict(checkpoint=cp['id'],operator_independent_hashes=True,objects={n:dict(sha256=h)for n,h in cp['files'].items()}))
  metrics=dict(trainer_state=pointer,steps=3,source_epoch=epoch,input_checkpoint=f.cp['id'],new_checkpoint=cp,original_job_sha256=sha(job),trainer_binding_sha256=sha(oldmanifest['trainer_state_binding']));save(f.root/(epoch+'-training-metrics.json'),metrics)
  ready=emit(f.controller,oldmanifest);self.assertEqual(ready['evidence']['original_job_id'],job['job_id']);self.assertEqual(ready['evidence']['training_startup_recovery']['original_failed_job_id'],original['job_id']);self.assertTrue(ready['evidence']['training_startup_recovery']['late_recovery']);self.assertEqual(require(f.root,oldmanifest,f.authority),ready)
  self.assertFalse((f.root/'roles'/(original['job_id']+'-report.json')).exists())

class RecoveryDispatchLifetimeTests(unittest.TestCase):
 def test_actual_dispatch_clips_86400_role_and_scoped_state_urls_to_existing_7200_declaration(self):
  from unittest.mock import Mock,patch
  from test_compact_training_integration import CompactPersistentIntegrationTests
  from subnet.remote_backend import RemoteJobs
  from subnet.backend_jobs import signed
  f=CompactPersistentIntegrationTests();f.setUp();self.addCleanup(f.doCleanups)
  original=f.job();old=copy.deepcopy(f.manifest)
  terminal=dict(phase='failed',job_id=original['job_id'],exit_code=1,runner_pid=10,runner_pid_ticks='10',child_pid=11,child_pid_ticks='11',started_at=23,finished_at=24)
  value=dict(version=r.VERSION,epoch=old['epoch'],original_signed_job=f.sign(original),original_job_sha256=sha(original),original_terminal=terminal,
   startup_witness=dict(version='operator-startup-failure-witness-v1',observed_at=30,exception='fresh-source-bootstrap-admission',execution_started=False,cuda_allocated=False,model_loaded=False,original_processes_absent=True,output_namespace_empty=True,physical_gpu_idle=True,evidence_sha256='f'*64),
   replacement_source_bundle={'sha256':'f'*64},replacement_job_label='same-declaration-recovery',created_at=31,expires_at=7231)
  manifest=dict(old,source_bundle=value['replacement_source_bundle'],training_startup_recovery=f.sign(value))
  jobs=RemoteJobs.__new__(RemoteJobs);jobs.controller=f.controller;jobs.state=f.root/'roles';jobs.state.mkdir(exist_ok=True)
  jobs.config={'job_ttl_seconds_by_role':{'train':86400}};jobs.metadata=dict(source_files=dict(original['source_files'],**{'subnet/training_startup_recovery.py':'1'*64}),runtime_versions=original['runtime_versions'])
  jobs.workspace='/synthetic';jobs.code='/synthetic/code';jobs.python='/synthetic/python';jobs.command=Mock();jobs.copy_to=Mock()
  jobs.remote_status=Mock(return_value={'phase':'complete'});jobs.checked=Mock(side_effect=lambda report,*a:report)
  jobs.copy_from=lambda remote,local:local.write_text('{}')
  lifetimes=[];presign=f.bucket.presign
  def bounded_presign(key,operation='get_object',expires=3600):lifetimes.append(expires);return presign(key,operation,expires)
  with patch.object(f.bucket,'presign',side_effect=bounded_presign),patch('subnet.remote_backend.time.time',return_value=32):
   jobs.run(value['replacement_job_label'],'train',manifest,submissions=original['submissions'],steps=original['steps'],training_policy=original['training_policy'])
  record=json.loads((jobs.state/(value['replacement_job_label']+'.json')).read_bytes())
  job=signed(json.loads((jobs.state/(record['job_id']+'-job.json')).read_bytes()),f.authority)
  self.assertEqual(job['expires_at'],7231);self.assertEqual(job['created_at'],32)
  self.assertTrue(lifetimes);self.assertTrue(all(v<=7199 for v in lifetimes))
  self.assertEqual(job['manifest']['payload']['training_startup_recovery'],f.sign(value))
  self.assertEqual(job['submissions'],original['submissions'])

class UnauditedPrecomputeRecovery(StartupRecovery):
 def setUp(self):
  super().setUp()
  self.old['training_input_policy']='committed-unaudited-training-v1'
  row=self.original['submissions'][0];row.update(url='https://example.invalid/bucket/public/same-epoch/submissions/frozen.json',size=123)
  self.original.update(manifest=self.sign(self.old),training_input_policy=self.old['training_input_policy'])
  self.job.update(training_input_policy=self.old['training_input_policy'],submissions=copy.deepcopy(self.original['submissions']))
  self.value.update(version=r.ADMISSION_VERSION,original_signed_job=self.sign(self.original),original_job_sha256=sha(self.original),execution_source_files=self.job['source_files'],authorized_input_objects=[dict(key='public/same-epoch/submissions/frozen.json',sha256=row['sha256'],size=row['size'])])
  self.manifest=dict(self.old,source_bundle=self.value['replacement_source_bundle'],training_startup_recovery=self.sign(self.value));self.job['manifest']=self.sign(self.manifest)
 def test_original_science_receipts_parent_and_failure_remain_immutable(self):
  self.job['submissions'][0]['url']='https://new.invalid/bucket/public/same-epoch/submissions/frozen.json?fresh=1'
  self.assertEqual(r.validate(self.job,self.manifest,self.authority),self.value)
  self.assertEqual(original_computation_manifest(self.manifest,self.authority),self.old)
 def test_same_hash_different_key_or_size_cannot_redirect_inputs(self):
  for field,value in [('url','https://example.invalid/bucket/public/another/submissions/frozen.json'),('size',124)]:
   job=copy.deepcopy(self.job);job['submissions'][0][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):r.validate(job,self.manifest,self.authority)
 def test_signed_execution_inventory_is_exact(self):
  job=copy.deepcopy(self.job);job['source_files']['subnet/extra.py']='1'*64
  with self.assertRaisesRegex(ValueError,'exact approved'):r.validate(job,self.manifest,self.authority)
