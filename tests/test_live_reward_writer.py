"""Portable CPU tests: real locks/process reads, signed mock evidence; no chain/GPU."""
import importlib.util,sys,tempfile,unittest,json,os,sqlite3,hashlib,copy,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from nacl.signing import SigningKey
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops import live_reward_writer as w
from subnet.backend_profiles import profile
from subnet.backend_jobs import SOURCE_FILES
from subnet.artifact_budget import for_manifest
from subnet.backend_jobs import file_map
from ops.live_reward_exporter import sign

class WriterTests(unittest.TestCase):
 def test_actual_process_identity_and_global_lock_exclusive(self):
  i=w.process_identity();self.assertEqual(i['writer_pid'],os.getpid());self.assertTrue(i['writer_start_ticks'].isdigit());self.assertTrue(i['boot_id'])
  with tempfile.TemporaryDirectory() as t:
   with w.global_lock(Path(t)/'owner.lock'):
    with self.assertRaises(BlockingIOError):
     with w.global_lock(Path(t)/'owner.lock'):pass
   with w.global_lock(Path(t)/'owner.lock'):pass
 def test_old_unit_status_queries_fail_closed(self):
  good=lambda *a,**k:SimpleNamespace(returncode=0,stdout='LoadState=loaded\nActiveState=inactive\nUnitFileState=disabled\n')
  self.assertEqual(len(w.observe_units(good)),4)
  self.assertEqual(len(w.observe_units(lambda args,**kw:SimpleNamespace(returncode=0,stdout='LoadState=loaded\nActiveState=inactive\nUnitFileState='+('static' if args[3].endswith('.service') else 'disabled')+'\n'))),4)
  with self.assertRaises(ValueError):w.observe_units(lambda *a,**kw:SimpleNamespace(returncode=0,stdout='LoadState=loaded\nActiveState=inactive\nUnitFileState=static\n'))
  for response in [SimpleNamespace(returncode=1,stdout=''),SimpleNamespace(returncode=0,stdout='LoadState=not-found\nActiveState=inactive\nUnitFileState=disabled\n'),SimpleNamespace(returncode=0,stdout='LoadState=loaded\nActiveState=active\nUnitFileState=disabled\n'),SimpleNamespace(returncode=0,stdout='LoadState=loaded\nActiveState=inactive\nUnitFileState=enabled\n')]:
   with self.assertRaises(ValueError):w.observe_units(lambda *a,**k:response)
 def test_cutover_anchor_signature_owner_path_and_lock_binding(self):
  key=SigningKey.generate();authority=key.verify_key.encode().hex();anchor=sign(dict(netuid=120,owner_hotkey=w.OWNER,compute_epoch_prefix='nonpayable-live-reward-math-v1-'),key)
  c=dict(version='live-single-writer-runtime-v1',netuid=120,owner_hotkey=w.OWNER,anchor_sha256=w.sha(anchor),global_lock_path=str(Path('/run/user')/str(os.getuid())/('affine-live-reward-120-'+hashlib.sha256(w.OWNER.encode()).hexdigest()+'.lock')),compute_state='/tmp/compute',reward_state='/tmp/reward',chain_state='/tmp/reward',queue_database='/tmp/queue',authority_seed_file='/tmp/seed',runtime_versions=dict(torch='test',transformers='test',toploc='test'))
  w.authenticate_cutover(sign(c,key),anchor,authority)
  for field,value in [('owner_hotkey','OTHER'),('netuid',121),('global_lock_path','/tmp/alternate.lock'),('chain_state','/tmp/other'),('compute_state','relative')]:
   bad=dict(c);bad[field]=value
   with self.assertRaises(ValueError):w.authenticate_cutover(sign(bad,key),anchor,authority)
  bad=sign(c,key);bad['payload']['netuid']=121
  with self.assertRaises(Exception):w.authenticate_cutover(bad,anchor,authority)
 def test_guard_actual_bytes_extra_or_missing_marker_refused(self):
  with tempfile.TemporaryDirectory() as t:
   hook=Path(t)/'validator.py';hook.write_text('reviewed original suppression hook\n');marker=Path(t)/'marker';marker.write_text('active')
   c={'legacy_guard_files':[dict(kind='validator-hook',path=str(hook),sha256=w.file_hash(hook)),dict(kind='suppression-marker',path=str(marker),sha256=w.file_hash(marker))]}
   # Exact actual production marker is mandatory even if arbitrary supplied file hashes match.
   with self.assertRaises(ValueError):w.guard_files(c)
   hook.write_text('changed')
   with self.assertRaises(ValueError):w.guard_files(c)
 def test_same_hour_deferred_retry_no_skipping_uncertain_execution(self):
  with tempfile.TemporaryDirectory() as t:
   state=Path(t);anchor={'effective_at':3700}
   self.assertIsNone(w.choose_hour(state,anchor,7000));self.assertEqual(w.choose_hour(state,anchor,8000),7200)
   (state/'writer-cursor.json').write_text(json.dumps(dict(window_end=7200,status='deferred_rate_limit')))
   self.assertEqual(w.choose_hour(state,anchor,16000),7200)
   (state/'writer-cursor.json').write_text(json.dumps(dict(window_end=7200,status='submitted')));self.assertEqual(w.choose_hour(state,anchor,16000),10800)
   (state/'writer-cursor.json').write_text(json.dumps(dict(window_end=10800,status='submitting')))
   with self.assertRaises(RuntimeError):w.choose_hour(state,anchor,16000)

class RuntimeTests(unittest.TestCase):
 def test_real_exporter_signed_handoff_dryrun_and_deferred_same_hour(self):
  from contextlib import contextmanager
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);compute=root/'compute';compute.mkdir();(compute/'controller.json').write_text(json.dumps({'active':None}));reward=root/'reward';reward.mkdir();seed=root/'seed';key=SigningKey.generate();seed.write_text(key.encode().hex());seed.chmod(0o600);authority=key.verify_key.encode().hex()
   anchor=sign(dict(version='live-reward-cutover-v1',netuid=120,owner_hotkey=w.OWNER,effective_at=3700,compute_epoch_prefix='nonpayable-live-reward-math-v1-',live_epoch_prefix='live-math-reward-v1-',approved_compute_sources=['b'*64]),key)
   c=dict(version='live-single-writer-runtime-v1',netuid=120,owner_hotkey=w.OWNER,anchor_sha256=w.sha(anchor),global_lock_path=str(Path('/run/user')/str(os.getuid())/('affine-live-reward-120-'+hashlib.sha256(w.OWNER.encode()).hexdigest()+'.lock')),compute_state=str(compute),reward_state=str(reward),chain_state=str(reward),queue_database=str(root/'queue'),authority_seed_file=str(seed),runtime_versions=dict(torch='test',transformers='test',toploc='test'))
   events=[]
   class Adapter:
    def registrations(self):events.append('registrations');return {}
    def submit_hour(self,*args,**kwargs):events.append(('submit',kwargs['execute']));return dict(status='deferred_rate_limit' if kwargs['execute'] else 'zero_points_no_submission',window_end=args[2])
   adapter=Adapter()
   actual_lock=w.global_lock
   @contextmanager
   def lock(path):
    with actual_lock(root/'real-test.lock'):
     events.append('lock');yield
   units=[dict(unit=u,running=False,enabled=False,status_query_succeeded=True) for u in w.UNITS]
   with patch.object(w,'global_lock',lock),patch.object(w,'guard_files'),patch.object(w,'observe_units',return_value=units),patch.object(w,'verify_completed_evidence',return_value=[]),patch.object(w.time,'time',return_value=8000):
    result=w.run_once(sign(c,key),anchor,authority,adapter_factory=lambda *a,**k:adapter)
    self.assertEqual(result['status'],'zero_points_no_submission');self.assertFalse((reward/'writer-cursor.json').exists());self.assertLess(events.index('registrations'),events.index(('submit',False)))
    proposal=w.read(reward/'hour-7200-reward-units.json');self.assertEqual(w.signed(proposal,authority)['points'],{})
    receipt=w.signed(w.read(reward/'actual-writer-observation.json'),authority);self.assertEqual(receipt['writer_pid'],os.getpid())
    w.run_once(sign(c,key),anchor,authority,execute=True,adapter_factory=lambda *a,**k:adapter)
    self.assertEqual(w.read(reward/'writer-cursor.json')['status'],'deferred_rate_limit')
    w.run_once(sign(c,key),anchor,authority,execute=True,adapter_factory=lambda *a,**k:adapter)
    self.assertEqual(w.read(reward/'writer-cursor.json')['window_end'],7200)
    from unittest.mock import Mock
    factory=Mock()
    with patch.object(w,'verify_completed_evidence',side_effect=ValueError('original evidence refusal')):
     with self.assertRaises(ValueError):w.run_once(sign(c,key),anchor,authority,adapter_factory=factory)
    factory.assert_not_called()
    class Uncertain(Adapter):
     def submit_hour(self,*args,**kwargs):raise OSError('test uncertain transport')
    with self.assertRaises(OSError):w.run_once(sign(c,key),anchor,authority,execute=True,adapter_factory=lambda *a,**k:Uncertain())
    self.assertEqual(w.read(reward/'writer-cursor.json')['status'],'submitting')
    with self.assertRaises(RuntimeError):w.run_once(sign(c,key),anchor,authority,execute=True,adapter_factory=factory)
    factory.assert_not_called()
    (reward/'writer-cursor.json').unlink()
    class LateFinalize(Adapter):
     def registrations(self):
      epoch='nonpayable-live-reward-math-v1-DELAYED'
      (compute/(epoch+'-manifest.json')).write_text(json.dumps(dict(epoch=epoch,deadline=7100)))
      (compute/'controller.json').write_text(json.dumps(dict(active=dict(epoch=epoch,phase='collect'))))
      return {}
    with patch.object(w.exporter,'run_once') as export_spy:
     with self.assertRaisesRegex(ValueError,'lacks original scores'):w.run_once(sign(c,key),anchor,authority,adapter_factory=lambda *a,**k:LateFinalize())
     export_spy.assert_not_called()
 def test_evidence_failure_refuses_before_adapter_and_export(self):
  # Failure order tested on actual runner; no real chain/network construction.
  with patch.object(w,'authenticate_cutover',return_value=({'global_lock_path':'unused'},{'effective_at':1})),patch.object(w,'global_lock',side_effect=ValueError('lock refusal')):
   from unittest.mock import Mock
   factory=Mock()
   with self.assertRaises(ValueError):w.run_once({}, {}, 'CPU',adapter_factory=factory)
   factory.assert_not_called()

def lineage_fixture(tmp):
 operator=SigningKey.generate();worker=SigningKey.generate();authority=operator.verify_key.encode().hex();wid=worker.verify_key.encode().hex()
 revision,bp,np=profile('cuda-bf16-eager-sm90-v1');cpfiles={'config.json':'a'*64,'model.safetensors':'a'*64};cp={'files':cpfiles,'id':file_map(cpfiles)}
 policy=dict(mode='sampled',version='bounded-random-v1',epoch_budget=2,escalation_budget=2,minimum_per_miner=1,maximum_per_miner=3,penalties=dict(invalid_batch_multiplier=.5,zero_epoch_after=0,penalize_structural=False))
 m=dict(epoch='nonpayable-live-reward-math-v1-CPU',payable=False,checkpoint=cp,source_bundle={'sha256':'b'*64,'size':1},model_runtime_revision=revision,backend_profile=bp,numerical_policy=np,audit_policy=policy,max_batches=3,K=1,L=1)
 jm=copy.deepcopy(m);jm.update(audit_seed='c'*64,audit_frozen_receipts={'MINER':{'sha256':'d'*64}});jm['audit_policy']['submission_counts']={'d'*64:1}
 files={name:'e'*64 for name in SOURCE_FILES};runtime=dict(torch='test',transformers='test',toploc='test')
 job=dict(schema=1,job_id='verify-CPU',role='verify',created_at=100.,expires_at=200.,manifest=sign(jm,operator),source_files=files,runtime_versions=runtime,submissions=[dict(url='https://bucket.r2.cloudflarestorage.com/frozen?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=TEST',sha256='d'*64)])
 audit=dict(epoch=m['epoch'],submission_sha256='d'*64,accepted=[],outcomes=[])
 remote=dict(job_id=job['job_id'],job_sha256=w.sha(job),operator=authority,role='verify',epoch=m['epoch'],checkpoint=cp['id'],source_files=files,runtime_versions=runtime,backend_profile=bp,numerical_policy=np,success=True,chain_transactions=False,completed_at=150.,execution_resources_enforced=True,audits=[audit])
 request=sign(dict(action='report',job_id=job['job_id'],report=remote),worker)
 db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row;db.execute('CREATE TABLE jobs(id,status,envelope,worker,report_request,report,report_digest)')
 envelope=sign(job,operator);db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?)',(job['job_id'],'complete',json.dumps(envelope),wid,json.dumps(request),json.dumps(remote),w.sha(remote)))
 state=Path(tmp);(state/'roles').mkdir();(state/'roles'/'verify-CPU-job.json').write_text(json.dumps(envelope))
 bound=dict(audit,remote_job_id=job['job_id'],backend_profile=bp,execution_resources_enforced=True)
 return state,m,bound,authority,dict(runtime_versions=runtime,verifier_identities=[wid]),db,files,operator,job,remote
class EvidenceTests(unittest.TestCase):
 def test_original_signed_job_worker_report_source_lease_and_frozen_binding(self):
  with tempfile.TemporaryDirectory() as t:
   state,m,a,auth,c,db,files,key,job,remote=lineage_fixture(t)
   got=w.verify_audit_lineage(state,m,a,auth,c,db,160.,files);self.assertEqual(got['submission_sha256'],'d'*64)
   for field,value in [('submission_sha256','f'*64),('remote_job_id','../OTHER')]:
    bad=dict(a);bad[field]=value
    with self.assertRaises((ValueError,KeyError)):w.verify_audit_lineage(state,m,bad,auth,c,db,160.,files)
   with self.assertRaises(ValueError):w.verify_audit_lineage(state,m,a,auth,dict(c,runtime_versions={'torch':'wrong'}),db,160.,files)
   with self.assertRaises(ValueError):w.verify_audit_lineage(state,m,a,auth,c,db,160.,dict(files,**{'subnet/model.py':'f'*64}))
 def test_actual_coordinator_transport_and_full_evidence_reader(self):
  import io,tarfile
  from subnet.source_bootstrap import TASK_ASSET
  with tempfile.TemporaryDirectory() as t:
   state,m,a,auth,c,old_db,files,key,job,remote=lineage_fixture(t)
   old_db.close();worker=SigningKey.generate();wid=worker.verify_key.encode().hex();c['verifier_identities']=[wid]
   members={name:b'# public CPU fixture, no executed model\n' for name in SOURCE_FILES};members.update({'subnet/__init__.py':b'','subnet/cli.py':b'# CPU fixture\n',TASK_ASSET:b'[]'})
   buffer=io.BytesIO()
   with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
    for name,data in members.items():
     entry=tarfile.TarInfo(name);entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
   body=buffer.getvalue();archive=state/'source.tar.gz';archive.write_bytes(body);descriptor=dict(sha256=hashlib.sha256(body).hexdigest(),size=len(body));descriptorpath=state/'descriptor.json';descriptorpath.write_text(json.dumps(sign(descriptor,key)))
   source=dict(sha256=descriptor['sha256'],archive_path=str(archive),descriptor_path=str(descriptorpath));c['source']=source
   m['source_bundle']=descriptor;job['manifest']['payload']['source_bundle']=descriptor;job['manifest']=sign(job['manifest']['payload'],key)
   job['source_files']={name:hashlib.sha256(data).hexdigest() for name,data in members.items() if name.startswith('subnet/')};envelope=sign(job,key);(state/'roles'/'verify-CPU-job.json').write_text(json.dumps(envelope))
   remote.update(job_sha256=w.sha(job),source_files=job['source_files']);clock=[100.]
   queuepath=state/'queue.sqlite3';queue=w.Coordinator(queuepath,auth,{wid:['verify']},clock=lambda:clock[0]);queue.enqueue(envelope)
   claim=queue.request(sign(dict(action='claim',role='verify',at=100.,nonce='FIRST-CLAIM-00001'),worker))['claim'];clock[0]=160.
   queue.request(sign(dict(action='report',job_id=job['job_id'],token=claim['token'],report=remote,at=160.,nonce='FIRST-REPORT-0001'),worker))
   (state/(m['epoch']+'-first-signed-manifest.json')).write_text(json.dumps(sign(m,key)))
   (state/(m['epoch']+'-signed-compute-scores.json')).write_text(json.dumps(sign(dict(receipts={'MINER':{'sha256':'d'*64}}),key)))
   (state/(m['epoch']+'-signed-compute-audit-MINER.json')).write_text(json.dumps(sign(a,key)));c.update(compute_state=str(state),queue_database=str(queuepath))
   result=w.verify_completed_evidence(c,auth,160.);self.assertEqual(len(result),1);self.assertEqual(result[0]['worker'],wid)
   archive.write_bytes(body+b'bad')
   with self.assertRaisesRegex(ValueError,'archive bytes'):w.verify_completed_evidence(c,auth,160.)
 def test_authenticated_worker_report_corruption_and_late_completion_refuse(self):
  with tempfile.TemporaryDirectory() as t:
   state,m,a,auth,c,db,files,key,job,remote=lineage_fixture(t)
   original=db.execute('SELECT report_request FROM jobs').fetchone()[0];bad=json.loads(original);bad['payload']['report']['success']=False;db.execute('UPDATE jobs SET report_request=?',(json.dumps(bad),))
   with self.assertRaises(Exception):w.verify_audit_lineage(state,m,a,auth,c,db,160.,files)
   db.execute('UPDATE jobs SET report_request=?',(original,))
   envelope=json.loads((state/'roles'/'verify-CPU-job.json').read_text());envelope['payload']['expires_at']=150.;envelope=sign(envelope['payload'],key);(state/'roles'/'verify-CPU-job.json').write_text(json.dumps(envelope));db.execute('UPDATE jobs SET envelope=?',(json.dumps(envelope),))
   with self.assertRaises(ValueError):w.verify_audit_lineage(state,m,a,auth,c,db,160.,files)
if __name__=='__main__':unittest.main()
