import base64,hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from subnet.storage import Identity,canonical
from subnet.remote_backend import RemoteJobs,RemoteMinerReserved,save

class Dispatch(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.jobs=RemoteJobs.__new__(RemoteJobs);self.jobs.state=Path(self.tmp.name);self.jobs.config={};self.jobs.workspace='/owned';self.jobs.code='/source';self.jobs.python='python';self.jobs.metadata=dict(source_files={},runtime_versions={})
  self.authority=Identity();self.jobs.controller=SimpleNamespace(authority=self.authority,signed=self.signed)
  self.jobs.command=Mock();self.jobs.copy_to=Mock();self.jobs.remote_status=Mock(return_value={'phase':'running'});self.jobs.copy_from=Mock(side_effect=AssertionError('no report wait'))
  self.m=dict(epoch='e',checkpoint={'id':'cp'},hourly_execution_policy={'version':'bounded-hourly-phases-v1'},deadline=600)
 def signed(self,p):return dict(payload=p,signer=self.authority.id,signature=base64.b64encode(self.authority.key.sign(canonical(p)).signature).decode())
 def test_new_dispatch_and_restart_adopt_exact_original_without_wait_or_relaunch(self):
  with patch('subnet.remote_backend.time.sleep',side_effect=AssertionError('no waiting')):
   first=self.jobs.run('e-mine','mine',self.m,dispatch_only=True,miner_id='owned')
   self.assertTrue(first['new_job_started']);self.assertFalse(first['terminal_observed']);self.assertNotIn('success',first)
   original=(self.jobs.state/(first['original_job_id']+'-job.json')).read_bytes();commands=self.jobs.command.call_count
   second=self.jobs.run('e-mine','mine',self.m,dispatch_only=True,miner_id='owned',capability={'new_url':'never substituted'})
   self.assertFalse(second['new_job_started']);self.assertEqual(first['original_job_id'],second['original_job_id']);self.assertEqual(original,(self.jobs.state/(first['original_job_id']+'-job.json')).read_bytes());self.assertEqual(self.jobs.command.call_count,commands);self.assertEqual(self.jobs.copy_to.call_count,1);self.jobs.remote_status.assert_not_called()
 def test_live_absent_ambiguous_or_timeout_prior_blocks_new_physical_job(self):
  first=self.jobs.run('e-mine','mine',self.m,dispatch_only=True,miner_id='owned');commands=self.jobs.command.call_count
  for status in ({'phase':'running'},{'phase':'not_launched'},{'phase':'unknown'},TimeoutError('SSH')):
   self.jobs.remote_status=Mock(side_effect=status if isinstance(status,Exception)else None,return_value=status)
   with self.subTest(status=status),self.assertRaises(RemoteMinerReserved):self.jobs.run('next-mine','mine',dict(self.m,epoch='next'),dispatch_only=True,miner_id='owned')
   self.assertFalse((self.jobs.state/'next-mine.json').exists());self.assertEqual(self.jobs.command.call_count,commands)
 def test_terminal_original_is_retained_and_releases_without_success_claim(self):
  for phase in ('complete','failed'):
   with self.subTest(phase=phase):
    self.setUp();first=self.jobs.run('e-mine','mine',self.m,dispatch_only=True,miner_id='owned');self.jobs.remote_status=Mock(return_value={'phase':phase,'runner_pid':12,'exit_code':0 if phase=='complete'else 1})
    second=self.jobs.run('next-mine','mine',dict(self.m,epoch='next'),dispatch_only=True,miner_id='owned');self.assertNotEqual(first['original_job_id'],second['original_job_id']);self.assertTrue((self.jobs.state/(first['original_job_id']+'-physical-terminal.json')).exists());self.jobs.copy_from.assert_not_called()
 def test_original_manifest_or_identity_cannot_change(self):
  self.jobs.run('e-mine','mine',self.m,dispatch_only=True,miner_id='owned')
  for m,miner in ((dict(self.m,deadline=601),'owned'),(self.m,'other')):
   with self.assertRaisesRegex(ValueError,'original miner request changed'):self.jobs.run('e-mine','mine',m,dispatch_only=True,miner_id=miner)
 def test_no_dispatch_only_other_roles_or_historical_policy(self):
  for role,m in (('train',self.m),('evaluate',self.m),('mine',dict(epoch='e',checkpoint={'id':'cp'}))):
   with self.assertRaises(ValueError):self.jobs.run('x',role,m,dispatch_only=True)

class Scheduler(unittest.TestCase):
 def test_real_service_captures_at_deadline_while_original_miner_remains_live(self):
  from test_capture_status import Controls
  from subnet.gpu_service import run
  c,m,status,g,ids=Controls().fixture();status['initial_published']=True;status['active'].update(phase='mine',identities=[ids[0].id]);clock=[1.]
  save(c.state/'controller.json',status);save(c.state/'e-manifest.json',m)
  dispatched=[]
  def dispatch(*a,**kw):
   self.assertTrue(kw['dispatch_only']);dispatched.append(a[0]);return dict(original_job_id='still-live',dispatch_only=True,new_job_started=True,terminal_observed=False)
  c.jobs.run=Mock(side_effect=dispatch)
  def finalize(manifest,cache):
   self.assertEqual(clock[0],20);self.assertEqual(json.loads((c.state/'controller.json').read_text())['active']['phase'],'collect');raise RuntimeError('captured at deadline with live original')
  c.finalize=finalize
  cfg=dict(state=str(c.state),bucket={},remote={'workspace':'/owned'},source_bundle={},owned_miner_dispatch=True,search_budget=2)
  with patch('subnet.gpu_service.time.time',side_effect=lambda:clock[0]),patch('subnet.gpu_service.time.sleep',side_effect=lambda n:clock.__setitem__(0,min(20,clock[0]+n))),patch('subnet.gpu_service.Bucket',return_value=g.bucket),patch('subnet.gpu_service.Gateway',return_value=g),patch('subnet.gpu_service.RemoteController',return_value=c),patch('subnet.gpu_service.ChainAdapter'),patch('subnet.gpu_service.owned_dispatch_allowed',return_value=True),patch('subnet.gpu_service.owned_dispatch_identities',return_value=[ids[0].id]),patch('subnet.gpu_service.owned_mining_job_fields',return_value={}),patch('subnet.gpu_service.log.exception'),self.assertRaisesRegex(RuntimeError,'captured at deadline'):
   # Use legacy staging here: hourly scheduling is independent of artifact transport.
   m.pop('submission_transport_policy');save(c.state/'e-manifest.json',m)
   run(cfg,once=True)
  self.assertEqual(len(dispatched),1);self.assertLess(clock[0],30)

class PhysicalProbe(unittest.TestCase):
 def test_report_before_child_wait_does_not_release_physical_miner(self):
  from subnet.remote_runner import probe
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);(p/'jobs'/'original').mkdir(parents=True);(p/'jobs'/'original'/'report.json').write_text('{}');(p/'runner-status').mkdir()
   (p/'runner-status'/'original.json').write_text(json.dumps(dict(phase='running',runner_pid=10,runner_pid_ticks='ticks',child_pid=11,child_pid_ticks='ticks')))
   with patch('subnet.remote_runner.ticks',return_value='ticks'):
    self.assertEqual(probe(d,'original')['phase'],'complete');self.assertEqual(probe(d,'original',physical=True)['phase'],'running')
   (p/'runner-status'/'original.json').write_text(json.dumps(dict(phase='complete',exit_code=0,finished_at=30)))
   self.assertEqual(probe(d,'original',physical=True)['phase'],'complete')
