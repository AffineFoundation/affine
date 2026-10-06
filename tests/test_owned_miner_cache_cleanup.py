import copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from subnet.backend_jobs import canonical,file_map
from subnet.cache_lifecycle import CacheLifecycle
from ops.owned_miner_cache_cleanup import VERSION,retire
from test_owned_cached_evaluation import SignedOwnedJobControls

class OwnedMinerTerminalDisposalControls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);f=SignedOwnedJobControls();f.setUp();self.f=f
  self.files={name:hashlib.sha256(name.encode()).hexdigest()for name in ['config.json','model.safetensors']};self.cp={'id':file_map(self.files),'files':self.files};self.directory=self.root/'checkpoints'/self.cp['id'];self.directory.mkdir(parents=True)
  for name in self.files:(self.directory/name).write_bytes(name.encode())
  self.cache=CacheLifecycle(self.root)
  with self.cache.lease_checkpoint(self.cp['id']):self.cache.record_checkpoint(self.cp['id'],self.files)
  self.job=copy.deepcopy(f.job);m=copy.deepcopy(f.m);m['checkpoint']=self.cp;m['source_bundle']={'sha256':'4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373'};self.job['manifest']=f.sign(m);self.job['role']='mine';self.job['miner_id']='owned-miner'
  self.report=dict(job_id=self.job['job_id'],checkpoint=self.cp['id'],job_sha256=hashlib.sha256(canonical(self.job)).hexdigest(),success=True,miner_id='owned-miner')
  self.ack=dict(version=VERSION,workspace=str(self.root),original_job=f.sign(self.job),original_report=self.report,job_sha256=self.report['job_sha256'],report_sha256=hashlib.sha256(canonical(self.report)).hexdigest(),checkpoint=self.cp,durable_report_full_readback=True,miner_id='owned-miner')
  (self.root/'runner-status').mkdir();self.marker=self.root/'runner-status'/(self.job['job_id']+'.json');self.marker.write_text(json.dumps(dict(job_id=self.job['job_id'],phase='complete',exit_code=0,runner_pid=99999999,runner_pid_ticks='1',child_pid=99999998,child_pid_ticks='2')))
 def run_cleanup(self):return retire(self.f.sign(self.ack),self.f.authority,str(self.root),self.job['source_files'],'4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373')
 def test_genuine_ACK_removes_only_exact_completed_owned_model(self):
  extra=self.root/'checkpoints'/'unrelated';extra.mkdir();(extra/'keep').write_bytes(b'keep');report=self.root/'report.json';report.write_bytes(b'preserve')
  r=self.run_cleanup();self.assertEqual(r['removed'],[self.cp['id']]);self.assertFalse(self.directory.exists());self.assertTrue(extra.exists());self.assertTrue(report.exists());self.assertTrue(self.marker.exists())
 def test_no_ack_bad_source_wrong_report_map_and_wrong_namespace_rejected(self):
  for change in ['durability','source','report','map','namespace']:
   saved=copy.deepcopy(self.ack)
   if change=='durability':self.ack['durable_report_full_readback']=False
   if change=='source':
    changed=copy.deepcopy(self.job);changed['source_files']={};self.ack['original_job']=self.f.sign(changed)
   if change=='report':self.ack['report_sha256']='0'*64
   if change=='map':self.ack['checkpoint']={'id':'wrong','files':{}}
   if change=='namespace':self.ack['workspace']=str(self.root.parent)
   with self.subTest(change=change),self.assertRaises((ValueError,KeyError)):self.run_cleanup()
   self.assertTrue(self.directory.exists());self.ack=saved
 def test_active_inherited_checkpoint_lease_defers_then_releases(self):
  with self.cache.lease_checkpoint(self.cp['id']):
   self.assertEqual(self.run_cleanup()['status'],'deferred');self.assertTrue(self.directory.exists())
  self.assertEqual(self.run_cleanup()['removed'],[self.cp['id']])
 def test_changed_inode_or_hardlink_never_deleted(self):
  p=self.directory/'config.json';old=p.read_bytes();p.unlink();p.write_bytes(old)
  self.assertEqual(self.run_cleanup()['status'],'deferred');self.assertTrue(p.exists())
  with self.cache.lease_checkpoint(self.cp['id']):self.cache.record_checkpoint(self.cp['id'],self.files)
  os.link(p,self.root/'external-hardlink');self.assertEqual(self.run_cleanup()['status'],'deferred');self.assertTrue(p.exists())
 def test_live_original_process_never_deleted(self):
  marker=json.loads(self.marker.read_text());marker.update(runner_pid=os.getpid(),runner_pid_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]);self.marker.write_text(json.dumps(marker))
  self.assertEqual(self.run_cleanup()['reason'],'original-process-still-live');self.assertTrue(self.directory.exists())
 def test_nonterminal_marker_and_missing_hydration_receipt_rejected(self):
  marker=json.loads(self.marker.read_text());marker['phase']='running';self.marker.write_text(json.dumps(marker))
  with self.assertRaisesRegex(ValueError,'terminal'):self.run_cleanup()
  marker['phase']='complete';self.marker.write_text(json.dumps(marker));self.cache._receipt(self.cp['id']).unlink()
  with self.assertRaisesRegex(ValueError,'hydration'):self.run_cleanup()
  self.assertTrue(self.directory.exists())
 def test_external_receipt_path_or_extra_members_rejected(self):
  p=self.cache._receipt(self.cp['id']);v=json.loads(p.read_text());v['path']='../external';p.write_text(json.dumps(v))
  with self.assertRaisesRegex(ValueError,'hydration inventory'):self.run_cleanup()
  self.assertTrue(self.directory.exists())
 def test_other_active_namespace_job_protects_completed_owned_checkpoint(self):
  marker=dict(job_id='active',phase='running',runner_pid=os.getpid(),runner_pid_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]);(self.root/'runner-status/active.json').write_text(json.dumps(marker))
  self.assertEqual(self.run_cleanup()['reason'],'owned-namespace-active-process');self.assertTrue(self.directory.exists())
 def test_wrong_miner_or_unsuccessful_original_refused(self):
  for key,value in [('miner_id','foreign'),('success',False)]:
   old=copy.deepcopy(self.ack);self.ack['original_report'][key]=value;self.ack['report_sha256']=hashlib.sha256(canonical(self.ack['original_report'])).hexdigest()
   with self.assertRaises(ValueError):self.run_cleanup()
   self.assertTrue(self.directory.exists());self.ack=old
 def test_symlink_checkpoint_file_refused_without_touching_target(self):
  p=self.directory/'config.json';p.unlink();target=self.root/'external';target.write_bytes(b'protected');p.symlink_to(target)
  self.assertEqual(self.run_cleanup()['status'],'deferred');self.assertTrue(target.exists())
if __name__=='__main__':unittest.main()

class MinerDurabilityAdapterControls(OwnedMinerTerminalDisposalControls):
 def test_failed_full_R2_ACK_never_invokes_remote_cleanup(self):
  from ops.owned_miner_cache_cleanup import retire_completed
  state=self.root/'local';(state/'roles').mkdir(parents=True);(state/'roles'/(self.job['job_id']+'-job.json')).write_bytes(canonical(self.f.sign(self.job)))
  out=self.root/'output';out.mkdir();(out/(self.job['job_id']+'-original-report.json')).write_bytes(canonical(self.report))
  remote=SimpleNamespace(workspace=str(self.root),code='source',python='python',metadata={'source_files':self.job['source_files']},remote_status=Mock(return_value=dict(phase='complete',exit_code=0)),checked=Mock(return_value=self.report),command=Mock(),copy_from=Mock(),copy_to=Mock())
  controller=SimpleNamespace(state=state,authority=SimpleNamespace(id=self.f.authority),signed=self.f.sign,bucket=SimpleNamespace(get=Mock(return_value=b'corrupt'),put=Mock()))
  with self.assertRaisesRegex(ValueError,'ACK full readback'):retire_completed(controller,remote,{'4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373':self.job['source_files']},output=out,owned_miners=['owned-miner'])
  remote.command.assert_not_called();remote.copy_to.assert_not_called();self.assertTrue(self.directory.exists())
 def test_running_original_is_neutral_deferred_without_report_fetch(self):
  from ops.owned_miner_cache_cleanup import retire_completed
  state=self.root/'local';(state/'roles').mkdir(parents=True);(state/'roles'/(self.job['job_id']+'-job.json')).write_bytes(canonical(self.f.sign(self.job)))
  remote=SimpleNamespace(remote_status=Mock(return_value=dict(phase='running')),copy_from=Mock(),command=Mock())
  controller=SimpleNamespace(state=state,authority=SimpleNamespace(id=self.f.authority));result=retire_completed(controller,remote,{'4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373':self.job['source_files']},output=self.root/'output',owned_miners=['owned-miner'])
  self.assertEqual(result,[dict(status='deferred',reason='original-mine-not-successfully-terminal')]);remote.copy_from.assert_not_called();remote.command.assert_not_called()

class MinerOperatorScopeControls(unittest.TestCase):
 def test_exact_entry_config_route_source_and_private_authority_preflight(self):
  from ops import automatic_owned_miner_cleanup as op
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);tree=root/'tree';(tree/'subnet').mkdir(parents=True);runtime=tree/'subnet/one.py';runtime.write_bytes(b'original');state=root/'state';state.mkdir();seed=state/'authority.seed';seed.write_bytes(b'existing');seed.chmod(0o600);config=dict(state=str(state),source_bundle={'sha256':'source'},remote={'roles':{'mine':{'host':'fixed'}}});path=root/'config.json';path.write_bytes(canonical(config));cleanup=root/'cleanup.py';cleanup.write_bytes(b'CPU-only')
   scope=dict(version='owned-miner-terminal-cleanup-operator-v1',created_at=0,expires_at=10**12,config_sha256=op.sha(path),state=str(state),max_jobs_per_cycle=8,sources={'source':{'subnet/one.py':op.sha(runtime)}},miner_endpoint={'host':'fixed'},runtime_tree=str(tree),entrypoint_sha256=op.sha(op.__file__),cleanup_module=str(cleanup),cleanup_module_sha256=op.sha(cleanup),controller={'unit':'exact.service','pid':1,'ticks':2,'invocation':'original'})
   with patch.object(op.subprocess,'check_output',return_value='MainPID=0\nInvocationID=\n'):
    self.assertEqual(op.validate(scope,config,path),tree)
    for key,value in [('config_sha256','bad'),('miner_endpoint',{'host':'foreign'}),('cleanup_module_sha256','bad'),('max_jobs_per_cycle',True)]:
     bad=copy.deepcopy(scope);bad[key]=value
     with self.subTest(key=key),self.assertRaises(ValueError):op.validate(bad,config,path)
    with patch.object(op.subprocess,'check_output',return_value='MainPID=42\nInvocationID=foreign\n'):
     with self.assertRaisesRegex(ValueError,'different controller'):op.validate(scope,config,path)
    runtime.write_bytes(b'mutated')
    with self.assertRaisesRegex(ValueError,'runtime'):op.validate(scope,config,path)

class DurableMinerScopeRenewalControls(unittest.TestCase):
 def test_expired_same_config_scope_renews_only_identical_controller(self):
  from ops import automatic_owned_miner_cleanup as op
  from nacl.signing import SigningKey
  import time
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);tree=root/'tree';(tree/'subnet').mkdir(parents=True);runtime=tree/'subnet/one.py';runtime.write_bytes(b'original');state=root/'state';state.mkdir();key=SigningKey.generate();seed=state/'authority.seed';seed.write_text(key.encode().hex());seed.chmod(0o600)
   config=dict(state=str(state),source_bundle={'sha256':'source'},remote={'roles':{'mine':{'host':'fixed'}}});path=root/'config.json';path.write_bytes(canonical(config));cleanup=root/'cleanup.py';cleanup.write_bytes(b'CPU-only');unit=root/'controller.service';unit.write_bytes(b'exact unit');launcher=root/'launch.py';launcher.write_bytes(b'exact launcher')
   scope=dict(version='owned-miner-terminal-cleanup-operator-v1',created_at=0,expires_at=1,config_sha256=op.sha(path),state=str(state),max_jobs_per_cycle=8,sources={'source':{'subnet/one.py':op.sha(runtime)}},miner_endpoint={'host':'fixed'},runtime_tree=str(tree),entrypoint_sha256=op.sha(op.__file__),cleanup_module=str(cleanup),cleanup_module_sha256=op.sha(cleanup),controller={'unit':'exact.service','pid':1,'ticks':2,'invocation':'original'},renewal_authorization={'version':'identical-controller-cleanup-renewal-v1','unit_path':str(unit),'unit_sha256':op.sha(unit),'launcher_path':str(launcher),'launcher_sha256':op.sha(launcher),'process_argv_sha256':'not-used-idle'})
   target=root/'scope.json';target.write_bytes(b'original signed scope')
   reply='MainPID=0\nInvocationID=\nFragmentPath='+str(unit)+'\n'
   with patch.object(op.subprocess,'check_output',return_value=reply),patch.object(op,'AUTH',key.verify_key.encode().hex()):
    result=op.renew_authorized_scope(scope,config,path,target);self.assertGreater(result['expires_at'],time.time());self.assertEqual(result['expires_at']-result['created_at'],3600);self.assertEqual(op.authenticate(op.read(target)),result)
    restarted=copy.deepcopy(scope);restarted['renewal_authorization']['process_argv_sha256']=hashlib.sha256(Path('/proc/self/cmdline').read_bytes()).hexdigest()
    running='MainPID='+str(os.getpid())+'\nInvocationID=actual-restart\nFragmentPath='+str(unit)+'\n'
    with patch.object(op.subprocess,'check_output',return_value=running):
     rebound=op.renew_authorized_scope(restarted,config,path,target)
     self.assertEqual(rebound['controller']['pid'],os.getpid());self.assertEqual(rebound['controller']['invocation'],'actual-restart');self.assertEqual(rebound['miner_endpoint'],scope['miner_endpoint']);self.assertEqual(rebound['config_sha256'],scope['config_sha256'])
    for kind in ['foreign-route','new-source','changed-unit','changed-launcher','unknown-argv']:
     trial=copy.deepcopy(scope);original=target.read_bytes()
     if kind=='foreign-route':trial['miner_endpoint']={'host':'foreign'}
     if kind=='new-source':trial['sources']={}
     if kind=='changed-unit':trial['renewal_authorization']['unit_sha256']='0'*64
     if kind=='changed-launcher':trial['renewal_authorization']['launcher_sha256']='0'*64
     if kind=='unknown-argv':
      trial['renewal_authorization']['process_argv_sha256']='0'*64
      reply='MainPID='+str(os.getpid())+'\nInvocationID=foreign\nFragmentPath='+str(unit)+'\n'
     with self.subTest(kind=kind),patch.object(op.subprocess,'check_output',return_value=reply),self.assertRaises(ValueError):op.renew_authorized_scope(trial,config,path,target)
     self.assertEqual(target.read_bytes(),original)
