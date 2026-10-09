import copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from subnet.backend_jobs import canonical,file_map
from subnet.cache_lifecycle import CacheLifecycle
from ops.owned_cached_evaluator_cleanup import VERSION,retire,retire_completed
from test_owned_cached_evaluation import SignedOwnedJobControls

class OwnedTerminalDisposalControls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);f=SignedOwnedJobControls();f.setUp();self.f=f
  self.files={name:hashlib.sha256(name.encode()).hexdigest()for name in ['config.json','model.safetensors']};self.cp={'id':file_map(self.files),'files':self.files};self.directory=self.root/'checkpoints'/self.cp['id'];self.directory.mkdir(parents=True)
  for name in self.files:(self.directory/name).write_bytes(name.encode())
  self.cache=CacheLifecycle(self.root)
  with self.cache.lease_checkpoint(self.cp['id']):self.cache.record_checkpoint(self.cp['id'],self.files)
  self.job=copy.deepcopy(f.job);m=copy.deepcopy(f.m);m['checkpoint']=self.cp;m['source_bundle']={'sha256':'4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373'};self.job['manifest']=f.sign(m)
  self.report=dict(job_id=self.job['job_id'],checkpoint=self.cp['id'],job_sha256=hashlib.sha256(canonical(self.job)).hexdigest(),success=True)
  self.ack=dict(version=VERSION,workspace=str(self.root),original_job=f.sign(self.job),original_report=self.report,job_sha256=self.report['job_sha256'],report_sha256=hashlib.sha256(canonical(self.report)).hexdigest(),checkpoint=self.cp,durable_report_full_readback=True)
  (self.root/'runner-status').mkdir();self.marker=self.root/'runner-status'/(self.job['job_id']+'.json');self.marker.write_text(json.dumps(dict(job_id=self.job['job_id'],phase='complete',exit_code=0,runner_pid=99999999,runner_pid_ticks='1',child_pid=99999998,child_pid_ticks='2')))
 def run_cleanup(self):return retire(self.f.sign(self.ack),self.f.authority,str(self.root),self.job['source_files'])
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
 def test_durable_fullreadback_failure_prevents_remote_cleanup(self):
  state=self.root/'local-state';(state/'checkpoint-evaluations').mkdir(parents=True);(state/'roles').mkdir();label='newlabel';(state/'checkpoint-evaluations'/'request.json').write_text(json.dumps(dict(status='complete',request={'label':label})));(state/'roles'/(self.job['job_id']+'-job.json')).write_bytes(canonical(self.f.sign(self.job)));(state/'roles'/(self.job['job_id']+'-report.json')).write_bytes(canonical(self.report))
  remote=SimpleNamespace(workspace=str(self.root),code='source',python='python',config={'checkpoint_caches':{}},checked=Mock(return_value=self.report),command=Mock());jobs=SimpleNamespace(instance=Mock(return_value=remote),original=Mock(return_value=({'job_sha256':self.report['job_sha256']},self.job,self.f.m)));controller=SimpleNamespace(state=state,authority=SimpleNamespace(id=self.f.authority),signed=self.f.sign,bucket=SimpleNamespace(get=Mock(return_value=b'corrupt'),put=Mock()))
  with self.assertRaisesRegex(ValueError,'ACK readback'):retire_completed(controller,jobs,{'source_sha256':'qualified'})
  jobs.instance.assert_called_once_with('qualified',original=self.job)
  remote.command.assert_not_called();self.assertTrue(self.directory.exists())
if __name__=='__main__':unittest.main()
