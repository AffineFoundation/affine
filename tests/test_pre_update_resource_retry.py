import sys,json,hashlib,tempfile,unittest,copy,time
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
from ops.trainer_lifecycle import pre_update_retry as r
class Controls(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.d=Path(self.t.name);self.root='root';self.epoch='epoch91'
  self.manifest=dict(epoch=self.epoch,checkpoint={'id':'base'},trainer_state_binding=dict(global_step_before=0,parent=None,genesis_sha256='genesis'))
  self.job=dict(job_id='original',manifest={'payload':self.manifest},submissions=[{'sha256':'data'}],steps=1,source_files={'math':'exact'},created_at=1,expires_at=2,persistent_training={'output_namespace':'old'},unaudited_training_execution={'x':'old'},learner_selection_operator_admission={'x':'old'})
  self.status=dict(job_id='original',phase='failed',exit_code=1,actual_wait=True)
  self.auth=dict(version=r.VERSION,epoch=self.epoch,original_job_id='original',original_job_sha256=r.digest(self.job),original_status_sha256=r.digest(self.status),worker_log_sha256=None,replacement_label=self.epoch+'-train-resource-r1',genesis_sha256='genesis',checkpoint='base',created_at=time.time()-1,expires_at=time.time()+60)
  self.log='worker: local_cache.admit(plan\nValueError: alternating optimizer memory and model disk budget\n';self.paths={k:self.d/k for k in ['authorization','original_job','original_status','worker_log']}
  self.row=dict(path='unused',retry_grant=str(self.d/'retry'),remote_retry_grant='/remote/retry',receipt=str(self.d/'receipt'))
  self.guards=SimpleNamespace(read=lambda x:json.loads(Path(x).read_bytes()),signed=lambda x:x['payload'],file_hash=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest())
  self.p=dict(authority=self.root,pre_update_memory_retry=self.row,prepared_reset_dispatch={'local_root':str(self.d)})
  (self.d/'reset.ROOT-SIGNED.private.json').write_text(json.dumps({'payload':{'original':'reset'}}));self.refresh()
 def refresh(self):
  self.paths['worker_log'].write_text(self.log);self.auth['worker_log_sha256']=self.guards.file_hash(self.paths['worker_log']);self.auth['original_job_sha256']=r.digest(self.job);self.auth['original_status_sha256']=r.digest(self.status)
  for k,v in [('authorization',{'payload':self.auth}),('original_job',{'payload':self.job}),('original_status',self.status)]:self.paths[k].write_text(json.dumps(v))
  for k,path in self.paths.items():self.row[k]=dict(path=str(path),sha256=self.guards.file_hash(path))
 def mounted(self):
  recovery=ModuleType('subnet.training_startup_recovery');recovery.label=lambda c,e:e+'-train'
  router=ModuleType('subnet.role_router')
  new=copy.deepcopy(self.job);new.update(job_id='retry',created_at=time.time(),expires_at=time.time()+60);new['persistent_training']={'output_namespace':'new'}
  self.new=new;path=self.d/'newjob'
  class Routed:pass
  def prepare(obj,label,manifest,submissions,steps):path.write_text(json.dumps({'payload':self.new}));return dict(job_path=str(path),already_issued=False)
  Routed.prepare_training_dispatch=prepare;router.RoutedJobs=Routed
  backend=ModuleType('subnet.remote_backend');backend.save=lambda p,v:Path(p).write_text(json.dumps(v))
  parent=ModuleType('subnet');parent.training_startup_recovery=recovery;parent.role_router=router
  ctx=patch.dict(sys.modules,{'subnet':parent,'subnet.training_startup_recovery':recovery,'subnet.role_router':router,'subnet.remote_backend':backend});ctx.start();self.addCleanup(ctx.stop)
  self.copies=[];obj=Routed();obj.controller=SimpleNamespace(authority=SimpleNamespace(id=self.root),signed=lambda x:{'payload':x});obj.roles={'train':SimpleNamespace(_approved_training_execution_client=SimpleNamespace(copy_to=lambda a,b:self.copies.append((a,b))))}
  r.install(self.p,self.guards);return obj,recovery
 def test_exact_preupdate_scope(self):r.validate_scope(self.p,self.guards)
 def test_signature_artifact_digest_drift(self):
  self.paths['original_job'].write_text('{}')
  with self.assertRaises(ValueError):r.validate_scope(self.p,self.guards)
 def test_wrong_terminal_rejected(self):
  for field,value in [('phase','running'),('exit_code',0),('actual_wait',False)]:
   with self.subTest(field=field):
    old=self.status[field];self.status[field]=value;self.refresh()
    with self.assertRaises(ValueError):r.validate_scope(self.p,self.guards)
    self.status[field]=old
 def test_postupdate_trace_rejected(self):
  self.log='train_epoch(\n'+self.log;self.refresh()
  with self.assertRaises(ValueError):r.validate_scope(self.p,self.guards)
 def test_wrong_genesis_rejected(self):
  self.auth['genesis_sha256']='other';self.refresh()
  with self.assertRaises(ValueError):r.validate_scope(self.p,self.guards)
 def test_parent_state_rejected(self):
  self.manifest['trainer_state_binding']['global_step_before']=1;self.refresh()
  with self.assertRaises(ValueError):r.validate_scope(self.p,self.guards)
 def test_label_scope_and_idempotent_grant(self):
  obj,labels=self.mounted();self.assertEqual(labels.label(obj.controller,'other'),'other-train');self.assertEqual(labels.label(obj.controller,self.epoch),self.auth['replacement_label'])
  obj.prepare_training_dispatch(self.auth['replacement_label'],self.manifest,self.job['submissions'],1);first=(self.d/'retry').read_bytes();obj.prepare_training_dispatch(self.auth['replacement_label'],self.manifest,self.job['submissions'],1)
  self.assertEqual(first,(self.d/'retry').read_bytes());self.assertEqual(len(self.copies),2)
 def test_changed_inputs_rejected_before_prepare(self):
  obj,_=self.mounted()
  with self.assertRaises(ValueError):obj.prepare_training_dispatch(self.auth['replacement_label'],self.manifest,[],1)
  self.assertFalse((self.d/'newjob').exists())
 def test_changed_scientific_bytes_rejected(self):
  obj,_=self.mounted();self.new['source_files']={'math':'changed'}
  with self.assertRaises(ValueError):obj.prepare_training_dispatch(self.auth['replacement_label'],self.manifest,self.job['submissions'],1)
  self.assertFalse((self.d/'retry').exists())
 def test_original_job_identity_rejected(self):
  obj,_=self.mounted();self.new['job_id']='original'
  with self.assertRaises(ValueError):obj.prepare_training_dispatch(self.auth['replacement_label'],self.manifest,self.job['submissions'],1)
 def test_expired_new_authorization_rejected(self):
  self.auth['expires_at']=time.time()-1;self.refresh();obj,_=self.mounted()
  with self.assertRaises(ValueError):obj.prepare_training_dispatch(self.auth['replacement_label'],self.manifest,self.job['submissions'],1)
if __name__=='__main__':unittest.main()
