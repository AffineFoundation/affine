import copy,json,shutil,sys,os
from pathlib import Path
import unittest
import test_durable_learner_service as fixtures
from ops import durable_learner_service as m,durable_audit_services as g

CAPTURE=dict(version='bounded-parallel-token-capture-v2',workers=8,
 max_document_bytes=2000000,max_inflight_bytes=16000000,completion_order='first-completed',
 journal_version='fsynced-per-epoch-capture-v1',state_checkpoint_documents=16)

class OverlayTests(unittest.TestCase):
 def setUp(self):
  self.fixture=fixtures.LearnerRecovery('test_restart_preserves_original_and_allows_epoch_advance');self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
  self.f=self.fixture.fixture;self.p=copy.deepcopy(self.fixture.p)
  # Keep all177 original scientific pins; replace one fixture filename so
  # prepare_runtime can exercise a genuine import target without model work.
  (self.f.runtime/'subnet/file_0.py').rename(self.f.runtime/'subnet/gpu_service.py')
  source=g.signed(g.read(self.p['source_approval']['path']),self.f.auth)
  files={str(v.relative_to(self.f.runtime)):g.file_hash(v)for v in self.f.runtime.rglob('*.py')}
  source['full_source_files']=source['runtime_source_files']=files
  self.p['source_approval']=self.f.document('source-approval.json',source)
  self.overlay=self.f.root/'coordinator-overlay';shutil.copytree(self.f.runtime,self.overlay)
  (self.overlay/'subnet/gpu_service.py').write_text("ORIGIN='CPU-overlay'\nMODEL_LOADED=False\n")
  overrides={'subnet/gpu_service.py':g.file_hash(self.overlay/'subnet/gpu_service.py')}
  self.p['operator_overlay']=dict(version=m.OVERLAY_VERSION,root=str(self.overlay),full_source_files=dict(files,**overrides),overrides=overrides,baseline_source_sha256=self.f.source,baseline_inventory_sha256=g.digest(files),learner_capture_policy=CAPTURE)
  cfg=g.read(self.f.config);cfg.update(learner_capture_policy=CAPTURE,submission_transport_policy='small-commitment-pairs-v2',hourly_execution_policy={'fixture':True});self.f.config.write_text(json.dumps(cfg));self.p['config']['file_sha256']=g.file_hash(self.f.config)
 def validate(self,p=None):return m.validate_policy(self.f.sign(p or self.p),self.f.auth)
 def test_signed_overlay_validates_without_changing_remote_source_or_original_status(self):
  status=self.fixture.status.read_bytes();self.validate();self.assertEqual(self.fixture.status.read_bytes(),status)
  self.assertEqual(self.p['source_sha256'],self.f.source);self.assertEqual(len(g.signed(g.read(self.p['source_approval']['path']),self.f.auth)['runtime_source_files']),177)
 def test_actual_import_uses_pinned_overlay_and_does_not_load_model(self):
  self.validate();saved={n:v for n,v in sys.modules.items()if n=='subnet'or n.startswith('subnet.')};paths=sys.path[:]
  try:
   module=m.prepare_runtime(self.p);self.assertEqual(module.ORIGIN,'CPU-overlay');self.assertFalse(module.MODEL_LOADED);self.assertEqual(Path(module.__file__),self.overlay/'subnet/gpu_service.py')
  finally:
   for n in list(sys.modules):
    if n=='subnet'or n.startswith('subnet.'):del sys.modules[n]
   sys.modules.update(saved);sys.path[:]=paths
 def test_unlisted_or_scientific_override_rejected_even_with_valid_ROOT_signature(self):
  for name in ['subnet/gpu_runtime.py','subnet/epoch_optimizer.py']:
   p=copy.deepcopy(self.p);p['operator_overlay']['overrides'][name]='a'*64;p['operator_overlay']['full_source_files'][name]='a'*64
   with self.assertRaises(ValueError):self.validate(p)
 def test_overlay_extra_file_or_changed_pin_refuses(self):
  p=self.overlay/'extra.py';p.write_text('unlisted')
  with self.assertRaises(ValueError):self.validate()
  p.unlink();(self.overlay/'subnet/gpu_service.py').write_text('drift')
  with self.assertRaises(ValueError):self.validate()
 def test_baseline_sha_inventory_or_member_mutation_refuses(self):
  for k in ['baseline_source_sha256','baseline_inventory_sha256']:
   p=copy.deepcopy(self.p);p['operator_overlay'][k]='0'*64
   with self.assertRaises(ValueError):self.validate(p)
  (self.f.runtime/'subnet/backend_jobs.py').write_text('mutated')
  with self.assertRaises(ValueError):self.validate()
 def test_policy_bool_or_different_signed_config_capture_refuses(self):
  p=copy.deepcopy(self.p);p['operator_overlay']['learner_capture_policy']['state_checkpoint_documents']=True
  with self.assertRaises(ValueError):self.validate(p)
  p=copy.deepcopy(self.p);p['operator_overlay']['learner_capture_policy']['workers']=16;p['operator_overlay']['learner_capture_policy']['max_inflight_bytes']=32000000
  with self.assertRaises(ValueError):self.validate(p)
 def test_same_root_symlink_or_foreign_owner_refuses(self):
  p=copy.deepcopy(self.p);p['operator_overlay']['root']=str(self.f.runtime)
  with self.assertRaises(ValueError):self.validate(p)
  link=self.f.root/'overlay-link';link.symlink_to(self.overlay,target_is_directory=True);p['operator_overlay']['root']=str(link)
  with self.assertRaises(ValueError):self.validate(p)
  from unittest.mock import patch
  original=Path.lstat
  def foreign(path):
   st=original(path)
   if path==self.overlay:
    from types import SimpleNamespace
    return SimpleNamespace(st_mode=st.st_mode,st_uid=os.getuid()+1)
   return st
  with patch('pathlib.Path.lstat',foreign):
   with self.assertRaises(ValueError):self.validate()
 def test_overlay_none_unknown_fields_and_dropped_inventory_refuse(self):
  p=copy.deepcopy(self.p);p['operator_overlay']=None
  with self.assertRaises(ValueError):self.validate(p)
  p=copy.deepcopy(self.p);p['operator_overlay']['allow_unsigned']=True
  with self.assertRaises(ValueError):self.validate(p)
  p=copy.deepcopy(self.p);p['operator_overlay']['full_source_files'].pop('subnet/backend_jobs.py')
  with self.assertRaises(ValueError):self.validate(p)

if __name__=='__main__':unittest.main()
