import tempfile,unittest,shutil
from pathlib import Path
from ops.prepare_empty_recovery_source import prepare
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=Path(tempfile.mkdtemp());self.source=self.tmp/'source';self.source.mkdir();(self.source/'subnet').mkdir()
  root=Path(__file__).resolve().parents[1];self.helper=root/'subnet/empty_epoch_policy.py'
  # Apply the reviewed public replay patch to its tracked qualified base.
  shutil.copy2(root/'subnet/remote_backend.py',self.source/'subnet/remote_backend.py')
  import subprocess
  subprocess.run(['git','apply',str(root/'ops/replay_source_patches/remote_backend.py.patch')],cwd=self.source,check=True)
  code="""def contract(config,round_number):
    result_placeholder=None
    return dict(heldout_indices={},
        backend_profile=BACKEND_PROFILE,model_id=config.get('model_id','HuggingFaceTB/SmolLM2-1.7B-Instruct'))

def heldout(config,manifest):
    pass

def run():
            if active['phase']=='mine':
                if time.time()<manifest['deadline']:
                    pass
            if active['phase']=='collect':
                result,reports=controller.finalize(manifest,status['checkpoint_path']);save(state/(epoch+'-verified.json'),reports)
"""
  (self.source/'subnet/gpu_service.py').write_text(code)
  import hashlib
  from unittest.mock import patch
  from ops import prepare_empty_recovery_source as recipe
  qualified=dict(recipe.BASE)
  qualified['gpu_service.py']=hashlib.sha256(code.encode()).hexdigest()
  self.base_patch=patch.object(recipe,'BASE',qualified);self.base_patch.start();self.addCleanup(self.base_patch.stop)
 def tearDown(self):shutil.rmtree(self.tmp)
 def test_new_output_preserves_base_and_integrates_signed_policy(self):
  before={p.name:p.read_bytes() for p in (self.source/'subnet').glob('*.py')}
  output=self.tmp/'result';prepare(self.source,output,self.helper)
  self.assertEqual(before,{p.name:p.read_bytes() for p in (self.source/'subnet').glob('*.py')})
  remote=(output/'subnet/remote_backend.py').read_text();service=(output/'subnet/gpu_service.py').read_text()
  self.assertLess(remote.index("operator_test_policy=kwargs.pop"),remote.index('manifest=super().open'))
  self.assertIn("manifest['operator_test_policy']=operator_test_policy",remote)
  self.assertIn("dispatch_allowed(manifest) and time.time()<manifest['deadline']",service)
  self.assertIn('validate_empty_completion(manifest,result,reports)',service)
 def test_rejects_destination_alias_inside_source(self):
  alias=self.tmp/'alias';alias.symlink_to(self.source,target_is_directory=True)
  with self.assertRaises(ValueError):prepare(self.source,alias/'new',self.helper)
 def test_rejects_private_source_symlink(self):
  secret=self.tmp/'secret';secret.write_text('private');(self.source/'link').symlink_to(secret)
  with self.assertRaises(ValueError):prepare(self.source,self.tmp/'new',self.helper)
 def test_mutated_base_rejected_without_destination(self):
  p=self.source/'subnet/remote_backend.py';p.write_text(p.read_text()+'\n# mutation\n')
  target=self.tmp/'new'
  with self.assertRaises(ValueError):prepare(self.source,target,self.helper)
  self.assertFalse(target.exists())
 def controller(self):
  import importlib.util
  output=self.tmp/'runtime';prepare(self.source,output,self.helper)
  spec=importlib.util.spec_from_file_location('subnet._empty_reviewed_remote',output/'subnet/remote_backend.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  c=object.__new__(module.RemoteController)
  class Bucket:
   def __init__(self):self.writes=[]
   def json(self,key,value):self.writes.append((key,value))
  c.bucket=Bucket();c.state=self.tmp/'runtime-state';c.state.mkdir();c.signed=lambda value:dict(payload=value)
  return c
 def test_first_published_challenge_has_empty_policy(self):
  from subnet.controller import Controller
  from subnet.empty_epoch_policy import selected
  from unittest.mock import patch
  c=self.controller();policy=selected({'epoch_prefix':'nonpayable-test','controlled_empty_rounds':[8]},8)
  def open_base(instance,epoch,*args,**kwargs):
   self.assertNotIn('operator_test_policy',kwargs)
   value=dict(epoch=epoch,payable=False);instance.bucket.json('public/'+epoch+'/manifest.json',instance.signed(value));return value
  with patch.object(Controller,'open',open_base):
   value=c.open('nonpayable-test-8',{}, {},operator_test_policy=policy)
  self.assertEqual(value['operator_test_policy'],policy)
  self.assertEqual(len(c.bucket.writes),1)
  self.assertEqual(c.bucket.writes[0][1]['payload']['operator_test_policy'],policy)
 def test_unapproved_empty_policy_fails_before_open(self):
  from subnet.controller import Controller
  from unittest.mock import patch
  c=self.controller()
  with patch.object(Controller,'open') as base:
   with self.assertRaises(ValueError):c.open('nonpayable-test',{}, {},operator_test_policy={'miner_dispatch':False})
   base.assert_not_called();self.assertEqual(c.bucket.writes,[])
if __name__=='__main__':unittest.main()
