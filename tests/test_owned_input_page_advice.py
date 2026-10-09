import contextlib,copy,hashlib,importlib.util,io,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from ops.trainer_lifecycle import owned_input_page_advice as peer
class PostloadAdvice(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.target=self.root/'checkpoints'/('a'*64);self.target.mkdir(parents=True);self.files={'config.json':b'{}','model.safetensors':b'original-immutable-model'}
  for n,b in self.files.items():(self.target/n).write_bytes(b)
  self.manifest={'epoch':'epoch-91','checkpoint':{'id':'a'*64,'files':{n:hashlib.sha256(b).hexdigest()for n,b in self.files.items()}}};self.guard=Mock(return_value={'original_guard':True})
  guard=self.guard
  class StateCache:
   def admit(cache,plan,**kw):return guard(cache,plan,**kw)
  self.module=SimpleNamespace(StateCache=StateCache);self.cache=StateCache();self.cache.workspace=self.root;self.cache.manifest=self.manifest
  self.owner=peer.OwnedInputPageAdvice();self.owner.observe=Mock(side_effect=[{'available_ram_bytes':10},{'available_ram_bytes':20}]);self.owner.install_cache(self.module)
  self.backend=SimpleNamespace(checkpoint=Mock(return_value=self.target));self.owner.install_checkpoint(self.backend)
 def authenticate(self):self.backend.checkpoint(self.manifest,self.root)
 def run_admit(self):
  with contextlib.redirect_stdout(io.StringIO()):return self.cache.admit({'unchanged':True},reclaimable_parent_bytes=123)
 def test_original_checkpoint_authentication_precedes_stats_capture(self):
  self.authenticate();self.assertEqual(self.owner.context['files'],self.manifest['checkpoint']['files'])
 def test_missing_original_authentication_never_advises_or_calls_guard(self):
  with patch.object(os,'posix_fadvise')as advice,self.assertRaisesRegex(ValueError,'original authenticated'):self.run_admit()
  advice.assert_not_called();self.guard.assert_not_called()
 def test_original_checkpoint_failure_retains_no_authorization(self):
  self.backend=SimpleNamespace(checkpoint=Mock(side_effect=ValueError('original hash failed')));self.owner.install_checkpoint(self.backend)
  with self.assertRaisesRegex(ValueError,'original hash'):self.authenticate()
  self.assertIsNone(self.owner.context)
 def test_real_syscall_keeps_model_bytes_stats_and_original_guard_arguments(self):
  self.authenticate();before={n:peer.OwnedInputPageAdvice.snapshot((self.target/n).stat())for n in self.files};result=self.run_admit()
  self.guard.assert_called_once_with(self.cache,{'unchanged':True},reclaimable_parent_bytes=123);self.assertTrue(result['original_guard']);e=result['postload_input_page_advice'];self.assertEqual(e['advised_bytes'],sum(map(len,self.files.values())));self.assertEqual(e['bytes_deleted'],0);self.assertFalse(e['resource_guard_changed']);self.assertEqual(e['before']['available_ram_bytes'],10);self.assertEqual(e['after']['available_ram_bytes'],20)
  for n,b in self.files.items():self.assertEqual((self.target/n).read_bytes(),b);self.assertEqual(peer.OwnedInputPageAdvice.snapshot((self.target/n).stat()),before[n])
 def test_original_insufficient_RAM_guard_still_fails(self):
  self.authenticate();self.guard.side_effect=ValueError('alternating optimizer memory and model disk budget')
  with self.assertRaisesRegex(ValueError,'alternating optimizer'):self.run_admit()
  self.guard.assert_called_once()
 def test_changed_model_before_advice_refuses(self):
  self.authenticate();(self.target/'model.safetensors').write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'changed before'):self.run_admit()
  self.guard.assert_not_called()
 def test_changed_model_during_advice_refuses(self):
  self.authenticate()
  def change(fd,*a):(self.target/'config.json').write_bytes(b'changed')
  with patch.object(os,'posix_fadvise',side_effect=change),self.assertRaisesRegex(ValueError,'during'):self.run_admit()
  self.guard.assert_not_called()
 def test_symlink_replacement_refuses(self):
  self.authenticate();p=self.target/'model.safetensors';p.unlink();p.symlink_to(self.target/'config.json')
  with self.assertRaises((OSError,ValueError)):self.run_admit()
  self.guard.assert_not_called()
 def test_hardlinked_checkpoint_refuses(self):
  os.link(self.target/'config.json',self.root/'outside-link')
  with self.assertRaisesRegex(ValueError,'unshared'):self.authenticate()
 def test_unapproved_manifest_inventory_refuses(self):
  self.authenticate();self.cache.manifest=copy.deepcopy(self.manifest);self.cache.manifest['checkpoint']['files']['config.json']='f'*64
  with self.assertRaisesRegex(ValueError,'original authenticated'):self.run_admit()
 def test_other_epoch_or_workspace_refuses(self):
  self.authenticate();self.cache.manifest=copy.deepcopy(self.manifest);self.cache.manifest['epoch']='another'
  with self.assertRaisesRegex(ValueError,'original authenticated'):self.run_admit()
 def test_outside_owned_checkpoint_root_refuses(self):
  self.backend=SimpleNamespace(checkpoint=Mock(return_value=self.root));self.owner.install_checkpoint(self.backend)
  with self.assertRaisesRegex(ValueError,'exact owned'):self.authenticate()
 def test_optimizer_files_untouched(self):
  optimizer=self.root/'.optimizer-state-cache';optimizer.mkdir();p=optimizer/'retained-state';p.write_bytes(b'never advise optimizer from input hook');old=p.stat();self.authenticate()
  actual=os.posix_fadvise;paths=[]
  def advise(fd,*args):paths.append(os.readlink('/proc/self/fd/'+str(fd)));return actual(fd,*args)
  with patch.object(os,'posix_fadvise',side_effect=advise):self.run_admit()
  self.assertTrue(all(str(self.target)in p for p in paths));self.assertEqual(p.read_bytes(),b'never advise optimizer from input hook');self.assertEqual(p.stat().st_ino,old.st_ino)
 def test_continuation_uses_original_authenticated_owned_export_cache(self):
  target=self.root/'jobs'/'previous-epoch-train'/'checkpoint-persistent-final';target.mkdir(parents=True)
  for n,b in self.files.items():(target/n).write_bytes(b)
  self.backend=SimpleNamespace(checkpoint=Mock(return_value=target));self.owner.install_checkpoint(self.backend)
  self.assertEqual(self.backend.checkpoint(self.manifest,self.root,target),target)
  self.assertEqual(self.owner.context['path'],str(target));self.run_admit();self.guard.assert_called_once()
  for n,b in self.files.items():self.assertEqual((target/n).read_bytes(),b)
 def test_explicit_cache_outside_workspace_or_hidden_optimizer_path_refuses(self):
  with tempfile.TemporaryDirectory()as other:
   for target in (Path(other),self.root/'.optimizer-state-cache'/'owned'):
    target.mkdir(parents=True,exist_ok=True)
    for n,b in self.files.items():(target/n).write_bytes(b)
    backend=SimpleNamespace(checkpoint=Mock(return_value=target));self.owner.install_checkpoint(backend)
    with self.assertRaisesRegex(ValueError,'exact owned'):backend.checkpoint(self.manifest,self.root,target)
if __name__=='__main__':unittest.main()
