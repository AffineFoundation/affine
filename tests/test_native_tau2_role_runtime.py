"""Wrapper admission controls, not numerical or cross-host qualification."""
import copy,hashlib,sys,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from subnet import native_tau2_role_runtime as role
class Runtime(unittest.TestCase):
 def descriptor(self):
  root=Path(role.__file__).resolve().parent.parent
  return {'kind':'auxiliary','model_runtime_revision':role.REVISION,'runtime_profile':copy.deepcopy(role.PROFILE),'renderer':'native-tau2-complete-chat-tools-v2','interpreter_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),'runtime_versions':{'torch':'approved'},'source_files':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['subnet/native_role_model_only.py','subnet/native_tau2_role_runtime.py']},'checkpoint':{'files':{'config.json':'a'*64}},'vocab_size':49152,'max_context':8192}
 def controls(self):
  return patch.multiple(role,role_descriptor=Mock(),profile=Mock(),verify_sources=Mock(),version=Mock(return_value='approved'))
 def test_exact_model_only_loader_and_geometry(self):
  d=self.descriptor();loaded=Mock();loaded.model.config=types.SimpleNamespace(vocab_size=49152,max_position_embeddings=8192)
  with self.controls(),patch.object(role,'ModelOnlyCPURuntime',return_value=loaded) as loader:
   runtime=role.ModelOnlyRoleCPU('trusted-weights',d)
   loader.assert_called_once_with('trusted-weights',d['checkpoint']['files'],threads=4);self.assertEqual(runtime.approved_descriptor(),d)
   runtime.compute([1],[2]);loaded.compute.assert_called_once_with([1],[2]);d['max_context']=1;self.assertEqual(runtime.approved_descriptor()['max_context'],8192)
 def test_old_runtime_and_extra_context_rejected(self):
  for field,value in [('model_runtime_revision','cpu-float32-eager-v2-bounded-toploc'),('renderer','different'),('runtime_profile',{'threads':1})]:
   d=self.descriptor();d[field]=value
   with self.controls(),patch.object(role,'ModelOnlyCPURuntime') as loader,self.assertRaises(ValueError):role.ModelOnlyRoleCPU('path',d)
   loader.assert_not_called()
 def test_loader_source_substitution_rejected_before_model(self):
  d=self.descriptor();d['source_files']['subnet/native_role_model_only.py']='0'*64
  with self.controls(),patch.object(role,'ModelOnlyCPURuntime') as loader,self.assertRaises(ValueError):role.ModelOnlyRoleCPU('path',d)
  loader.assert_not_called()
 def test_loaded_vocabulary_or_context_drift_rejected(self):
  for vocab,ctx in [(50000,8192),(49152,4096)]:
   loaded=Mock();loaded.model.config=types.SimpleNamespace(vocab_size=vocab,max_position_embeddings=ctx)
   with self.controls(),patch.object(role,'ModelOnlyCPURuntime',return_value=loaded),self.assertRaises(ValueError):role.ModelOnlyRoleCPU('path',self.descriptor())
if __name__=='__main__':unittest.main()

class Registry(unittest.TestCase):
 def test_old_and_uploaded_runtime_revisions_fail_closed(self):
  for revision in ['cpu-float32-eager-v2-bounded-toploc','subnet.untrusted:Runtime',None]:
   with patch.object(role,'ModelOnlyRoleCPU') as cpu,self.assertRaises(ValueError):role.load_role('path',{'model_runtime_revision':revision})
   cpu.assert_not_called()
 def test_auxiliary_cannot_choose_agent_cuda_dispatch(self):
  with self.assertRaises(ValueError):role.load_role('path',{'model_runtime_revision':role.CUDA_REVISION,'kind':'auxiliary','training_eligible':False})
 def test_exact_cpu_registry_target(self):
  d={'model_runtime_revision':role.REVISION}
  with patch.object(role,'ModelOnlyRoleCPU',return_value='approved') as cpu:
   self.assertEqual(role.load_role('path',d),'approved');cpu.assert_called_once_with('path',d)
