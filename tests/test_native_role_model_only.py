"""Mocked loader and tiny-tensor controls, not pretrained numerical qualification."""
import hashlib,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from subnet import native_role_model_only as role

class ModelOnlyTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  for name,data in [('config.json',b'{}'),('model.safetensors',b'fake-test-only-safe-frame'),('tokenizer.json',b'{}')]:
   (self.root/name).write_bytes(data)
  self.files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.iterdir()}
 def test_empty_and_unsafe_maps_rejected_before_transformers_import(self):
  for files in [{},{'../config.json':'a'*64},{'config.json':'not-sha'},{'model.pt':'a'*64},{'/model.safetensors':'a'*64}]:
   with self.subTest(files=files),patch.dict(sys.modules,{'transformers':None}),self.assertRaises(ValueError):role.ModelOnlyCPURuntime(self.root,files)
 def test_extra_missing_modified_and_pickle_model_files_rejected(self):
  for mode in ['extra','missing','modified','pickle']:
   with self.subTest(mode=mode):
    self.setUp();self.addCleanup(self.tmp.cleanup)
    if mode=='extra':(self.root/'unexpected.json').write_bytes(b'{}')
    elif mode=='missing':(self.root/'tokenizer.json').unlink()
    elif mode=='modified':(self.root/'config.json').write_bytes(b'{"changed":1}')
    else:(self.root/'pytorch_model.bin').write_bytes(b'not-loaded')
    with self.assertRaises(ValueError):role.validate_checkpoint(self.root,self.files)
 def test_executable_symlink_and_nonregular_inputs_rejected(self):
  (self.root/'custom_model.py').write_bytes(b'raise RuntimeError("never execute")')
  with self.assertRaisesRegex(ValueError,'executable'):role.validate_checkpoint(self.root,self.files)
  (self.root/'custom_model.py').unlink();(self.root/'model.safetensors').unlink();(self.root/'model.safetensors').symlink_to(self.root/'config.json')
  with self.assertRaisesRegex(ValueError,'regular'):role.validate_checkpoint(self.root,self.files)
 def test_loader_has_no_environment_configure_or_session_dependency(self):
  import torch
  tokenizer=Mock();model=Mock();model.eval.return_value=model
  auto_token=types.SimpleNamespace(from_pretrained=Mock(return_value=tokenizer));auto_model=types.SimpleNamespace(from_pretrained=Mock(return_value=model))
  build=Mock();verify=Mock();parts=Mock(return_value='bounded-parts');poly=types.ModuleType('toploc.poly');toploc=types.ModuleType('toploc');toploc.build_proofs_base64=build;toploc.verify_proofs_base64=verify;toploc.poly=poly
  c=types.ModuleType('toploc.C');csrc=types.ModuleType('toploc.C.csrc');utils=types.ModuleType('toploc.C.csrc.utils');utils.get_fp_parts=parts
  fake={'transformers':types.SimpleNamespace(AutoTokenizer=auto_token,AutoModelForCausalLM=auto_model),'toploc':toploc,'toploc.poly':poly,'toploc.C':c,'toploc.C.csrc':csrc,'toploc.C.csrc.utils':utils,'subnet.environments':None,'subnet.harness':None,'subnet.model':None}
  with patch.dict(sys.modules,fake),patch.object(torch,'set_num_threads') as threads:
   runtime=role.ModelOnlyCPURuntime(self.root,self.files)
   threads.assert_called_once_with(4);self.assertIs(runtime.tokenizer,tokenizer);self.assertIs(runtime.model,model);self.assertFalse(hasattr(runtime,'configure'));self.assertFalse(hasattr(runtime,'spec'));self.assertIs(runtime.build_proofs,build)
   auto_model.from_pretrained.assert_called_once_with(self.root,local_files_only=True,dtype=torch.float32,attn_implementation='eager',trust_remote_code=False,use_safetensors=True)
   auto_token.from_pretrained.assert_called_once_with(self.root,local_files_only=True,trust_remote_code=False)
   self.assertEqual(poly.get_fp_parts('tensor'),'bounded-parts');parts.assert_called_once_with('tensor',num_threads=4)
 def test_threads_cannot_change_declared_numerical_policy(self):
  for threads in [1,True,4.0]:
   with self.subTest(threads=threads),self.assertRaisesRegex(ValueError,'four-thread'):role.ModelOnlyCPURuntime(self.root,self.files,threads=threads)
 def test_exact_qualified_forward_slices_and_bf16_hidden_segments(self):
  import torch,numpy as np
  runtime=role.ModelOnlyCPURuntime.__new__(role.ModelOnlyCPURuntime)
  hidden=torch.arange(20,dtype=torch.float32).reshape(1,5,4);logits=torch.arange(35,dtype=torch.float32).reshape(1,5,7)/10
  def forward(tokens,**kwargs):
   self.assertEqual(tokens.tolist(),[[1,2,3,4,5]]);self.assertEqual(kwargs,{'output_hidden_states':True,'use_cache':False});return types.SimpleNamespace(hidden_states=[hidden],logits=logits)
  runtime.model=forward;acts,lp=runtime.compute([1,2,3],[4,5])
  self.assertEqual([tuple(a.shape) for a in acts],[(3,4),(1,4),(1,4)]);self.assertTrue(all(a.dtype==torch.bfloat16 and a.is_contiguous() for a in acts));np.testing.assert_array_equal(lp,torch.log_softmax(logits[0,2:4].float(),-1).numpy());self.assertEqual(lp.dtype,np.float32)

 def test_unpinned_nested_artifact_tree_rejected_and_readme_ignored(self):
  (self.root/'README.md').write_text('benign documentation');role.validate_checkpoint(self.root,self.files)
  nested=self.root/'nested';nested.mkdir();(nested/'model.safetensors').write_bytes(b'unpinned')
  with self.assertRaisesRegex(ValueError,'flat regular'):role.validate_checkpoint(self.root,self.files)
