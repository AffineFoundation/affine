"""No CUDA/model load: strict descriptor and mocked adapter conformance."""
import copy,pathlib,types,unittest
from subnet import native_role_cuda as role
from subnet.long_context_runtime import POLICY,file_sha,digest

class RoleTests(unittest.TestCase):
 def setUp(self):
  root=pathlib.Path(__file__).resolve().parent.parent;names=['subnet/native_role_cuda.py','subnet/long_context_runtime.py','subnet/proofs.py','subnet/native_tau2_model.py','subnet/native_tau2_probe.py','subnet/harness.py'];files={'config.json':'1'*64,'model.safetensors':'2'*64}
  self.descriptor={'kind':'agent','training_eligible':True,'native_role_revision':role.REVISION,'renderer':role.RENDERER,'max_context':32768,'vocab_size':151936,'max_output_tokens':128,'runtime_profile':{'policy':POLICY,'native_role_revision':role.REVISION},'numerical_policy':{'logprobs_atol':1e-5,'logprobs_rtol':0,'TOPLOC_errors':0},'source_files':{n:file_sha(root/n) for n in names},'harness_source_sha256':file_sha(root/'subnet/harness.py'),'checkpoint':{'id':digest(files),'files':files}}
 def test_exact_source_and_model_descriptor_validates_without_loading(self):role.validate_descriptor(self.descriptor)
 def test_wrong_model_geometry_source_and_tolerances_rejected(self):
  for field in ['max_context','vocab_size','source','tolerance','policy_type']:
   d=copy.deepcopy(self.descriptor)
   if field=='max_context':d[field]=8192
   elif field=='vocab_size':d[field]=49152
   elif field=='source':d['source_files']['subnet/native_role_cuda.py']='a'*64
   elif field=='policy_type':d['runtime_profile']['policy']['tf32']=0
   else:d['numerical_policy']['logprobs_atol']=.01
   with self.subTest(field=field),self.assertRaises(ValueError):role.validate_descriptor(d)
 def test_adapter_never_configures_environment_or_auxiliary_model(self):
  calls=[];compute=lambda p,o:('acts','lp');build=lambda *a,**k:['proof'];verify=lambda *a,**k:[]
  def factory(path,files):
   calls.append((path,files));return types.SimpleNamespace(tokenizer=object(),model=types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=151936,max_position_embeddings=32768)),compute=compute,build_proofs=build,verify_proofs=verify)
  runtime=role.NativeAgentCUDARuntime('/approved',self.descriptor,runtime_factory=factory)
  self.assertEqual(len(calls),1);self.assertFalse(hasattr(runtime,'configure'));self.assertFalse(hasattr(runtime,'spec'));self.assertIs(runtime.compute,compute);self.assertIs(runtime.build_proofs,build)
  returned=runtime.approved_descriptor();returned['kind']='user';self.assertEqual(runtime.approved_descriptor()['kind'],'agent')
 def test_user_role_or_changed_filemap_rejected_before_factory(self):
  d=copy.deepcopy(self.descriptor);d['kind']='auxiliary'
  with self.assertRaises(ValueError):role.NativeAgentCUDARuntime('/unused',d,runtime_factory=lambda *_:self.fail('factory called'))
  d=copy.deepcopy(self.descriptor);d['checkpoint']['files']['config.json']='a'*64
  with self.assertRaisesRegex(ValueError,'checkpoint descriptor'):role.NativeAgentCUDARuntime('/unused',d,runtime_factory=lambda *_:self.fail('factory called'))
