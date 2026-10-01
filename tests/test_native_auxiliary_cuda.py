"""Descriptor controls only; genuine CUDA qualification lives in sealed jobs."""
import copy,json,pathlib,tempfile,unittest
from subnet.native_auxiliary_cuda import POLICY,REVISION,CHECKPOINT,validate_descriptor
from subnet.long_context_runtime import file_sha
ROOT=pathlib.Path(__file__).resolve().parent.parent
class TestNativeAuxiliaryCUDA(unittest.TestCase):
    def descriptor(self):
        fixed={'kind': 'auxiliary', 'training_eligible': False, 'checkpoint': {'files': {'chat_template.jinja': '872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788', 'config.json': 'ff877f0ce2c6168d60f4b349f591633d7fe5cc4d300dbf9cfb40634bb73a5684', 'generation_config.json': '21c38e9ee40390023368e1afdbe53bb1fc322557b04ebe9d75742f0c0c177b45', 'model.safetensors': 'c667673e379730e717781e5f449c3bff0a85713fed4bc5b7c46457f2b032fcc9', 'tokenizer.json': 'bf346d64f6f0fbcefb4c1b6928a98241467dff36c6fbae5fe1785c4ff90667f4', 'tokenizer_config.json': '61a6c7fcaf88a9f7e8cd21eebd10bccce6eb9b2a31c64d1efbdadfec5e86f97e'}, 'id': '39818e714a6e4e47b3fdd07e4eeb9cac619cf010fcc83a30068708531eac7d06'}, 'max_context': 8192, 'max_output_tokens': 128, 'vocab_size': 49152, 'renderer': 'native-tau2-complete-chat-tools-v2', 'numerical_policy': {'TOPLOC_errors': 0, 'logprobs_atol': 1e-05, 'logprobs_rtol': 0}, 'harness_source_sha256': '4b565259163fc1bf6df3775a89d10fd953a65cccd9423910b56a398f183bec04'}
        names=['subnet/native_auxiliary_cuda.py','subnet/long_context_runtime.py','subnet/proofs.py','subnet/native_tau2_model.py','subnet/harness.py']
        fixed.update(native_role_revision=REVISION,model_runtime_revision=REVISION,runtime_profile={'policy':POLICY,'native_role_revision':REVISION},source_files={n:file_sha(ROOT/n) for n in names})
        return fixed
    def test_fixed_auxiliary_mask_and_checkpoint(self):
        d=self.descriptor();validate_descriptor(d)
        for key,value in [('kind','agent'),('training_eligible',True),('max_context',32768),('vocab_size',151936),('model_runtime_revision','cpu-v2')]:
            bad=copy.deepcopy(d);bad[key]=value
            with self.assertRaises(ValueError):validate_descriptor(bad)
        bad=copy.deepcopy(d);bad['checkpoint']['files']['config.json']='0'*64
        with self.assertRaises(ValueError):validate_descriptor(bad)
    def test_strict_profile_source_numerical_types(self):
        d=self.descriptor()
        for path,value in [(('runtime_profile','policy','toploc_threads'),4),(('numerical_policy','TOPLOC_errors'),False),(('runtime_profile','policy','sdpa_backend'),'MATH')]:
            bad=copy.deepcopy(d);target=bad
            for k in path[:-1]:target=target[k]
            target[path[-1]]=value
            with self.assertRaises(ValueError):validate_descriptor(bad)
        bad=copy.deepcopy(d);bad['source_files']['subnet/native_auxiliary_cuda.py']='0'*64
        with self.assertRaises(ValueError):validate_descriptor(bad)
    def test_no_default_environment_constructor(self):
        text=(ROOT/'subnet/native_auxiliary_cuda.py').read_text()
        self.assertNotIn('EnvironmentSpec',text);self.assertNotIn('configure(',text)
        self.assertIn('SDPBackend.FLASH_ATTENTION',text)
