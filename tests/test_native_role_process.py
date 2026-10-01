"""Real subprocess toy controls ONLY; not numerical/model qualification."""
import base64,hashlib,io,json,os,struct,sys
import numpy as np
import unittest
class Raises:
    @staticmethod
    def raises(error):return unittest.TestCase().assertRaises(error)
pytest=Raises()
from subnet.native_role_process import ProcessRoleRuntime,read_frame,write_frame,unpack_array
from subnet.native_role_worker import Worker
D={'vocab_size':8,'max_context':32,'max_output_tokens':4,'kind':'agent'}
SCRIPT='''
import sys,numpy as np
from collections import UserDict
from types import SimpleNamespace
from subnet.native_role_worker import serve
class Tokenizer:
 def encode(self,text,**kw):return [len(text)%8]
 def decode(self,tokens,**kw):return 'toy'
 def apply_chat_template(self,messages,**kw):return UserDict({'input_ids':[1,2,3],'attention_mask':[1,1,1]})
class Runtime:
 tokenizer=Tokenizer()
 def approved_descriptor(self):return {'vocab_size':8,'max_context':32,'max_output_tokens':4,'kind':'agent'}
 def compute(self,prompt,output):
  print('PRIVATE MODEL LOG')
  return object(),np.full((len(output),8),-2,dtype=np.float32)
 def build_proofs(self,acts,**kw):return ['synthetic-not-TOPLOC']
 def verify_proofs(self,acts,proofs,**kw):return [SimpleNamespace(exp_mismatches=np.int64(0),mant_err_mean=np.float32(0),mant_err_median=np.float32(0))]
 def sample(self,prompt,seed,temperature,top_p,max_tokens):return [seed%8]
serve(Runtime(),Runtime().approved_descriptor(),sys.stdin.buffer,sys.stdout.buffer)
'''
def launch():return ProcessRoleRuntime([sys.executable,'-c',SCRIPT],D,env={'OMP_NUM_THREADS':'2'})
def test_real_subprocess_toy_compute_proof_tokenizer_and_logs():
    old=os.environ.get('OMP_NUM_THREADS')
    with launch() as runtime:
        assert runtime.tokenizer.encode('hello')==[5]
        assert runtime.tokenizer.decode([1])=='toy'
        assert runtime.tokenizer.apply_chat_template([{'role':'user','content':'all context'}])==[1,2,3]
        handle,lp=runtime.compute([1],[2,3]);assert lp.shape==(2,8)
        assert runtime.build_proofs(handle)==['synthetic-not-TOPLOC']
        checks=runtime.verify_proofs(handle,['synthetic-not-TOPLOC'])
        assert type(checks[0].exp_mismatches) is int and type(checks[0].mant_err_mean) is float
        assert checks[0].exp_mismatches==0
        assert runtime.sample([1],7,.7,1.,2)==[7]
    assert os.environ.get('OMP_NUM_THREADS')==old
    assert runtime.process.poll() is not None

def test_stale_handle_and_arbitrary_command_reject():
    with launch() as runtime:
        old,_=runtime.compute([1],[2]);runtime.compute([1],[3])
        with pytest.raises(ValueError):runtime.build_proofs(old)
        with pytest.raises(ValueError):runtime._rpc('eval',code='import os')
        with pytest.raises(ValueError):runtime._rpc('build_proofs',handle=old,decode_batching_size=16,topk=128)

def test_unknown_tokenizer_options_and_geometry():
    with launch() as runtime:
        with pytest.raises(ValueError):runtime._rpc('tokenizer',method='save_pretrained',options={})
        with pytest.raises(ValueError):runtime.tokenizer.apply_chat_template([{'role':'user','content':'x'}],tokenize=False)
        with pytest.raises(ValueError):runtime.compute([True],[2])
        with pytest.raises(ValueError):runtime.compute([1]*32,[2])
        with pytest.raises(ValueError):runtime.sample([1],True,.7,1.,1)

def test_read_prefix_budget_before_body():
    class Stream:
        calls=0
        def read(self,n):
            self.calls+=1
            assert self.calls==1
            return struct.pack('!I',101)
    with pytest.raises(ValueError):read_frame(Stream(),limit=100)
    with pytest.raises(ValueError):read_frame(io.BytesIO(b'\0\0'))
    with pytest.raises(ValueError):read_frame(io.BytesIO(struct.pack('!I',10)+b'{}'))

def test_array_forged_header_hash_dtype_nonfinite():
    def pack(array):
        stream=io.BytesIO();np.save(stream,array,allow_pickle=False);raw=stream.getvalue()
        return {'npy_base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest(),'shape':[1,8],'dtype':'float32'}
    assert unpack_array(pack(np.ones((1,8),dtype=np.float32)),1,8).shape==(1,8)
    for array in (np.ones((2,8),dtype=np.float32),np.ones((1,8),dtype=np.float64),np.full((1,8),np.nan,dtype=np.float32)):
        with pytest.raises(ValueError):unpack_array(pack(array),1,8)
    bad=pack(np.ones((1,8),dtype=np.float32));bad['sha256']='0'*64
    with pytest.raises(ValueError):unpack_array(bad,1,8)

def test_approved_readiness_exact_types():
    with pytest.raises(ValueError):ProcessRoleRuntime([sys.executable,'-c',SCRIPT],{**D,'max_output_tokens':4.0})

def test_json_nonfinite_and_write_bound():
    with pytest.raises(ValueError):read_frame(io.BytesIO(struct.pack('!I',3)+b'NaN'))
    with pytest.raises(ValueError):write_frame(io.BytesIO(),{'huge':'x'*100},limit=10)

class TestNativeRoleProcess(unittest.TestCase):
    """Synthetic subprocess controls; no model qualification."""

for _name,_function in list(globals().items()):
    if _name.startswith("test_") and callable(_function):
        setattr(TestNativeRoleProcess,_name,lambda self,f=_function:f())
del _name,_function

class TestStartupRejection(unittest.TestCase):
    def test_private_operator_config_mode(self):
        import tempfile,pathlib,subprocess
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'worker.json'
            path.write_text('{}');path.chmod(0o644)
            result=subprocess.run([sys.executable,'-m','subnet.native_role_worker','--config',str(path)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(result.stdout,b'')
            self.assertIn(b'private bounded operator config',result.stderr)
    def test_source_rejection_precedes_registry_and_model(self):
        import tempfile,pathlib,subprocess
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'worker.json'
            path.write_text(json.dumps({'version':'native-role-process-json-v1','checkpoint':'/does-not-exist','descriptor':{'source_files':{}}}));path.chmod(0o600)
            result=subprocess.run([sys.executable,'-m','subnet.native_role_worker','--config',str(path)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(result.stdout,b'')
            self.assertIn(b'worker source closure',result.stderr)

