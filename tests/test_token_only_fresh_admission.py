"""Actual fresh-process worker/bootstrap/source admission, before GPU/network."""
import subprocess,sys,unittest
from pathlib import Path
import test_backend_fresh_admission as existing
from subnet import forced_sampling as f,fast_prefill_audit as fast,token_only_protocol as p

class TokenFreshAdmission(unittest.TestCase):
 def test_real_signed_worker_deferred_admission_and_fresh_source_loader(self):
  root=Path(__file__).resolve().parents[1]
  contract=f.new_contract(dict(version=fast.THREEWAY_VERSION,max_attempts=16,uncertainty_adjudication='numerical-inconclusive-no-replay-v1',calibration=dict(version=fast.CALIBRATION,checkpoint='a'*64,model_runtime_revision='cuda-bf16-eager-sm86-v1',backend_profile_sha256='b'*64,harness_sha256='c'*64,report_sha256='d'*64,cdf_abs_error=.00001,logprob_atol=.00001,toploc_exp_mismatches=0,toploc_mant_err_mean=0,toploc_mant_err_median=0)))
  prefix=existing.SCRIPT.split("os.environ['CUBLAS_WORKSPACE_CONFIG']")[0]
  for mode in ['valid','missing-policy','tampered','preimport']:
   script=prefix+'''\nmanifest.update(sampling_contract=CONTRACT,sampling_source_hash=SOURCE,token_artifact_policy=POLICY,submission_transport_policy='small-commitment-token-pairs-v3',max_batches=1)
for name in ('token_only_protocol','token_only_runtime','threeway_prefill_research','native_session_validation'):
 job['source_files']['subnet/'+name+'.py']=b.digest(root/'subnet'/ (name+'.py'))
with tempfile.TemporaryDirectory()as directory:
 seed=Path(directory)/'seed';seed.write_text('11'*32);seed.chmod(0o600)
 job['miner_identity_file']=str(seed);job['miner_id']=SigningKey(bytes.fromhex('11'*32)).verify_key.encode().hex()
 job['capability']['batch_put_urls']=[job['capability']['put_url']];job['capability']['training_put_urls']=[job['capability']['put_url']]
 if CONTROL=='missing-policy':manifest.pop('token_artifact_policy')
 if CONTROL=='tampered':job['source_files']['subnet/token_only_protocol.py']='0'*64
 if CONTROL=='preimport':import subnet.token_only_protocol
 job['manifest']=sign(manifest)
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 def boundary(*args):
  for name in ('subnet.token_only_protocol','subnet.forced_sampling'):
   assert isinstance(sys.modules[name].__loader__,b.importlib.abc.Loader)
  assert 'subnet.model'not in sys.modules and 'subnet.gpu_runtime'not in sys.modules
  if 'torch'in sys.modules:assert not sys.modules['torch'].cuda.is_initialized()
  raise RuntimeError('qualified checkpoint boundary')
 with patch.object(b,'version',return_value='approved'),patch.object(b,'checkpoint',side_effect=boundary)as cp:
  try:b.execute(sign(job),authority,str(Path(directory)/'workspace'))
  except (ValueError,RuntimeError)as error:
   expected={'valid':'qualified checkpoint boundary','missing-policy':'token policy','tampered':'worker source mismatch','preimport':'requires fresh process'}[CONTROL]
   assert expected in str(error),repr(error)
  else:raise AssertionError('must stop before models')
  assert cp.call_count==(1 if CONTROL=='valid'else 0)
print('TOKEN_FRESH_ADMISSION_PASS')
'''
   script='CONTRACT='+repr(contract)+'\nSOURCE='+repr(f.source_hash())+'\nPOLICY='+repr(p.POLICY)+'\nCONTROL='+repr(mode)+'\n'+script
   result=subprocess.run([sys.executable,'-B','-c',script,'valid'],cwd=root,capture_output=True,text=True,timeout=30)
   self.assertEqual(result.returncode,0,result.stderr);self.assertIn('TOKEN_FRESH_ADMISSION_PASS',result.stdout)
