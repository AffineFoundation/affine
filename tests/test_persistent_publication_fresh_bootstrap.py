"""Fresh signed v4 publication admission; no model/network/GPU execution."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import test_persistent_training_integration as fixture
from training_receipt_fixtures import signed_receipt
from subnet import backend_jobs as b

SCRIPT=r'''
import json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import subnet.backend_jobs as b
p,mode=sys.argv[1:];envelope=json.loads(Path(p).read_text())
if mode=='preimport':import subnet.persistent_publication
if mode=='unrelated':import subnet.epoch_optimizer
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
with tempfile.TemporaryDirectory()as workspace:
 def stop(*args):
  assert type(sys.modules['subnet.persistent_publication'].__loader__).__name__=='Loader'
  raise RuntimeError('authenticated fresh checkpoint boundary')
 with patch.object(b,'version',return_value='approved'),patch.object(b,'checkpoint',side_effect=stop),patch.object(b.time,'time',return_value=50):
  try:b.execute(envelope,envelope['signer'],workspace)
  except (ValueError,RuntimeError)as exc:
   expected={'valid':'authenticated fresh checkpoint boundary','preimport':'before runtime imports: subnet.persistent_publication','unrelated':'before runtime imports: subnet.'}[mode]
   assert expected in str(exc),str(exc)
  else:raise AssertionError('test must stop before model')
print('PASS '+mode)
'''

class FreshPersistentPublication(unittest.TestCase):
 def test_real_signed_validation_import_is_reloaded_but_prior_compute_is_rejected(self):
  fx=fixture.PersistentIntegrationTests();fx.setUp();self.addCleanup(fx.doCleanups)
  fx.manifest['persistent_publication_policy']=dict(version='parallel-persistent-publication-v1',state_readback='local-full',checkpoint_readback_workers=1)
  receipt,*_=signed_receipt(fx.key,fx.manifest,fx.miner,fx.receipts[fx.miner],fx.batch)
  fx.submission['verifier_receipt']=receipt
  job=fx.job();root=Path(b.__file__).resolve().parent.parent
  job['source_files']['subnet/persistent_publication.py']='b'*64
  job['source_files']={name:b.digest(root/name)for name in job['source_files']}
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'signed.json';p.write_text(json.dumps(fx.sign(job)))
   for mode in ('valid','preimport','unrelated'):
    result=subprocess.run([sys.executable,'-B','-c',SCRIPT,str(p),mode],cwd=root,env=dict(os.environ,PYTHONPATH=str(root)),capture_output=True,text=True,timeout=60)
    self.assertEqual(result.returncode,0,result.stderr);self.assertIn('PASS '+mode,result.stdout)
