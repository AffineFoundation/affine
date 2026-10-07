"""Actual signed mining bootstrap -> authenticated finder -> full validation."""
import json,subprocess,sys,unittest
from pathlib import Path
import test_backend_jobs as fixture
class Controls(unittest.TestCase):
 def test_v5_and_legacy_signed_mining_bootstrap_stays_pure(self):
  for version,budget in [('forced-inverse-cdf-prefill-miner-bound-v5',1000),('forced-inverse-cdf-prefill-support-v3',128)]:
   f=fixture.MiningEpochWindow();f.setUp();f.manifest.update(K=2,L=2,max_batches=3,sampling_contract=dict(version=version,max_attempts=budget));f.job.update(search_budget=budget,seed_start=0,manifest=f.sign(f.manifest));payload=[f.sign(f.job),f.authority]
   code="""import json,sys;from pathlib import Path
from subnet import backend_jobs as b
e,a=json.loads(sys.stdin.read());b._validate(e,a,now=50,resolve_source=False)
assert 'subnet.forced_sampling'not in sys.modules
b.install_source_loader(Path(b.__file__).parent.parent)
b.validate(e,a,now=50)
"""
   p=subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parent.parent,input=json.dumps(payload),text=True,capture_output=True)
   self.assertEqual(p.returncode,0,p.stderr)
