"""Bootstrap request admission must not preload unauthenticated sampler code."""
import unittest,json,subprocess,sys
from pathlib import Path
import test_v5_calibration_independent_integration as fixture
class Controls(unittest.TestCase):
 def test_request_then_fresh_loader_then_full_context_in_fresh_process(self):
  fx=fixture.Controls();fx.setUp();root=Path(__file__).resolve().parent.parent
  code="""import json,sys;from pathlib import Path
from subnet import backend_jobs as b,successor_calibration as c
req,m=json.loads(sys.stdin.read());c.request(req)
assert 'subnet.forced_sampling'not in sys.modules and 'subnet.fast_prefill_audit'not in sys.modules
b.install_source_loader(Path(b.__file__).parent.parent)
from subnet import successor_calibration as fresh
context=fresh.draw_context(m,req);assert context['miner']==req['miner']
"""
  p=subprocess.run([sys.executable,'-B','-c',code],cwd=root,input=json.dumps([fx.req,fx.manifest]),text=True,capture_output=True)
  self.assertEqual(p.returncode,0,p.stderr)
 def test_full_calibration_semantics_not_relaxed_by_pure_bootstrap(self):
  from subnet import successor_calibration as c
  fx=fixture.Controls();fx.setUp();req=json.loads(json.dumps(fx.req));req['draw_contract']['calibration']['cdf_abs_error']=-1
  c.request(req)
  with self.assertRaises(ValueError):c.draw_context(fx.manifest,req)
