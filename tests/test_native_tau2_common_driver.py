"""Driver admission orchestration only; no model execution in these controls."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from nacl.signing import SigningKey
from subnet import native_tau2_common_driver as d

class Driver(unittest.TestCase):
 def test_missing_fixed_role_paths_rejected_before_loading(self):
  manifest={'roles':{'agent':{},'user':{}}}
  with patch.object(d,'validate_epoch',return_value=manifest),patch.object(d,'CPURoleRuntime') as runtime:
   with self.assertRaises(ValueError):d.dependencies({},'authority',{}, {'agent':'path'})
   runtime.assert_not_called()
 def test_each_distinct_approved_role_descriptor_reaches_its_runtime(self):
  manifest={'roles':{'agent':{'checkpoint':'current'},'user':{'checkpoint':'fixed'}}}
  with patch.object(d,'validate_epoch',return_value=manifest),patch.object(d,'CPURoleRuntime',side_effect=lambda p,r:(p,r)):
   checked,runtimes,policies=d.dependencies({},'authority',{}, {'agent':'agent-path','user':'user-path'})
  self.assertEqual(runtimes['agent'],('agent-path',manifest['roles']['agent']));self.assertEqual(runtimes['user'],('user-path',manifest['roles']['user']));self.assertEqual(policies,{'agent':None,'user':None})
 def test_verifier_rejects_artifact_epoch_substitution_before_compute(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td);(out/'signed-epoch.json').write_text(json.dumps({'mutated':'epoch'}))
   with patch.object(d,'dependencies',return_value=({}, {}, {})),patch.object(d,'verify_receipt') as infer:
    with self.assertRaises(ValueError):d.verify({},'authority',{}, {},SigningKey.generate(),0,0,'data','public','private',out)
    infer.assert_not_called()
 def test_array_filename_cannot_escape_operator_artifacts(self):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td);(out/'signed-epoch.json').write_text('{}');(out/'signed-receipts.json').write_text(json.dumps([{'payload':{'probabilities_file':'../private-seed','role':'user'}}]))
   with patch.object(d,'dependencies',return_value=({}, {}, {})),patch.object(d,'checked_records'),patch.object(d,'verify_receipt') as infer:
    with self.assertRaises(ValueError):d.verify({},'authority',{}, {},SigningKey.generate(),0,0,'data','public','private',out)
    infer.assert_not_called()
if __name__=='__main__':unittest.main()
