import pathlib,tempfile,json,unittest
from types import SimpleNamespace
from nacl.signing import SigningKey
from ops.native_tau2_private_rejection_observer import install
from subnet.native_tau2_common_search_contract import authenticate
class PrivateDiagnosticTests(unittest.TestCase):
 def test_records_exact_private_cause_and_preserves_original_diagnostic(self):
  key=SigningKey.generate();value={'reason':'unchanged'};base=SimpleNamespace(rejection_diagnostic=lambda *a:value)
  with tempfile.TemporaryDirectory() as d:
   path=pathlib.Path(d)/'private.json';original=install(base,path,key,{'recovery_attempt':1});self.assertIs(base.rejection_diagnostic(None,None,RuntimeError('waiting-disk-capacity')),value)
   p=authenticate(json.loads(path.read_text()),key.verify_key.encode().hex());self.assertEqual(p['records'][0]['exception_message'],'waiting-disk-capacity');self.assertFalse(p['model_or_native_policy_changed']);self.assertEqual(path.stat().st_mode&0o777,0o600);self.assertIs(original(None,None,None),value)
 def test_private_exception_messages_are_bounded(self):
  key=SigningKey.generate();base=SimpleNamespace(rejection_diagnostic=lambda *a:{})
  with tempfile.TemporaryDirectory() as d:
   path=pathlib.Path(d)/'private.json';install(base,path,key,{});base.rejection_diagnostic(None,None,RuntimeError('x'*20000));p=json.loads(path.read_text())['payload'];self.assertEqual(len(p['records'][0]['exception_message']),4096);self.assertLessEqual(len(p['records'][0]['private_traceback']),16384)
 def test_sink_failure_does_not_replace_original_return(self):
  from unittest.mock import patch
  key=SigningKey.generate();result={'reason':'original'};base=SimpleNamespace(rejection_diagnostic=lambda *a:result)
  with tempfile.TemporaryDirectory() as d:
   install(base,pathlib.Path(d)/'private.json',key,{})
   with patch('ops.native_tau2_private_rejection_observer.os.open',side_effect=OSError('capacity')):self.assertIs(base.rejection_diagnostic(None,None,RuntimeError('original')),result)
 def test_original_diagnostic_exception_unchanged(self):
  key=SigningKey.generate();error=ValueError('original diagnostic error')
  def fail(*a):raise error
  base=SimpleNamespace(rejection_diagnostic=fail)
  with tempfile.TemporaryDirectory() as d:
   install(base,pathlib.Path(d)/'private.json',key,{})
   try:base.rejection_diagnostic(None,None,RuntimeError('roleerror'))
   except ValueError as actual:self.assertIs(actual,error)
   else:self.fail('original exception suppressed')
