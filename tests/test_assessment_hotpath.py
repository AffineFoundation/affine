import copy,unittest
from unittest.mock import patch
from subnet.continuous_audit_policy import valid_digest
from subnet import numerical_resolution as n
import test_numerical_resolution as fixtures
class SyntaxControls(unittest.TestCase):
 def test_exact_original_language(self):
  class Text(str):pass
  values=[None,True,0,[],{},b'a'*64,Text('a'*64),'','a'*63,'a'*65,'a'*64,'0123456789abcdef'*4]
  values += ['a'*i+chr(c)+'a'*(63-i)for i in(0,31,63)for c in range(256)]
  values += ['a'*i+'é'+'a'*(63-i)for i in(0,31,63)]
  for value in values:
   old=type(value)is str and len(value)==64 and all(c in'0123456789abcdef'for c in value)
   self.assertEqual(valid_digest(value),old,repr(value))
class MemoControls(unittest.TestCase):
 def setUp(self):self.f=fixtures.ReviewedUnknownControls();self.f.setUp();self.a=self.f.archives[0]
 def call(self):return n.reference(self.a['ack'],self.a['archive'],self.f.authority)
 def test_authenticate_once_exact_inputs_only(self):
  with patch.object(n,'_reference_uncached',wraps=n._reference_uncached)as oracle:
   with n.authenticated_reference_cache():self.assertEqual(self.call(),self.call());self.assertEqual(oracle.call_count,1)
   self.call();self.assertEqual(oracle.call_count,2)
 def test_shared_results_cannot_be_mutated(self):
  with n.authenticated_reference_cache():
   first=self.call();expected=copy.deepcopy(first);first[0]['at']=999;first[2].clear();second=self.call();self.assertEqual(second,expected);second[2].clear();self.assertEqual(self.call(),expected)
 def test_changed_archive_ack_and_authority_never_hit(self):
  with n.authenticated_reference_cache():
   self.call()
   for ack,raw,authority in [(self.a['ack'],self.a['archive']+b'x',self.f.authority),(dict(self.a['ack'],signature='AA=='),self.a['archive'],self.f.authority),(self.a['ack'],self.a['archive'],self.f.verifier),(self.a['ack'],bytearray(self.a['archive']),self.f.authority)]:
    with self.assertRaises(Exception):n.reference(ack,raw,authority)
 def test_scopes_and_exception_reset(self):
  with patch.object(n,'_reference_uncached',wraps=n._reference_uncached)as oracle:
   with n.authenticated_reference_cache():
    self.call()
    with self.assertRaises(RuntimeError):
     with n.authenticated_reference_cache():self.call();raise RuntimeError('test')
    self.call();self.assertEqual(oracle.call_count,2)
   self.call();self.assertEqual(oracle.call_count,3)
 def test_actual_resolution_equal_and_policy_changes_still_rejected(self):
  expected=self.f.resolved()
  with n.authenticated_reference_cache():
   self.assertEqual(self.f.resolved(),expected);self.assertEqual(self.f.resolved(),expected)
   self.f.kw['cutoff']=29
   with self.assertRaises(ValueError):self.f.resolved()
if __name__=='__main__':unittest.main()
