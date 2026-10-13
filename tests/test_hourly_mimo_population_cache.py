import copy,unittest
from ops.mimo_population_cache import AuthenticatedPopulationCache
class CachedPopulation(unittest.TestCase):
 def setUp(self):
  self.calls=0
  def auth(doc,authority,*,manifest_guard):
   self.calls+=1
   if doc['signature']!='good' or doc['payload']['value']!=3:raise ValueError('invalid')
   manifest_guard(doc);return doc['payload']
  self.cache=AuthenticatedPopulationCache(auth);self.guard=lambda _:None;self.doc={'payload':{'value':3},'signature':'good','signer':'authority'}
 def call(self,doc=None,authority='a',guard=None):return self.cache(self.doc if doc is None else doc,authority,manifest_guard=guard or self.guard)
 def test_exact_repeat_and_defensive_copy(self):
  first=self.call();first['value']=99
  self.assertEqual(self.call(),{'value':3});self.assertEqual(self.calls,1)
 def test_signature_and_payload_changes_rechecked(self):
  self.call()
  for key in ('signature','payload'):
   doc=copy.deepcopy(self.doc)
   if key=='signature':doc[key]='bad'
   else:doc[key]['value']=4
   with self.assertRaises(ValueError):self.call(doc)
  self.assertEqual(self.calls,3);self.assertEqual(self.call(),{'value':3})
 def test_authority_and_guard_are_bound(self):
  self.call();self.call(authority='b');self.call(guard=lambda _:None);self.assertEqual(self.calls,3)
 def test_new_invocation_cold(self):
  self.call();other=AuthenticatedPopulationCache(self.cache.original);other(self.doc,'a',manifest_guard=self.guard);self.assertEqual(self.calls,2)
 def test_errors_not_cached(self):
  bad=copy.deepcopy(self.doc);bad['signature']='bad'
  for _ in range(2):
   with self.assertRaises(ValueError):self.call(bad)
  self.assertEqual(self.calls,2);self.assertEqual(len(self.cache.entries),0)
if __name__=='__main__':unittest.main()
