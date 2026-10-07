import copy
import unittest
from ops.research_pod_ownership import bind, before_launch, completed, heartbeat, verify, rent_once, VERSION, OWNER

class Registry:
 def __init__(self):self.rows={};self.calls=[];self.fail=False
 def load(self):return copy.deepcopy(self.rows)
 def register(self,name,**kwargs):
  self.calls.append(('register',name))
  if self.fail:raise OSError('registry fsync failure')
  self.rows[name]=dict(kwargs,meta=kwargs['meta'],released_at=None)
 def touch(self,name):self.calls.append(('touch',name))

class Ownership(unittest.TestCase):
 def setUp(self):self.r=Registry();self.b=dict(version=VERSION,name='affine-research-node',pod_id='exact-provider-id',purpose='bounded qualification',price_usd_h=5.76,rental_intent_sha256='a'*64,retention_authorized=True)
 def test_register_before_launch_and_keep_retained_after_completion(self):
  order=[]
  def launch():order.append(len(self.r.calls));return 17
  self.assertEqual(before_launch(self.r,self.b,launch),17);self.assertEqual(order,[1]);completed(self.r,self.b,'b'*64)
  row=verify(self.r,self.b);self.assertEqual(row['owner'],OWNER);self.assertEqual(row['expected_hours'],0);self.assertFalse(row.get('release_requested_at'));self.assertFalse(row['released_at'])
 def test_failed_register_never_launches(self):
  self.r.fail=True;calls=[]
  with self.assertRaises(OSError):before_launch(self.r,self.b,lambda:calls.append(1))
  self.assertEqual(calls,[])
 def test_no_adoption_of_foreign_active_owner(self):
  self.r.rows[self.b['name']]={'owner':'manual:another','expected_hours':0,'meta':{},'released_at':None}
  with self.assertRaises(ValueError):bind(self.r,self.b)
  self.assertEqual(self.r.calls,[])
 def test_rebound_provider_identity_refuses_launch(self):
  bind(self.r,self.b);b=dict(self.b,pod_id='different-provider');calls=[]
  with self.assertRaises(ValueError):before_launch(self.r,b,lambda:calls.append(1))
  self.assertFalse(calls)
 def test_heartbeat_refuses_done_or_missing_binding(self):
  with self.assertRaises(ValueError):heartbeat(self.r,self.b)
  bind(self.r,self.b);self.r.rows[self.b['name']]['release_requested_at']=1
  with self.assertRaises(ValueError):heartbeat(self.r,self.b)
 def test_registration_idempotent_no_global_reaper_changes(self):
  bind(self.r,self.b);bind(self.r,self.b);self.assertEqual([c[0]for c in self.r.calls],['register','touch'])
 def test_retention_requires_explicit_authorization(self):
  with self.assertRaises(ValueError):bind(self.r,dict(self.b,retention_authorized=False))
 def test_released_name_can_only_rebind_new_exact_rental(self):
  bind(self.r,self.b);self.r.rows[self.b['name']]['released_at']=3;new=dict(self.b,pod_id='new-provider',rental_intent_sha256='c'*64);bind(self.r,new);verify(self.r,new)
 def test_rental_name_owned_before_provider_call(self):
  def rent():
   self.assertEqual(self.r.rows[self.b['name']]['meta']['provider_pod_id'],'pending-original-rental');return {'pod_id':'actual-original'}
  receipt,bound=rent_once(self.r,self.b,rent);self.assertEqual(bound['pod_id'],'actual-original');verify(self.r,bound)
 def test_rental_registration_failure_prevents_billing(self):
  self.r.fail=True;calls=[]
  with self.assertRaises(OSError):rent_once(self.r,self.b,lambda:calls.append(1))
  self.assertEqual(calls,[])
 def test_rental_timeout_preserves_owner_and_prevents_reissue(self):
  calls=[]
  def rent():calls.append(1);raise TimeoutError('observation timeout')
  with self.assertRaises(TimeoutError):rent_once(self.r,self.b,rent)
  with self.assertRaises(ValueError):rent_once(self.r,self.b,rent)
  self.assertEqual(calls,[1]);self.assertEqual(self.r.rows[self.b['name']]['expected_hours'],0)
if __name__=='__main__':unittest.main()
