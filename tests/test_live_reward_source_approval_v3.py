import copy,unittest
from test_live_reward_source_approval_v2 import ReplacementApprovalTests
from ops.live_reward_exporter import sign
from subnet.live_reward_bridge import sha

class EightNodeApproval(ReplacementApprovalTests):
 def v3(self,count=8):
  docs=self.replacement_chain([self.ids,self.ids+[format(i,'064x')for i in range(5,count+1)]])
  p=copy.deepcopy(docs[1]['payload']);p['version']='live-compute-source-approval-v3';docs[1]=sign(p,self.key);return docs
 def test_explicit_v3_allows_eight_and_preserves_historical_scope(self):
  docs=self.v3();c,_=self.apply_chain(docs)
  self.assertEqual(len(c['_source_authorizations'][docs[1]['payload']['source']['sha256']]['verifier_identities']),8)
  self.assertEqual(c['_source_authorizations'][docs[0]['payload']['source']['sha256']]['verifier_identities'],self.ids)
 def test_old_v2_cannot_admit_eight(self):
  docs=self.v3();p=copy.deepcopy(docs[1]['payload']);p['version']='live-compute-source-approval-v2';docs[1]=sign(p,self.key)
  with self.assertRaises(ValueError):self.apply_chain(docs)
 def test_v3_cannot_downgrade_to_v2(self):
  roster=self.ids+[format(i,'064x')for i in range(5,9)]
  docs=self.replacement_chain([self.ids,roster,self.ids])
  previous=docs[0]
  for i in range(1,len(docs)):
   p=copy.deepcopy(docs[i]['payload']);p['previous_authorization_sha256']=sha(previous)
   if i==1:p['version']='live-compute-source-approval-v3'
   else:
    for j,r in enumerate(p['retirements']):
     value=copy.deepcopy(r['payload']);value['previous_authorization_sha256']=sha(previous);p['retirements'][j]=sign(value,self.key)
   docs[i]=sign(p,self.key);previous=docs[i]
  with self.assertRaisesRegex(ValueError,'downgrade'):self.apply_chain(docs)
 def test_nine_nodes_refused(self):
  with self.assertRaises(ValueError):self.apply_chain(self.v3(9))

if __name__=='__main__':unittest.main()
