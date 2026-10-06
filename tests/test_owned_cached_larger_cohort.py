import base64,copy,unittest
from nacl.signing import SigningKey
from subnet.backend_jobs import canonical
from ops.owned_cached_larger_cohort import prepare,aggregate,POLICY,SOURCE
class LargerCohortTests(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();self.manifest={'source_bundle':{'sha256':SOURCE},'checkpoint':{'id':'parent'},'environments':[{'env_id':'affine_math','indices':list(range(6746)),'spec':{'num_samples':7496}}]};self.harness={'version':'text-tools-long-kv-v3','policy':'autoregressive','max_output_tokens':1024,'temperature':.7,'top_p':1.};self.job={'role':'evaluate','owned_evaluation_policy':POLICY,'manifest':self.sign(self.manifest),'heldout':[{'indices':list(range(6746,6778)),'harness':self.harness}]}
 def sign(self,p):return {'payload':p,'signer':self.authority,'signature':base64.b64encode(self.key.sign(canonical(p)).signature).decode()}
 def plan(self):return prepare(self.sign(self.job),self.authority)
 def reports(self,p):return [{'heldout':[{'index':i,'seed':s,'native_graded':True,'verified':False,'proof_verification_performed':False,'trust_scope':POLICY['trust_scope'],'classification':'positive','reward':1}for i,s in zip(g['indices'],g['seeds'])],'heldout_failures':[]}for g in p['groups']]
 def test_deterministic_disjoint_heldout128_four32_no_dispatch(self):
  p=self.plan();self.assertEqual(p,self.plan());indices=[i for g in p['groups']for i in g['indices']];self.assertEqual(len(set(indices)),128);self.assertTrue(all(i>=6778 for i in indices));self.assertFalse(p['dispatch_allowed']);self.assertEqual([len(g['indices'])for g in p['groups']],[32]*4)
 def test_wrong_source_cap_policy_partial_training_split_fail(self):
  for change in ('source','cap','policy','split'):
   job=copy.deepcopy(self.job);m=copy.deepcopy(self.manifest)
   if change=='source':m['source_bundle']['sha256']='old'
   if change=='split':m['environments'][0]['indices']=list(range(256))
   if change=='cap':job['heldout'][0]['harness']['max_output_tokens']=128
   if change=='policy':job['owned_evaluation_policy']={}
   job['manifest']=self.sign(m)
   with self.assertRaises(ValueError):prepare(self.sign(job),self.authority)
 def test_aggregate_complete_and_missing_failure_never_zero(self):
  p=self.plan();rs=self.reports(p);self.assertEqual(aggregate(p,rs)['successes'],128)
  for change in ('missing','duplicate','seed','infra','boolreward','class'):
   bad=copy.deepcopy(rs)
   if change=='missing':bad.pop()
   if change=='duplicate':bad[0]['heldout'][1]=bad[0]['heldout'][0]
   if change=='seed':bad[0]['heldout'][0]['seed']+=1
   if change=='infra':bad[0]['heldout_failures']=[{'error':'native error'}]
   if change=='boolreward':bad[0]['heldout'][0]['reward']=True
   if change=='class':bad[0]['heldout'][0]['classification']='negative'
   with self.assertRaises(ValueError):aggregate(p,bad)
 def test_changed_selection_seed_changes_cohort_and_signature_tamper_fails(self):
  p=self.plan();other=prepare(self.sign(self.job),self.authority,selection_seed='other');self.assertNotEqual(p['cohort_sha256'],other['cohort_sha256']);original=self.sign(self.job);original['payload']['role']='mine'
  with self.assertRaises(Exception):prepare(original,self.authority)
