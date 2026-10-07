import copy,hashlib,tempfile,unittest
from pathlib import Path
from ops.negative_quality_count_plan import paired_quality_refs,resource_budget,select_eos_quota
from ops.run_same_parent_negative_quality_trial import prospective_budget,copy_verified_parent_shard


def row(attempt,label='negative',eos=True):
 output=[attempt+1,151645]if eos else[attempt+1]*1024
 rollout={'index':1,'seed':attempt,'task_hash':'a'*64,'classification':label,'reward':int(label=='positive'),'turns':[{'prompt':[1],'output':output,'text':str(attempt),'observations':[]}]}
 return {'index':1,'attempt':attempt,'status':'verified','native_verified':True,'sampler_verified':True,'rollout':rollout}


class Controls(unittest.TestCase):
 def streams(self):return {1:[row(i,'positive'if i<2 else'negative',eos=i<4)for i in range(16)]}
 def test_same_positive_refs_required(self):
  cap=[{'index':1,'positive':{'rollout_sha256':'a'*64},'negative':{'rollout_sha256':'b'*64},'gradient_task_weight':1.}];eos=copy.deepcopy(cap);eos[0]['negative']['rollout_sha256']='c'*64
  self.assertTrue(paired_quality_refs(cap,eos,[1])['same_positive_all_tasks'])
  eos[0]['positive']['rollout_sha256']='d'*64
  with self.assertRaises(ValueError):paired_quality_refs(cap,eos,[1])
 def test_matched_nested_quota_holds_negative_quality_and_task_weight(self):
  result=select_eos_quota(self.streams(),[1],minimum_tasks=1);self.assertTrue(result['training_supply_gate_met']);s=result['selections'][0]
  self.assertEqual(s['K1']['negative'],s['K2']['negative'][:1]);self.assertEqual(s['K1']['positive'],s['K2']['positive'][:1]);self.assertEqual(s['pair_weight_K2']*2,s['task_weight']);self.assertFalse(result['dispatch_allowed'])
 def test_capped_second_negative_cannot_fill_EOS_quota(self):
  streams=self.streams();streams[1][3]=row(3,eos=False);r=select_eos_quota(streams,[1],minimum_tasks=1);self.assertFalse(r['training_supply_gate_met']);self.assertEqual(r['matched_tasks'],0)
 def test_duplicate_content_not_extra_quota_and_conflicting_label_refused(self):
  streams=self.streams();streams[1][3]['rollout']['turns']=copy.deepcopy(streams[1][2]['rollout']['turns']);self.assertFalse(select_eos_quota(streams,[1],minimum_tasks=1)['training_supply_gate_met'])
  streams[1][3]['rollout']['classification']='positive';streams[1][3]['rollout']['reward']=1
  with self.assertRaises(ValueError):select_eos_quota(streams,[1],minimum_tasks=1)
 def test_all_original_attempts_verification_and_population_required(self):
  for mutate in [lambda s:s[1].pop(),lambda s:s[1][0].update(sampler_verified=False),lambda s:s[1][0].update(attempt=1),lambda s:s[1][0]['rollout'].update(task_hash='b'*64)]:
   s=self.streams();mutate(s)
   with self.assertRaises(ValueError):select_eos_quota(s,[1],minimum_tasks=1)
  with self.assertRaises(ValueError):select_eos_quota(self.streams(),[2],minimum_tasks=1)
 def test_disposable_optimizer_budget_exact_no_output_export(self):
  b=resource_budget([{'numel':7615616512},{'numel':0}],15242726226,91387491264)
  self.assertEqual(b['output_optimizer_export_bytes'],0);self.assertFalse(b['durable_research_optimizer_claim']);self.assertEqual(b['cold_total_GET_bytes'],106630217490)
 def test_new_identity_and_explicit_no_optimizer_GET_export(self):
  s={'optimizer_state_export_allowed':False,'optimizer_RAM_disposable_research_only':True,'optimizer_payload_GETs_in_training_allowed':False,'restore_concurrency':4,'retained_pod_id':'NEW-PENDING-PHYSICAL','old_interrupted_scope_sha256':'397f6e23d98b5ae184dd47591fee1b94fa06ee482f6991de03e5d59be44e67c2','workspace':'/root/affine-CP24-negative-quality-v4-NEW/base'}
  self.assertTrue(prospective_budget(s))
  for k,v in [('retained_pod_id','ba5ad596-d42a-4bcc-8643-f18e2272ccd5'),('workspace','/root/affine-CP24-completed-negative-research-v1/original'),('optimizer_state_export_allowed',True),('optimizer_payload_GETs_in_training_allowed',True),('restore_concurrency',True)]:
   bad={**s,k:v}
   with self.assertRaises(ValueError):prospective_budget(bad)
 def test_warm_parent_copy_is_fullhash_exclusive_private_without_network(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d).resolve();p=root/'original';p.write_bytes(b'CPU fixture');p.chmod(0o600);r={'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};q=root/'copy';copy_verified_parent_shard(p,q,r);self.assertEqual(q.read_bytes(),p.read_bytes());self.assertEqual(q.stat().st_mode&0o777,0o600)
   with self.assertRaises(FileExistsError):copy_verified_parent_shard(p,q,r)
   p.write_bytes(b'corrupt')
   with self.assertRaises(ValueError):copy_verified_parent_shard(p,root/'new',r)
   self.assertFalse((root/'new').exists())
