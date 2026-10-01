import copy,unittest
from test_replay_training import ReplayTrainingTests
from subnet.replay_training import admitted,audit_admitted,verified_pairs
from subnet import verified_replay_pool as r
class ArchiveAdmission(ReplayTrainingTests):
 def archived_inputs(self):
  live=copy.deepcopy(self.manifest);live['harness_source_hash']='1'*64
  current=self.sign(live);pool=copy.deepcopy(self.inputs['pool']['payload']);pool['current_manifest_sha256']=r.digest(current)
  pool['entries'][0]['current_manifest_sha256']=r.digest(current);pool['pool_sha256']=r.digest({k:v for k,v in pool.items()if k!='pool_sha256'})
  return copy.deepcopy(live),{'manifest':current,'pool':self.sign(pool),'reuse_counts':{}}
 def test_verified_archive_pin_allows_readonly_metadata(self):
  live,envelope=self.archived_inputs()
  self.assertEqual(len(audit_admitted(live,envelope,self.authority,expected_archive_harness_source_hash='1'*64)[1]['selected']),1)
 def test_wrong_or_missing_archive_pin_refused(self):
  live,envelope=self.archived_inputs()
  for value in [None,'','2'*64,True]:
   with self.assertRaises(ValueError):audit_admitted(live,envelope,self.authority,expected_archive_harness_source_hash=value)
 def test_job_fields_cannot_override_live_source_admission(self):
  live,envelope=self.archived_inputs();live['expected_archive_harness_source_hash']='1'*64;live['skip_source_checks']=True
  with self.assertRaisesRegex(ValueError,'source mismatch'):admitted(live,envelope,self.authority)
 def test_fresh_verification_never_uses_archived_metadata_override(self):
  live,envelope=self.archived_inputs()
  class Runtime:
   def configure(self,*args):raise AssertionError('must refuse before model')
  with self.assertRaisesRegex(ValueError,'source mismatch'):verified_pairs(Runtime(),live,envelope,self.authority)
 def test_archive_api_still_checks_signed_projected_registry(self):
  live,envelope=self.archived_inputs();live['sample_harness_registry']={'e':dict(indices=[0,1],harness=dict(live['environments'][0]['harness'],candidates=['forged','no']))}
  with self.assertRaisesRegex(ValueError,'projected'):audit_admitted(live,envelope,self.authority,expected_archive_harness_source_hash='1'*64)
if __name__=='__main__':unittest.main()
