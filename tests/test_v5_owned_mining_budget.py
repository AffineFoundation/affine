"""Signed job admission for the versioned owned-miner nonce budget; CPU only."""
import unittest,copy
import test_backend_jobs as fixtures
from subnet.backend_jobs import validate
from subnet.forced_sampling import MINER_VERSION
class Controls(unittest.TestCase):
 def setUp(self):
  self.f=fixtures.MiningEpochWindow();self.f.setUp()
  self.f.manifest.update(K=2,L=2,max_batches=3,sampling_contract=dict(version=MINER_VERSION,max_attempts=1000))
 def check(self,budget,seed=0):
  self.f.job.update(search_budget=budget,seed_start=seed,manifest=self.f.sign(self.f.manifest))
  return validate(self.f.sign(self.f.job),self.f.authority,now=50)
 def test_signed_v5_job_admits_full1000_attempt_budget(self):self.assertEqual(self.check(1000)[0]['search_budget'],1000)
 def test_signed_v5_job_admits_last_nonce(self):self.assertEqual(self.check(1,999)[0]['seed_start'],999)
 def test_v5_out_of_range_and_bool_refuse(self):
  for budget,seed in [(1001,0),(0,0),(1,1000),(2,999),(True,0),(1,True),(1,-1)]:
   with self.subTest(budget=budget,seed=seed),self.assertRaises(ValueError):self.check(budget,seed)
 def test_legacy_max128_preserved(self):
  self.f.manifest.pop('sampling_contract');self.assertEqual(self.check(128)[0]['search_budget'],128)
  with self.assertRaisesRegex(ValueError,'mining search budget'):self.check(129)
 def test_support_v3_still_max128(self):
  self.f.manifest['sampling_contract']['version']='forced-inverse-cdf-prefill-support-v3';self.assertEqual(self.check(128)[0]['search_budget'],128)
  with self.assertRaisesRegex(ValueError,'mining search budget'):self.check(1000)
 def test_signed_deadline_prevents_budget_admission(self):
  self.f.manifest['deadline']=50
  with self.assertRaisesRegex(ValueError,'epoch window closed'):self.check(1000)
if __name__=='__main__':unittest.main()
