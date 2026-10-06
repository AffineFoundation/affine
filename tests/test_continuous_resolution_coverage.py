import copy,unittest
import test_continuous_audit_policy as controls
from subnet import continuous_audit_policy as p
class ResolutionCoverage(unittest.TestCase):
 def setUp(self):
  self.f=controls.PolicyControls();self.f.setUp();self.f.p['version']=p.RESOLUTION_VERSION
 def test_unknown_only_zero_points_without_fraud_or_blacklist(self):
  s=self.f.calculate(obs=[self.f.observation('numerical_ambiguous')]);m=s['miners']['b'*64]
  self.assertEqual(s['points']['b'*64],0);self.assertEqual(s['weights']['b'*64],0);self.assertEqual(m['confirmed_invalid_current'],0);self.assertFalse(m['blacklisted']);self.assertEqual(m['reward_multiplier'],1);self.assertFalse(m['unresolved_is_fraud'])
 def test_infra_and_pending_do_not_change_prior_or_resolution_coverage(self):
  none=self.f.calculate();infra=self.f.calculate(obs=[self.f.observation('infrastructure_error')]);self.assertEqual(none['points'],infra['points']);self.assertEqual(infra['miners']['b'*64]['resolution_coverage_factor'],1)
 def test_current_unknown_cannot_be_rescued_by_valid_history(self):
  old=dict(self.f.row,epoch='old',round=0,batch_sha256='9'*64,checkpoint='8'*64);good=self.f.observation();good.update(epoch='old',checkpoint='8'*64,batch_sha256='9'*64);unknown=self.f.observation('numerical_ambiguous',job='2'*64)
  s=self.f.calculate(rows=[old,self.f.row],obs=[good,unknown]);self.assertEqual(s['points']['b'*64],0);self.assertEqual(s['miners']['b'*64]['confirmed_invalid_recent'],0)
 def test_recent_unknown_reduces_new_epoch_credit_then_ages_out(self):
  old=dict(self.f.row,epoch='old',round=0,batch_sha256='9'*64,checkpoint='8'*64);unknown=self.f.observation('numerical_ambiguous');unknown.update(epoch='old',checkpoint='8'*64,batch_sha256='9'*64);good=self.f.observation(job='2'*64)
  s=self.f.calculate(rows=[old,self.f.row],obs=[unknown,good]);self.assertAlmostEqual(s['miners']['b'*64]['resolution_coverage_factor'],1/1.8)
  # New infrastructure alone never converts UNKNOWN history into validity.
  infra=self.f.observation('infrastructure_error',job='3'*64);s=self.f.calculate(rows=[old,self.f.row],obs=[unknown,infra]);self.assertEqual(s['points']['b'*64],0)
  self.f.p['recent_epochs']=1;s=self.f.calculate(rows=[old,self.f.row],obs=[unknown,infra]);self.assertEqual(s['points']['b'*64],.5)
 def test_partial_resolution_scales_points_duplicates_do_not_game_coverage(self):
  other=dict(self.f.row,index=1,batch_sha256='9'*64);a=self.f.observation();b=self.f.observation('numerical_ambiguous',job='2'*64);b['batch_sha256']='9'*64
  s=self.f.calculate(rows=[self.f.row,other],obs=[a,b]);self.assertEqual(s['miners']['b'*64]['resolution_coverage_factor'],.5);self.assertAlmostEqual(s['points']['b'*64],2/3)
  repeated=self.f.calculate(rows=[self.f.row,other],obs=[a,b,b]);self.assertEqual(s,repeated)
 def test_invalid_remains_resolved_and_penalized_not_unknown(self):
  s=self.f.calculate(obs=[self.f.observation('confirmed_invalid')]);m=s['miners']['b'*64];self.assertEqual(m['resolution_coverage_factor'],1);self.assertEqual(m['confirmed_invalid_current'],1);self.assertEqual(m['reward_multiplier'],.25);self.assertAlmostEqual(s['points']['b'*64],1/12)
 def test_cutoff_does_not_count_future_unknown_and_hourly_accepts_v3(self):
  s=self.f.calculate(obs=[self.f.observation('numerical_ambiguous',at=31)]);self.assertEqual(s['points']['b'*64],.5);s['cutoff']=3600;h=p.hourly_aggregate([controls.signed(self.f.authority,s)],self.f.root,3600);self.assertEqual(h['points'],s['points'])
 def test_historical_v1_v2_snapshot_bytes_keep_frozen_original_digests(self):
  # Generated from immutable pre-change1877ecd0 with these original fixtures.
  expected={p.LEGACY_VERSION:['1a850a8bb889430c3cbf2f42b93bed8eb7115fd5bea23ee689ed7954b9f3d08e','db3afef0d95bf3c9633683c36139b60d0cdeabab9487d164f1d861bf64a2c250','6827ff43be1f7ac2895af63af558f7b0ea6294e2b0dae6805f5fa13477d12ed3','c17e30f962aa921dbc8aa8a99183345916dda76817ce6647734490594e7671b5','c17e30f962aa921dbc8aa8a99183345916dda76817ce6647734490594e7671b5'],p.VERSION:['68e2d82617b303e861df84dd793de0b065bb041a1fbedb7da8b5c93563149cdf','b5f65778c8a1776f45120eb6535bc762056e132a8d2f22a045aaf5fc91234f12','f85bc4739c6363499140e08f9617bab2bbe80944c879ae2ad14e2fb28b3f28bf','62379f03063620c5872761e9463ac51afc4287cd9d573e652697268afb4a9821','62379f03063620c5872761e9463ac51afc4287cd9d573e652697268afb4a9821']}
  for version,digests in expected.items():
   self.f.p['version']=version
   for outcome,digest in zip((None,'verified_valid','confirmed_invalid','numerical_ambiguous','infrastructure_error'),digests):
    with self.subTest(version=version,outcome=outcome):self.assertEqual(p.digest(self.f.calculate(obs=[]if outcome is None else[self.f.observation(outcome)])),digest)
 def test_v2_outputs_exactly_same_without_coverage_fields(self):
  self.f.p['version']=p.VERSION;s=self.f.calculate(obs=[self.f.observation('numerical_ambiguous')]);self.assertEqual(s['points']['b'*64],.5);self.assertNotIn('resolution_coverage_factor',s['miners']['b'*64])
if __name__=='__main__':unittest.main()
