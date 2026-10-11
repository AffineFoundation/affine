import copy
import unittest
import test_positive_sft_execution_route as route_fixture
from subnet import unaudited_training_execution as a
from subnet.research_parent_adoption import MODULE
from subnet.training_receipts import sha

class ResearchParentSourceTests(unittest.TestCase):
 def setUp(self):
  self.fixture=route_fixture.PositiveSFTRouteTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
 def candidate(self):
  r=copy.deepcopy(self.fixture.release['payload'])
  r['execution_source_files'].update({MODULE:'a'*64,'subnet/optimizer_state_cache.py':'b'*64})
  r['changed_source_files']={n:h for n,h in r['execution_source_files'].items() if r['original_source_files'].get(n)!=h}
  return r
 def test_new_release_requires_new_qualification_map(self):
  r=self.candidate();a._sources(r)
  with self.assertRaisesRegex(ValueError,'integration qualification'):a.release(self.fixture.sign(r),self.fixture.authority)
  q=r['execution_qualification']['payload'];q['execution_source_files_sha256']=sha(r['execution_source_files']);r['execution_qualification']=self.fixture.sign(q)
  self.assertEqual(a.release(self.fixture.sign(r),self.fixture.authority)['execution_source_files'][MODULE],'a'*64)
 def test_old_release_unchanged_and_sampler_change_still_rejected(self):
  a.release(self.fixture.release,self.fixture.authority)
  r=self.candidate();r['execution_source_files']['subnet/cached_sampling.py']='0'*64
  r['changed_source_files']={n:h for n,h in r['execution_source_files'].items() if r['original_source_files'].get(n)!=h}
  with self.assertRaisesRegex(ValueError,'only approved'):a._sources(r)
 def test_historical_and_missing_adoption_source_pin_rejected(self):
  r=self.candidate();r['execution_source_files'].pop(MODULE);r['changed_source_files'].pop(MODULE)
  with self.assertRaisesRegex(ValueError,'prospective SFT source pins'):a._sources(r)
  r=copy.deepcopy(self.fixture.fx.genesis_release['payload']);r['execution_source_files'].update({MODULE:'a'*64,'subnet/optimizer_state_cache.py':'b'*64})
  r['changed_source_files']={n:h for n,h in r['execution_source_files'].items() if r['original_source_files'].get(n)!=h}
  with self.assertRaisesRegex(ValueError,'prospective SFT source pins'):a._sources(r)

if __name__=='__main__':unittest.main()
