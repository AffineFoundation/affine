"""Nine admitted tasks retain the unchanged across-miner zero rule."""
import copy,unittest
from subnet.scoring import score,unique_points
from test_manifest_batch_capacity import BatchCapacity
class Cap9Scoring(unittest.TestCase):
 def test_cross_miner_collision_still_zero_for_both(self):
  fixture=BatchCapacity();case,manifest,batches=fixture.fixture(9);self.addCleanup(fixture.doCleanups)
  reports={'one':{'accepted':batches},'two':{'accepted':[copy.deepcopy(batches[8])]}}
  self.assertEqual(unique_points(reports),{'one':8,'two':0});self.assertEqual(score(reports)['weights'],{'one':1.,'two':0.})
 def test_nine_unique_task_points_do_not_depend_on_extra_rollouts(self):
  fixture=BatchCapacity();case,manifest,batches=fixture.fixture(9);self.addCleanup(fixture.doCleanups)
  report={'one':{'accepted':batches}}
  self.assertEqual(unique_points(report),{'one':9})
  report['one']['accepted'].append(copy.deepcopy(batches[8]));self.assertEqual(unique_points(report),{'one':9})
if __name__=='__main__':unittest.main()
