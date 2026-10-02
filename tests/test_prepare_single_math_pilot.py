import importlib.util,unittest
from pathlib import Path
p=Path(__file__).parents[1]/'ops/prepare_single_math_pilot.py';s=importlib.util.spec_from_file_location('prep',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def rows():
 return [dict(data=dict(problem=f'public original fixture {i}',subject=f'subject{i%7}',level=f'Level {1+i%5}')) for i in range(7496)]
class Preparation(unittest.TestCase):
 def test_whole_public_population_and_separate_owned_subset(self):
  r=rows();c=m.configuration({'id':'affine_math'},r,Path('/tmp/preparation'))
  train=set(c['environments'][0]['indices']);reserved=set(c['reserved_heldout_indices']['affine_math'])
  self.assertEqual((len(train),len(reserved)),(6746,750));self.assertFalse(train&reserved)
  self.assertEqual(len(train|reserved),7496)
  schedule=c['owned_mining_schedule'];visited=[i for row in schedule for i in row['affine_math']]
  self.assertEqual(len(schedule),422);self.assertEqual(len(visited),len(set(visited)))
  self.assertEqual(set(visited),train);self.assertTrue(all(1<=len(row['affine_math'])<=16 for row in schedule))
  self.assertEqual(len(c['heldout'][0]['indices']),32);self.assertTrue(set(c['heldout'][0]['indices'])<=reserved)
  self.assertNotIn('indices_per_environment_per_epoch',c);self.assertIsNone(c['initial_checkpoint'])
  self.assertEqual(c['environments'][0]['harness']['policy'],'autoregressive')
 def test_duplicate_or_wrong_count_refused(self):
  r=rows();r[1]['data']['problem']=r[0]['data']['problem']
  with self.assertRaises(ValueError):m.population(r)
  with self.assertRaises(ValueError):m.population(rows()[:-1])
if __name__=='__main__':unittest.main()
