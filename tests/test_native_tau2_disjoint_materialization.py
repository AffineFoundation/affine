import copy,unittest
from ops.materialize_native_tau2_disjoint import select,scenario_hash

def row(i,group,persona=None):return {'id':str(i),'user_scenario':{'instructions':{'task_instructions':group,'reason_for_call':'public issue'},'persona':persona},'initial_state':{'variant':i}}
class Disjoint(unittest.TestCase):
 def test_persona_and_initial_state_variants_share_group(self):
  self.assertEqual(scenario_hash(row(0,'same','A')),scenario_hash(row(1,'same','B')))
 def test_original_bodies_preserved_and_groups_disjoint(self):
  rows=[row(i,f'group{i//3}',str(i)) for i in range(24)];before=copy.deepcopy(rows);chosen,total=select(rows,8)
  self.assertEqual(rows,before);self.assertEqual(total,8);self.assertEqual([r['id'] for r in chosen],list(map(str,[0,3,6,9,12,15,18,21])))
  self.assertFalse({scenario_hash(t) for t in chosen[:4]}&{scenario_hash(t) for t in chosen[4:]})
 def test_single_scenario_inventory_cannot_claim_disjointness(self):
  with self.assertRaises(ValueError):select([row(i,'same') for i in range(32)],32)
 def test_five_actual_style_groups_support_32_variant_tasks(self):
  chosen,groups=select([row(i,f'group{i//20}',str(i)) for i in range(100)],32)
  self.assertEqual(groups,5);self.assertEqual(len(chosen),32)
  self.assertEqual(len({scenario_hash(t) for t in chosen[:16]}),2)
  self.assertEqual(len({scenario_hash(t) for t in chosen[16:]}),3)
  self.assertFalse({scenario_hash(t) for t in chosen[:16]}&{scenario_hash(t) for t in chosen[16:]} )
 def test_missing_instructions_and_duplicate_ids_rejected(self):
  with self.assertRaises(ValueError):select([{'id':'0','user_scenario':{}}],4)
  with self.assertRaises(ValueError):select([row(0,'a'),row(0,'b'),row(2,'c'),row(3,'d')],4)
if __name__=='__main__':unittest.main()
