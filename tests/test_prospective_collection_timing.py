import copy,json,tempfile,types,unittest
from pathlib import Path
from ops.trainer_lifecycle.prospective_collection_timing import VERSION,validate,install,profiles

OLD=dict(version='bounded-hourly-phases-v1',mine_seconds=600,freeze_seconds=60,audit_seconds=600,train_publication_seconds=2100,weight_seconds=120,slack_seconds=120)
CURRENT=dict(OLD,mine_seconds=1800,audit_seconds=60,train_publication_seconds=1500,weight_seconds=60)
FUTURE=dict(CURRENT,mine_seconds=1200,slack_seconds=720)

class Timing(unittest.TestCase):
 def setUp(self):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)
  self.a=dict(version=VERSION,first_round=91,previous_duration=600,duration=1800,previous_hourly_policy=OLD,hourly_policy=CURRENT,successor_first_round=93,successor_duration=1200,successor_hourly_policy=FUTURE)
  self.old=dict(duration=600,hourly_execution_policy=OLD)
  self.cfg=dict(state=str(self.root),duration=1800,hourly_execution_policy=CURRENT)
 def state(self,n):return dict(round=n,active=dict(epoch='test-'+str(n)))
 def issued(self,n,duration,policy):
  p=self.root/('test-'+str(n)+'-manifest.json');p.write_text(json.dumps(dict(start=100,deadline=100+duration,hourly_execution_policy=policy)));return p
 def test_existing92_is_byte_unchanged_and_admitted(self):
  p=self.issued(92,1800,CURRENT);raw=p.read_bytes();self.assertEqual(validate(self.a,self.old,self.cfg,self.state(92)),{'duration','hourly_execution_policy'});self.assertEqual(p.read_bytes(),raw)
 def test_future93_correct_issued_profile_survives_restart(self):
  self.issued(93,1200,FUTURE);validate(self.a,self.old,self.cfg,self.state(93))
 def test_issued93_old_profile_refuses_retroactive_cut(self):
  self.issued(93,1800,CURRENT)
  with self.assertRaisesRegex(ValueError,'already published'):validate(self.a,self.old,self.cfg,self.state(93))
 def test_issued92_new_profile_refuses_rewriting_history(self):
  self.issued(92,1200,FUTURE)
  with self.assertRaisesRegex(ValueError,'already published'):validate(self.a,self.old,self.cfg,self.state(92))
 def test_unpublished_future_can_open(self):validate(self.a,self.old,self.cfg,self.state(93))
 def test_physical_original_config_must_stay_exact(self):
  for field,value in(('duration',1200),('hourly_execution_policy',FUTURE)):
   with self.subTest(field=field),self.assertRaises(ValueError):validate(self.a,self.old,dict(self.cfg,**{field:value}),self.state(92))
 def test_successor_policy_budgets_and_types_cannot_drift(self):
  for field,value in(('successor_first_round',True),('successor_first_round',92),('successor_duration',600),('successor_hourly_policy',dict(FUTURE,freeze_seconds=120)),('hourly_policy',dict(CURRENT,train_publication_seconds=2100))):
   with self.subTest(field=field),self.assertRaises(ValueError):profiles(dict(self.a,**{field:value}))
 def test_contract_only_changes_two_metadata_fields_at_boundary(self):
  original=dict(duration=1800,hourly_execution_policy=CURRENT,K=4,L=4,max_batches=3,sampling={'source':'original'},jobTTL=3600)
  s=types.SimpleNamespace(contract=lambda c,n:copy.deepcopy(original));install(s,self.a)
  self.assertEqual(s.contract({},92),original)
  result=s.contract({},93);self.assertEqual(result,dict(original,duration=1200,hourly_execution_policy=FUTURE));self.assertEqual(sum(v for k,v in result['hourly_execution_policy'].items()if k!='version'),3600)
  self.assertEqual(original['duration'],1800)
 def test_wrong_original_contract_fails_closed(self):
  s=types.SimpleNamespace(contract=lambda c,n:dict(duration=600,hourly_execution_policy=OLD));install(s,self.a)
  with self.assertRaises(ValueError):s.contract({},93)
 def test_round_must_match_exact_active_epoch(self):
  with self.assertRaises(ValueError):validate(self.a,self.old,self.cfg,dict(round=93,active=dict(epoch='test-92')))
if __name__=='__main__':unittest.main()
