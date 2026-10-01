import json,pathlib,tempfile,unittest
from unittest.mock import Mock,patch
from subnet import native_tau2_common_streaming_driver as s
class StreamingDriverTests(unittest.TestCase):
 def fixtures(self,out):
  epoch={'payload':{'epoch':'new'}};receipts=[{'payload':{'role':role,'probabilities_file':f'role-{i}.npy'}} for i,role in enumerate(('user','agent','user'))]
  for name,value in [('signed-epoch.json',epoch),('signed-receipts.json',receipts),('signed-native-simulation.json',{})]:(out/name).write_text(json.dumps(value))
  manifest={'epoch':'new','environment':{'id':'tau2','version':'controlled'},'sampler_provenance':'curated'}
  return epoch,receipts,manifest
 def test_every_role_array_independently_checked_in_order(self):
  with tempfile.TemporaryDirectory() as d:
   out=pathlib.Path(d);epoch,receipts,manifest=self.fixtures(out);runtimes={'user':object(),'agent':object()};native={'task_hash':'a'*64,'reward':1,'full_native_trajectory_verified':True}
   with patch.object(s.base,'dependencies',return_value=(manifest,runtimes,{'user':None,'agent':None})),patch.object(s,'checked_records'),patch.object(s,'read_array',side_effect=[b'u0',b'a1',b'u2']) as reads,patch.object(s,'verify_receipt',return_value={'verified':True}) as checks,patch.object(s,'replay',return_value=native),patch.object(s.base,'sign',return_value={}),patch.object(s,'admit_sample',return_value={}),patch.object(s.base,'save'):
    report=s.verify(epoch,'authority',{}, {},object(),0,0,None,None,None,out,object());self.assertTrue(report['all_model_roles_verified']);self.assertEqual(checks.call_count,3);self.assertEqual([c.args[2] for c in checks.call_args_list],[b'u0',b'a1',b'u2']);self.assertEqual([c.args[2] for c in reads.call_args_list],['role-0.npy','role-1.npy','role-2.npy'])
 def test_corrupt_storage_stops_before_native_grade(self):
  with tempfile.TemporaryDirectory() as d:
   out=pathlib.Path(d);epoch,receipts,manifest=self.fixtures(out)
   with patch.object(s.base,'dependencies',return_value=(manifest,{},{})),patch.object(s,'checked_records'),patch.object(s,'read_array',side_effect=ValueError('hash mismatch')),patch.object(s,'replay') as replay:
    with self.assertRaises(ValueError):s.verify(epoch,'authority',{}, {},object(),0,0,None,None,None,out,object())
    replay.assert_not_called()
 def test_wrong_ordinal_path_rejects_before_storage(self):
  with tempfile.TemporaryDirectory() as d:
   out=pathlib.Path(d);epoch,receipts,manifest=self.fixtures(out);receipts[0]['payload']['probabilities_file']='role-9.npy';(out/'signed-receipts.json').write_text(json.dumps(receipts))
   with patch.object(s.base,'dependencies',return_value=(manifest,{},{})),patch.object(s,'checked_records'),patch.object(s,'read_array') as read:
    with self.assertRaisesRegex(ValueError,'ordinal'):s.verify(epoch,'authority',{}, {},object(),0,0,None,None,None,out,object())
    read.assert_not_called()
if __name__=='__main__':unittest.main()
