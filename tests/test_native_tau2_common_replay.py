import copy,unittest
from subnet.native_tau2_common_replay import ReplayResponses,compare_native
from subnet.native_tau2_common_contract import digest
class NativeReplay(unittest.TestCase):
 def test_exact_native_request_role_and_full_context(self):
  request={'model':'fixed-user','messages':[{'role':'user','content':'full native context'}],'tools':[{'name':'whole-schema'}]}
  responses=ReplayResponses([{'request_sha256':digest(request),'response':{'native':'response'}}])
  with self.assertRaises(ValueError):responses.response(dict(request,model='changing-agent'))
  with self.assertRaises(ValueError):responses.response(dict(request,tools=[]))
  self.assertEqual(responses.cursor,0);self.assertEqual(responses.response(request),{'native':'response'})
  with self.assertRaises(ValueError):responses.response(request)
 def fixture(self):
  manifest={'epoch':'epoch','environment':{'id':'original'},'roles':{'user':{'checkpoint':'fixed'}}}
  task={'id':'genuine'}
  result={'manifest_sha256':digest(manifest),'epoch':'epoch','environment_id':'original','environment_index':4,'fixed_user_sha256':digest(manifest['roles']['user']),'task_hash':digest(task),'task':task,'simulation':{'messages':[{'role':'tool','content':'exact-original','timestamp':'then'}],'termination_reason':'USER_STOP','reward_info':{'reward':1,'native_all':{'database':True}}}}
  return manifest,result
 def test_only_wallclock_metadata_can_differ(self):
  manifest,old=self.fixture();new=copy.deepcopy(old);new['simulation']['messages'][0]['timestamp']='later'
  self.assertTrue(compare_native(old,new,manifest,4))
 def test_forged_observation_reward_or_fixed_auxiliary_rejected(self):
  manifest,old=self.fixture()
  for field in ['observation','reward','auxiliary','task']:
   new=copy.deepcopy(old)
   if field=='observation':new['simulation']['messages'][0]['content']='fabricated'
   elif field=='reward':new['simulation']['reward_info']['native_all']['database']=False
   elif field=='auxiliary':new['fixed_user_sha256']='c'*64
   else:new['task']['id']='changed'
   with self.subTest(field=field),self.assertRaises(ValueError):compare_native(old,new,manifest,4)
if __name__=='__main__':unittest.main()
