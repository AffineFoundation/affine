import copy,unittest
from pathlib import Path
from subnet.native_tau2_common_cpu import FirstTaskPublicPolicy,POLICY_REVISION,hash_file
class PublicPolicy(unittest.TestCase):
 def descriptor(self,role):
  root=Path(__file__).resolve().parents[1]
  return {'revision':POLICY_REVISION,'scope':'public-request-derived-curated-output','role':role,'request_model':'model-'+role,'agent_seed_stride':256,'source_files':{n:hash_file(root/n) for n in ['subnet/native_tau2_public_policy.py','subnet/native_tau2_common_cpu.py']}}
 def test_agent_attempt_can_select_different_public_guidance(self):
  p=FirstTaskPublicPolicy(self.descriptor('agent'),'agent','model-agent');request={'model':'model-agent','messages':[{'role':'user','content':'visible goal'}],'tools':[]}
  self.assertNotEqual(p.select(request,0),p.select(request,256));self.assertEqual(p.select(request,0),p.select(request,0))
 def test_fixed_auxiliary_policy_uses_visible_native_tool_results(self):
  p=FirstTaskPublicPolicy(self.descriptor('user'),'user','model-user')
  request={'model':'model-user','messages':[{'role':'user','content':'Please check your network status and enable roaming'},{'role':'assistant','tool_calls':[{'id':'a','function':{'name':'check_network_status'}}]},{'role':'tool','tool_call_id':'a','content':'roaming_enabled: false'}],'tools':[{'function':{'name':'toggle_roaming'}}]}
  output=p.select(request,7);self.assertIn('toggle_roaming',output);self.assertNotIn('PRIVATE',output)
  other=copy.deepcopy(request);other['messages'][-1]['content']='roaming_enabled: true';self.assertNotEqual(p.select(other,7),output)
 def test_unbound_policy_source_or_wrong_role_rejected(self):
  d=self.descriptor('user');d['source_files']['subnet/native_tau2_public_policy.py']='c'*64
  with self.assertRaises(ValueError):FirstTaskPublicPolicy(d,'user','model-user')
  p=FirstTaskPublicPolicy(self.descriptor('user'),'user','model-user')
  with self.assertRaises(ValueError):p.select({'model':'wrong'},0)
if __name__=='__main__':unittest.main()
