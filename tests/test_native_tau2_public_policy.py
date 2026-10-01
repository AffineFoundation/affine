import json,unittest
from subnet.native_tau2_public_policy import public_action,format_candidates

def request(result):
    return {'model':'native-model-user','tools':[{'function':{'name':n}} for n in ('check_network_status','toggle_roaming','run_speed_test')],'messages':[{'role':'user','content':'Check your network status; enable roaming if disabled.'},{'role':'assistant','tool_calls':[{'id':'t','function':{'name':'check_network_status','arguments':'{}'}}]},{'role':'tool','tool_call_id':'t','content':result}]}
class PublicPolicyTests(unittest.TestCase):
    def test_public_observation_drives_actual_roaming_action(self):
        action,reason=public_action(request('Data Roaming Enabled: No'));self.assertIn('toggle_roaming',action);self.assertEqual(reason,'visible-disabled-roaming-plus-agent-instruction')
    def test_enabled_roaming_does_not_toggle_it_back_off(self):
        action,_=public_action(request('Data Roaming Enabled: Yes'));self.assertIn('run_speed_test',action)
    def test_unresolved_speed_never_synthesizes_stop(self):
        r=request('Speed test failed: No Connection.');r['messages'][1]['tool_calls'][0]['function']['name']='run_speed_test';action,_=public_action(r);self.assertNotIn('###STOP###',action)
    def test_candidates_preserve_equivalent_tool_semantics(self):
        values=format_candidates('{"tool_call":{"name":"run_speed_test","arguments":{}}}');self.assertNotEqual(values[0],values[1]);self.assertEqual(json.loads(values[0]),json.loads(values[1]))
