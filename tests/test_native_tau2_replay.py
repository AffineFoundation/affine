import unittest
from subnet.native_tau2_replay import message_view

class NativeReplayTests(unittest.TestCase):
    def test_clock_changes_do_not_change_native_semantics(self):
        one={'messages':[{'role':'user','content':'Actual model observation','timestamp':'first'}]}
        two={'messages':[{'role':'user','content':'Actual model observation','timestamp':'second'}]}
        self.assertEqual(message_view(one),message_view(two))
    def test_user_observation_and_tool_call_changes_remain_visible(self):
        actual={'messages':[{'role':'user','content':'Actual model observation','timestamp':'first'},{'role':'assistant','tool_calls':[{'name':'native_tool','arguments':{'id':'original'}}]}]}
        fake={'messages':[{'role':'user','content':'Fabricated observation','timestamp':'second'},{'role':'assistant','tool_calls':[{'name':'native_tool','arguments':{'id':'changed'}}]}]}
        self.assertNotEqual(message_view(actual),message_view(fake))
        self.assertEqual(message_view(actual)[1]['tool_calls'][0]['arguments']['id'],'original')

if __name__=='__main__':unittest.main()
