import unittest
from unittest.mock import Mock
from ops.probe_long_context_eog import trace_rows,verify_expected
from subnet import harness
from subnet.long_context_runtime import digest
class NativeEOGInputTests(unittest.TestCase):
    def test_exact_tool_observations_and_all_schemas_preserved(self):
        tools=[{'name':'public','inputSchema':{'type':'object'}}];public={'public':{'messages':[{'role':'user','content':'task'}],'tools':tools},'events':[{'name':'public','arguments':{'x':1},'observation':'exact\nresult'},{'name':'public','arguments':{},'observation':'next'}]}
        actual,rows=trace_rows(public,harness)
        self.assertEqual(actual,tools);self.assertEqual(rows[1]['messages'][-1],{'role':'user','content':'Tool result: exact\nresult'});self.assertEqual(rows[0]['action_sha256'],digest({'name':'public','arguments':{'x':1}}));self.assertEqual(public['public']['messages'],[{'role':'user','content':'task'}])
    def test_edited_history_or_action_rejected_before_model_recompute(self):
        runtime=Mock()
        for artifact in ({'prompt':[2],'output':[3]},{'prompt':[1],'output':[4]}):
            with self.assertRaisesRegex(ValueError,'native public'):verify_expected(runtime,artifact,None,[1],[3])
        runtime.verify.assert_not_called()
