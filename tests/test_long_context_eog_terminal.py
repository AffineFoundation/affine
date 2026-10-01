import unittest
from ops.probe_long_context_eog_terminal import trace_rows
from subnet import harness

class TerminalTests(unittest.TestCase):
 def test_complete_observations_and_terminal_action_bound(self):
  public={'public':{'messages':[{'role':'user','content':'public request'}],'tools':[{'name':'x'}]},'events':[{'name':'x','arguments':{'a':1},'observation':'exact native observation'}]}
  tools,rows=trace_rows(public,harness,'DONE')
  self.assertEqual(len(rows),2);self.assertEqual(rows[-1]['messages'][-1]['content'],'Tool result: exact native observation')
  self.assertEqual(rows[-1]['text'],'DONE');self.assertTrue(rows[-1]['terminal']);self.assertEqual(tools,public['public']['tools'])
  self.assertNotEqual(rows[-1]['action_sha256'],rows[0]['action_sha256'])
 def test_tool_call_cannot_masquerade_as_terminal(self):
  public={'public':{'messages':[{'role':'user','content':'x'}],'tools':[]},'events':[]}
  with self.assertRaisesRegex(ValueError,'no tool calls'):trace_rows(public,harness,'{"tool_call":{"name":"x","arguments":{}}}')

class TerminalSealerTests(unittest.TestCase):
 def test_six_proofs_cannot_claim_complete_terminal_control(self):
  from pathlib import Path
  from subnet.long_context_runtime import POLICY,digest
  from ops.seal_long_context_eog_terminal import validate_fetched
  job={'role':'long-context-proof-probe','experiment':'native-public-eog-terminal-v3','policy':POLICY,'terminal_text':'DONE'}
  report={'job_hash':digest(job),'completed':True,'full_model_recompute':True,'terminal_text':'DONE','terminal_model_proof':True,'records':[{'full_proof_verified':True}]*6}
  with self.assertRaisesRegex(ValueError,'complete fresh'):validate_fetched(job,report,Path('/unused'),b'')
 def test_terminal_text_cannot_change_after_generation(self):
  from pathlib import Path
  from ops.seal_long_context_eog_terminal import validate_fetched
  with self.assertRaisesRegex(ValueError,'terminal model binding'):validate_fetched({'terminal_text':'DONE'},{'terminal_text':'forged','terminal_model_proof':True},Path('/unused'),b'')
