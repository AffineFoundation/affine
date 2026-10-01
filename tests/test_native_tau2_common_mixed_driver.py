"""Registry/rejection orchestration controls; no numerical qualification."""
import types,unittest
from unittest.mock import Mock,patch
from subnet import native_tau2_common_mixed_driver as d
class Mixed(unittest.TestCase):
 def test_distinct_roles_use_separate_processes_and_close_on_partial_failure(self):
  import tempfile
  manifest={'roles':{'agent':{'checkpoint':'gpu'},'user':{'checkpoint':'fixedcpu'}}}
  with tempfile.TemporaryDirectory() as directory:
   rows={name:{'argv':['trusted-'+name],'environment':{},'stderr_path':directory+'/'+name+'.log'} for name in ['agent','user']}
   first=Mock()
   with patch.object(d,'validate_epoch',return_value=manifest),patch.object(d,'ProcessRoleRuntime',side_effect=[first,ValueError('second worker rejected')]) as load,self.assertRaises(ValueError):d.dependencies({},'authority',{},rows)
   self.assertEqual(load.call_count,2);first.close.assert_called_once();self.assertEqual(d.STARTED_WORKERS,[])
 def test_context_rejection_reports_only_safe_counts_and_hash(self):
  endpoint=types.SimpleNamespace(records=[{}]*10,manifest={'roles':{'agent':{'request_model':'agent','max_output_tokens':128,'max_context':8192}}},renderer=lambda t,r:[1]*8113,runtimes={'agent':types.SimpleNamespace(tokenizer='tokenizer')})
  report=d.rejection_diagnostic(endpoint,{'model':'agent','messages':[{'content':'private secret'}]},ValueError('complete context exceeds approved role budget'))
  self.assertEqual(report['reason'],'complete_context_budget_exceeded');self.assertEqual(report['prompt_tokens'],8113);self.assertEqual(report['completed_role_count'],10)
  self.assertNotIn('private',str(report));self.assertNotIn('messages',report)
 def test_infrastructure_failure_is_not_reclassified_as_budget_error(self):
  endpoint=types.SimpleNamespace(records=[],manifest={'roles':{'agent':{'request_model':'agent','max_output_tokens':128,'max_context':8192}}},renderer=Mock(side_effect=RuntimeError('private config')),runtimes={'agent':types.SimpleNamespace(tokenizer=None)})
  report=d.rejection_diagnostic(endpoint,{'model':'agent'},OSError('private host'))
  self.assertEqual(report['reason'],'role_request_rejected');self.assertEqual(report['exception_type'],'OSError');self.assertEqual(report['diagnostic_exception_type'],'RuntimeError');self.assertNotIn('private',str(report))
if __name__=='__main__':unittest.main()
