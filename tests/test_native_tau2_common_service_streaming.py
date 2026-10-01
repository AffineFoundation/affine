import json,pathlib,subprocess,tempfile,unittest
from unittest.mock import patch,Mock
from nacl.signing import SigningKey
from subnet import native_tau2_common_service_streaming as s
class StreamingCoordinatorTests(unittest.TestCase):
 def test_launch_disconnect_persists_hold_and_no_resubmission(self):
  with tempfile.TemporaryDirectory() as d:
   c=object.__new__(s.Coordinator);c.key=SigningKey.generate();c.ssh=lambda text:['ssh',text];folder=pathlib.Path(d);job={'scope':'one'}
   with patch.object(c,'scp') as scp,patch.object(s.subprocess,'check_output',side_effect=subprocess.CalledProcessError(255,['ssh'])):
    with self.assertRaisesRegex(RuntimeError,'unresolved remote launch'):c.remote(folder,'/root/owned',job,'train')
    record=json.loads((folder/'train-remote-process.json').read_text());self.assertEqual(record['launch_state'],'unresolved');self.assertEqual(record['job_sha256'],s.digest(job))
    with self.assertRaisesRegex(RuntimeError,'unresolved remote launch'):c.remote(folder,'/root/owned',job,'train')
    self.assertEqual(scp.call_count,1)
 def test_restart_does_not_open_new_epoch_after_unresolved_launch(self):
  with tempfile.TemporaryDirectory() as d:
   c=object.__new__(s.Coordinator);c.state=pathlib.Path(d);c.status=Mock();c.open=Mock();folder=c.state/'epoch-one';folder.mkdir();s.write(folder/'train-remote-process.json',{'launch_state':'unresolved'})
   with self.assertRaisesRegex(RuntimeError,'unresolved remote launch'):c.loop()
   c.open.assert_not_called()
 def test_report_from_other_signed_job_rejected(self):
  job={'training_policy':s.TRAIN_POLICY,'scope':'current'};report={'job_sha256':s.digest({'scope':'old'}),'training_policy':s.TRAIN_POLICY,'optimizer_steps':1,'full_model_finetune':True,'auxiliary_tokens_in_loss':False}
  with self.assertRaisesRegex(ValueError,'signed job identity'):s.validate_training_report(report,job)
 def test_report_binds_exact_current_job_and_policy(self):
  job={'training_policy':s.TRAIN_POLICY,'scope':'current'};report={'job_sha256':s.digest(job),'training_policy':s.TRAIN_POLICY,'optimizer_steps':1,'full_model_finetune':True,'auxiliary_tokens_in_loss':False};self.assertIs(s.validate_training_report(report,job),report)
  report['auxiliary_tokens_in_loss']=True
  with self.assertRaisesRegex(ValueError,'objective'):s.validate_training_report(report,job)
if __name__=='__main__':unittest.main()

class CompletionGateTests(unittest.TestCase):
 def setUp(self):
  self.contract={'fixed_auxiliary_descriptor':{'weights':'fixed'},'heldout_tasks':[{'index':i,'seed':20260930+i,'task_hash':str(i)} for i in range(16,32)]};self.contract['dataset_id']=s.digest(self.contract)
  self.report={'all_tasks_completed':True,'completed_count':16,'error_count':0,'dataset_id':self.contract['dataset_id'],'fixed_user_sha256':s.digest(self.contract['fixed_auxiliary_descriptor']),'records':[dict(t,verified=True,reward=0.) for t in self.contract['heldout_tasks']]}
 def gate(self,other=None,contract=None):return s.evaluation_completion_gate(self.report,other or self.report,self.contract,contract or self.contract)
 def test_all16_exact_fresh_reports_same_fixed_dataset_can_complete(self):self.assertTrue(self.gate()['complete'])
 def test_partial_error_cannot_claim_full_pipeline(self):
  import copy
  other=copy.deepcopy(self.report);other.update(all_tasks_completed=False,completed_count=15,error_count=1);other['records'][-1]={'index':31,'verified':False,'error_type':'ContextError'};self.assertFalse(self.gate(other)['complete'])
 def test_all_completed_flag_cannot_hide_missing_row(self):
  import copy
  other=copy.deepcopy(self.report);other['records'].pop();self.assertFalse(self.gate(other)['complete'])
 def test_changed_fixed_dataset_cannot_complete(self):
  import copy
  contract=copy.deepcopy(self.contract);contract['fixed_auxiliary_descriptor']={'weights':'drifted'};contract['dataset_id']=s.digest({k:v for k,v in contract.items() if k!='dataset_id'});self.assertFalse(self.gate(contract=contract)['complete'])
 def test_wrong_seed_cannot_complete(self):
  import copy
  other=copy.deepcopy(self.report);other['records'][0]['seed']+=1;self.assertFalse(self.gate(other)['complete'])
