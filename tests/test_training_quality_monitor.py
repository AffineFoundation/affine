import copy,unittest,base64
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.training_quality_monitor import compare,reconcile,evaluation_indices
class Quality(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
 def sign(self,x):return {'payload':x,'signer':self.authority,'signature':base64.b64encode(self.key.sign(canonical(x)).signature).decode()}
 def inputs(self,n=32,before=20,after=18):
  record=dict(env_id='math',dataset_id='same',taskset_hash='same',heldout_indices=list(range(n)),fixed_task_ids=list(range(n)),harness_config={},model_runtime_revision='fp32',requested_count=n,completed_count=n,successes=before,evaluation_failures=[])
  b=dict(phase='before',epoch='e',checkpoint='cp0',status='complete',public_optimizer_steps=4,records=[record]);a=copy.deepcopy(b);a.update(phase='after',checkpoint='cp1',public_optimizer_steps=5);a['records'][0]['successes']=after
  t=dict(version='authenticated-training-quality-input-v1',epoch='e',input_checkpoint='cp0',output_checkpoint='cp1',optimizer_step_before=4,optimizer_step_after=5,parent_state_sha256='s0',output_state_sha256='s1',inference_weights_changed=True,diagnostics=dict(training_pair_margin_delta=[.01],updates=[dict(loss=.69,gradient_norm_before_clip=.3)]))
  return b,a,t
 def result(self,*args):return compare(*(self.sign(x)for x in self.inputs(*args)),self.authority)
 def test_small32_noise_never_confirmed(self):self.assertFalse(self.result()['hold']);self.assertFalse(self.result()['suites'][0]['confirmed_regression'])
 def test_severe_confirmed_regression(self):self.assertTrue(reconcile([self.result(750,600,300)])['hold_future_training'])
 def test_duplicate_evidence_does_not_inflate_streak(self):
  row=self.result();self.assertEqual(reconcile([row,row])['observed_updates'],1)
 def test_changed_eval_tasks_rejected(self):
  b,a,t=self.inputs();a['records'][0]['dataset_id']='different'
  with self.assertRaises(ValueError):compare(*(self.sign(x)for x in (b,a,t)),self.authority)
 def test_nonfinite_gradient_rejected(self):
  b,a,t=self.inputs();t['diagnostics']['updates'][0]['loss']=float('inf')
  with self.assertRaises(ValueError):compare(*(self.sign(x)for x in (b,a,t)),self.authority)
 def test_repeated_confirmed_regression_holds_with_exact_lineage(self):
  rows=[]
  for i in range(3):
   r=self.result(750,500,400);r.update(evidence_id=str(i),input_checkpoint='cp'+str(i),output_checkpoint='cp'+str(i+1),optimizer_step_before=i,optimizer_step_after=i+1,parent_state_sha256='s'+str(i),output_state_sha256='s'+str(i+1));rows.append(r)
  self.assertFalse(rows[0]['hold']);self.assertTrue(reconcile(rows)['hold_future_training'])
  rows[1]['parent_state_sha256']='foreign'
  with self.assertRaises(ValueError):reconcile(rows)
 def test_eval_rotation_and_full750_no_leak(self):
  reserved=list(range(750));mining=list(range(750,1000));self.assertEqual(len(evaluation_indices(reserved,mining,0)),128);self.assertEqual(evaluation_indices(reserved,mining,23),reserved)
  with self.assertRaises(ValueError):evaluation_indices(reserved,[0],0)
 def test_changed_signature_rejected(self):
  b,a,t=self.inputs();doc=self.sign(a);doc['payload']['checkpoint']='forged'
  with self.assertRaises(Exception):compare(self.sign(b),doc,self.sign(t),self.authority)
 def test_infra_not_penalty(self):
  b,a,t=self.inputs();a['status']='error';row=compare(*(self.sign(x)for x in (b,a,t)),self.authority);self.assertEqual(row['status'],'unknown_infrastructure');self.assertFalse(row['hold'])
if __name__=='__main__':unittest.main()
