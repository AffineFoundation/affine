from pathlib import Path
import copy,importlib.util,sys,unittest
from unittest.mock import patch
from contextlib import nullcontext
from ops import current_assessment_writer as writer,current_assessment_compact_queue as compact
import test_current_assessment_writer as f

class WriterBinding(unittest.TestCase):
 setUp=f.WriterControls.setUp
 invoke=f.WriterControls.invoke
 def enabled(self):
  p=copy.deepcopy(self.policy['payload']);p['compact_queue_evidence']=compact.POLICY.copy();p['module_hashes'][str(Path(compact.__file__))]=writer.file_hash(compact.__file__);self.policy=f.sign(p,self.key);return p
 def inspect_kwargs(self,expected):
  observed=[]
  def loader(*args,**kwargs):observed.append(kwargs);return self.evidence
  with patch.object(writer,'authenticate_cutover',return_value=(self.c,{})),patch.object(writer,'global_lock',return_value=nullcontext()),patch.object(writer,'guard_files'),patch.object(writer,'observe_units',return_value=[]),patch.object(writer,'writer_gate'),patch.object(writer.time,'time',return_value=7300):
   writer.run_once(self.policy,self.cutover,self.anchor,self.auth,adapter_factory=self.adapter,evidence_loader=loader)
  self.assertEqual(len(observed),1)
  if expected is None:self.assertNotIn('compact_queue_evidence',observed[0])
  else:self.assertEqual(observed[0]['compact_queue_evidence'],expected)
 def test_explicit_flag_is_forwarded_to_actual_evidence_loader(self):
  self.enabled();self.inspect_kwargs(compact.POLICY)
 def test_new_module_pin_required_before_chain_or_evidence(self):
  p=self.enabled();p['module_hashes'].pop(str(Path(compact.__file__)));self.policy=f.sign(p,self.key)
  with self.assertRaisesRegex(ValueError,'execution modules'):self.invoke()
  self.assertEqual(self.calls,[])
 def test_unknown_or_widened_policy_refused(self):
  p=self.enabled();p['compact_queue_evidence']['maximum_rows']+=1;self.policy=f.sign(p,self.key)
  with self.assertRaisesRegex(ValueError,'bounded actual-byte'):self.invoke()
  self.assertEqual(self.calls,[])
 def test_no_option_preserves_original_evidence_kwargs(self):
  self.inspect_kwargs(None)
if __name__=='__main__':unittest.main(verbosity=2)
