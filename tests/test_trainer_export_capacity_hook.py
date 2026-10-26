import unittest
from ops.trainer_export_capacity_hook import retry_capacity,DISK_ERROR
class CapacityHookTests(unittest.TestCase):
 def test_no_reconcile_when_existing_gate_passes(self):
  self.assertEqual(retry_capacity(lambda:'pass',lambda:self.fail('unneeded')),'pass')
 def test_disk_failure_reconciles_then_rechecks_original(self):
  calls=[]
  def gate():
   calls.append('gate')
   if len(calls)==1:raise ValueError(DISK_ERROR)
   return {'required_bytes':123}
  self.assertEqual(retry_capacity(gate,lambda:calls.append('reconcile')),{'required_bytes':123});self.assertEqual(calls,['gate','reconcile','gate'])
 def test_memory_or_other_failures_never_relaxed(self):
  def gate():raise ValueError('memory')
  with self.assertRaisesRegex(ValueError,'memory'):retry_capacity(gate,lambda:self.fail('wrong gate'))
 def test_stale_catalog_does_not_block_now_sufficient_capacity(self):
  count=[0]
  def gate():
   count[0]+=1
   if count[0]==1:raise ValueError(DISK_ERROR)
   return True
  def old():raise ValueError('stale catalog')
  self.assertTrue(retry_capacity(gate,old));self.assertEqual(count[0],2)
 def test_failed_or_consumed_catalog_never_bypasses_disk_guard(self):
  def gate():raise ValueError(DISK_ERROR)
  with self.assertRaisesRegex(ValueError,DISK_ERROR):retry_capacity(gate,lambda:{'already_absent':['old']})
if __name__=='__main__':unittest.main()
