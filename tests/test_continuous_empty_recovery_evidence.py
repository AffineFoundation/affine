import unittest
from ops.check_gpu_continuous_evidence import epoch_pending,bind_empty_recovery
class EmptyEvidenceTests(unittest.TestCase):
 def test_partial_empty_evaluations_are_pending(self):
  self.assertTrue(epoch_pending(None,{'status':'closed_no_accepted_batches'},False,0,16))
  self.assertTrue(epoch_pending(None,{'status':'closed_no_accepted_batches'},False,15,16))
  self.assertFalse(epoch_pending(None,{'status':'closed_no_accepted_batches'},False,16,16))
 def test_normal_epoch_needs_metrics_but_abort_has_own_evidence(self):
  self.assertTrue(epoch_pending(None,None,False,16,16))
  self.assertFalse(epoch_pending(None,None,True,16,16))
  self.assertFalse(epoch_pending({'status':'aborted'},None,False,0,16))
 def empty(self):return [{'epoch':'empty','next_epoch':'next','checkpoint':'old'}]
 def test_open_successor_does_not_prove_recovery(self):
  rows=self.empty();bind_empty_recovery(rows,[]);self.assertFalse(rows[0]['recovery_verified'])
 def test_verified_real_training_successor_proves_recovery(self):
  rows=self.empty();bind_empty_recovery(rows,[{'epoch':'next','checkpoint':'new','steps':1,'points':{'miner':1}}]);self.assertTrue(rows[0]['recovery_verified']);self.assertEqual(rows[0]['recovery_checkpoint'],'new')
 def test_zero_reward_or_unchanged_weights_not_recovery(self):
  for cp,steps,points in [('old',1,{'miner':1}),('new',0,{'miner':1}),('new',1,{})]:
   rows=self.empty();bind_empty_recovery(rows,[{'epoch':'next','checkpoint':cp,'steps':steps,'points':points}]);self.assertFalse(rows[0]['recovery_verified'])
if __name__=='__main__':unittest.main()
