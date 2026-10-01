import unittest,copy
from subnet.empty_epoch_policy import selected,dispatch_allowed,validate_empty_completion
class Tests(unittest.TestCase):
 def test_only_designated_round_skips_dispatch(self):
  config={'epoch_prefix':'nonpayable-example','controlled_empty_rounds':[8]}
  self.assertIsNone(selected(config,7));self.assertIsNone(selected(config,9));policy=selected(config,8)
  manifest={'epoch':'nonpayable-8','payable':False,'operator_test_policy':policy}
  self.assertFalse(dispatch_allowed(manifest));self.assertTrue(dispatch_allowed({'epoch':'nonpayable-9'}))
 def test_malformed_config_or_payable_scope_rejected(self):
  for rounds in ([True],[1,2],[-1]):
   with self.assertRaises(ValueError):selected({'epoch_prefix':'nonpayable-x','controlled_empty_rounds':rounds},1)
  with self.assertRaises(ValueError):selected({'epoch_prefix':'real','controlled_empty_rounds':[1]},1)
 def test_signed_scope_or_credit_mutation_rejected(self):
  policy=selected({'epoch_prefix':'nonpayable-x','controlled_empty_rounds':[8]},8)
  m={'epoch':'nonpayable-x','payable':False,'operator_test_policy':policy}
  validate_empty_completion(m,{'points':{},'weights':{}},{})
  for points,weights,reports in [({'uid':1},{},{}),({}, {'uid':1},{}),({}, {},{'miner':{'accepted':[{}]}})]:
   with self.assertRaises(ValueError):validate_empty_completion(m,{'points':points,'weights':weights},reports)
  changed=copy.deepcopy(m);changed['operator_test_policy']['miner_dispatch']=True
  with self.assertRaises(ValueError):dispatch_allowed(changed)
if __name__=='__main__':unittest.main()
