import unittest
from ops.check_gpu_continuous_evidence import training_pair_index

class Tests(unittest.TestCase):
    def pair(self):
        return {'env_id':'affine_math'}, {'env_id':'affine_math','index':3}, {'env_id':'affine_math','index':3}
    def test_historical_environment_definition_has_no_batch_index(self):
        self.assertEqual(training_pair_index(*self.pair()),3)
    def test_fresh_batch_requires_same_index(self):
        b,p,n=self.pair();b['index']=3
        self.assertEqual(training_pair_index(b,p,n),3)
        b['index']=4
        with self.assertRaises(ValueError):training_pair_index(b,p,n)
    def test_different_negative_task_or_environment_rejected(self):
        b,p,n=self.pair();n['index']=4
        with self.assertRaises(ValueError):training_pair_index(b,p,n)
        n['index']=3;n['env_id']='affine_logic'
        with self.assertRaises(ValueError):training_pair_index(b,p,n)
    def test_boolean_or_missing_index_rejected(self):
        for value in [True,None,-1]:
            b,p,n=self.pair();p['index']=n['index']=value
            with self.assertRaises(ValueError):training_pair_index(b,p,n)

if __name__=='__main__':unittest.main()
