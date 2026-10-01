import unittest
from ops.materialize_tasksets import split

class ProspectiveTaskSplit(unittest.TestCase):
    def test_fixed_heldout_cannot_enter_training_search(self):
        training,heldout=split(32,16)
        self.assertFalse(set(training)&set(heldout))
        self.assertEqual(set(training)|set(heldout),set(range(32)))
        self.assertEqual(heldout,list(range(16,32)))
    def test_empty_or_boolean_partitions_are_rejected(self):
        for count,training_count in ((32,0),(32,32),(True,1),(32,True),(1025,16)):
            with self.subTest(count=count,training_count=training_count),self.assertRaises(ValueError):split(count,training_count)

if __name__=='__main__':unittest.main()
