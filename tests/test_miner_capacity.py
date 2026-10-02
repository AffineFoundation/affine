import unittest
from types import SimpleNamespace
from unittest.mock import patch
from subnet.miner import Miner
from subnet.batches import UploadBudgetExceeded


class MinerCapacityTests(unittest.TestCase):
    def test_rejected_candidate_cannot_poison_acknowledged_local_state(self):
        miner = Miner.__new__(Miner)
        miner.manifest = dict(epoch='test', checkpoint={'id': 'approved'}, K=1, L=1)
        original = [({'already-uploaded': True}, [])]
        miner.batches = original
        runtime = SimpleNamespace(spec=SimpleNamespace(version='original'))
        runtime.for_environment = lambda *args: runtime
        runtime.rollout = lambda index, seed: (dict(reward=1 if seed % 2 == 0 else 0,
            classification='positive' if seed % 2 == 0 else 'negative', turns=[{'output': [seed]}]), [])
        miner.runtime = runtime
        miner.runtimes = {}
        definition = dict(env_id='affine_math', spec={})
        with patch('subnet.miner.entry', return_value=definition), patch('subnet.miner.harness_for', return_value={}), patch('subnet.miner.pack', side_effect=UploadBudgetExceeded('full')):
            with self.assertRaises(UploadBudgetExceeded):
                miner.search(9, max_attempts=2)
        self.assertIs(miner.batches, original)
        self.assertEqual(len(miner.batches), 1)


if __name__ == '__main__':
    unittest.main()
