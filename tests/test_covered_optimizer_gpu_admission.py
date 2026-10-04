from types import SimpleNamespace
import unittest
from ops.probe_covered_epoch_optimizer import admit_gpu


class GPUAdmissionTests(unittest.TestCase):
    def fake_gpu(self, initialized=False, available=True, sm=(9, 0)):
        state = {'initialized': initialized, 'property_queries': 0}
        def properties():
            state['initialized'] = True
            state['property_queries'] += 1
            return sm
        cuda = SimpleNamespace(is_initialized=lambda: state['initialized'],
                               is_available=lambda: available,
                               get_device_capability=properties)
        return SimpleNamespace(cuda=cuda), state

    def test_fresh_process_accepts_property_query_that_initializes_cuda(self):
        torch, state = self.fake_gpu()
        admit_gpu(torch)
        self.assertTrue(state['initialized'])
        self.assertEqual(state['property_queries'], 1)

    def test_initialized_process_refused_before_any_property_query(self):
        torch, state = self.fake_gpu(initialized=True)
        with self.assertRaises(ValueError): admit_gpu(torch)
        self.assertEqual(state['property_queries'], 0)

    def test_missing_gpu_and_unqualified_hardware_refused(self):
        for options in ({'available': False}, {'sm': (8, 6)}):
            torch, _ = self.fake_gpu(**options)
            with self.assertRaises(ValueError): admit_gpu(torch)


if __name__ == '__main__':
    unittest.main()
