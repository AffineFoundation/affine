"""Coverage integrity tests; no GPU/production qualification claim."""
import copy
import unittest
from .probe import exact_probe


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.indices=[1,3,9]
        self.result=dict(checkpoint_id='fixture',partial=False,samples=24,planned_samples=24,
            envs=dict(affine_gatepool=dict(samples=24,tasks=3,
                per_task={str(i):dict(n=8,L=[1]*8) for i in self.indices})))

    def check(self):return exact_probe(self.result,self.indices,'fixture')

    def test_complete_plan(self):
        self.assertEqual(self.check()['samples'],24)
        self.assertNotIn('indices',self.check())

    def test_counter_only_forgery(self):
        del self.result['envs']['affine_gatepool']['per_task']['9']
        with self.assertRaises(ValueError):self.check()

    def test_wrong_draw_same_total(self):
        c=self.result['envs']['affine_gatepool']['per_task'];c['10']=c.pop('9')
        with self.assertRaises(ValueError):self.check()

    def test_duplicate_alias_not_another_draw(self):
        c=self.result['envs']['affine_gatepool']['per_task'];c['01']=c.pop('1')
        with self.assertRaises(ValueError):self.check()

    def test_uneven_counts_cannot_cancel(self):
        c=self.result['envs']['affine_gatepool']['per_task'];c['1']['n']=7;c['3']['n']=9
        with self.assertRaises(ValueError):self.check()

    def test_missing_rollout_lengths(self):
        self.result['envs']['affine_gatepool']['per_task']['1']['L'].pop()
        with self.assertRaises(ValueError):self.check()

    def test_wrong_checkpoint(self):
        self.result['checkpoint_id']='different'
        with self.assertRaises(ValueError):self.check()

    def test_partial_even_if_all_counters_match(self):
        self.result['partial']=True
        with self.assertRaises(ValueError):self.check()

    def test_environment_summary(self):
        self.result['envs']['affine_gatepool']['tasks']=2
        with self.assertRaises(ValueError):self.check()

    def test_duplicate_plan(self):
        self.indices=[1,1,9]
        with self.assertRaises(ValueError):self.check()

    def test_boolean_is_not_sample_count(self):
        self.result['envs']['affine_gatepool']['per_task']['1']['n']=True
        with self.assertRaises(ValueError):self.check()

    def test_probe_errors_are_left_to_native_scorer(self):
        self.result['envs']['affine_gatepool']['per_task']['1']['x']=8
        self.assertTrue(self.check()['complete'])


if __name__=='__main__':unittest.main()
