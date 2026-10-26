import unittest
from dashboard.run_projection import project


class RunProjectionTests(unittest.TestCase):
    def test_new_run_does_not_include_old_results_or_modify_history(self):
        old = dict(id='nonpayable-live-reward-math-v1--123-76', source='live-reward-math', start=100)
        new = dict(id='nonpayable-live-reward-math-v1--456-78', source='live-reward-math', start=200)
        forged = dict(id='nonpayable-live-reward-math-v1--456-79', source='live-reward-math', start=99)
        epochs = [old, new, forged]
        results = [dict(epoch_id=old['id'], timestamp=250), dict(epoch_id=new['id'], timestamp=210),
                   dict(epoch_id=new['id'], timestamp=99)]
        selected, evaluations = project(epochs, results, dict(first_round=78, started_at=150, run_id='fresh'))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['display_epoch'], 1)
        self.assertEqual(len(evaluations), 1)
        self.assertNotIn('display_epoch', new)
        self.assertEqual(len(epochs), 3)

    def test_absent_boundary_keeps_existing_behavior(self):
        epochs, evaluations = [dict(id='old')], []
        self.assertEqual(project(epochs, evaluations, None), (epochs, evaluations))
