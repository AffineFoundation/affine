import copy
import unittest
from ops.training_optimization_analysis import paired_comparison


def evaluation(rewards, checkpoint):
    return dict(records=[dict(index=i, seed=i+100, task_hash='task-'+str(i), prompt_sha256='prompt-'+str(i), cohort_sha256='cohort', protocol='same', reward=r, classification='positive' if r else 'negative', native_graded=True, TOPLOC_claimed=False, checkpoint=checkpoint) for i,r in enumerate(rewards)], tasks=len(rewards), checkpoint=checkpoint, repeat_controls_passed=True, actual_dtype='torch.bfloat16', actual_runtime_revision='fixed', batch_size=8)


class AnalysisTests(unittest.TestCase):
    def test_same_tasks_paired_not_difference_of_unrelated_averages(self):
        result=paired_comparison(evaluation([1,0,0,1],'a'),evaluation([1,1,1,1],'b'),resamples=1000)
        self.assertEqual(result['gained'],2)
        self.assertEqual(result['lost'],0)
        self.assertEqual(result['improvement_percentage_points'],50)
        self.assertFalse(result['production_learning_proven'])

    def test_prompt_runtime_and_missing_tasks_rejected(self):
        a=evaluation([0,1],'a');b=evaluation([1,1],'b')
        for field in ['prompt_sha256','cohort_sha256','task_hash']:
            bad=copy.deepcopy(b);bad['records'][0][field]='changed'
            with self.assertRaises(ValueError):paired_comparison(a,bad,resamples=1000)
        bad=copy.deepcopy(b);bad['batch_size']=4
        with self.assertRaises(ValueError):paired_comparison(a,bad,resamples=1000)
        with self.assertRaises(ValueError):paired_comparison(a,evaluation([1],'b'),resamples=1000)

    def test_grader_errors_and_forged_rewards_rejected(self):
        a=evaluation([0,1],'a');b=evaluation([1,1],'b')
        bad=copy.deepcopy(b);bad['failures']=[dict(infrastructure_failure=True)]
        with self.assertRaises(ValueError):paired_comparison(a,bad,resamples=1000)
        bad=copy.deepcopy(b);bad['records'][0]['reward']=True
        with self.assertRaises(ValueError):paired_comparison(a,bad,resamples=1000)


if __name__=='__main__':unittest.main()
