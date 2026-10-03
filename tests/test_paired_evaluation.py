import copy,unittest
from subnet.paired_evaluation import summarize

class PairedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cohort={'indices':[1,2,3,4],'seeds':[101,102,103,104]}
        def rows(successes):
            return [dict(env_id='affine_math',index=i,seed=s,task_hash=f'{i:064x}',
                verified=True,classification='positive' if i in successes else 'negative',
                reward=1 if i in successes else 0) for i,s in zip(self.cohort['indices'],self.cohort['seeds'])]
        self.before=rows({1});self.after=rows({1,2,3})
    def test_counts_pairing_and_exact_probability(self):
        result=summarize(self.cohort,self.before,list(reversed(self.after)))
        self.assertEqual((result['baseline_correct'],result['learned_correct']),(1,3))
        self.assertEqual((result['paired_gains'],result['paired_losses']),(2,0))
        self.assertEqual(result['paired_exact_two_sided_p'],.5)
        self.assertEqual(result['accuracy_change'],.5)
        self.assertFalse(result['execution_authenticated_here'])
        self.assertFalse(result['stable_long_term_improvement_proven'])
    def test_unchanged_population_is_not_gain(self):
        result=summarize(self.cohort,self.before,self.before)
        self.assertEqual(result['paired_exact_two_sided_p'],1)
        self.assertEqual(result['accuracy_change'],0)
    def test_missing_or_duplicate_tasks_cannot_shrink_cohort(self):
        for rows in [self.after[:-1],self.after[:-1]+[self.after[0]]]:
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                summarize(self.cohort,self.before,rows)
    def test_seed_task_environment_or_verification_mutations_refused(self):
        for key,value in [('seed',1000),('task_hash','a'*64),('env_id','other'),('verified',False)]:
            changed=copy.deepcopy(self.after);changed[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                summarize(self.cohort,self.before,changed)
    def test_grader_error_or_conflicting_reward_is_not_failure_sample(self):
        for key,value in [('error_type','TimeoutError'),('reward',.5),('reward',True),('reward',float('nan')),('classification','neutral')]:
            changed=copy.deepcopy(self.after);changed[0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                summarize(self.cohort,self.before,changed)
    def test_cohort_identity_not_boolean_or_repeated(self):
        for indices in [[1,1,3,4],[True,2,3,4]]:
            with self.subTest(indices=indices),self.assertRaises(ValueError):
                summarize(dict(self.cohort,indices=indices),self.before,self.after)

if __name__=='__main__':unittest.main()
