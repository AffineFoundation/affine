import copy,unittest
from ops.continuous_owned_cached_evaluator import config_admission,FIXED32_INDICES
from subnet.owned_cached_evaluation import POLICY

class RestartEvaluator(unittest.TestCase):
    def setUp(self):
        source='4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373'
        cp='6493a901bd009f0800d5eed97d19aee78947cc586afeb27abfb2c72032ad1924'
        self.c=dict(version='continuous-owned-cached-base-restart-1024-v4',dispatch_allowed=True,owned_evaluation_policy=POLICY,state='/isolated/fresh',production_state='/existing/production',source_sha256=source,source_bundle={'sha256':source},heldout=[dict(indices=list(FIXED32_INDICES),seed=20261002,harness=dict(version='text-tools-long-kv-v3',policy='autoregressive',max_output_tokens=1024,temperature=.7,top_p=1.))],evaluation_token_cap=1024,stop_after_pair=False,evaluation_job_ttl_seconds=1800,before_optimizer_steps=0,completed_history=None,evaluation_experiment_id='owned-cached-native-fixed32-cap1024-v1',evaluation_mode='independent-checkpoints-v1',legacy_evaluator_scheduler_must_remain_stopped=True,run_id='completed-math-base7b-restart-20261008-v1',before_checkpoint=cp,base_checkpoint_descriptor={'id':cp})
    def test_fresh_baseline_zero_preserves_fixed_cohort(self):
        self.assertEqual(config_admission(self.c),POLICY)
    def test_old_history_or_nonzero_genesis_or_changed_base_refused(self):
        for field,value in [('before_optimizer_steps',10),('completed_history',{'state':'old'}),('before_checkpoint','a'*64)]:
            c=copy.deepcopy(self.c);c[field]=value
            with self.assertRaises(ValueError):config_admission(c)
    def test_cannot_mix_harness_or_population(self):
        for field,value in [('seed',99),('indices',list(range(32))),('harness',dict(version='text-tools-long-kv-v3',policy='autoregressive',max_output_tokens=2048,temperature=.7,top_p=1.))]:
            c=copy.deepcopy(self.c);c['heldout'][0][field]=value
            with self.assertRaises(ValueError):config_admission(c)
