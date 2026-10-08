"""One quota knob reaches signed openings and bounded native pair budgets."""
import copy
import json
import unittest
from unittest.mock import patch
from subnet.batch_quotas import configured_quotas, normalize_config
from subnet.gpu_service import contract, initial_manifest
from subnet.controller import Controller
from subnet import forced_sampling as forced
from ops.native_training_outcome_filter import MULTI_VERSION, K2L2_VERSION, validate_limits
from test_class_quota_opening import QuotaOpeningTests
from test_fast_prefill_audit import Controls


class SingleKnobTests(unittest.TestCase):
    def test_even_sizes_and_legacy_configuration_without_mutation(self):
        for count in (4,8,16,32,128):
            config={'samples_per_batch':count};before=copy.deepcopy(config)
            self.assertEqual(configured_quotas(config),(count//2,count//2))
            self.assertEqual(normalize_config(config)['K'],count//2)
            self.assertEqual(config,before)
        self.assertEqual(normalize_config({'K':1,'L':2}),{'K':1,'L':2})
        for count in (True,8.,'8',0,2,7,130):
            with self.subTest(count=count),self.assertRaises(ValueError):configured_quotas({'samples_per_batch':count})
        for K in (2,True):
            with self.assertRaises(ValueError):configured_quotas({'samples_per_batch':8,'K':K})

    def test_one_number_reaches_first_signed_manifest_and_initial_state(self):
        for count in (8,16,128):
            case=QuotaOpeningTests();case.setUp()
            try:
                _,fixture=Controls().support_runtime()
                c=copy.deepcopy(fixture['sampling_contract'])
                c.update(version=forced.MINER_VERSION,max_attempts=1000)
                config=dict(case.config,samples_per_batch=count,max_batches=3,sampling_policy={k:v for k,v in c.items()if k not in ('randomness','verification','generation')})
                with patch('subnet.gpu_service.definitions',return_value=[case.row]):
                    selected=contract(config,0);initial=initial_manifest(config,case.checkpoint)
                self.assertEqual((initial['K'],initial['L']),(count//2,count//2))
                selected.pop('duration');selected.pop('heldout_indices');selected.pop('environments')
                manifest=case.opening(**selected)
                published=json.loads(case.bucket.objects['public/nonpayable-quota/manifest.json'])['payload']
                self.assertEqual(published,manifest)
                self.assertEqual((published['samples_per_batch'],published['K'],published['L'],published['max_batches']),(count,count//2,count//2,3))
                forced.binding(published,case.miner)
                with self.assertRaises(ValueError):forced.binding(dict(published,samples_per_batch=count+2),case.miner)
            finally:case.doCleanups()

    def test_native_budget_derives_from_manifest_not_second_configuration_number(self):
        policy=dict(version=MULTI_VERSION,max_pairs='manifest',workers=2,per_grade_seconds=2,wall_seconds=10,max_reply_bytes=1024)
        for count in (4,8,16,128):
            manifest=normalize_config({'samples_per_batch':count})
            self.assertEqual(validate_limits(policy,manifest=manifest)['max_pairs'],256*count//2)
        self.assertEqual(policy['max_pairs'],'manifest')
        with self.assertRaises(ValueError):validate_limits(policy)
        with self.assertRaises(ValueError):validate_limits(dict(policy,version=K2L2_VERSION),manifest={'K':2,'L':2})
        with self.assertRaises(ValueError):validate_limits(policy,manifest={'samples_per_batch':8,'K':2,'L':2})
