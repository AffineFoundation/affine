import copy
import unittest
from ops.harness_length_cutover import validate_cutover
from subnet.harness import normalize
from subnet import successor_calibration as calibration, forced_sampling


class LengthCutover(unittest.TestCase):
    def setUp(self):
        self.old = {'K': 4, 'L': 4, 'max_batches': 3, 'environments': [
            {'spec': {'id': 'affine_math', 'max_output_tokens': 2048},
             'harness': {'version': 'text-tools-long-v2', 'max_output_tokens': 1024}}]}
        self.new = copy.deepcopy(self.old)
        self.new['environments'][0]['harness']['max_output_tokens'] = 2048
        self.grant = dict(version='signed-harness-output-budget-v1',
                          previous_config_sha256='old', new_config_sha256='new',
                          minimum_round=72, old_max_output_tokens=1024,
                          new_max_output_tokens=2048, env_id='affine_math')

    def validate(self):
        return validate_cutover(self.new, self.old, self.grant,
                                previous_config_sha256='old', new_config_sha256='new')

    def test_only_output_budget_changes(self):
        self.validate()
        self.new['K'] = 2
        with self.assertRaises(ValueError):
            self.validate()

    def test_other_harness_changes_rejected(self):
        self.new['environments'][0]['harness']['temperature'] = 1
        with self.assertRaises(ValueError):
            self.validate()

    def test_unqualified_length_rejected(self):
        self.grant['new_max_output_tokens'] = 4096
        with self.assertRaises(ValueError):
            self.validate()

    def test_long_contract_supported_by_calibration_and_harness(self):
        harness = normalize(self.new['environments'][0]['harness'])
        req = calibration.request(dict(version=calibration.MINER_CALIBRATION_VERSION,
            env_id='affine_math', harness=harness, task_indices=[0, 1],
            max_tokens=2048, miner='a'*64,
            draw_contract=forced_sampling.new_contract(dict(
                version=forced_sampling.MINER_VERSION, max_attempts=1000,
                support_adjudication='exact-cached-replay-v1',
                calibration=dict(version='cached-prefill-calibration-v1', checkpoint='b'*64,
                    model_runtime_revision='test', backend_profile_sha256='c'*64,
                    harness_sha256='d'*64, report_sha256='e'*64,
                    cdf_abs_error=1e-5, logprob_atol=1e-5, toploc_exp_mismatches=0,
                    toploc_mant_err_mean=0, toploc_mant_err_median=0)))))
        self.assertEqual(req['max_tokens'], 2048)


if __name__ == '__main__':
    unittest.main()
