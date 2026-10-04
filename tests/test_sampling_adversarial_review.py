"""Adversarial CPU controls for sampling and reward admission, not GPU qualification.

Attacks rebuild genuine target-model probabilities and TOPLOC proofs. This
specifically checks that valid computation evidence cannot substitute for the
prescribed sampler. Signature tests use fresh temporary test identities only.
"""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from nacl.signing import SigningKey

from subnet import forced_sampling as sampling
from subnet import live_reward_bridge as bridge
from subnet.audit_policy import InvalidSample


def load_fixture(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sampling_fixture = load_fixture('sampling_review_fixture', 'test_forced_sampling.py')
reward_fixture = load_fixture('sampling_review_reward_fixture', 'test_live_reward_bridge.py')


class TwoTurnSession:
    """Fixed authentic feedback; action text does not change the outcome label."""
    def reset(self, index, seed):
        self.turn = 0
        return {'task_hash': 'c' * 64,
                'messages': [{'role': 'user', 'content': 'task'}]}

    def step(self, action):
        self.turn += 1
        done = self.turn == 2
        return {'done': done, 'reward': float(done),
                'classification': 'positive' if done else 'neutral',
                'observations': [] if done else [{'role': 'user', 'content': 'feedback'}]}

    def close(self):
        pass


class SamplingAdversarialReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = sampling_fixture.ForcedSamplingTests()
        self.fixture.setUp()

    def runtime(self):
        runtime = self.fixture.runtime()
        runtime.spec.max_turns = 2

        def prompt(messages, tools=()):
            result = [0, 1]
            for message in messages[1:]:
                if message['role'] == 'assistant':
                    result.extend(map(int, message['content'].split()))
                else:
                    result.append(2)
            return result

        runtime.prompt = prompt
        return runtime

    def honest(self, predicate=lambda output: True):
        runtime = self.runtime()
        for attempt in range(16):
            rollout, arrays = runtime.rollout(2, attempt)
            if predicate(rollout['turns'][-1]['output']):
                self.assertTrue(self.runtime().verify(rollout, arrays))
                return runtime, rollout, arrays
        self.fail('control model did not produce the required honest trajectory')

    def fresh_computation(self, runtime, rollout, arrays, output):
        turn = rollout['turns'][-1]
        turn['output'] = list(output)
        turn['text'] = runtime.tokenizer.decode(output)
        acts, arrays[-1] = runtime.compute(turn['prompt'], output)
        turn['proofs'] = runtime.build_proofs(acts, decode_batching_size=16, topk=128)

    def assert_legacy_accepts_forced_rejects(self, rollout, arrays):
        legacy = self.runtime()
        legacy.sampling_context = None
        self.assertTrue(legacy.verify(rollout, arrays))
        with self.assertRaisesRegex(InvalidSample, 'sampling replay mismatch'):
            self.runtime().verify(rollout, arrays)

    def test_second_turn_arbitrary_tokens_with_genuine_target_model_proofs(self):
        with patch('subnet.model.create_session', return_value=TwoTurnSession()):
            runtime, rollout, arrays = self.honest()
            output = list(rollout['turns'][-1]['output'])
            output[-1] = (output[-1] + 1) % 8
            self.fresh_computation(runtime, rollout, arrays, output)
            self.assert_legacy_accepts_forced_rejects(rollout, arrays)

    def test_prefix_truncation_with_genuine_proofs_cannot_skip_required_suffix(self):
        with patch('subnet.model.create_session', return_value=TwoTurnSession()):
            runtime, rollout, arrays = self.honest(lambda output: len(output) >= 2)
            self.fresh_computation(runtime, rollout, arrays, rollout['turns'][-1]['output'][:-1])
            self.assert_legacy_accepts_forced_rejects(rollout, arrays)

    def test_extra_output_after_sampler_eos_with_genuine_proofs_is_rejected(self):
        with patch('subnet.model.create_session', return_value=TwoTurnSession()):
            runtime, rollout, arrays = self.honest(lambda output: len(output) < 4 and output[-1] == 7)
            self.fresh_computation(runtime, rollout, arrays, rollout['turns'][-1]['output'] + [0])
            self.assert_legacy_accepts_forced_rejects(rollout, arrays)

    def test_reseeding_only_second_turn_with_fresh_proofs_is_rejected(self):
        with patch('subnet.model.create_session', return_value=TwoTurnSession()):
            runtime, rollout, arrays = self.honest()
            original = rollout['turns'][-1]['output']
            chosen = next(runtime.sample_output(rollout['turns'][-1]['prompt'], attempt, [], 1, 2, 'c' * 64)
                          for attempt in range(16)
                          if runtime.sample_output(rollout['turns'][-1]['prompt'], attempt, [], 1, 2, 'c' * 64) != original)
            self.fresh_computation(runtime, rollout, arrays, chosen)
            self.assert_legacy_accepts_forced_rejects(rollout, arrays)

    def test_context_substitution_with_genuine_target_model_proofs_is_rejected(self):
        with patch('subnet.model.create_session', return_value=TwoTurnSession()):
            runtime, rollout, arrays = self.honest()
            rollout['turns'][-1]['prompt'].append(3)
            self.fresh_computation(runtime, rollout, arrays, rollout['turns'][-1]['output'])
            with self.assertRaisesRegex(InvalidSample, 'context'):
                self.runtime().verify(rollout, arrays)

    def test_checkpoint_or_epoch_receipt_substitution_is_rejected(self):
        with patch('subnet.model.create_session', return_value=TwoTurnSession()):
            _, rollout, arrays = self.honest()
            for field in ('checkpoint', 'epoch'):
                foreign = copy.deepcopy(self.runtime().sampling_context)
                foreign[field] = 'd' * 64 if field == 'checkpoint' else 'different-epoch'
                changed = copy.deepcopy(rollout)
                changed['sampling'] = sampling.receipt(foreign, rollout['seed'])
                with self.subTest(field=field), self.assertRaisesRegex(InvalidSample, 'binding'):
                    self.runtime().verify(changed, arrays)


class RewardSamplingAdversarialReviewTests(unittest.TestCase):
    def fixture(self):
        key, inputs, registrations, reports = reward_fixture.fixture()
        manifest = inputs['manifest_document']['payload']
        manifest['sampling_contract'] = sampling.new_contract({'version': sampling.VERSION, 'max_attempts': 16})
        manifest['sampling_source_hash'] = sampling.source_hash()
        context = sampling.binding(manifest)
        for report in reports.values():
            report['sampling_assurance'] = sampling.assurance(manifest)
            for batch in report['accepted']:
                for attempt, rollout in enumerate(batch['rollouts']):
                    rollout['seed'] = attempt
                    rollout['sampling'] = sampling.receipt(context, attempt)
        inputs['manifest_document'] = reward_fixture.sign(manifest, key)
        opening = inputs['opening_document']['payload']
        opening['first_manifest_sha256'] = bridge.sha(inputs['manifest_document'])
        inputs['opening_document'] = reward_fixture.sign(opening, key)
        inputs['audit_documents'] = {miner: reward_fixture.sign(report, key) for miner, report in reports.items()}
        bridge.project(**inputs)
        return key, inputs, registrations, reports

    def test_controller_signed_legacy_assurance_cannot_receive_new_sampling_reward(self):
        key, inputs, _, reports = self.fixture()
        miner = next(iter(reports))
        reports[miner]['sampling_assurance'] = sampling.assurance({'epoch': 'legacy'})
        inputs['audit_documents'][miner] = reward_fixture.sign(reports[miner], key)
        with self.assertRaisesRegex(ValueError, 'sampling assurance'):
            bridge.project(**inputs)

    def test_valid_report_assurance_cannot_mask_one_wrong_rollout_receipt(self):
        key, inputs, _, reports = self.fixture()
        miner = next(iter(reports))
        reports[miner]['accepted'][0]['rollouts'][1]['seed'] = 2
        inputs['audit_documents'][miner] = reward_fixture.sign(reports[miner], key)
        with self.assertRaisesRegex(ValueError, 'sampling receipt'):
            bridge.project(**inputs)

    def test_miner_cannot_self_sign_positive_audit_for_reward(self):
        _, inputs, _, reports = self.fixture()
        miner = next(iter(reports))
        inputs['audit_documents'][miner] = reward_fixture.sign(reports[miner], SigningKey.generate())
        with self.assertRaises(Exception):
            bridge.project(**inputs)

    def test_valid_sampling_receipts_do_not_authorize_unaudited_rewards(self):
        key, inputs, _, reports = self.fixture()
        miner = next(iter(reports))
        reports[miner]['outcomes'][0]['fully_audited'] = False
        inputs['audit_documents'][miner] = reward_fixture.sign(reports[miner], key)
        with self.assertRaisesRegex(ValueError, 'fully audited'):
            bridge.project(**inputs)


if __name__ == '__main__':
    unittest.main()
