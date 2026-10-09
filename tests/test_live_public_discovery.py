import base64
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nacl.signing import SigningKey
from ops import publish_live_public_discovery as discovery
from subnet.storage import canonical


class PublicDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.key = SigningKey.generate()
        self.authority = self.key.verify_key.encode().hex()
        self.contract = dict(version='forced-inverse-cdf-prefill-miner-bound-v5', max_attempts=1000,
                             generation='cached-eager-inverse-cdf', verification='prefill-cdf-calibrated',
                             randomness='1'*64, support_adjudication='exact-cached-replay-v1')
        self.env = dict(spec=dict(id='affine_math', version='prime-v1-2-completed-math'),
                        harness=dict(max_output_tokens=2048))
        self.config = dict(source_bundle=dict(sha256='a'*64), K=4, L=4, max_batches=3,
                           sampling_policy={key: self.contract[key] for key in ('version', 'max_attempts', 'support_adjudication')},
                           submission_transport_policy='small-commitment-pairs-v2',
                           probability_artifact_policy=dict(version='selected-token-logprobs-v1'),
                           environments=[self.env])
        self.controller = dict(active=dict(epoch='epoch86', phase='mine'),
                               checkpoint=dict(id='b'*64, files={'model':'c'*64}))
        self.manifest = dict(epoch='epoch86', start=10, deadline=30, checkpoint=self.controller['checkpoint'],
                             source_bundle=self.config['source_bundle'], K=4, L=4, max_batches=3,
                             sampling_contract=self.contract, submission_transport_policy=self.config['submission_transport_policy'],
                             probability_artifact_policy=self.config['probability_artifact_policy'],
                             capabilities={'secret-upload-location':'never-project'},
                             environments=[dict(self.env, env_id='affine_math', indices=[1, 2])])
        self.pointer = dict(authority=self.authority, expires_at=40,
                            current_url='https://test.r2.cloudflarestorage.com/b/current?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=signed')

    def envelope(self, payload):
        return dict(payload=payload, signer=self.authority,
                    signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())

    def project(self, **kwargs):
        return discovery.project(self.config, self.controller, self.pointer, self.envelope(self.manifest),
                                 self.authority, live=kwargs.get('live', True), now=kwargs.get('now', 20))

    def test_exact_contract_with_completed_grader_and_2048_budget(self):
        result = self.project()
        self.assertTrue(result['accepting_submissions'])
        self.assertEqual(result['sampling_contract'], self.contract)
        self.assertEqual(result['max_output_tokens'], 2048)
        self.assertEqual(result['samples_per_batch'], 8)
        self.assertEqual(result['nonce_max'], 999)
        self.assertNotIn('capabilities', result)
        self.assertFalse(result['chain_weight_submission'])

    def test_missing_contract_never_gets_invented_v5_default(self):
        del self.manifest['sampling_contract']
        with self.assertRaisesRegex(ValueError, 'explicit signed sampling'):
            self.project()

    def test_wrong_epoch_source_checkpoint_quota_or_harness_rejected(self):
        original = copy.deepcopy(self.manifest)
        for field, value in [('epoch', 'old'), ('source_bundle', {'sha256':'d'*64}),
                             ('checkpoint', {'id':'e'*64, 'files':{}}), ('K', 2),
                             ('submission_transport_policy', 'direct-r2-v1'),
                             ('environments', [dict(self.env, env_id='affine_math', harness={'max_output_tokens':1024})])]:
            with self.subTest(field=field):
                self.manifest = copy.deepcopy(original)
                self.manifest[field] = value
                with self.assertRaises(ValueError): self.project()
        self.manifest = original

    def test_signature_and_expired_capability_fail(self):
        envelope = self.envelope(self.manifest)
        envelope['payload'] = dict(self.manifest, deadline=100)
        with self.assertRaises(Exception):
            discovery.project(self.config, self.controller, self.pointer, envelope, self.authority, live=True, now=20)
        self.pointer['expires_at'] = 20
        with self.assertRaises(ValueError): self.project()

    def test_dead_process_training_and_deadline_close(self):
        self.assertFalse(self.project(live=False)['accepting_submissions'])
        self.assertFalse(self.project(now=30)['accepting_submissions'])
        self.controller['active']['phase'] = 'train'
        result = self.project()
        self.assertFalse(result['accepting_submissions'])
        self.assertNotIn('sampling_contract', result)

    def test_failure_replaces_stale_open_document_and_returns_no_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'mining.json'
            target.write_text(json.dumps(dict(accepting_submissions=True, sampling_version='invented',
                                             current_url='secret-capability')))
            with patch.object(discovery, 'actual_selector', side_effect=ValueError('bad config')):
                result = discovery.publish(None, target, self.authority)
            saved = json.loads(target.read_text())
            self.assertFalse(saved['accepting_submissions'])
            self.assertNotIn('current_url', saved)
            self.assertNotIn('sampling_version', saved)
            self.assertNotIn('current_url', result)
            self.assertEqual(target.stat().st_mode & 0o777, 0o644)

    def test_wrapped_signed_learner_policy_and_real_lifetime_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root/'completed_math_restart_service.py'; runner.write_text('# pinned entry point\n')
            config = root/'config.json'; config.write_text(json.dumps(self.config))
            policy = root/'policy.json'
            policy.write_text(json.dumps(self.envelope(dict(execute_allowed=True, source_sha256='a'*64,
                runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
                config=dict(path=str(config), sha256=hashlib.sha256(config.read_bytes()).hexdigest())))))
            proc = root/'proc'; process = proc/'42'; process.mkdir(parents=True)
            argv = ['python', '-I', '-B', str(runner), '--policy', str(policy)]
            (process/'cmdline').write_bytes(b'\0'.join(arg.encode() for arg in argv)+b'\0')
            fields = ['S']+['0']*18+['991']+['0']*10
            (process/'stat').write_text('42 (python) '+' '.join(fields))
            inspect = lambda: f'MainPID=42\nActiveState=active\nExecStart=python -I -B {runner} --policy {policy} ; ignore\n'
            cfg, record = discovery.actual_selector(self.authority, proc=proc, inspect_unit=inspect)
            self.assertEqual(cfg, self.config)
            self.assertEqual(record['child_ticks'], '991')
            config.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'config drift'):
                discovery.actual_selector(self.authority, proc=proc, inspect_unit=inspect)


if __name__ == '__main__':
    unittest.main()
