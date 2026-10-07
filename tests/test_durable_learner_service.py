import copy
import json
from pathlib import Path
import unittest
import test_durable_audit_services as audit_tests
from ops import durable_learner_service as m
from ops import durable_audit_services as g

class LearnerRecovery(unittest.TestCase):
    def setUp(self):
        self.fixture = audit_tests.DurableServices('test_real_sqlite_restart_and_no_deadline')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        for i in range(174):
            (f.runtime / 'subnet' / f'file_{i}.py').write_text('# scientific pin\n')
        inventory = {str(p.relative_to(f.runtime)): g.file_hash(p) for p in f.runtime.rglob('*.py')}
        evidence = f.root / 'source-evidence.json'
        evidence.write_text('{}')
        source = {'version': 'ordinary-orchestration-only-source-approval-v1', 'approved': True,
                  'source_sha256': f.source, 'optimizer_reset': False, 'historical_relabel': False,
                  'full_source_files': inventory, 'runtime_source_files': inventory,
                  'evidence': {'readback': {'path': str(evidence), 'file_sha256': g.file_hash(evidence)}}}
        source_row = f.document('source-approval.json', source)
        translation = f.root / 'translation.json'; translation.write_text('{}')
        qualified = {'version': 'ordinary-orchestration-only-training-qualification-approval-v1',
                     'approved': True, 'candidate_source_sha256': f.source,
                     'translation_path': str(translation), 'translation_file_sha256': g.file_hash(translation)}
        qual_row = f.document('qualification.json', qualified)
        reward_row = f.document('activation.json', {'approved_sources': [f.source]})
        cfg = {'state': str(f.state), 'source_bundle': {'sha256': f.source},
               'persistent_training_admission': {'source_sha256': f.source, 'gpu_qualification_sha256': g.file_hash(translation)},
               'persistent_training_qualification_translation': {'path': str(translation), 'sha256': g.file_hash(translation), 'approval_path': qual_row['path'], 'approval_sha256': qual_row['file_sha256']},
               'preparation_only': False, 'activation_allowed': True, 'activation_approved': True,
               'deployment_gate': {'execution_allowed': True}, 'training_input_policy': 'committed-unaudited-training-v1',
               'continuous_reward_activation_document': g.read(reward_row['path'])}
        f.config.write_text(json.dumps(cfg))
        self.status = f.state / 'controller.json'
        self.status.write_text(json.dumps({'initial_published': True, 'persistent_state_committed': True, 'trainer_state': {'optimizer_steps': 19}, 'active': {'phase': 'train', 'original_job': 'original-29'}}))
        self.p = {'version': m.VERSION, 'execute_allowed': True, 'authority': f.auth, 'identity': f.p['identity'],
                  'config': {'path': str(f.config), 'file_sha256': g.file_hash(f.config)},
                  'source_root': str(f.runtime), 'source_sha256': f.source, 'source_approval': source_row,
                  'qualification_approval': qual_row, 'reward_activation': reward_row,
                  'authority_seed': f.p['authority_seed'], 'singleton_lock': str(f.root / 'learner.lock'),
                  'excluded_units': ['old-learner.service'], 'runner_file_sha256': g.file_hash(Path(m.__file__).resolve()),
                  'guards_file_sha256': g.file_hash(Path(g.__file__).resolve())}

    def validate(self, p=None):
        return m.validate_policy(self.fixture.sign(p or self.p), self.fixture.auth)

    def test_restart_preserves_original_and_allows_epoch_advance(self):
        before = self.status.read_bytes()
        self.validate(); self.validate()
        self.assertEqual(self.status.read_bytes(), before)
        s = g.read(self.status); s['trainer_state']['optimizer_steps'] = 20; s['active'] = {'phase': 'opening', 'original_job': 'new-30'}
        self.status.write_text(json.dumps(s)); self.validate()
        self.assertEqual(g.read(self.status)['trainer_state']['optimizer_steps'], 20)

    def test_capture_recovery_installed_window_expiry_restart_is_read_only(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        import test_late_capture_recovery as fixtures
        from subnet import late_capture_recovery as recovery
        helper=fixtures.LateRecovery();gateway,key,first,authorization=helper.fixture()
        with tempfile.TemporaryDirectory()as td,patch.object(recovery,'AUTHORITY',key.id):
            root=Path(td);state=gateway.epochs['e'];recovery.attach(gateway,'e',authorization,first,at=31)
            state['miners']=sorted(state['miners'])
            (root/'gateway.json').write_bytes(g.canonical({'epochs':{'e':state}}))
            (root/'controller.json').write_bytes(g.canonical({'active':{'epoch':'next','phase':'opening'}}))
            (root/'config.json').write_bytes(g.canonical({'state':str(root)}))
            (root/'authorization.json').write_bytes(g.canonical(authorization));(root/'first.json').write_bytes(g.canonical(first))
            row={'epoch':'e','authorization':{'path':str(root/'authorization.json')},'first_signed_manifest':{'path':str(root/'first.json')}}
            policy={'capture_recovery':row,'config':{'path':str(root/'config.json')}}
            writes=[]
            class FakeGateway:
                def __init__(self):self.epochs=g.read(root/'gateway.json')['epochs']
                def persist(self):writes.append(1)
            service=SimpleNamespace(Gateway=FakeGateway);before=(root/'gateway.json').read_bytes()
            with patch('time.time',return_value=1000):
                m.install_capture_recovery(service,policy)
                actual=service.Gateway()
            self.assertEqual(writes,[]);self.assertEqual((root/'gateway.json').read_bytes(),before)
            self.assertEqual(recovery.cutoff(actual.epochs['e'],'e',at=1000),25)
            self.assertEqual(actual.epochs['e']['commitment_binding']['freeze_until'],25)

    def test_capture_recovery_cpu_check_rejects_typed_window_even_with_valid_signature(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        import test_late_capture_recovery as fixtures
        from subnet import late_capture_recovery as recovery
        helper=fixtures.LateRecovery();gateway,key,first,authorization=helper.fixture()
        authorization=helper.sign(key,dict(authorization['payload'],operational_until=99999))
        with tempfile.TemporaryDirectory()as td,patch.object(recovery,'AUTHORITY',key.id):
            root=Path(td);state=gateway.epochs['e'];state['miners']=sorted(state['miners'])
            (root/'gateway.json').write_bytes(g.canonical({'epochs':{'e':state}}));(root/'config.json').write_bytes(g.canonical({'state':str(root)}))
            (root/'authorization.json').write_bytes(g.canonical(authorization));(root/'first.json').write_bytes(g.canonical(first))
            policy={'capture_recovery':{'epoch':'e','authorization':{'path':str(root/'authorization.json')},'first_signed_manifest':{'path':str(root/'first.json')}},'config':{'path':str(root/'config.json')}}
            service=SimpleNamespace(Gateway=lambda:None)
            with self.assertRaisesRegex(ValueError,'bounded late capture'):
                m.install_capture_recovery(service,policy)

    def test_no_fresh_initialization_or_optimizer_reset(self):
        for key in ('initial_published', 'persistent_state_committed', 'trainer_state'):
            s = g.read(self.status); old = s[key]; s[key] = False
            self.status.write_text(json.dumps(s))
            with self.assertRaises(ValueError): self.validate()
            s[key] = old; self.status.write_text(json.dumps(s))

    def test_runtime_approval_drift_and_injection(self):
        (self.fixture.runtime / 'subnet' / 'extra.py').write_text('# injected')
        with self.assertRaises(ValueError): self.validate()
        (self.fixture.runtime / 'subnet' / 'extra.py').unlink()
        p = copy.deepcopy(self.p); p['guards_file_sha256'] = '0'*64
        with self.assertRaises(ValueError): self.validate(p)

    def test_config_and_qualification_cannot_change(self):
        cfg = g.read(self.fixture.config); cfg['persistent_training_qualification_translation']['sha256'] = '0'*64
        self.fixture.config.write_text(json.dumps(cfg))
        with self.assertRaises(ValueError): self.validate()
        p = copy.deepcopy(self.p); p['config']['file_sha256'] = g.file_hash(self.fixture.config)
        with self.assertRaises(ValueError): self.validate(p)

    def test_historical_hex_approval_preserves_original_bytes(self):
        row = self.p['source_approval']
        document = g.read(row['path'])
        import base64
        document['signature'] = base64.b64decode(document['signature']).hex()
        Path(row['path']).write_text(json.dumps(document))
        row['file_sha256'] = g.file_hash(row['path'])
        with self.assertRaises(Exception): self.validate()
        row['signature_encoding'] = 'hex'
        before = Path(row['path']).read_bytes()
        self.validate()
        self.assertEqual(Path(row['path']).read_bytes(), before)

    def test_signed_policy_cannot_add_launch_expiry_or_disable(self):
        for change in ({'expires_at': 1}, {'execute_allowed': False}):
            p = copy.deepcopy(self.p); p.update(change)
            with self.assertRaises(ValueError): self.validate(p)

if __name__ == '__main__': unittest.main()
