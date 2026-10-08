import copy, unittest
from nacl.signing import SigningKey
from subnet.learner_selection_operator_bridge import approval, FILES
from subnet.math_completion import FIELD, VERSION, ENVIRONMENT_VERSION
from training_receipt_fixtures import sign

class CompletedPeer(unittest.TestCase):
    def setUp(self):
        self.key = SigningKey.generate(); self.authority = self.key.verify_key.encode().hex()
        names = ['subnet/batch_quotas.py', 'subnet/sampling_uniqueness.py', 'subnet/trajectory_identity.py', 'subnet/trainer_local_state.py', 'subnet/math_completion.py']
        names += ['subnet/module%d.py' % i for i in range(177)]
        self.manifest = dict(epoch='fresh-1', source_bundle={'sha256':'a'*64}, sampling_contract={'version':'forced-inverse-cdf-prefill-miner-bound-v5','max_attempts':1000}, K=4, L=4, samples_per_batch=8, max_batches=3, optimizer_state_export_policy='trainer-local-only-v1', environments=[{'spec':dict(id='affine_math', adapter='prime_v1', version=ENVIRONMENT_VERSION, max_turns=1, config={FIELD:VERSION})}])
        self.p = dict(version='cpu-selection-peer-completed-math-local-trainer-authorization-v4', source_sha256='a'*64, scientific_source_files={n:'b'*64 for n in names}, operator_files={n:'c'*64 for n in FILES}, minimum_round=78, epoch_prefix='fresh-', peer_entry_sha256='d'*64, peer_runner_sha256='e'*64, backend_execution_allowed=True)
    def check(self, p=None, m=None):
        return approval(sign(self.key,p or self.p), self.authority, m or self.manifest)
    def test_completed_math_requires_exact_runtime_and_local_genesis(self):
        self.assertEqual(self.check()['version'], self.p['version'])
        for field in ('subnet/math_completion.py', 'subnet/trainer_local_state.py'):
            p=copy.deepcopy(self.p);p['scientific_source_files'].pop(field)
            with self.assertRaises(ValueError):self.check(p)
    def test_unmarked_task_or_other_state_policy_cannot_use_completed_grant(self):
        for kind in ('marker','state','version'):
            m=copy.deepcopy(self.manifest)
            if kind=='marker':m['environments'][0]['spec']['config']={}
            if kind=='state':m['optimizer_state_export_policy']='upload-only-independent-full-v1'
            if kind=='version':m['environments'][0]['spec']['version']='prime-v1-1'
            with self.assertRaises(ValueError):self.check(m=m)
    def test_changed_quota_not_implicitly_allowed(self):
        m=copy.deepcopy(self.manifest);m['K']=2
        with self.assertRaises(ValueError):self.check(m=m)
