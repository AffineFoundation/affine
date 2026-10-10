"""Synthetic signed CPU peer admission; no scientific execution or production keys."""
import copy
import unittest
from nacl.signing import SigningKey
from subnet import learner_selection_operator_bridge as bridge
from training_receipt_fixtures import sign

class PeerCapacity(unittest.TestCase):
    def fixture(self, implementation=bridge, effective_lr=False):
        from subnet.math_completion import FIELD, VERSION, ENVIRONMENT_VERSION
        key=SigningKey.generate();authority=key.verify_key.encode().hex()
        names=['subnet/batch_quotas.py','subnet/sampling_uniqueness.py','subnet/trajectory_identity.py','subnet/trainer_local_state.py','subnet/math_completion.py']
        if effective_lr:names+=['subnet/fp32_gradient_accumulation.py','subnet/unaudited_training_execution.py','subnet/learning_rate_transition.py']
        names+=['subnet/module%d.py'%i for i in range(177)]
        manifest=dict(epoch='future-1',source_bundle={'sha256':'a'*64},sampling_contract={'version':'forced-inverse-cdf-prefill-miner-bound-v5','max_attempts':1000},K=4,L=4,samples_per_batch=8,max_batches=3,optimizer_state_export_policy='trainer-local-only-v1',environments=[{'spec':dict(id='affine_math',adapter='prime_v1',version=ENVIRONMENT_VERSION,max_turns=1,config={FIELD:VERSION})}])
        policy=dict(version=implementation.GENESIS_LR_AUTH_VERSION if effective_lr else 'cpu-selection-peer-completed-math-local-trainer-authorization-v4',source_sha256='a'*64,scientific_source_files={n:'b'*64 for n in names},operator_files={n:'c'*64 for n in implementation.FILES},minimum_round=78,epoch_prefix='future-',peer_entry_sha256='d'*64,peer_runner_sha256='e'*64,backend_execution_allowed=True)
        return key,authority,manifest,policy

    def check_variant(self,implementation,effective_lr=False):
        key,authority,m,p=self.fixture(implementation,effective_lr)
        document=sign(key,p)
        for cap in (3,9):
            with self.subTest(cap=cap):self.assertEqual(implementation.approval(document,authority,dict(m,max_batches=cap)),p)
        for change in ({'max_batches':True},{'max_batches':0},{'max_batches':257},{'K':2},{'samples_per_batch':4},{'source_bundle':{'sha256':'f'*64}},{'epoch':'old-1'}):
            with self.subTest(change=change),self.assertRaises(ValueError):implementation.approval(document,authority,dict(m,**change))
        wrong=copy.deepcopy(document);wrong['payload']['minimum_round']=0
        with self.assertRaises(Exception):implementation.approval(wrong,authority,m)
        for missing in ('subnet/batch_quotas.py','subnet/math_completion.py','subnet/trainer_local_state.py'):
            broken=copy.deepcopy(p);broken['scientific_source_files'].pop(missing)
            with self.assertRaises(ValueError):implementation.approval(sign(key,broken),authority,m)

    def test_completed_peer_caps_retain_auth_and_completion_guards(self):self.check_variant(bridge)

    def test_historical179_peer_is_still_only_three_batches(self):
        key,authority,m,p=self.fixture();m=dict(m,K=2,L=2,samples_per_batch=4)
        p=copy.deepcopy(p);p['version']=bridge.K2L2_AUTH_VERSION
        for field in ('subnet/batch_quotas.py','subnet/math_completion.py','subnet/trainer_local_state.py'):p['scientific_source_files'].pop(field)
        doc=sign(key,p);self.assertEqual(bridge.approval(doc,authority,m),p)
        with self.assertRaises(ValueError):bridge.approval(doc,authority,dict(m,max_batches=6))

if __name__=='__main__':unittest.main()
