"""Signed admission controls. Qualifications here are synthetic test records."""
from copy import deepcopy
import unittest
from nacl.exceptions import BadSignatureError

from test_mimo_public_worker import PublicWorkerTests
from subnet.mimo_release_policy import (ReleasePolicy, build_policy, policy_for_manifest,
    POLICY_KEY, CHECKPOINT_KEY, sign, digest)


class ReleasePolicyTests(unittest.TestCase):
    def setUp(self):
        self.f=PublicWorkerTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.policy=ReleasePolicy(self.f.policy,self.f.operator.id,allow_public=True)

    def test_public_requires_signed_qualified_policy_and_separate_local_opt_in(self):
        with self.assertRaisesRegex(ValueError,'booleans'):
            ReleasePolicy(self.f.policy,self.f.operator.id,allow_public='true')
        with self.assertRaisesRegex(ValueError,'locally disabled'):
            ReleasePolicy(self.f.policy,self.f.operator.id)
        m=deepcopy(self.f.m);m.pop(POLICY_KEY)
        with self.assertRaisesRegex(ValueError,'qualified release'):
            policy_for_manifest(m,self.f.operator.id,allow_public=True)
        self.policy.validate_manifest(self.f.m)

    def test_private_mode_cannot_smuggle_payable_manifest(self):
        with self.assertRaisesRegex(ValueError,'private release cannot'):
            build_policy(self.f.operator,self.f.m,task_assets=self.f.assets,enabled=True)

    def test_source_read_url_refresh_preserves_scope_and_sampling_draws(self):
        from subnet.forced_sampling import binding
        original=deepcopy(self.f.m);renewed=deepcopy(original)
        original['source_bundle']['url']='https://old-alias.invalid'
        renewed['source_bundle']['url']='https://new-alias.invalid'
        renewed['source_bundle']['read_url']=renewed['source_bundle']['url']
        self.policy.validate_manifest(original);self.policy.validate_manifest(renewed)
        a=binding(original,self.f.f.identity.id);b=binding(renewed,self.f.f.identity.id)
        self.assertEqual(a,b)
        renewed['source_bundle']['read_url']='https://conflicting.invalid'
        with self.assertRaisesRegex(ValueError,'conflicting source'):
            self.policy.validate_manifest(renewed)

    def test_missing_wrong_scope_or_failed_qualification_rejected(self):
        for mutation in ('missing','scope','failed','signature'):
            with self.subTest(mutation=mutation):
                value=deepcopy(self.f.policy['payload'])
                if mutation=='missing':value['qualifications'].pop()
                else:
                    q=value['qualifications'][0]['payload']
                    if mutation=='scope':q['scope_sha256']='0'*64
                    elif mutation=='failed':q['passed']=False
                    else:q['evidence_sha256']='1'*64
                    if mutation!='signature':value['qualifications'][0]=sign(self.f.operator,q)
                with self.assertRaises((ValueError,BadSignatureError)):
                    ReleasePolicy(sign(self.f.operator,value),self.f.operator.id,allow_public=True)

    def test_changed_source_taskset_sampler_or_runtime_needs_new_policy(self):
        mutations=[('max_batches',10),('source_bundle',{'sha256':'0'*64}),
                   ('mimo_runtime',dict(self.f.m['mimo_runtime'],enable_thinking=False)),
                   ('environments',[])]
        for field,value in mutations:
            m=deepcopy(self.f.m);m[field]=value
            with self.assertRaisesRegex(ValueError,'differs from qualified'):
                self.policy.validate_manifest(m)

    def test_checkpoint_rotation_requires_bound_successful_handover_certificate(self):
        m=deepcopy(self.f.m);old=m['checkpoint']['id']
        cp={'files':dict(m['checkpoint']['files'],**{'config.json':'8'*64})}
        cp['id']=digest(cp['files']);m['checkpoint']=cp
        m['sampling_contract']['calibration']['checkpoint']=cp['id']
        with self.assertRaisesRegex(ValueError,'successor checkpoint'):
            self.policy.validate_manifest(m)
        m[CHECKPOINT_KEY]=self.policy.authorize_checkpoint(cp,input_checkpoint=old,
            optimizer_step=1,publication_receipt_sha256='9'*64,identity=self.f.operator)
        self.policy.validate_manifest(m)
        changed=deepcopy(m);changed[CHECKPOINT_KEY]['payload']['optimizer_step']=2
        with self.assertRaises((ValueError,BadSignatureError)):self.policy.validate_manifest(changed)


if __name__=='__main__':unittest.main()
