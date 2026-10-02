import copy
import unittest
from subnet.backend_profiles import profile, resolve, for_config, HOPPER_REVISION, LEGACY_REVISION
import test_backend_jobs

class GPUProfiles(unittest.TestCase):
    def test_legacy_default_remains_exact(self):
        from subnet.backend_jobs import REVISION, BACKEND_PROFILE, NUMERICAL_POLICY
        self.assertEqual(for_config({}), (REVISION, BACKEND_PROFILE, NUMERICAL_POLICY))
        self.assertEqual(REVISION, LEGACY_REVISION)

    def test_hopper_requires_matching_signed_hardware_and_policy(self):
        revision, backend, policy = profile(HOPPER_REVISION)
        manifest = dict(model_runtime_revision=revision, backend_profile=backend, numerical_policy=policy)
        self.assertEqual(resolve(manifest), (revision,backend,policy))
        for key, value in [('sm',[8,6]), ('tf32',True), ('tf32',0), ('dtype','float32')]:
            changed = copy.deepcopy(manifest); changed['backend_profile'][key]=value
            with self.assertRaises(ValueError): resolve(changed)
        changed=copy.deepcopy(manifest);changed['numerical_policy']['logprob_atol']=1
        with self.assertRaises(ValueError):resolve(changed)
        with self.assertRaises(ValueError):for_config({'model_runtime_revision':'arbitrary-gpu'})

    def test_hopper_job_admission_does_not_accept_cross_profile_claims(self):
        fixture=test_backend_jobs.BackendJobAuthorization();fixture.setUp()
        revision, backend, policy=profile(HOPPER_REVISION)
        fixture.manifest.update(model_runtime_revision=revision, backend_profile=backend, numerical_policy=policy)
        fixture.job['manifest']=fixture.sign(fixture.manifest)
        from subnet.backend_jobs import validate
        _, actual=validate(fixture.sign(fixture.job),fixture.authority,now=50)
        self.assertEqual(actual['backend_profile']['sm'],[9,0])
        fixture.manifest['backend_profile']['sm']=[8,6]
        fixture.job['manifest']=fixture.sign(fixture.manifest)
        with self.assertRaises(ValueError):validate(fixture.sign(fixture.job),fixture.authority,now=50)
