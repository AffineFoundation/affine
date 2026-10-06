import copy
import hashlib
import unittest
from pathlib import Path

from ops.continuous_owned_heldout128 import digest
from ops.continuous_owned_heldout128_factory import prepare_packet
import test_owned_cached_group_operator as fixture


class FactoryTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture.OperatorTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.policy=dict(template_original_job=self.f.jobs[0],source_files=self.f.files,
            runtime_versions=self.f.scope['runtime_versions'],cohort_sha256=digest(self.f.groups),groups=self.f.groups,
            endpoint=dict(host='approved',port=22,workspace=str(self.f.root),python='/approved/python',known_hosts='/approved/knownhosts'),
            remote_root=str(self.f.root/'groups'),remote_source_path=str(self.f.root/'frozen-source'),
            group_lifetime_seconds=1000,expires_at=10000,group_scope_template=self.f.scope)
        required=('ops/owned_cached_group_operator.py','ops/owned_cached_group_retention.py','ops/owned_cached_larger_cohort.py','ops/owned_cached_group_frozen_cache.py')
        base=Path(__file__).resolve().parents[1]
        self.policy['cpu_dependencies']={n:dict(path=str(base/n),sha256=hashlib.sha256((base/n).read_bytes()).hexdigest())for n in required}
        self.policy['group_scope_template']['operator_dependency_pins']={n:v['sha256']for n,v in self.policy['cpu_dependencies'].items()}
        self.policy['group_scope_template']['operator_file_sha256']=self.policy['group_scope_template']['operator_dependency_pins'][required[0]]
        self.publication=dict(checkpoint_descriptor=self.f.sign(self.f.cp),optimizer_publication=self.f.sign({'descriptor':{'optimizer_steps':14}}))
    def prepare(self,refresh=lambda m,t:m):
        identity=digest([digest(self.policy),self.f.cp['id'],self.policy['cohort_sha256']])
        return prepare_packet(self.policy,self.publication,identity,self.f.authority,self.f.sign,now=10,refresh_manifest=refresh)
    def test_four_signed_separate_originals_same_checkpoint_and_cohort(self):
        packet=self.prepare();jobs=[j['payload']for j in packet['original_jobs']]
        self.assertEqual(len({j['job_id']for j in jobs}),4)
        self.assertEqual([j['heldout'][0]['indices']for j in jobs],[g['indices']for g in self.f.groups])
        self.assertTrue(all(j['manifest']['payload']['checkpoint']==self.f.cp for j in jobs))
        self.assertEqual(packet['expires_at'],1010)
        scope=packet['scope']['payload'];self.assertEqual(scope['endpoint']['workspace'],packet['workspace'])
        self.assertEqual(scope['endpoint']['code'],scope['source_path'])
    def test_source_runtime_checkpoint_cache_or_other_mode_reject(self):
        for field,value in [('checkpoint_cache',{'foreign':'cache'}),('trusted_evaluation_policy',{}),('successor_calibration',{})]:
            old=self.policy['template_original_job'];job=copy.deepcopy(old['payload']);job[field]=value
            self.policy['template_original_job']=self.f.sign(job)
            with self.assertRaises(ValueError):self.prepare()
            self.policy['template_original_job']=old
    def test_url_refresh_cannot_change_any_scientific_setting(self):
        def bad(m,t):m['harness_source_hash']='changed-not-in-short-whitelist';return m
        with self.assertRaises(ValueError):self.prepare(bad)
    def test_url_only_refresh_preserves_signed_bytes_identity(self):
        def refresh(m,t):
            m['checkpoint']['read_urls']={n:'private-transport'for n in m['checkpoint']['files']}
            m['source_bundle']['read_url']='private-source-transport';return m
        packet=self.prepare(refresh)
        self.assertEqual(packet['scope']['payload']['checkpoint']['id'],self.f.cp['id'])
    def test_fixed_identity_refuses_new_random_group_label(self):
        with self.assertRaises(ValueError):
            prepare_packet(self.policy,self.publication,'a'*64,self.f.authority,self.f.sign,now=10,refresh_manifest=lambda m,t:m)


    def test_bad_dependency_names_and_hashes_rejected_before_signer(self):
        self.policy['group_scope_template']['operator_dependency_pins']={'wrong': 'a'*64}
        with self.assertRaises(ValueError):self.prepare()


if __name__=='__main__':unittest.main()
