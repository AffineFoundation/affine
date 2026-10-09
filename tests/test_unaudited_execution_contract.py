"""Synthetic signed CPU controls only; no actual qualification or dispatch."""
import copy
import sys
import unittest
from unittest.mock import patch

from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'tests')]
from nacl.signing import SigningKey
from training_receipt_fixtures import sign
from test_committed_training_inputs import LearnerAdmissionTests
from subnet import committed_training_inputs as learner
from subnet.persistent_cpu_adamw import HYPERPARAMETERS,POLICY
from subnet.persistent_training_protocol import VERSION as LINEAGE_VERSION
from subnet.training_receipts import sha
from subnet import unaudited_training_execution as amendment

class AmendmentTests(unittest.TestCase):
    def setUp(self):
        fx=LearnerAdmissionTests();fx.setUp();self.addCleanup(fx.doCleanups)
        self.key=fx.operator;self.authority=fx.authority
        self.sign=lambda value:sign(self.key,value)
        self.submissions=[fx.obj]
        m=copy.deepcopy(fx.manifest)
        params=[dict(name='weight',shape=[1],numel=1)]
        parent=dict(namespace='private/trainer-state/synthetic/original',
            descriptor_key='private/trainer-state/synthetic/original/authority-state.json',
            publication_sha256='1'*64,descriptor_sha256='2'*64,genesis_sha256='3'*64,
            optimizer_steps=2,inference_checkpoint=m['checkpoint']['id'],parameters_sha256=sha(params))
        binding=dict(version=LINEAGE_VERSION,policy=POLICY,epoch=m['epoch'],
            input_checkpoint=m['checkpoint']['id'],hyperparameters=copy.deepcopy(HYPERPARAMETERS),
            parameters=params,parameters_sha256=sha(params),source_sha256=m['source_bundle']['sha256'],
            gpu_qualification_sha256='4'*64,genesis_sha256=parent['genesis_sha256'],
            genesis=None,parent=parent,global_step_before=2)
        m['trainer_state_binding']=binding;m['training_runtime']={'version':'synthetic-original-bf16'}
        self.public=self.sign(m)
        m=learner.coverage_manifest(m,self.submissions,seed='5'*64,captured_at=21)
        context=self.sign(dict(original_signed_manifest=self.sign(m),parent_binding_sha256=sha(binding)))
        grades=self.sign(dict(context_sha256=sha(context),sampling_assurance='unaudited'))
        subset=self.sign(dict(context_sha256=sha(context),grade_receipt_sha256=sha(grades['payload']),
            accepted_submissions=self.submissions,accepted_inventory_sha256=sha(learner.receipt_inventory(self.submissions)),
            sampling_assurance='unaudited',claims_rewritten=False))
        self.native_docs=dict(context=context,grades=grades,subset=subset)
        m['native_training_eligibility_receipt']=dict(version='native-outcome-accepted-subset-v1',
            context_sha256=sha(context),grades_sha256=sha(grades),subset_sha256=sha(subset),
            authorization_sha256='6'*64,sampling_assurance='unaudited',
            proof_verification_performed=False,claims_rewritten=False,cheating_penalties=False)
        self.manifest=self.sign(m)
        before={n:'a'*64 for n in amendment.SCIENTIFIC_CHANGES if 'fp32_gradient_accumulation'not in n}
        before.update({'subnet/committed_training_inputs.py':'b'*64,'subnet/cached_sampling.py':'c'*64})
        after=dict(before,**{n:'d'*64 for n in amendment.SCIENTIFIC_CHANGES},**{'subnet/unaudited_training_execution.py':'9'*64})
        runtime=dict(torch='synthetic',transformers='synthetic',toploc='synthetic')
        q=dict(version=amendment.QUALIFICATION,method=amendment.METHOD,
            execution_source_bundle_sha256='e'*64,execution_source_files_sha256=sha(after),
            runtime_versions=runtime,training_runtime_sha256=sha(m['training_runtime']),
            actual_GPU_execution=True,passed=True,report_sha256='f'*64,
            optimizer_reset=False,objective_changed=False,hyperparameters_changed=True,effective_learning_rate=5e-7,base_hyperparameters_sha256=sha(HYPERPARAMETERS),state_version='persistent-fp32-trainer-state-v2-effective-lr')
        self.value=dict(version=amendment.VERSION,method=amendment.METHOD,epoch=m['epoch'],job_id='prospective-train',
            **amendment.preparation_scope(self.public,self.manifest,self.submissions,self.native_docs,self.authority),
            original_source_bundle_sha256=m['source_bundle']['sha256'],training_source_bundle={'sha256':'e'*64},
            original_source_files=before,execution_source_files=after,
            changed_source_files={n:h for n,h in after.items()if before.get(n)!=h},
            execution_qualification=self.sign(q),runtime_versions=runtime,
            training_runtime_sha256=sha(m['training_runtime']),training_policy=POLICY,
            training_input_policy=learner.VERSION,steps=1,trainer_binding_sha256=sha(binding),
            parent_descriptor_sha256=parent['descriptor_sha256'],genesis_sha256=binding['genesis_sha256'],
            optimizer_step_before=2,created_at=22,expires_at=120,execution_release_sha256='8'*64,effective_learning_rate=5e-7)
        self.value['learning_rate_authorization']=self.sign(amendment._grant_payload(self.value,m))
        self.job=dict(role='train',job_id='prospective-train',manifest=self.manifest,
            source_files=after,runtime_versions=runtime,training_policy=POLICY,
            training_input_policy=learner.VERSION,steps=1,submissions=self.submissions,
            created_at=23,expires_at=100)
    def envelope(self):
        return self.sign(dict(self.job,**{amendment.FIELD:self.sign(self.value)}))
    def run_check(self):return amendment.validate(self.envelope(),self.authority,now=30)
    def test_correct_unaudited_execution_changes_no_manifest(self):
        before=copy.deepcopy(self.job['manifest'])
        with patch('subnet.batches.unpack',side_effect=AssertionError('audit barrier')):
            result=self.run_check()
        self.assertEqual(result['method'],amendment.METHOD)
        self.assertEqual(self.job['manifest'],before)
        self.assertNotIn('verifier_receipt',self.job['submissions'][0])
        self.assertEqual(amendment.execution_bundle(self.envelope(),self.authority),{'sha256':'e'*64})
    def test_new_bundle_without_signature_cannot_route(self):
        env=self.envelope();env['payload'][amendment.FIELD]['payload']['training_source_bundle']['sha256']='0'*64
        with self.assertRaises(ValueError):amendment.execution_bundle(env,self.authority)
    def test_foreign_authority_refused(self):
        env=sign(SigningKey.generate(),self.envelope()['payload'])
        with self.assertRaises(ValueError):amendment.validate(env,self.authority)
    def test_role_limited_to_training(self):
        self.job['role']='verify'
        with self.assertRaises(ValueError):self.run_check()
    def test_current_job_id_bound(self):
        self.job['job_id']='different'
        with self.assertRaises(ValueError):self.run_check()
    def test_original_manifest_cannot_change(self):
        m=copy.deepcopy(self.manifest['payload']);m['K']=2;self.job['manifest']=self.sign(m)
        with self.assertRaises(ValueError):self.run_check()
    def test_original_miner_source_remains_original(self):
        m=copy.deepcopy(self.manifest['payload']);m['source_bundle']['sha256']='e'*64
        self.job['manifest']=self.sign(m);self.value['original_signed_manifest_sha256']=sha(self.job['manifest'])
        with self.assertRaises(ValueError):self.run_check()
    def test_input_inventory_cannot_change(self):
        self.job['submissions']=self.submissions*2
        with self.assertRaises(ValueError):self.run_check()
    def test_parent_descriptor_counter_genesis_bound(self):
        for key in ('parent_descriptor_sha256','genesis_sha256','optimizer_step_before','trainer_binding_sha256'):
            old=self.value[key];self.value[key]=old+1 if isinstance(old,int)else'0'*64
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_check()
            self.value[key]=old
    def test_native_receipt_cannot_be_promoted_to_audited(self):
        self.value['native_eligibility_receipt']['sampling_assurance']='verified'
        with self.assertRaises(ValueError):self.run_check()
    def test_old_gpu_qualification_is_not_new_math_qualification(self):
        q=copy.deepcopy(self.value['execution_qualification']['payload']);q['execution_source_bundle_sha256']='7'*64
        self.value['execution_qualification']=self.sign(q)
        with self.assertRaisesRegex(ValueError,'qualification'):self.run_check()
    def test_cpu_only_qualification_refused(self):
        q=copy.deepcopy(self.value['execution_qualification']['payload']);q['actual_GPU_execution']=False
        self.value['execution_qualification']=self.sign(q)
        with self.assertRaisesRegex(ValueError,'qualification'):self.run_check()
    def test_undeclared_or_sampler_source_delta_refused(self):
        self.value['execution_source_files']['subnet/cached_sampling.py']='8'*64
        self.value['changed_source_files']['subnet/cached_sampling.py']='8'*64
        with self.assertRaisesRegex(ValueError,'source delta'):self.run_check()
    def test_runtime_cannot_change(self):
        self.job['runtime_versions']=dict(torch='other')
        with self.assertRaises(ValueError):self.run_check()
    def test_expiry_and_lifetime_are_bounded(self):
        with self.assertRaises(ValueError):amendment.validate(self.envelope(),self.authority,now=101)
        self.value['expires_at']=1e10
        with self.assertRaises(ValueError):self.run_check()
    def test_native_subset_signature_and_membership_checked_before_issue(self):
        docs=copy.deepcopy(self.native_docs);docs['subset']['payload']['accepted_submissions']=[]
        with self.assertRaises(ValueError):amendment.preparation_scope(self.public,self.manifest,self.submissions,docs,self.authority)
    def test_public_contract_unchanged_before_issue(self):
        m=copy.deepcopy(self.public['payload']);m['L']=8
        with self.assertRaises(ValueError):amendment.preparation_scope(self.sign(m),self.manifest,self.submissions,self.native_docs,self.authority)
    def test_report_must_disclose_actual_execution(self):
        env=self.envelope();evidence=amendment.provenance(env,self.authority)
        report=dict(job_sha256=sha(env['payload']),source_files=self.job['source_files'],**{amendment.FIELD:evidence})
        amendment.validate_provenance(report,env,self.authority)
        self.assertEqual(evidence['input_assurance'],'unaudited');self.assertFalse(evidence['optimizer_reset'])
        self.assertEqual(evidence['optimizer_step_after'],3)
        report[amendment.FIELD]['execution_source_bundle_sha256']='7'*64
        with self.assertRaises(ValueError):amendment.validate_provenance(report,env,self.authority)

if __name__=='__main__':unittest.main()
