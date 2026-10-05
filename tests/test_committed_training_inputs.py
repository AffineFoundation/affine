import copy,hashlib,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from nacl.signing import SigningKey
from training_receipt_fixtures import sign,transport_fixture
from subnet import committed_training_inputs as learner
from subnet.storage import canonical
from subnet.training_receipts import sha

class LearnerAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.operator=SigningKey.generate();self.miner=SigningKey.generate()
        self.authority=self.operator.verify_key.encode().hex();self.identity=self.miner.verify_key.encode().hex()
        fx=transport_fixture(self.operator,policy='bf16-cpu-fp32-master-task-normalized-persistent-v4')
        self.manifest=fx['manifest'];self.manifest['training_input_policy']=learner.VERSION
        self.batch=fx['batch']
        for rollout in self.batch['rollouts']:rollout['environment_version']='synthetic-v1'
        self.build()
    def build(self):
        doc=dict(version=learner.ARTIFACT_VERSION,epoch=self.manifest['epoch'],checkpoint=self.manifest['checkpoint']['id'],miner=self.identity,slot=0,batch=self.batch)
        self.data=canonical(doc);self.path=self.root/'document.json';self.path.write_bytes(self.data)
        child=dict(slot=0,env_id='math',index=0,batch_sha256=sha(self.batch),sha256='a'*64,size=999,
            training_sha256=hashlib.sha256(self.data).hexdigest(),training_size=len(self.data))
        original=sign(self.miner,dict(version='small-commitment-pairs-v2',epoch=self.manifest['epoch'],miner=self.identity,checkpoint=self.manifest['checkpoint']['id'],source=self.manifest['source_bundle']['sha256'],batches=[child]))
        admission=dict(version=learner.VERSION,epoch=self.manifest['epoch'],checkpoint=self.manifest['checkpoint']['id'],source_sha256=self.manifest['source_bundle']['sha256'],miner_identity=self.identity,slot=0,original_commitment=original,commitment_sha256=sha(original),proof_sha256='a'*64,batch_sha256=sha(self.batch),document_sha256=child['training_sha256'],document_size=len(self.data),captured_at=21,assurance='unaudited')
        self.obj=dict(sha256=child['training_sha256'],size=len(self.data),learner_admission=sign(self.operator,admission))
    def admit(self):return learner.admitted_submission(self.path,self.obj,self.manifest,self.authority)
    def test_unaudited_needs_no_proof_zip_or_grader(self):
        with patch('subnet.batches.unpack',side_effect=AssertionError('proof')),patch('subnet.model.Runtime.compute',side_effect=AssertionError('model')):
            summary,pairs=self.admit()
        self.assertEqual(summary['assurance'],'unaudited');self.assertFalse(summary['trainer_verification_performed']);self.assertEqual(len(pairs),1)
        self.assertNotIn('accepted',summary)
    def test_tampered_bytes_rejected(self):
        self.path.write_bytes(self.data+b' ')
        with self.assertRaises(ValueError):self.admit()
    def test_operator_signature_cannot_replace_miner_commitment(self):
        value=self.obj['learner_admission']['payload'];value['proof_sha256']='b'*64
        self.obj['learner_admission']=sign(self.operator,value)
        with self.assertRaisesRegex(ValueError,'commitment'):self.admit()
    def test_heldout_rejected_even_signed(self):
        self.manifest['heldout_indices']={'math':[0]}
        with self.assertRaisesRegex(ValueError,'heldout'):self.admit()
    def test_claimed_positive_negative_required(self):
        self.batch['rollouts'][1]['classification']='positive';self.build()
        with self.assertRaisesRegex(ValueError,'quota'):self.admit()
    def test_noninteger_token_rejected_even_signed(self):
        self.batch['rollouts'][0]['turns'][0]['output']=[True];self.build()
        with self.assertRaisesRegex(ValueError,'token'):self.admit()
    def test_historical_policy_cannot_opt_in(self):
        self.manifest['training_input_policy']='authenticated-verifier-compact-inputs-v2'
        with self.assertRaisesRegex(ValueError,'explicit'):self.admit()
    def test_job_population_cannot_change_or_claim_audited(self):
        self.manifest=learner.coverage_manifest(self.manifest,[self.obj],seed='c'*64,captured_at=21)
        job=dict(role='train',training_policy=self.manifest['training_policy'],training_input_policy=learner.VERSION,
            source_files={'subnet/committed_training_inputs.py':'d'*64},submissions=[self.obj])
        learner.validate_job(job,self.manifest,self.authority)
        job['submissions']=[self.obj,self.obj]
        with self.assertRaises(ValueError):learner.validate_job(job,self.manifest,self.authority)
    def test_native_prompt_no_grading_no_model(self):
        _,pairs=self.admit();definition,pos,neg=pairs[0]
        definition['spec'].update(id='affine_math',max_turns=1,max_output_tokens=64);pos['env_seed']=neg['env_seed']=0
        runtime=SimpleNamespace(tokenizer=object(),model=SimpleNamespace(config=SimpleNamespace(vocab_size=10)))
        session=SimpleNamespace(reset=lambda *a:dict(task_hash='6'*64,messages=[],tools=[]),close=lambda:None)
        with patch('subnet.native_math_prompt.NativeMathPromptSession',return_value=session),patch('subnet.harness.render',return_value=[1,2]):
            learner.validate_native_prompt(runtime,pairs,self.manifest)
            pos['turns'][0]['prompt']=[8]
            with self.assertRaisesRegex(ValueError,'prompt'):learner.validate_native_prompt(runtime,pairs,self.manifest)

if __name__=='__main__':unittest.main()

class LearnerCollectionTests(LearnerAdmissionTests):
    def controller(self):
        from test_persistent_training_integration import MemoryBucket
        bucket=MemoryBucket();bucket.objects['private/frozen/document']=self.data
        original=self.obj['learner_admission']['payload']['original_commitment']
        receipt=dict(commitment_document=original,training_documents=[dict(slot=0,sha256=self.obj['sha256'],size=self.obj['size'],frozen_key='private/frozen/document',captured_at=21)])
        gateway=SimpleNamespace(capture_learner=lambda epoch:{self.identity:receipt},freeze=lambda epoch:(_ for _ in()).throw(AssertionError('heavyfreeze')))
        return SimpleNamespace(state=self.root,bucket=bucket,gateway=gateway,authority=SimpleNamespace(id=self.authority),signed=lambda payload:sign(self.operator,payload))
    def test_collect_private_small_inputs_without_audits_and_resume(self):
        controller=self.controller()
        manifest,inputs,population=learner.collect(controller,self.manifest)
        self.assertEqual(len(inputs),1);self.assertEqual(population['assurance'],'unaudited')
        self.assertEqual(manifest['training_coverage']['assurance'],'unaudited')
        self.assertFalse((self.root/(self.manifest['epoch']+'-scores.json')).exists())
        controller.gateway.capture_learner=lambda epoch:(_ for _ in()).throw(AssertionError('recapture'))
        self.assertEqual(learner.collect(controller,self.manifest),(manifest,inputs,population))
    def test_cheap_structure_bad_document_excluded_not_fraud(self):
        self.batch['rollouts'][0]['turns'][0]['output']=[True];self.build()
        _,inputs,population=learner.collect(self.controller(),self.manifest)
        self.assertEqual(inputs,[]);self.assertEqual(population['exclusions'][0]['reason'],'structural_ineligible')
        self.assertNotIn('fraud',population)
    def test_context_cannot_change_on_resume(self):
        controller=self.controller();learner.collect(controller,self.manifest)
        changed=copy.deepcopy(self.manifest);changed['source_bundle']['sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'context'):learner.collect(controller,changed)

class UnauditedPersistentEvidenceTests(unittest.TestCase):
    def test_real_fp32_publication_accepts_truthful_unaudited_admission(self):
        from test_persistent_training_integration import PersistentIntegrationTests
        fx=PersistentIntegrationTests();fx.setUp()
        try:
            for rollout in fx.batch['rollouts']:rollout['environment_version']='synthetic-v1'
            report,job=fx.report()
            setup=LearnerAdmissionTests();setup.setUp()
            try:
                manifest=copy.deepcopy(fx.manifest)
                manifest['training_input_policy']=learner.VERSION
                setup.operator=fx.key;setup.authority=fx.authority
                setup.manifest=manifest;setup.batch=fx.batch
                for rollout in setup.batch['rollouts']:rollout['environment_version']='synthetic-v1'
                setup.build();summary,_=setup.admit()
                obj=setup.obj;obj['url']=fx.submission['url']
                manifest=learner.coverage_manifest(manifest,[obj],seed=fx.manifest['training_coverage']['seed'],captured_at=21)
                job.update(training_input_policy=learner.VERSION,submissions=[obj],manifest=fx.sign(manifest))
                job['source_files']['subnet/committed_training_inputs.py']='b'*64
                job['source_files']['subnet/native_math_prompt.py']='b'*64
                report['training_admissions']=[summary]
                report['training'].update(training_input_policy=learner.VERSION,all_pairs_authenticated_verifier_receipts=False,input_assurance='unaudited')
                from subnet.backend_jobs import validate
                validate(fx.sign(job),fx.authority,now=30)
                missing=copy.deepcopy(job);missing['source_files'].pop('subnet/native_math_prompt.py')
                with self.assertRaisesRegex(ValueError,'prompt eligibility source pin'):
                    validate(fx.sign(missing),fx.authority,now=30)
                from subnet.persistent_training_protocol import validate_report
                validate_report(report,job,manifest)
                self.assertEqual(report['persistent_training_state']['descriptor']['optimizer_steps'],3)
                self.assertEqual(report['audits'],[])
                report['training_admissions'][0]['fully_audited']=True
                with self.assertRaisesRegex(ValueError,'exact unaudited'):validate_report(report,job,manifest)
            finally:setup.doCleanups()
        finally:fx.doCleanups()

class LearnerFreshBootstrapTests(unittest.TestCase):
    def test_declared_pure_learner_reload_without_gpu_preload_exception(self):
        import subprocess,sys
        from test_persistent_publication_bootstrap import SCRIPT
        script=SCRIPT.replace('persistent_publication','committed_training_inputs')
        for mode in ('reload','runtime-preload','undeclared','bad-hash'):
            with self.subTest(mode=mode):
                result=subprocess.run([sys.executable,'-B','-c',script,mode],cwd=Path(__file__).resolve().parent.parent,capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)

class AuditPopulationHandoffTests(LearnerCollectionTests):
    def test_original_signed_population_handoff_includes_actual_round(self):
        import sys
        controller=self.controller();calls=[]
        def register(manifest_document,receipts,round_number,committed_at,authority,*,eligible_pairs=None):
            calls.append((manifest_document,receipts,round_number,eligible_pairs))
            self.assertNotIn('training_coverage',manifest_document['payload']) if 'training_coverage'not in self.manifest else None
            return dict(version='continuous-audit-population-v1',manifest_document=manifest_document,receipts=receipts,round=round_number,committed_at=committed_at,records=[])
        with patch.dict(sys.modules,{'subnet.continuous_audit_service':SimpleNamespace(register_population=register)}):
            learner.collect(controller,self.manifest,round_number=12)
        self.assertEqual(calls[0][2],12)
        self.assertEqual(len(calls[0][3]),1)
        self.assertEqual(calls[0][3][0]['miner'],self.identity)
        audit=__import__('json').loads((self.root/(self.manifest['epoch']+'-continuous-audit-population.json')).read_bytes())
        self.assertEqual(audit['payload']['round'],12)
        self.assertEqual(audit['payload']['manifest_document']['payload'],self.manifest)

    def test_structurally_bad_declared_pair_remains_audit_candidate_not_reward_eligible(self):
        import sys
        self.batch['rollouts'][0]['turns'][0]['output']=[True];self.build()
        calls=[]
        def register(manifest_document,receipts,round_number,committed_at,authority,*,eligible_pairs=None):
            calls.append((receipts,eligible_pairs))
            return dict(version='continuous-audit-population-v1',manifest_document=manifest_document,receipts=receipts,round=round_number,committed_at=committed_at,records=[],eligible_evidence_ids=[])
        with patch.dict(sys.modules,{'subnet.continuous_audit_service':SimpleNamespace(register_population=register)}):
            _,inputs,_=learner.collect(self.controller(),self.manifest,round_number=12)
        self.assertEqual(inputs,[])
        self.assertIn(self.identity,calls[0][0])
        self.assertEqual(calls[0][1],[])
