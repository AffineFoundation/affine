import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from nacl.signing import SigningKey
from training_receipt_fixtures import sign
from subnet import learner_blacklist_selection as filtering,committed_training_inputs as learner
from subnet.training_receipts import sha,computation_binding

class BlacklistSelectionTests(unittest.TestCase):
    def setUp(self):
        self.key=SigningKey.generate();self.auth=self.key.verify_key.encode().hex()
        self.bad='a'*64;self.good='b'*64;self.new='c'*64
        self.manifest=dict(epoch='future',checkpoint={'id':'d'*64},source_bundle={'sha256':'e'*64},start=3650,deadline=3700)
        self.policy=dict(version='continuous-probabilistic-audit-v3',recent_epochs=8,decay=.8,prior_alpha=1,prior_beta=1,invalid_multiplier=.1,zero_epoch_after=2,blacklist_after=3,blacklist_epochs=4)
        self.assessment=dict(version='hourly-current-miner-assessment-v1',cutoff=3600,evidence_cutoff=3600,assessment_stale=False,writer_policy_sha256='f'*64,miner_estimates={self.bad:dict(blacklisted=True,confirmed_invalid_recent=3,latest_bad_round=36,current_estimate_round=36,unresolved_is_fraud=False,infrastructure_counted_in_coverage=False),self.good:dict(blacklisted=False,reward_multiplier=0,numerical_ambiguous_recent_weight=99)})
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.controller=SimpleNamespace(state=Path(self.tmp.name),authority=SimpleNamespace(id=self.auth))
        self.rows=[self.obj(m,i)for i,m in enumerate([self.bad,self.good,self.new])]
        self.enable()
    def obj(self,m,i):
        p=dict(epoch='future',checkpoint='d'*64,miner_identity=m,document_sha256=str(i)*64)
        return dict(sha256=str(i)*64,size=10,learner_admission=sign(self.key,p))
    def enable(self):
        p=dict(version=filtering.VERSION,checkpoint='d'*64,source_sha256='e'*64,target_round=37,maximum_age_seconds=3600,assessment_document=sign(self.key,self.assessment),writer_policy_sha256='f'*64,audit_policy=self.policy)
        self.manifest[filtering.FIELD]=sign(self.key,p)
    def select(self,at=3800):
        with patch('time.time',return_value=at):return learner.select_training_documents(self.controller,self.manifest,self.rows,{},round_number=37)
    def test_default_off_identical_old_selection_and_no_status_dependency(self):
        self.manifest.pop(filtering.FIELD)
        selected,status=self.select();self.assertEqual(selected,self.rows);self.assertNotIn('blacklist_selection',status)
    def test_new_and_UNKNOWN_zero_reward_remain_eligible(self):
        selected,status=self.select();self.assertEqual(selected,self.rows[1:]);self.assertEqual(status['blacklist_excluded_count'],1)
        self.assertEqual(status['eligible_count'],3);self.assertEqual(status['training_eligible_count'],2)
    def test_frozen_retry_not_rechecked_at_later_hour(self):
        first=self.select();second=self.select(at=100000);self.assertEqual(first,second)
    def test_first_selection_after_cutoff_expiry_uses_signed_opening(self):
        original=sign(self.key,self.manifest)
        self.manifest=filtering.authenticate(original,self.auth)
        selected,status=self.select(at=100000)
        self.assertEqual(selected,self.rows[1:])
        self.assertEqual(status['blacklist_selection']['assessment_cutoff'],3600)
        self.assertEqual(filtering.authenticate(original,self.auth),self.manifest)
    def test_stale_at_opening_remains_rejected(self):
        self.manifest.update(start=7201,deadline=7300)
        with self.assertRaisesRegex(ValueError,'fresh original'):self.select(at=7400)
    def test_partial_and_malformed_epoch_window_refuse(self):
        for change in ('missing-start','missing-deadline','empty','too-long','bool','nonfinite'):
            m=copy.deepcopy(self.manifest)
            if change=='missing-start':m.pop('start')
            if change=='missing-deadline':m.pop('deadline')
            if change=='empty':m['deadline']=m['start']
            if change=='too-long':m['deadline']=m['start']+7201
            if change=='bool':m['start']=True
            if change=='nonfinite':m['deadline']=float('inf')
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'opening/deadline'):
                filtering.admit(m[filtering.FIELD],m,self.auth,at=100000,round_number=37)
    def test_minimal_preopening_context_keeps_real_time_freshness(self):
        m=dict(checkpoint=self.manifest['checkpoint'],source_bundle=self.manifest['source_bundle'])
        filtering.admit(self.manifest[filtering.FIELD],m,self.auth,at=3800,round_number=37)
        with self.assertRaisesRegex(ValueError,'fresh original'):
            filtering.admit(self.manifest[filtering.FIELD],m,self.auth,at=7201,round_number=37)
    def test_tampered_signed_opening_is_not_authenticated(self):
        document=sign(self.key,self.manifest);document['payload']['start']-=3600
        with self.assertRaises(ValueError):filtering.authenticate(document,self.auth)
    def test_replaced_policy_cannot_change_retry_inputs(self):
        self.select();self.assessment['miner_estimates'][self.bad]['blacklisted']=False;self.enable()
        with self.assertRaisesRegex(ValueError,'immutable'):self.select()
    def test_stale_future_tampered_and_wrong_parent_refuse(self):
        for kind in ['stale','future','tamper','parent']:
            with self.subTest(kind=kind):
                m=copy.deepcopy(self.manifest)
                if kind=='stale':m[filtering.FIELD]['payload']['assessment_document']['payload']['assessment_stale']=True;m[filtering.FIELD]['payload']['assessment_document']=sign(self.key,m[filtering.FIELD]['payload']['assessment_document']['payload']);m[filtering.FIELD]=sign(self.key,m[filtering.FIELD]['payload'])
                if kind=='future':at=3500
                else:at=3800
                if kind=='tamper':m[filtering.FIELD]['payload']['assessment_document']['payload']['miner_estimates'][self.bad]['blacklisted']=False
                if kind=='parent':m['checkpoint']['id']='9'*64
                with self.assertRaises(ValueError):filtering.partition(self.rows,m,self.auth,at=at,round_number=37)
    def test_expiry_target_round_and_BOOL_refuse(self):
        self.manifest[filtering.FIELD]['payload']['target_round']=40;self.manifest[filtering.FIELD]=sign(self.key,self.manifest[filtering.FIELD]['payload'])
        kept,_=filtering.partition(self.rows,self.manifest,self.auth,at=3800,round_number=40);self.assertEqual(kept,self.rows)
        self.manifest[filtering.FIELD]['payload']['target_round']=True;self.manifest[filtering.FIELD]=sign(self.key,self.manifest[filtering.FIELD]['payload'])
        with self.assertRaises(ValueError):filtering.partition(self.rows,self.manifest,self.auth,at=3800)
    def test_original_population_not_mutated(self):
        before=copy.deepcopy(self.rows);self.select();self.assertEqual(self.rows,before)
    def test_computation_binding_binds_policy(self):
        self.manifest['learner_blacklist_selection_round']=37
        round_before=sha(computation_binding(self.manifest));self.manifest['learner_blacklist_selection_round']=38
        self.assertNotEqual(round_before,sha(computation_binding(self.manifest)))
        before=sha(computation_binding(self.manifest));self.manifest[filtering.FIELD]['signature']='different';self.assertNotEqual(before,sha(computation_binding(self.manifest)))

class CollectionBlacklistTests(unittest.TestCase):
    def test_end_to_end_collection_keeps_audit_population_and_freezes_job_binding(self):
        from test_committed_training_inputs import LearnerCollectionTests
        from subnet.continuous_audit_service import register_population
        fx=LearnerCollectionTests();fx.setUp();self.addCleanup(fx.tmp.cleanup)
        c=fx.controller();original_capture=c.gateway.capture_learner
        def capture(epoch):
            rows=original_capture(epoch)
            for receipt in rows.values():receipt['sha256']=sha(receipt['commitment_document'])
            return rows
        c.gateway.capture_learner=capture
        base=copy.deepcopy(fx.manifest);base['start']=10
        captured=[]
        def registered(*args,**kw):
            value=register_population(*args,**kw);captured.append(value);return value
        with patch('subnet.continuous_audit_service.register_population',side_effect=registered),patch('time.time',return_value=3701):
            _,old_inputs,old_pop=learner.collect(c,base,round_number=1)
        # Same authentic original committed documents in a fresh owned controller state.
        c.state=Path(fx.tmp.name)/'filtered';c.state.mkdir()
        status=dict(version='hourly-current-miner-assessment-v1',cutoff=0,evidence_cutoff=0,assessment_stale=False,writer_policy_sha256='f'*64,miner_estimates={fx.identity:dict(blacklisted=True,confirmed_invalid_recent=3,latest_bad_round=1,current_estimate_round=1,unresolved_is_fraud=False,infrastructure_counted_in_coverage=False)})
        audit=dict(version='continuous-probabilistic-audit-v3',recent_epochs=8,decay=.8,prior_alpha=1,prior_beta=1,invalid_multiplier=.1,zero_epoch_after=2,blacklist_after=3,blacklist_epochs=4)
        base[filtering.FIELD]=sign(fx.operator,dict(version=filtering.VERSION,checkpoint=base['checkpoint']['id'],source_sha256=base['source_bundle']['sha256'],target_round=1,maximum_age_seconds=3600,assessment_document=sign(fx.operator,status),writer_policy_sha256='f'*64,audit_policy=audit))
        base['learner_blacklist_selection_round']=1
        # The FIRST filtered capture happens after the assessment's wall-clock
        # cutoff expires, while this signed opening was originally fresh.
        with patch('subnet.continuous_audit_service.register_population',side_effect=registered),patch('time.time',return_value=3701):
            trained,new_inputs,new_pop=learner.collect(c,base,round_number=1)
        self.assertEqual(len(old_inputs),1);self.assertEqual(new_inputs,[])
        self.assertEqual(captured[0]['records'],captured[1]['records'])
        self.assertEqual(captured[0]['eligible_evidence_ids'],captured[1]['eligible_evidence_ids'])
        self.assertEqual(old_pop['eligible_inventory'],new_pop['eligible_inventory'])
        self.assertEqual(new_pop['eligible_count'],1)
        self.assertEqual(trained['learner_blacklist_selection_snapshot'],new_pop['training_selection']['blacklist_selection'])
        c.gateway.capture_learner=lambda e:(_ for _ in()).throw(AssertionError('recapture'))
        self.assertEqual(learner.collect(c,base,round_number=1),(trained,new_inputs,new_pop))
    def test_training_job_cannot_reinsert_blacklisted_input_or_unbind_snapshot(self):
        from test_committed_training_inputs import LearnerAdmissionTests
        fx=LearnerAdmissionTests();fx.setUp();self.addCleanup(fx.tmp.cleanup)
        m=fx.manifest;m['start']=10;m['deadline']=20
        audit=dict(version='continuous-probabilistic-audit-v3',recent_epochs=8,decay=.8,prior_alpha=1,prior_beta=1,invalid_multiplier=.1,zero_epoch_after=2,blacklist_after=3,blacklist_epochs=4)
        assessment=dict(version='hourly-current-miner-assessment-v1',cutoff=0,evidence_cutoff=0,assessment_stale=False,writer_policy_sha256='f'*64,miner_estimates={fx.identity:dict(blacklisted=True,confirmed_invalid_recent=3,latest_bad_round=1,current_estimate_round=1,unresolved_is_fraud=False,infrastructure_counted_in_coverage=False)})
        m[filtering.FIELD]=sign(fx.operator,dict(version=filtering.VERSION,checkpoint=m['checkpoint']['id'],source_sha256=m['source_bundle']['sha256'],target_round=1,maximum_age_seconds=3600,assessment_document=sign(fx.operator,assessment),writer_policy_sha256='f'*64,audit_policy=audit))
        m['learner_blacklist_selection_round']=1
        m=learner.coverage_manifest(m,[fx.obj],seed='c'*64,captured_at=21)
        m['learner_blacklist_selection_snapshot']=filtering.admit(m[filtering.FIELD],m,fx.authority,at=21)
        job=dict(role='train',training_policy=m['training_policy'],training_input_policy=learner.VERSION,source_files={'subnet/committed_training_inputs.py':'d'*64,'subnet/learner_blacklist_selection.py':'d'*64},submissions=[fx.obj])
        with self.assertRaisesRegex(ValueError,'blacklisted'):learner.validate_job(job,m,fx.authority)
        m.pop('learner_blacklist_selection_snapshot')
        with self.assertRaisesRegex(ValueError,'snapshot'):learner.validate_job(job,m,fx.authority)

class AutomaticOpeningTests(BlacklistSelectionTests):
    def setUp(self):
        super().setUp()
        self.path=self.controller.state/'assessment.json';self.write_assessment()
        self.controller.signed=lambda v:sign(self.key,v)
        self.approval=dict(version=filtering.AUTHORIZATION_VERSION,source_sha256='e'*64,writer_policy_sha256='f'*64,audit_policy=self.policy,maximum_age_seconds=3600,assessment_path=str(self.path))
        self.config=dict(source_bundle={'sha256':'e'*64},training_input_policy=learner.VERSION,**{filtering.AUTHORIZATION_FIELD:sign(self.key,self.approval)})
    def write_assessment(self):
        from subnet.storage import canonical
        self.path.write_bytes(canonical(sign(self.key,self.assessment)))
    def test_approved_mechanism_autonomously_binds_new_parent_round_and_new_snapshot(self):
        with patch('time.time',return_value=3800):
            first=filtering.prepare_opening(self.controller,self.config,dict(checkpoint={'id':'d'*64},round=37),{})
        self.assessment['cutoff']=self.assessment['evidence_cutoff']=7200;self.assessment['miner_estimates'][self.bad]['blacklisted']=False;self.write_assessment()
        with patch('time.time',return_value=7400):
            second=filtering.prepare_opening(self.controller,self.config,dict(checkpoint={'id':'9'*64},round=38),{})
        saved=copy.deepcopy(first)
        # Execute actual production opening branch with an existing manifest.
        import ast,json
        from subnet import gpu_service
        tree=ast.parse(Path(gpu_service.__file__).read_text());branch=next(n for n in ast.walk(tree)if isinstance(n,ast.If)and isinstance(n.test,ast.Call)and isinstance(n.test.func,ast.Attribute)and n.test.func.attr=='exists'and isinstance(n.test.func.value,ast.Name)and n.test.func.value.id=='manifestpath')
        manifestpath=self.controller.state/'saved-opening.json';manifestpath.write_text(json.dumps(saved))
        context=dict(manifestpath=manifestpath,json=json)
        # Any refresh reaches the real else branch and fails on missing actors.
        loop=ast.parse('for _ in range(1):\n pass').body[0];loop.body=[branch]
        exec(compile(ast.Module(body=[loop],type_ignores=[]),'actual-opening-branch','exec'),context)
        self.assertEqual(context['manifest'],saved)
        self.assertEqual(first[filtering.FIELD]['payload']['checkpoint'],'d'*64)
        self.assertEqual(second[filtering.FIELD]['payload']['checkpoint'],'9'*64)
        self.assertEqual(second['learner_blacklist_selection_round'],38)
        self.assertNotEqual(first[filtering.FIELD],second[filtering.FIELD])
        self.assertEqual(self.config[filtering.AUTHORIZATION_FIELD]['payload'],self.approval)
    def test_stale_and_path_symlink_refuse_before_opening(self):
        with patch('time.time',return_value=10000),self.assertRaisesRegex(ValueError,'fresh'):filtering.prepare_opening(self.controller,self.config,dict(checkpoint={'id':'d'*64},round=37),{})
        moved=self.path.with_name('kept.json');self.path.rename(moved);self.path.symlink_to(moved)
        with self.assertRaisesRegex(ValueError,'path'):filtering.prepare_opening(self.controller,self.config,dict(checkpoint={'id':'d'*64},round=37),{})
    def test_explicit_NULL_refuses(self):
        with self.assertRaises(ValueError):filtering.prepare_opening(self.controller,{filtering.AUTHORIZATION_FIELD:None},None,{})
        with self.assertRaises(ValueError):filtering.partition(self.rows,dict(self.manifest,**{filtering.FIELD:None}),self.auth,at=3800)
    def test_absent_mechanism_returns_exact_contract(self):
        contract={'old':'unchanged'};self.assertIs(filtering.prepare_opening(self.controller,{},None,contract),contract)
