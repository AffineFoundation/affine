import base64
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from nacl.signing import SigningKey
from ops.native_training_outcome_filter import AUTHORIZATION_VERSION,VERSION,digest
from ops.native_training_eligibility import NativeEligibilitySelector,NativeNoUpdate,bind_subset,_canonical
from subnet.committed_training_inputs import coverage_manifest


class SelectorIntegration(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.state=Path(self.directory.name);self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        def sign(payload):return dict(payload=payload,signer=self.authority,signature=base64.b64encode(self.key.sign(_canonical(payload)).signature).decode())
        self.sign=sign;self.controller=SimpleNamespace(state=self.state,authority=SimpleNamespace(id=self.authority),signed=sign)
        self.epoch='CPU-ONLY-never-dispatch-native-selection'
        self.manifest=dict(epoch=self.epoch,checkpoint={'id':'1'*64},source_bundle={'sha256':'2'*64},
            trainer_state_binding={'global_step_before':23},training_input_policy='committed-unaudited-training-v1')
        self.submissions=[]
        for i in range(2):
            data=_canonical({'fixture':i});original=self.state/('original-'+str(i));original.write_bytes(data)
            self.submissions.append(dict(sha256=__import__('hashlib').sha256(data).hexdigest(),size=len(data),url=original.as_uri(),learner_admission=self.sign({'fixture':i})))
        self.manifest=coverage_manifest(self.manifest,self.submissions,seed='a'*64,captured_at=10)
        self.population=self.state/(self.epoch+'-learner-population.json')
        self.population.write_bytes(_canonical(dict(version='committed-unaudited-training-v1',manifest=self.manifest,submissions=self.submissions,population={'assurance':'unaudited'})))
        self.selection=self.state/(self.epoch+'-learner-training-selection.json');self.selection.write_bytes(_canonical({'seed':'a'*64}))
        self.original_files={p:p.read_bytes()for p in (self.population,self.selection)}
        self.auth=self.sign(dict(version=AUTHORIZATION_VERSION,source_root='/approved/f213',source_files={'subnet/test.py':'b'*64}))
        self.selector=NativeEligibilitySelector(self.controller,self.auth,'/approved/tokenizer','/approved/bin/python')
        self.statuses=['accepted_native_labels','excluded_label_mismatch'];self.calls=0
    def grade(self,paths,context,*args):
        self.calls+=1;decisions=[];rows=[]
        for i,(obj,status) in enumerate(zip(self.submissions,self.statuses)):
            identity=str(i)*64;scores=(1,0) if status=='accepted_native_labels' else (0,1) if status=='excluded_label_mismatch' else (None,None)
            grades=[dict(claim=c,native_score=v,label_matches=None if v is None else (v==1)==(c=='positive'))for c,v in zip(('positive','negative'),scores)]
            rows.append(dict(pair_sha256=identity,status=status,grades=grades))
            decisions.append(dict(document_sha256=obj['sha256'],learner_admission_sha256=digest(obj['learner_admission']),pair_sha256=[identity],accepted=status=='accepted_native_labels'))
        return [],dict(version=VERSION,context_sha256=digest(context),sampling_assurance='unaudited',proof_verification_performed=False,claims_rewritten=False,cheating_penalties=False,rows=rows,document_decisions=decisions)
    def select(self):
        with patch('ops.native_training_eligibility.filter_eligibility_context',side_effect=self.grade):return self.selector.select(self.manifest,self.submissions)
    def assert_originals(self):
        for p,data in self.original_files.items():self.assertEqual(p.read_bytes(),data)
    def test_subset_new_coverage_original_population_and_lineage_preserved(self):
        manifest,subset=self.select();self.assertEqual(subset,self.submissions[:1]);self.assertEqual(manifest['trainer_state_binding'],self.manifest['trainer_state_binding'])
        self.assertNotEqual(manifest['training_coverage']['inventory_sha256'],self.manifest['training_coverage']['inventory_sha256']);self.assert_originals()
    def test_allaccepted_retains_identical_coverage(self):
        self.statuses=['accepted_native_labels']*2;manifest,subset=self.select();self.assertEqual(manifest,self.manifest);self.assertEqual(subset,self.submissions);self.assert_originals()
    def test_zeroaccepted_records_no_update_and_never_dispatches(self):
        self.statuses=['excluded_indeterminate','excluded_label_mismatch']
        with self.assertRaises(NativeNoUpdate):self.select()
        path=self.state/'native-outcome-eligibility'/self.epoch/'subset.ROOT-SIGNED.json';receipt=json.loads(path.read_bytes())['payload']
        self.assertEqual(receipt['disposition'],'no_update');self.assertEqual(receipt['accepted_count'],0);self.assert_originals()
    def test_restart_reuses_authentic_exact_subset_no_grading_or_get(self):
        expected=self.select();self.assertEqual(self.calls,1)
        (self.state/'roles').mkdir();(self.state/'roles'/(self.epoch+'-train.json')).write_text('issued original')
        with patch('ops.native_training_eligibility.filter_eligibility_context',side_effect=AssertionError('must not regrade')),patch('urllib.request.urlopen',side_effect=AssertionError('must not refetch')):
            self.assertEqual(self.selector.select(self.manifest,self.submissions),expected)
        self.assert_originals()
    def test_crash_after_grade_before_subset_reuses_grade_exactly(self):
        from ops.native_training_eligibility import _create
        def create(path,value):
            if path.name=='subset.ROOT-SIGNED.json':raise RuntimeError('simulated crash')
            return _create(path,value)
        with patch('ops.native_training_eligibility._create',side_effect=create):
            with self.assertRaisesRegex(RuntimeError,'simulated crash'):self.select()
        with patch('ops.native_training_eligibility.filter_eligibility_context',side_effect=AssertionError('no regrade')):self.selector.select(self.manifest,self.submissions)
    def test_existing_issued_job_without_subset_refuses_adoption(self):
        (self.state/'roles').mkdir();(self.state/'roles'/(self.epoch+'-train.json')).write_text('original')
        with self.assertRaisesRegex(ValueError,'issued training job'):self.select()
        self.assertEqual(self.calls,0)
    def test_changed_original_capture_selection_parent_or_policy_refuse(self):
        self.select()
        self.selection.write_text('changed')
        with self.assertRaisesRegex(ValueError,'context changed'):self.select()
        self.selection.write_bytes(self.original_files[self.selection]);self.manifest['trainer_state_binding']['global_step_before']=24
        with self.assertRaisesRegex(ValueError,'frozen population'):self.select()
    def test_changed_signed_subset_or_grade_cannot_be_reused(self):
        self.select();path=self.state/'native-outcome-eligibility'/self.epoch/'subset.ROOT-SIGNED.json';value=json.loads(path.read_text());value['payload']['accepted_count']=2;path.write_bytes(_canonical(value))
        with self.assertRaises(Exception):self.select()
    def test_grade_status_cannot_forge_acceptance(self):
        def forged(*args):
            _,r=self.grade(*args);r['document_decisions'][1]['accepted']=True;return [],r
        with patch('ops.native_training_eligibility.filter_eligibility_context',side_effect=forged):
            with self.assertRaisesRegex(ValueError,'completeness verdict'):self.selector.select(self.manifest,self.submissions)
    def test_missing_or_duplicate_original_decisions_refuse(self):
        for mutate in (lambda r:r['document_decisions'].pop(),lambda r:r['document_decisions'][1].update(pair_sha256=['0'*64])):
            # Direct binding tests avoid leaving a signed invalid-grade journal.
            context=self.sign(dict(submissions=self.submissions));_,receipt=self.grade([],context);mutate(receipt)
            with self.assertRaises(ValueError):bind_subset(context,receipt,self.submissions,self.authority)
    def test_hardlink_or_symlink_original_refuse(self):
        self.population.unlink();self.population.symlink_to(self.selection)
        with self.assertRaisesRegex(ValueError,'journal file'):self.select()
    def test_controller_overlay_invokes_selector_before_capacity_and_job_signature(self):
        # Execute the real frozen f213 controller function with ONLY the new
        # four-line hook, stubbing transport after its first job construction.
        path=Path(__file__).parents[1]/'prospective/native-training-controller-overlay/subnet/persistent_training_controller.py'
        spec=importlib.util.spec_from_file_location('subnet.native_filter_controller_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        class CapturedDispatch(Exception):pass
        def run(label,role,manifest,checkpoint,**kwargs):
            self.assertEqual(role,'train');self.assertEqual(kwargs['submissions'],self.submissions[:1]);self.assertEqual(self.calls,1);raise CapturedDispatch()
        self.controller.native_training_eligibility_selector=self.selector
        self.controller.jobs=SimpleNamespace(training_resume=lambda *a:None,persistent_training_capacity=lambda *a,**kw:{'fixture':True},run=run)
        with patch.object(module,'validate_binding',return_value={'fixture':True}),patch('subnet.training_startup_recovery.declaration',return_value=None),patch('ops.native_training_eligibility.filter_eligibility_context',side_effect=self.grade):
            with self.assertRaises(CapturedDispatch):module.train(self.controller,self.manifest,[],None,steps=1)
        self.assert_originals()

    def load_controller(self):
        path=Path(__file__).parents[1]/'prospective/native-training-controller-overlay/subnet/persistent_training_controller.py'
        spec=importlib.util.spec_from_file_location('subnet.native_filter_controller_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    def test_recovery_cannot_replace_graded_inputs_before_dispatch(self):
        module=self.load_controller();self.controller.native_training_eligibility_selector=self.selector
        self.controller.jobs=SimpleNamespace(run=lambda *a,**kw:self.fail('must not dispatch'))
        with patch.object(module,'validate_binding',return_value={'fixture':True}),patch('subnet.training_startup_recovery.declaration',return_value={'payload':{'version':'test-recovery'}}),patch('subnet.training_startup_recovery.apply',side_effect=AssertionError('must not replace inputs')):
            with self.assertRaisesRegex(ValueError,'separately authorized native eligibility'):module.train(self.controller,self.manifest,[],None,steps=1)
        self.assertEqual(self.calls,0);self.assert_originals()
    def test_completed_allaccepted_metrics_reobserved_without_dispatch_or_grade(self):
        self.statuses=['accepted_native_labels']*2;derived,inputs=self.select();module=self.load_controller()
        self.controller.native_training_eligibility_selector=self.selector
        self.controller.jobs=SimpleNamespace(run=lambda *a,**kw:self.fail('must not dispatch'))
        from ops.native_training_eligibility import _inventory
        metrics={'learner_admission_inventory':_inventory(inputs)}
        (self.state/(self.epoch+'-training-metrics.json')).write_bytes(_canonical(metrics))
        job={'manifest':self.sign(derived),'steps':1,'submissions':inputs}
        (self.state/'roles').mkdir();(self.state/'roles/test-report.json').write_text('{}')
        class CompletedOriginalReobserved(Exception):pass
        def checked_report(*args):raise CompletedOriginalReobserved()
        with patch.object(module,'validate_binding',return_value={'fixture':True}),patch.object(module,'original_request',return_value=({'job_id':'test'},job)),patch.object(module,'validate_report',side_effect=checked_report),patch('subnet.training_startup_recovery.declaration',return_value=None),patch('ops.native_training_eligibility.filter_eligibility_context',side_effect=AssertionError('must not regrade')):
            with self.assertRaises(CompletedOriginalReobserved):module.train(self.controller,self.manifest,[],None,steps=1)
        self.assertEqual(self.calls,1);self.assert_originals()

if __name__=='__main__':unittest.main()
