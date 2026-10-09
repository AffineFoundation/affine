"""CPU fixtures only; no live keys, GPU, chain call, or mutable production state."""
import base64
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.live_reward_bridge import signed, sha
from subnet import learner_blacklist_selection as selection
from subnet import committed_training_inputs as learner
from subnet import current_assessment as calculation
from types import SimpleNamespace, ModuleType

SOURCE = Path(__file__).resolve().parents[1] / 'ops/training_assessment_writer_reuse.py'
spec = importlib.util.spec_from_file_location('candidate_training_snapshot', SOURCE)
candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)


class FreshWriterReuse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = SigningKey.generate(); self.authority = self.key.verify_key.encode().hex()
        self.config = self.root / 'audit.json'; self.config.write_bytes(b'{"scope":"CPU-fixture"}')
        self.policy = dict(audit_config=str(self.config), source_admission_sha256='b'*64,
                           numerical_resolution_policy_sha256='c'*64, module_hashes={}, verifiers=['d'*64])
        self.review = dict(producer_policy_sha256=sha(self.sign(self.policy)),
            writer_policy_sha256='e'*64, audit_config_file_sha256=hashlib.sha256(self.config.read_bytes()).hexdigest(),
            source_admission_sha256='b'*64, numerical_resolution_policy_sha256='c'*64,
            directory=str(self.root))
        self.output = self.root / 'learner.json'; self.cache = self.root / 'assessment-3600.json'
        self.bad, self.good, self.new = '1'*64, '2'*64, '3'*64
        self.audit = dict(version='continuous-probabilistic-audit-v3', recent_epochs=8, decay=.8,
            prior_alpha=1, prior_beta=1, invalid_multiplier=.1, zero_epoch_after=2, blacklist_after=3, blacklist_epochs=4)
        self.assessment = dict(version='hourly-current-miner-assessment-v1', cutoff=3600,
            evidence_cutoff=3600, assessment_stale=False, writer_policy_sha256='e'*64,
            half_life_hours=6, history_hours=168, hourly_alpha=1-2**(-1/6),
            smoothing_basis='estimated-valid-contribution-before-penalty', penalties_applied_after_smoothing=True,
            training_completion_required=False, unaudited_samples_claimed_verified=False,
            evidence_hashes={k:self.review[k] for k in ('audit_config_file_sha256','source_admission_sha256','numerical_resolution_policy_sha256')},
            miner_estimates={self.bad:dict(blacklisted=True,confirmed_invalid_recent=3,latest_bad_round=36,
                current_estimate_round=36,unresolved_is_fraud=False,infrastructure_counted_in_coverage=False),
                self.good:dict(blacklisted=False,reward_multiplier=0,numerical_ambiguous_recent_weight=99)})
        self.write_cache()

    def sign(self, value, key=None):
        key = key or self.key
        return dict(payload=copy.deepcopy(value),signer=key.verify_key.encode().hex(),
            signature=base64.b64encode(key.sign(canonical(value)).signature).decode())

    def write_cache(self):
        self.cache.write_bytes(canonical(self.sign(self.assessment)))

    def reuse(self, **kw):
        return candidate.reuse_cache(self.output,self.policy,self.review['producer_policy_sha256'],
            self.authority,signed=signed,now=kw.pop('now',lambda:3700),review=self.review,**kw)

    def test_exact_writer_envelope_copied_without_resigning_or_chain_receipt(self):
        original=self.cache.read_bytes();result=self.reuse()
        self.assertEqual(result['producer'],'hourly-writer');self.assertFalse(result['chain_transactions'])
        self.assertEqual(self.output.read_bytes(),original)
        self.assertEqual(signed(json.loads(original),self.authority)['writer_policy_sha256'],'e'*64)

    def test_copied_writer_snapshot_reused_on_retry(self):
        self.reuse();original=self.output.read_bytes();self.cache.unlink()
        self.assertEqual(self.reuse()['producer'],'hourly-writer');self.assertEqual(self.output.read_bytes(),original)

    def test_original_G_cache_rule_retained(self):
        a=copy.deepcopy(self.assessment);a['writer_policy_sha256']=self.review['producer_policy_sha256']
        self.output.write_bytes(canonical(self.sign(a)));self.cache.unlink()
        self.assertEqual(self.reuse()['producer'],'original')

    def test_stale_wrong_hour_and_policy_refused(self):
        for key,value in [('assessment_stale',True),('cutoff',0),('cutoff',7200),
                          ('evidence_cutoff',0),('writer_policy_sha256','9'*64),('cutoff',True)]:
            with self.subTest(key=key,value=value):
                a=copy.deepcopy(self.assessment);a[key]=value
                self.cache.write_bytes(canonical(self.sign(a)));self.assertIsNone(self.reuse())
                self.assertFalse(self.output.exists())

    def test_wrong_source_numerical_or_config_digest_refused(self):
        for key in self.assessment['evidence_hashes']:
            with self.subTest(key=key):
                a=copy.deepcopy(self.assessment);a['evidence_hashes'][key]='9'*64
                self.cache.write_bytes(canonical(self.sign(a)));self.assertIsNone(self.reuse())

    def test_config_drift_does_not_reuse_reviewed_writer(self):
        self.config.write_bytes(b'{}');self.assertIsNone(self.reuse())

    def test_malformed_signed_schema_is_a_miss(self):
        for field,value in [('evidence_hashes',[]),('miner_estimates',[]),('hourly_alpha','bad'),
                            ('miner_estimates',{self.good:dict(blacklisted='false')}),
                            ('training_completion_required',True),('half_life_hours',1)]:
            with self.subTest(field=field):
                a=copy.deepcopy(self.assessment);a[field]=value
                self.cache.write_bytes(canonical(self.sign(a)));self.assertIsNone(self.reuse())

    def test_bad_signature_signer_and_changed_status_refused(self):
        doc=self.sign(self.assessment);doc['payload']['miner_estimates'][self.bad]['blacklisted']=False
        for d in (doc,self.sign(self.assessment,SigningKey.generate())):
            self.cache.write_bytes(canonical(d));self.assertIsNone(self.reuse())

    def test_partial_missing_symlink_directory_and_oversized_refused(self):
        self.cache.write_bytes(b'{"payload":');self.assertIsNone(self.reuse())
        self.cache.unlink();self.assertIsNone(self.reuse())
        alternate=self.root/'elsewhere.json';alternate.write_bytes(canonical(self.sign(self.assessment)))
        self.cache.symlink_to(alternate);self.assertIsNone(self.reuse());self.cache.unlink()
        self.cache.mkdir();self.assertIsNone(self.reuse());self.cache.rmdir()
        with self.cache.open('wb') as f:f.truncate(candidate.MAX_ASSESSMENT_BYTES+1)
        self.assertIsNone(self.reuse())

    def test_non_owned_file_refused(self):
        real=candidate.os.getuid()
        with patch.object(candidate.os,'getuid',return_value=real+1):self.assertIsNone(self.reuse())

    def test_file_changes_while_reading_is_refused(self):
        real=candidate.os.fstat;count=0
        def changed(fd):
            nonlocal count
            s=real(fd);count+=1
            if count%2==0:
                return SimpleNamespace(st_size=s.st_size,st_mtime_ns=s.st_mtime_ns+1)
            return s
        with patch.object(candidate.os,'fstat',side_effect=changed):self.assertIsNone(self.reuse())

    def test_wrong_reviewed_producer_is_cache_miss(self):
        self.policy['source_admission_sha256']='9'*64;self.assertIsNone(self.reuse())

    def test_hour_rollover_does_not_accept_previous_hour(self):
        times=iter([3700,7200])
        self.assertIsNone(self.reuse(now=lambda:next(times)));self.assertFalse(self.output.exists())

    def test_atomic_failure_keeps_previous_output(self):
        self.output.write_bytes(b'prior')
        with patch.object(candidate.os,'replace',side_effect=OSError('fixture')):
            self.assertIsNone(self.reuse())
        self.assertEqual(self.output.read_bytes(),b'prior')
        self.assertEqual(list(self.root.glob('learner.json.*')),[])

    def selection_fixture(self):
        self.reuse();assessment=json.loads(self.output.read_bytes())
        manifest=dict(epoch='future',checkpoint={'id':'d'*64},source_bundle={'sha256':'f'*64},
            start=3650,deadline=3700,learner_blacklist_selection_round=37)
        manifest[selection.FIELD]=self.sign(dict(version=selection.VERSION,checkpoint='d'*64,
            source_sha256='f'*64,target_round=37,maximum_age_seconds=3600,assessment_document=assessment,
            writer_policy_sha256='e'*64,audit_policy=self.audit))
        objects=[dict(sha256=str(i)*64,size=10,learner_admission=self.sign(dict(epoch='future',
            checkpoint='d'*64,miner_identity=m,document_sha256=str(i)*64))) for i,m in enumerate((self.bad,self.good,self.new))]
        controller=SimpleNamespace(state=self.root,authority=SimpleNamespace(id=self.authority))
        return manifest,objects,controller

    def test_actual_learner_excludes_blacklisted_not_unknown_zero_or_new(self):
        manifest,objects,controller=self.selection_fixture();before=copy.deepcopy(objects)
        with patch('time.time',return_value=3800):
            selected,report=learner.select_training_documents(controller,manifest,objects,{},round_number=37)
        self.assertEqual(selected,objects[1:]);self.assertEqual(objects,before)
        self.assertEqual(report['eligible_count'],3);self.assertEqual(report['blacklist_excluded_count'],1)
        original=copy.deepcopy(manifest)
        self.cache.unlink();self.output.unlink()
        with patch('time.time',return_value=100000):
            again=learner.select_training_documents(controller,manifest,objects,{},round_number=37)
        self.assertEqual(again,(selected,report));self.assertEqual(manifest,original)

    def test_actual_expiry_at_target_round_unchanged(self):
        manifest,objects,_=self.selection_fixture();p=manifest[selection.FIELD]['payload'];p['target_round']=40
        manifest[selection.FIELD]=self.sign(p)
        kept,_=selection.partition(objects,manifest,self.authority,at=3800,round_number=40)
        self.assertEqual(kept,objects)

    def main_fixture(self):
        policyfile=self.root/'producer.json';policyfile.write_bytes(canonical(self.sign(self.policy)))
        seed=self.root/'FIXTURE-ONLY.seed';seed.write_text(self.key.encode().hex())
        return ['snapshot','--runtime',str(Path(learner.__file__).resolve().parents[1]),
            '--writer-policy',str(policyfile),'--authority-seed',str(seed),'--output',str(self.output)]

    def test_main_fast_path_has_zero_evidence_calls(self):
        evidence_module=ModuleType('ops.current_assessment_evidence');evidence_module.load_evidence=lambda *a,**kw:None
        with patch.dict(candidate.REVIEWED_REUSE,self.review,clear=True),patch.object(sys,'argv',self.main_fixture()),\
                patch.dict(sys.modules,{'ops.current_assessment_evidence':evidence_module,'subnet.current_assessment':calculation}),\
                patch('time.time',return_value=3700),patch.object(evidence_module,'load_evidence') as load,\
                contextlib.redirect_stdout(io.StringIO()):
            candidate.main();load.assert_not_called()
        self.assertEqual(self.output.read_bytes(),self.cache.read_bytes())

    def test_main_miss_uses_original_calculation_and_G_signature(self):
        self.assessment['assessment_stale']=True;self.write_cache()
        evidence=dict(snapshots=[],committed_at_by_epoch={},evidence_hashes={})
        evidence_module=ModuleType('ops.current_assessment_evidence');evidence_module.load_evidence=lambda *a,**kw:None
        with patch.dict(candidate.REVIEWED_REUSE,self.review,clear=True),patch.object(sys,'argv',self.main_fixture()),\
                patch.dict(sys.modules,{'ops.current_assessment_evidence':evidence_module,'subnet.current_assessment':calculation}),\
                patch('time.time',return_value=3700),patch.object(evidence_module,'load_evidence',return_value=evidence) as load,\
                contextlib.redirect_stdout(io.StringIO()):
            candidate.main();load.assert_called_once()
        result=signed(json.loads(self.output.read_bytes()),self.authority)
        self.assertEqual(result['writer_policy_sha256'],self.review['producer_policy_sha256'])
        self.assertFalse(result['assessment_stale']);self.assertEqual(result['cutoff'],3600)

    def test_original_refresh_failure_preserves_output_and_propagates(self):
        self.output.write_bytes(b'original artifact');self.cache.unlink()
        evidence_module=ModuleType('ops.current_assessment_evidence');evidence_module.load_evidence=lambda *a,**kw:None
        with patch.dict(candidate.REVIEWED_REUSE,self.review,clear=True),patch.object(sys,'argv',self.main_fixture()),\
                patch.dict(sys.modules,{'ops.current_assessment_evidence':evidence_module,'subnet.current_assessment':calculation}),\
                patch('time.time',return_value=3700),patch.object(evidence_module,'load_evidence',side_effect=RuntimeError('fixture outage')):
            with self.assertRaisesRegex(RuntimeError,'fixture outage'):candidate.main()
        self.assertEqual(self.output.read_bytes(),b'original artifact')


if __name__=='__main__':unittest.main()
