import importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import nullcontext
from nacl.signing import SigningKey
from ops import current_assessment_writer as w
from ops.live_reward_exporter import sign


class WriterControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name); self.key = SigningKey.generate()
        self.auth = self.key.verify_key.encode().hex(); seed = self.path/'seed'
        seed.write_text(self.key.encode().hex()); seed.chmod(0o600)
        self.c = dict(global_lock_path=str(self.path/'lock'), reward_state=str(self.path),
                      chain_state=str(self.path), authority_seed_file=str(seed))
        self.cutover = sign({'original':True},self.key); self.anchor=sign({'original':True},self.key)
        modules = ('ops.current_assessment_writer','subnet.numerical_resolution','subnet.continuous_audit_policy','ops.current_assessment_evidence','subnet.current_assessment','subnet.chain','ops.live_reward_writer','subnet.weight_submission_transaction','subnet.weight_submission_reconciliation')
        pins = {str(Path(importlib.util.find_spec(m).origin).resolve()): None for m in modules}
        pins = {p:w.file_hash(p) for p in set(pins) | w.sdk_seam_paths()}
        self.policy = sign(dict(version=w.NEVER_BURN_VERSION, half_life_hours=6, first_window=3600,
            netuid=120, owner_hotkey=w.OWNER,audit_config='/not-a-training-state',
            source_admission_sha256='a'*64, verifiers=['v'],module_hashes=pins,numerical_resolution_policy_sha256='b'*64,fallback_assessments=[],
            cutover_sha256=w.sha(self.cutover),anchor_sha256=w.sha(self.anchor),execute_enabled=True,
            zero_total_policy='no-owner-retain-v1',
            registration_change_policy='current-hotkey-snapshot-v1'),self.key)
        self.calls=[]
        controls=self
        class Adapter:
            def __init__(self,*a,**k):pass
            def registrations(self):return {'hotkey':{'public_key':'miner','uid':85}}
            def submit_hour(self,points,regs,end,execute=False,**kwargs):
                controls.calls.append((points,regs,end,execute))
                return dict(status=controls.status, window_end=end)
        self.adapter=Adapter;self.status='planned'
        self.evidence=dict(snapshots=[dict(epoch='failed-original-training',round=13,cutoff=7200,
            miners={'miner':dict(unique_eligible_batches=3,validity_probability=.5,reward_multiplier=1.)})],
            committed_at_by_epoch={'failed-original-training':7000},evidence_hashes={'source_admission_sha256':'a'*64,'numerical_resolution_policy_sha256':'b'*64})
    def invoke(self,result=None,*,execute=False,now=7300):
        def load(*a,**k):
            if isinstance(result,Exception):raise result
            return result or self.evidence
        with patch.object(w,'authenticate_cutover',return_value=(self.c,{})), \
             patch.object(w,'global_lock',return_value=nullcontext()), \
             patch.object(w,'guard_files'),patch.object(w,'observe_units',return_value=[]), \
             patch.object(w,'writer_gate'),patch.object(w.time,'time',return_value=now):
            return w.run_once(self.policy,self.cutover,self.anchor,self.auth,execute=execute,
                              adapter_factory=self.adapter,evidence_loader=load)
    def test_no_training_or_opening_files_required(self):
        self.invoke();self.assertGreater(self.calls[0][0]['hotkey'],0)
        self.assertEqual(self.calls[0][1]['hotkey']['uid'],85)
    def test_hour_retry_uses_immutable_assessment(self):
        self.status='deferred_rate_limit';self.invoke(execute=True)
        self.invoke(ValueError('training still broken'),execute=True)
        self.assertEqual(self.calls[0],self.calls[1])
    def test_outage_next_hour_uses_last_valid(self):
        self.invoke();self.invoke(TimeoutError(),now=11000)
        self.assertEqual(self.calls[0][0],self.calls[1][0])
        health=json.loads((self.path/w.ASSESSMENT_DIRECTORY/'last-run.json').read_text())
        self.assertTrue(health['assessment_stale'])
    def test_no_initial_evidence_fails_without_transaction(self):
        self.assertEqual(self.invoke(TimeoutError())['status'],'no_valid_assessment')
        self.assertEqual(self.calls,[])
    def test_invalid_fresh_evidence_uses_only_authenticated_history(self):
        self.invoke()
        self.invoke(ValueError('invalid signature'),now=11000)
        self.assertEqual(len(self.calls),2)
        self.assertEqual(self.calls[0][0],self.calls[1][0])
    def test_successful_hour_is_not_resubmitted(self):
        (self.path/'weights.json').write_text(json.dumps({'last_submitted_window':7200}))
        self.assertEqual(self.invoke()['status'],'already_submitted');self.assertEqual(self.calls,[])
    def test_uncertain_submission_cannot_blind_retry(self):
        directory=self.path/'current-assessment-v1';directory.mkdir()
        (directory/'submission.json').write_text(json.dumps({'status':'submitting'}))
        with self.assertRaisesRegex(RuntimeError,'uncertain'):self.invoke(execute=True)
        self.assertEqual(self.calls,[])
    def test_tampered_policy_and_execution_disabled_refuse(self):
        self.policy['payload']['half_life_hours']=1
        with self.assertRaises(Exception):self.invoke()
        body=dict(self.policy['payload'],half_life_hours=6,execute_enabled=False)
        self.policy=sign(body,self.key)
        with self.assertRaises(ValueError):self.invoke(execute=True)




class WriterRecoveryIntegration(unittest.TestCase):
    setUp = WriterControls.setUp
    invoke = WriterControls.invoke
    def test_preflight_exception_no_longer_sets_submitting(self):
        self.adapter.submit_hour=lambda *a,**k:(_ for _ in ()).throw(TimeoutError('preflight'))
        with self.assertRaises(TimeoutError):self.invoke(execute=True)
        self.assertFalse((self.path/'current-assessment-v1'/'submission.json').exists())
    def _pending(self,window):
        d=self.path/'current-assessment-v1';d.mkdir()
        (d/'submission.json').write_text(json.dumps(dict(status='submitting',window_end=window,attempt_directory='/fixture')))
        self.adapter.chain=type('Chain',(),{'close':lambda _:None})()
    def test_pending_recovery_never_calls_submit(self):
        self._pending(3600)
        with patch.object(w.SubmissionJournal,'recover',return_value=dict(preserve_fence=True,reason='incomplete_read')):
            result=self.invoke(execute=True)
        self.assertEqual(result['status'],'reconciliation_pending');self.assertEqual(self.calls,[])
    def test_resolved_old_window_only_submits_current_hour(self):
        self._pending(3600)
        def resolved(*args,**kw):
            (self.path/'weights.json').write_text(json.dumps({'last_submitted_window':3600}))
            return dict(preserve_fence=False,status='submitted_finalized')
        with patch.object(w.SubmissionJournal,'recover',side_effect=resolved):
            self.invoke(execute=True)
        self.assertEqual([call[2] for call in self.calls],[7200])
    def test_resolved_current_hour_returns_without_submit(self):
        self._pending(7200)
        def resolved(*args,**kw):
            (self.path/'weights.json').write_text(json.dumps({'last_submitted_window':7200}))
            return dict(preserve_fence=False,status='submitted_finalized')
        with patch.object(w.SubmissionJournal,'recover',side_effect=resolved):
            result=self.invoke(execute=True)
        self.assertEqual(result['status'],'already_submitted');self.assertTrue(result['recovered']);self.assertEqual(self.calls,[])

if __name__=='__main__':unittest.main()
