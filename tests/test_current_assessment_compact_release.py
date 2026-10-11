"""Compact reader activation preserves one cursor and authentic last-best weights."""
import copy,hashlib,json,unittest
from pathlib import Path
from unittest.mock import patch
from ops import current_assessment_writer as w,current_assessment_compact_queue as compact
from ops.live_reward_exporter import sign
import test_current_assessment_writer as fixture

class CompactReleaseControls(unittest.TestCase):
    invoke = fixture.WriterControls.invoke
    def setUp(self):
        fixture.WriterControls.setUp(self)
        p=copy.deepcopy(self.policy['payload'])
        p['compact_queue_evidence']=compact.POLICY.copy()
        p['module_hashes'][str(Path(compact.__file__).resolve())]=w.file_hash(compact.__file__)
        self.policy=sign(p,self.key)

    def adopt(self):
        old = self.policy
        directory = self.path/w.ASSESSMENT_DIRECTORY
        rows = []
        seen = set()
        for name in ('assessment-7200.json', 'last-valid-assessment.json', 'last-positive-assessment.json'):
            raw = (directory/name).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            target = self.path/(digest+'.json'); target.write_bytes(raw)
            rows.append(dict(path=str(target), sha256=digest, writer_policy_sha256=w.sha(old)))
        body = copy.deepcopy(old['payload'])
        body['fallback_assessments'].extend(rows)
        self.policy = sign(body, self.key)
        return old, rows

    def test_old_h5_last_best_preserved_on_new_reader_timeout(self):
        self.invoke(); old, rows = self.adopt()
        self.assertEqual(len(rows), 1)
        result = self.invoke(TimeoutError('new reader unavailable'), now=11000)
        self.assertEqual(result['status'], 'planned')
        self.assertEqual(self.calls[0][0], self.calls[1][0])
        current = json.loads((self.path/w.ASSESSMENT_DIRECTORY/'assessment-10800.json').read_bytes())['payload']
        self.assertTrue(current['assessment_stale'])
        self.assertEqual(current['evidence_cutoff'], 7200)
        self.assertEqual(current['writer_policy_sha256'], w.sha(self.policy))
        self.assertNotEqual(w.sha(old), w.sha(self.policy))
        self.assertNotIn(w.OWNER, self.calls[-1][0])

    def test_old_same_hour_reused_without_new_evidence_or_bytes_change(self):
        self.invoke(); self.adopt()
        target = self.path/w.ASSESSMENT_DIRECTORY/'assessment-7200.json'
        raw = target.read_bytes()
        self.invoke(AssertionError('must never recompute same hour'))
        self.assertEqual(target.read_bytes(), raw)
        self.assertEqual(self.calls[0], self.calls[1])

    def test_later_blacklist_constraints_never_resurrect_old_positive(self):
        self.invoke(); self.adopt()
        prior = json.loads((self.path/w.ASSESSMENT_DIRECTORY/'last-valid-assessment.json').read_bytes())['payload']
        penalty = copy.deepcopy(prior)
        penalty.update(cutoff=10800, evidence_cutoff=10800)
        penalty['points']['miner'] = 0
        penalty['miner_estimates']['miner']['blacklisted'] = True
        penalty['miner_estimates']['miner']['reward_multiplier'] = 0
        penalty['writer_policy_sha256'] = w.sha(self.policy)
        (self.path/w.ASSESSMENT_DIRECTORY/'last-valid-assessment.json').write_text(json.dumps(sign(penalty, self.key)))
        result = self.invoke(TimeoutError(), now=14600)
        self.assertEqual(result['status'], 'no_valid_registered_recipients')
        self.assertTrue(result['retained_onchain_weights'])
        self.assertEqual(len(self.calls), 1)

    def test_latest_old_zero_penalty_and_old_positive_both_survive_handoff(self):
        self.invoke()
        directory = self.path/w.ASSESSMENT_DIRECTORY
        old = self.policy
        positive = json.loads((directory/'last-positive-assessment.json').read_bytes())['payload']
        zero = copy.deepcopy(positive); zero.update(cutoff=10800, evidence_cutoff=10800)
        zero['points']['miner'] = 0
        zero['miner_estimates']['miner']['reward_multiplier'] = 0
        (directory/'last-valid-assessment.json').write_text(json.dumps(sign(zero, self.key)))
        _, rows = self.adopt()
        self.assertEqual(len(rows), 2)
        result = self.invoke(TimeoutError(), now=14600)
        self.assertEqual(result['status'], 'no_valid_registered_recipients')
        self.assertEqual(len(self.calls), 1)

    def test_bad_history_no_valid_assessment_retains_chain_without_adapter(self):
        self.invoke(); _, rows = self.adopt()
        Path(rows[0]['path']).write_text('invalid')
        self.calls.clear()
        result = self.invoke(TimeoutError(), now=11000)
        self.assertEqual(result['status'], 'no_valid_assessment')
        self.assertTrue(result['retained_onchain_weights'])
        self.assertFalse(result['chain_executed'])
        self.assertEqual(self.calls, [])

    def test_empty_current_registration_never_substitutes_owner(self):
        self.adapter.registrations = lambda _: {}
        result = self.invoke()
        self.assertEqual(result['status'], 'no_valid_registered_recipients')
        self.assertTrue(result['retained_onchain_weights'])
        self.assertEqual(self.calls, [])

    def test_old_uncertain_attempt_keeps_original_cursor_and_no_evidence(self):
        self.invoke(); self.adopt(); self.calls.clear()
        cursor = self.pending(3600); before = cursor.read_bytes()
        with patch.object(w.SubmissionJournal, 'recover', return_value=dict(preserve_fence=True, reason='incomplete_read')) as recover:
            result = self.invoke(AssertionError('evidence must not run'), execute=True, now=11000)
        self.assertEqual(result['status'], 'reconciliation_pending')
        self.assertEqual(cursor.read_bytes(), before)
        self.assertEqual(recover.call_count, 1)
        self.assertEqual(self.calls, [])

    def test_resolved_old_attempt_only_submits_current_hour_using_same_cursor(self):
        self.invoke(); self.adopt(); self.calls.clear()
        cursor = self.pending(3600)
        def resolved(*args, **kwargs):
            (self.path/'weights.json').write_text(json.dumps(dict(last_submitted_window=3600)))
            return dict(preserve_fence=False, status='submitted_finalized')
        with patch.object(w.SubmissionJournal, 'recover', side_effect=resolved):
            result = self.invoke(TimeoutError(), execute=True, now=11000)
        self.assertEqual([row[2] for row in self.calls], [10800])
        self.assertEqual(json.loads(cursor.read_bytes())['window_end'], 10800)

    def test_resolved_current_hour_never_replays_same_hour(self):
        self.invoke(); self.adopt(); self.calls.clear(); self.pending(7200)
        def resolved(*args, **kwargs):
            (self.path/'weights.json').write_text(json.dumps(dict(last_submitted_window=7200)))
            return dict(preserve_fence=False, status='submitted_finalized')
        with patch.object(w.SubmissionJournal, 'recover', side_effect=resolved):
            result = self.invoke(AssertionError('must not read'), execute=True)
        self.assertTrue(result['recovered']); self.assertEqual(result['status'], 'already_submitted')
        self.assertEqual(self.calls, [])

    def pending(self, window):
        path = self.path/'current-assessment-v1/submission.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(dict(status='submitting', window_end=window, attempt_directory='/fixture')))
        self.adapter.chain = type('Chain', (), {'close': lambda _: None})()
        return path

if __name__=="__main__":unittest.main()
