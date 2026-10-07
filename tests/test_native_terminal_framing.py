import copy
import unittest
from ops.native_training_outcome_filter import VERSION,_filter_admitted_pairs,validate_limits

class TerminalFraming(unittest.TestCase):
    def setUp(self):
        self.policy=dict(version=VERSION,workers=2,max_pairs=256,per_grade_seconds=2,wall_seconds=10,max_reply_bytes=1024,terminal_rule='max-or-eos-v1')
        self.p={'classification':'positive','task_hash':'trusted','turns':[{'output':[1,0],'text':'forged'}]}
        self.n={'classification':'negative','task_hash':'trusted','turns':[{'output':[2,0],'text':'forged'}]}
        self.pair=[{'env_id':'math','index':7},self.p,self.n]
        self.calls=[]
    def run_filter(self,pairs=None,resolve=None):
        return _filter_admitted_pairs(pairs or [self.pair],self.policy,resolve or (lambda *args:('42','trusted',3,{0},10)),lambda t:'correct'if t[0]==1 else 'wrong',lambda g,r,t:(self.calls.append(r)or int(r=='correct'),None))
    def test_eos_completed_and_non_eos_cap_are_accepted_without_sampling_claim(self):
        for output in ([2,0],[2,2,2],[2,2,0]):
            self.n['turns'][0]['output']=output
            accepted,r=self.run_filter();self.assertEqual(accepted,[self.pair]);self.assertEqual(r['terminal_rule'],'max-or-eos-v1');self.assertFalse(r['proof_verification_performed']);self.assertEqual(r['sampling_assurance'],'unaudited')
    def test_short_non_eos_prefix_excludes_entire_pair_before_any_grader(self):
        before=copy.deepcopy(self.pair);self.n['turns'][0]['output']=[2]
        original=copy.deepcopy(self.pair);accepted,r=self.run_filter();self.assertEqual(accepted,[]);self.assertEqual(self.calls,[]);self.assertEqual(self.pair,original);self.assertEqual(r['rows'][0]['status'],'excluded_terminal_rule');self.assertFalse(r['cheating_penalties']);self.assertFalse(r['claims_rewritten']);self.assertTrue(all(g['native_score']is None for g in r['rows'][0]['grades']))
    def test_output_after_eos_is_excluded_even_at_cap(self):
        self.n['turns'][0]['output']=[2,0,2];accepted,r=self.run_filter();self.assertEqual(accepted,[]);self.assertEqual(self.calls,[])
    def test_miner_done_eos_and_budget_metadata_do_not_change_trusted_rule(self):
        self.n['turns'][0].update(output=[2],done=True,max_output_tokens=1,eos_token_id=2,finish_reason='length');self.n.update(eos_token_id=2,max_output_tokens=1)
        accepted,r=self.run_filter();self.assertEqual(accepted,[]);self.assertEqual(self.calls,[]);self.assertEqual(r['rows'][0]['grades'][1]['approved_output_cap'],3)
    def test_one_bad_pair_does_not_block_good_pair_and_does_not_grade_bad_pair(self):
        bad=copy.deepcopy(self.pair);bad[0]['index']=8;bad[2]['turns'][0]['output']=[2]
        accepted,r=self.run_filter([self.pair,bad]);self.assertEqual(accepted,[self.pair]);self.assertEqual(len(self.calls),2);self.assertEqual([x['status']for x in r['rows']],['accepted_native_labels','excluded_terminal_rule'])
    def test_invalid_trusted_eos_and_unknown_signed_rule_fail_before_grading(self):
        for eos in ({None},{True},{10},{0,1},set()):
            with self.assertRaisesRegex(ValueError,'single tokenizer EOS'):self.run_filter(resolve=lambda *a:('42','trusted',3,eos,10))
        self.assertEqual(self.calls,[])
        self.policy['terminal_rule']='miner-selected';self.assertRaises(ValueError,validate_limits,self.policy)
    def test_legacy_signed_policy_interpretation_is_preserved(self):
        del self.policy['terminal_rule'];self.n['turns'][0]['output']=[2];accepted,r=self.run_filter();self.assertEqual(accepted,[self.pair]);self.assertNotIn('terminal_rule',r)

if __name__=='__main__':unittest.main()

class TerminalSubset(unittest.TestCase):
    def fixture(self):
        from test_native_training_eligibility import SelectorIntegration
        f=SelectorIntegration();f.setUp();self.addCleanup(f.directory.cleanup)
        f.statuses=['accepted_native_labels']*2
        auth=dict(f.auth['payload'],limits={'terminal_rule':'max-or-eos-v1'})
        from ops.native_training_eligibility import NativeEligibilitySelector
        f.auth=f.sign(auth);f.selector=NativeEligibilitySelector(f.controller,f.auth,'/approved/tokenizer','/approved/bin/python')
        base=f.grade
        def grade(*args):
            accepted,r=base(*args);r['terminal_rule']='max-or-eos-v1'
            for row in r['rows']:
                for g in row['grades']:g['terminal_framing_valid']=True
            r['rows'][0]['status']='excluded_terminal_rule';r['document_decisions'][0]['accepted']=False
            for g in r['rows'][0]['grades']:g.update(native_score=None,label_matches=None,reason='terminal_framing_exclusion')
            r['rows'][0]['grades'][1]['terminal_framing_valid']=False
            return accepted,r
        f.grade=grade;return f
    def test_terminal_excluded_subset_restarts_without_grading_original_claims_unchanged(self):
        from unittest.mock import patch
        f=self.fixture();expected=f.select();self.assertEqual(expected[1],f.submissions[1:]);f.assert_originals()
        with patch('ops.native_training_eligibility.filter_eligibility_context',side_effect=AssertionError('no regrade')):
            self.assertEqual(f.selector.select(f.manifest,f.submissions),expected)
    def test_terminal_receipt_cannot_mix_native_grade_or_authorization_rule(self):
        from unittest.mock import patch
        from ops.native_training_eligibility import bind_subset
        f=self.fixture();f.select()
        import json
        root=f.state/'native-outcome-eligibility'/f.epoch
        context=json.loads((root/'context.ROOT-SIGNED.json').read_bytes());receipt=json.loads((root/'grades.ROOT-SIGNED.json').read_bytes())['payload'];receipt['rows'][0]['grades'][0].update(native_score=1,label_matches=True)
        with self.assertRaisesRegex(ValueError,'cannot be graded'):bind_subset(context,receipt,f.submissions,f.authority)
        original=json.loads((root/'grades.ROOT-SIGNED.json').read_bytes())['payload']
        del original['terminal_rule']
        (root/'grades.ROOT-SIGNED.json').write_text(json.dumps(f.sign(original)))
        with self.assertRaisesRegex(ValueError,'authorized terminal rule'):f.selector.select(f.manifest,f.submissions)
