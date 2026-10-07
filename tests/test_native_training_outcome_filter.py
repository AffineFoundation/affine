import copy
import hashlib
import tempfile
import threading
import time
import unittest
from pathlib import Path
from ops.native_training_outcome_filter import (VERSION, _filter_admitted_pairs, _read, validate_limits)


class NativeLabels(unittest.TestCase):
    def setUp(self):
        self.policy=dict(version=VERSION,workers=4,max_pairs=256,per_grade_seconds=2,wall_seconds=10,max_reply_bytes=1024)
        self.definition={'env_id':'math','index':7}
        self.p={'classification':'positive','task_hash':'trusted','turns':[{'output':[1],'text':'forged'}]}
        self.n={'classification':'negative','task_hash':'trusted','turns':[{'output':[2],'text':'forged'}]}
        self.pair=[self.definition,self.p,self.n]
    def run_filter(self,pairs=None,grader=None,resolve=None,decode=None,clock=time.monotonic):
        return _filter_admitted_pairs(pairs or [self.pair],self.policy,
            resolve or (lambda *args:('42','trusted',3,{0},10)),
            decode or (lambda t:'correct' if t==[1] else 'wrong'),
            grader or (lambda g,r,t:(int(r=='correct'),None)),clock=clock)
    def test_tokens_graded_not_forged_text_original_unchanged(self):
        before=copy.deepcopy(self.pair);accepted,receipt=self.run_filter()
        self.assertEqual(accepted,[self.pair]);self.assertEqual(before,self.pair)
        self.assertFalse(receipt['proof_verification_performed']);self.assertEqual(receipt['sampling_assurance'],'unaudited')
        self.assertFalse(receipt['cheating_penalties']);self.assertFalse(receipt['claims_rewritten'])
        self.assertFalse(receipt['rows'][0]['grades'][0]['submitted_text_matches_decoded'])
    def test_reversed_native_outcomes_exclude_pair_without_relabel(self):
        accepted,r=self.run_filter(grader=lambda g,r,t:(int(r!='correct'),None))
        self.assertEqual(accepted,[]);self.assertEqual(r['rows'][0]['status'],'excluded_label_mismatch')
        self.assertEqual(self.p['classification'],'positive')
    def test_unavailable_timeout_or_empty_not_negative(self):
        for reason in ('native_timeout','native_exit_75','native_invalid_output'):
            accepted,r=self.run_filter(grader=lambda g,r,t:(None,reason))
            self.assertEqual(accepted,[]);self.assertEqual(r['rows'][0]['status'],'excluded_indeterminate')
            self.assertIsNone(r['rows'][0]['grades'][1]['native_score'])
    def test_nan_float_bool_not_binary_native_outcome(self):
        for value in (float('nan'),1.,True,2,-1,'0.0'):
            with self.assertRaisesRegex(ValueError,'exact native binary'):self.run_filter(grader=lambda *a:(value,None))
    def test_cap_negative_is_accurate_label_not_mismatch(self):
        self.n['turns'][0]['output']=[2,2,2]
        accepted,r=self.run_filter();self.assertEqual(accepted,[self.pair])
        grade=r['rows'][0]['grades'][1];self.assertTrue(grade['non_eos_cap']);self.assertEqual(grade['native_score'],0)
    def test_eos_at_cap_distinguished(self):
        self.n['turns'][0]['output']=[2,2,0]
        _,r=self.run_filter();self.assertFalse(r['rows'][0]['grades'][1]['non_eos_cap'])
    def test_wrong_task_class_turn_tokens_refuse_before_grade(self):
        mutations=[lambda p:p.update(task_hash='foreign'),lambda p:p.update(classification='negative'),
                   lambda p:p.update(turns=[]),lambda p:p['turns'][0].update(output=[True]),
                   lambda p:p['turns'][0].update(output=[10]),lambda p:p['turns'][0].update(output=[1]*4)]
        for mutate in mutations:
            pair=copy.deepcopy(self.pair);mutate(pair[1]);calls=[]
            with self.assertRaises(ValueError):self.run_filter([pair],grader=lambda *a:calls.append(a))
            self.assertEqual(calls,[])
    def test_population_prevalidation_no_partial_grading(self):
        bad=copy.deepcopy(self.pair);bad[1]['task_hash']='other';calls=[]
        with self.assertRaises(ValueError):self.run_filter([self.pair,bad],grader=lambda *a:calls.append(a))
        self.assertEqual(calls,[])
    def test_duplicate_pair_and_oversize_population_refuse(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):self.run_filter([self.pair,self.pair])
        self.policy['max_pairs']=1
        with self.assertRaisesRegex(ValueError,'pair count'):self.run_filter([self.pair,self.pair])
    def test_reply_bytes_limit_and_decode_type(self):
        for decode in (lambda t:'x'*1025,lambda t:None):
            with self.assertRaisesRegex(ValueError,'decoded'):self.run_filter(decode=decode)
    def test_deadline_never_grades_pending_cases(self):
        ticks=iter([0,0,11,11,12,12]);calls=[]
        accepted,r=self.run_filter(grader=lambda *a:calls.append(a),clock=lambda:next(ticks))
        self.assertEqual(calls,[]);self.assertEqual(accepted,[])
        self.assertEqual(r['rows'][0]['status'],'excluded_indeterminate')
    def test_per_case_timeout_bounded(self):
        observed=[]
        def grade(g,r,t):observed.append(t);return int(r=='correct'),None
        self.run_filter(grader=grade,clock=lambda:0);self.assertEqual(observed,[2,2])
    def test_concurrency_bounded(self):
        lock=threading.Lock();live=0;maximum=0
        pairs=[]
        for i in range(8):
            pair=copy.deepcopy(self.pair);pair[0]['index']=i;pairs.append(pair)
        def grade(g,r,t):
            nonlocal live,maximum
            with lock:live+=1;maximum=max(maximum,live)
            time.sleep(.01)
            with lock:live-=1
            return int(r=='correct'),None
        self.run_filter(pairs,grader=grade);self.assertGreater(maximum,1);self.assertLessEqual(maximum,4)
    def test_limits_explicit_default_off_strict_types(self):
        with self.assertRaises(ValueError):validate_limits(None)
        for k,v in [('workers',5),('max_pairs',257),('wall_seconds',601),('per_grade_seconds',True),('version','other')]:
            policy=dict(self.policy);policy[k]=v
            with self.assertRaises(ValueError):validate_limits(policy)
    def test_trusted_file_sha_symlink_hardlink_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trusted';path.write_bytes(b'x');sha=hashlib.sha256(b'x').hexdigest()
            self.assertEqual(_read(path,sha,10)[0],b'x')
            with self.assertRaises(ValueError):_read(path,'0'*64,10)
            symlink=Path(d)/'link';symlink.symlink_to(path)
            with self.assertRaises(ValueError):_read(symlink,sha,10)
            import os;os.link(path,Path(d)/'hard')
            with self.assertRaises(ValueError):_read(path,sha,10)

class GraderProcess(unittest.TestCase):
    def test_serialized_argument_bound_ascii_and_escaped_unicode(self):
        from ops.native_training_outcome_filter import PinnedGrader
        import sys
        with tempfile.TemporaryDirectory() as d:
            script=Path(d)/'grade.py';script.write_text('print("0.0")')
            g=PinnedGrader(sys.executable,script,hashlib.sha256(script.read_bytes()).hexdigest())
            for reply in ('x'*100000,'漢'*18000):
                score,reason=g('42',reply,1)
                self.assertIsNone(score);self.assertEqual(reason,'native_argument_limit')
            self.assertEqual(g('42','short',2),(0,None))
    def test_exact_subprocess_output_and_source_drift(self):
        from ops.native_training_outcome_filter import PinnedGrader
        import sys
        with tempfile.TemporaryDirectory() as d:
            for value,expected in [('0.0',0),('1.0',1),('NaN',None),('0',None)]:
                script=Path(d)/'grade.py';script.write_text('print('+repr(value)+')')
                g=PinnedGrader(sys.executable,script,hashlib.sha256(script.read_bytes()).hexdigest())
                self.assertEqual(g('42','reply',2)[0],expected)
            script.write_text('print("1.0")')
            with self.assertRaisesRegex(ValueError,'drift'):g('42','reply',2)
    def test_subprocess_timeout_and_missing_interpreter_indeterminate(self):
        from ops.native_training_outcome_filter import PinnedGrader
        import sys
        with tempfile.TemporaryDirectory() as d:
            script=Path(d)/'grade.py';script.write_text('import time;time.sleep(2)')
            sha=hashlib.sha256(script.read_bytes()).hexdigest()
            g=PinnedGrader(sys.executable,script,sha)
            self.assertEqual(g('42','reply',.05),(None,'native_timeout'))
            from unittest.mock import patch
            with patch('ops.native_training_outcome_filter.subprocess.run',side_effect=OSError('unavailable')):
                self.assertEqual(g('42','reply',1),(None,'native_spawn_unavailable'))
            foreign=Path(d)/'bin';foreign.mkdir();binary=foreign/'python';binary.write_text('foreign interpreter')
            with self.assertRaisesRegex(ValueError,'executable SHA'):PinnedGrader(binary,script,sha)

if __name__=='__main__':unittest.main()
