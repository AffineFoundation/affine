"""Synthetic CPU controls. No native grading, dataset, network or production key."""
import ast
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ops import native_training_outcome_filter as new
from ops import native_task_representative_wave as wave
from nacl.signing import SigningKey
from subnet.distributed_roles import authenticate
old = new  # Compare the supported16 and explicit32 signed worker configurations.

KEY = SigningKey(hashlib.sha256(b'workers32-synthetic-test-authority').digest())
AUTH = KEY.verify_key.encode().hex()


def signed(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return dict(payload=copy.deepcopy(payload), signer=AUTH,
                signature=base64.b64encode(KEY.sign(raw).signature).decode())


def policy(workers=32, **changes):
    row = dict(version=new.MULTI_VERSION, workers=workers, max_pairs=2048,
               per_grade_seconds=30, wall_seconds=600, max_reply_bytes=262144,
               terminal_rule='max-or-eos-v1')
    row.update(changes)
    return row


def pairs(count=64):
    rows = []
    for i in range(count):
        definition = dict(env_id='synthetic', spec={}, number=i)
        def rollout(positive):
            return dict(classification='positive' if positive else 'negative',
                        task_hash=str(i), turns=[dict(output=[100+i*2+int(positive), 99], text='fixture')])
        rows.append((definition, rollout(True), rollout(False)))
    return rows


def resolve(definition, positive, negative):
    return 'fixture-gold', str(definition['number']), 2, {99}, 100000


def decode(tokens):
    value = tokens[0]-100
    return str(value//2)+('p' if value % 2 else 'n')


def normalize(receipt):
    result = copy.deepcopy(receipt)
    for name in ('policy_sha256', 'elapsed_seconds', 'prepare_seconds', 'grading_parallel_wall_seconds'):
        result.pop(name)
    for row in result['rows']:
        for grade in row['grades']:
            grade.pop('native_elapsed_seconds', None)
    return result


def synthetic_grade(gold, reply, timeout):
    i = int(reply[:-1])
    if i % 13 == 0:
        return None, 'native_timeout'
    if i % 17 == 0:
        return None, 'native_argument_limit'
    score = int(reply.endswith('p'))
    return (1-score if i % 19 == 0 else score), None


def memory_request(workers=32, sizes=(1000, 2000)):
    manifest = signed(dict(K=4, L=4, max_batches=9,
        training_input_policy='committed-unaudited-training-v1',
        training_policy='bf16-cpu-fp32-master-task-normalized-persistent-v4',
        training_task_capacity=dict(version='signed-training-task-capacity-v1', max_tasks=512)))
    authorization = signed(dict(limits=policy(workers, max_pairs='manifest')))
    context = signed(dict(original_signed_manifest=manifest,
                          authorization_sha256=new.digest(authorization),
                          submissions=[dict(size=n) for n in sizes]))
    return dict(context=context, authorization=authorization, authority=AUTH,
                paths=['synthetic']*len(sizes), source_root='/synthetic',
                tokenizer_root='/synthetic', interpreter=sys.executable)


class Workers32Tests(unittest.TestCase):


    def test_signed_worker_range(self):
        for value in (1, 16, 32):
            self.assertEqual(new.validate_limits(policy(value))['workers'], value)
        for value in (0, -1, 33, 64, True, False, 32.0, '32', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'workers'):
                new.validate_limits(policy(value))

    def test_other_limits_and_schema_unchanged(self):
        for changes in ({'wall_seconds':601}, {'per_grade_seconds':61}, {'max_reply_bytes':262145},
                        {'max_pairs':16385}, {'max_pairs':0}, {'version':'future'},
                        {'terminal_rule':'other'}, {'unknown':1}):
            with self.subTest(changes=changes):
                for module in (old, new):
                    with self.assertRaises(ValueError):
                        module.validate_limits(policy(16, **changes))
        self.assertEqual(old.validate_limits(policy(16)), new.validate_limits(policy(16)))

    def test_manifest_pair_budget_unchanged(self):
        request = memory_request()
        manifest = authenticate(request['context']['payload']['original_signed_manifest'], AUTH)
        self.assertEqual(new.validate_limits(policy(max_pairs='manifest'), manifest=manifest)['max_pairs'], 2048)

    def test_1024_pair_order_scores_failures_and_atomic_documents_equal(self):
        population = pairs(1024)
        results = []
        for module, workers in ((old, 16), (new, 32)):
            accepted, receipt = module._filter_admitted_pairs(population, policy(workers), resolve, decode, synthetic_grade)
            decisions = []
            for start in range(0, len(population), 4):
                rows = receipt['rows'][start:start+4]
                decisions.append(dict(pair_sha256=[row['pair_sha256'] for row in rows],
                                      accepted=all(row['status']=='accepted_native_labels' for row in rows)))
            whole = module._complete_document_pairs(population, decisions, required_pairs=4)
            results.append((accepted, normalize(receipt), whole, decisions))
        self.assertEqual(results[0], results[1])

    def test_completion_order_does_not_change_rows(self):
        population = pairs(40)
        completed = []
        def grader(gold, reply, timeout):
            time.sleep(.012 if reply.startswith('0') else .0001)
            completed.append(reply)
            return synthetic_grade(gold, reply, timeout)
        a = old._filter_admitted_pairs(population, policy(16), resolve, decode, synthetic_grade)
        b = new._filter_admitted_pairs(population, policy(32), resolve, decode, grader)
        self.assertEqual(a[0], b[0])
        self.assertEqual(normalize(a[1]), normalize(b[1]))
        self.assertNotEqual(completed[0], '0p')

    def test_exact_32_concurrent_grades_and_complete_drain(self):
        barrier = threading.Barrier(32, timeout=5)
        lock = threading.Lock()
        active = peak = count = 0
        def grader(gold, reply, timeout):
            nonlocal active, peak, count
            with lock:
                active += 1
                count += 1
                peak = max(peak, active)
            try:
                barrier.wait()
                return int(reply.endswith('p')), None
            finally:
                with lock:
                    active -= 1
        accepted, receipt = new._filter_admitted_pairs(pairs(64), policy(32), resolve, decode, grader)
        self.assertEqual((active, peak, count, len(accepted)), (0, 32, 128, 64))

    def test_grade_exception_drains_running_workers(self):
        lock = threading.Lock()
        active = 0
        def grader(gold, reply, timeout):
            nonlocal active
            with lock:
                active += 1
            try:
                if reply == '0p':
                    raise ValueError('synthetic dependency drift')
                time.sleep(.002)
                return int(reply.endswith('p')), None
            finally:
                with lock:
                    active -= 1
        with self.assertRaisesRegex(ValueError, 'synthetic dependency drift'):
            new._filter_admitted_pairs(pairs(64), policy(), resolve, decode, grader)
        self.assertEqual(active, 0)

    def test_all_pairs_prevalidated_before_grading(self):
        population = pairs()
        population[-1][1]['turns'][0]['output'] = [True, 99]
        grader = unittest.mock.Mock(side_effect=AssertionError('should not grade'))
        for module, workers in ((old,16), (new,32)):
            with self.assertRaisesRegex(ValueError, 'output token bounds'):
                module._filter_admitted_pairs(population, policy(workers), resolve, decode, grader)
        grader.assert_not_called()

    def test_terminal_exclusion_never_grades_either_rollout(self):
        population = pairs(1)
        population[0][1]['turns'][0]['output'] = [99,99]
        grader = unittest.mock.Mock(side_effect=AssertionError('should not grade'))
        a = new._filter_admitted_pairs(population, policy(), resolve, decode, grader)
        self.assertEqual(a[0], [])
        self.assertEqual(a[1]['rows'][0]['status'], 'excluded_terminal_rule')
        grader.assert_not_called()

    def test_same_expired_budget_all_indeterminate(self):
        grader = unittest.mock.Mock(side_effect=AssertionError('should not grade'))
        outputs = []
        for module, workers in ((old,16), (new,32)):
            calls = 0
            def clock():
                nonlocal calls
                calls += 1
                return 0 if calls==1 else 601
            outputs.append(module._filter_admitted_pairs(pairs(), policy(workers), resolve, decode, grader, clock=clock))
        self.assertEqual(outputs[0][0], [])
        self.assertEqual(normalize(outputs[0][1]), normalize(outputs[1][1]))
        self.assertTrue(all(g['reason']=='filter_deadline' for row in outputs[1][1]['rows'] for g in row['grades']))
        grader.assert_not_called()

    def test_per_grade_timeout_stays_30(self):
        observed = []
        def grader(gold, reply, timeout):
            observed.append(timeout)
            return int(reply.endswith('p')), None
        new._filter_admitted_pairs(pairs(), policy(), resolve, decode, grader, clock=lambda:0)
        self.assertEqual(set(observed), {30})

    def test_memory_reserves_authenticated_child_count(self):
        with patch.dict(sys.modules, {'ops.native_training_outcome_filter':new}):
            for workers in (1,16,32):
                self.assertEqual(wave.required_memory_bytes(memory_request(workers)),
                                 64*3000+512*1024**2+workers*1073741824)

    def test_memory_tamper_rejected(self):
        for target in ('authorization','context','manifest'):
            request = memory_request()
            if target=='manifest':
                request['context']['payload']['original_signed_manifest']['payload']['K']=8
                request['context']=signed(request['context']['payload'])
            else:
                request[target]['payload']['tamper']=True
            with patch.dict(sys.modules, {'ops.native_training_outcome_filter':new}), self.assertRaises(Exception):
                wave.required_memory_bytes(request)

    def test_memory_wrong_authorization_binding_rejected(self):
        request = memory_request()
        request['authorization'] = signed(dict(limits=policy(16, max_pairs='manifest')))
        with patch.dict(sys.modules, {'ops.native_training_outcome_filter':new}), self.assertRaisesRegex(ValueError, 'authorization binding'):
            wave.required_memory_bytes(request)

    def test_memory_invalid_signed_limits_rejected(self):
        for workers in (0,33,True,'32'):
            with self.subTest(workers=workers), patch.dict(sys.modules, {'ops.native_training_outcome_filter':new}), self.assertRaises(ValueError):
                wave.required_memory_bytes(memory_request(workers))

    def test_memory_admission_precedes_filter_or_output(self):
        request = memory_request()
        eligibility = SimpleNamespace(_load=lambda _:json.dumps(request), _create=unittest.mock.Mock())
        import subnet.persistent_training_state as state
        grader = unittest.mock.Mock()
        fake_filter = SimpleNamespace(digest=new.digest, validate_limits=new.validate_limits,
                                      filter_eligibility_context=grader)
        with patch.dict(sys.modules, {'ops.native_training_outcome_filter':fake_filter,
                                      'ops.native_training_eligibility':eligibility}), \
             patch.object(state,'available_ram_bytes',return_value=1), \
             patch.object(sys,'argv',['fixture','--request','unused','--output','unused']), \
             self.assertRaisesRegex(ValueError,'decode and grader memory admission'):
            wave.main()
        grader.assert_not_called()
        eligibility._create.assert_not_called()

    def test_memory_admission_boundary_keeps_same_filter_arguments(self):
        request = memory_request()
        eligibility = SimpleNamespace(_load=lambda _:json.dumps(request), _create=unittest.mock.Mock())
        import subnet.persistent_training_state as state
        grader = unittest.mock.Mock(return_value=([],{'synthetic':True}))
        fake_filter = SimpleNamespace(digest=new.digest, validate_limits=new.validate_limits,
                                      filter_eligibility_context=grader)
        required=64*3000+512*1024**2+32*1073741824
        with patch.dict(sys.modules, {'ops.native_training_outcome_filter':fake_filter,
                                      'ops.native_training_eligibility':eligibility}), \
             patch.object(state,'available_ram_bytes',return_value=required), \
             patch.object(sys,'argv',['fixture','--request','unused','--output','unused']):
            wave.main()
        grader.assert_called_once_with(request['paths'],request['context'],request['authorization'],
                                       AUTH,request['source_root'],request['tokenizer_root'],request['interpreter'])
        eligibility._create.assert_called_once_with('unused',{'synthetic':True})


class IsolatedGraderControls(unittest.TestCase):
    """Exercise original PinnedGrader with subprocess boundary mocked, never verify.py."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        script=root/'synthetic.py';script.write_text('synthetic')
        self.grader=new.PinnedGrader.__new__(new.PinnedGrader)
        self.grader.executable=Path(sys.executable).resolve()
        self.grader.executable_stamp=new._stamp(self.grader.executable.lstat())
        self.grader.script=script;self.grader.stamp=new._stamp(script.lstat())
        self.grader.argv=[str(self.grader.executable),'-I','-S','-c','pass','synthetic-site',str(script)]

    def test_argv_resources_and_exact_output(self):
        with patch.object(new.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=b'1.0\n')) as run:
            self.assertEqual(self.grader('a','b',30),(1,None))
        args,kwargs=run.call_args
        self.assertEqual(args[0][1:4],['-I','-S','-c'])
        self.assertIn('RLIMIT_AS,(1073741824,1073741824)',args[0][4])
        self.assertIn('RLIMIT_CPU,(30,31)',args[0][4])
        self.assertEqual(kwargs,dict(stdin=subprocess.DEVNULL,capture_output=True,timeout=30))
        for raw,expected in ((b'0.0',(0,None)),(b'1',(None,'native_invalid_output')),
                             (b'nan',(None,'native_invalid_output')),(b'1.00',(None,'native_invalid_output'))):
            with patch.object(new.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=raw)):
                self.assertEqual(self.grader('a','b',30),expected)

    def test_timeout_exit_spawn_and_argument_unknown_semantics(self):
        cases=((subprocess.TimeoutExpired('synthetic',1),'native_timeout'),(OSError('synthetic'),'native_spawn_unavailable'))
        for error,reason in cases:
            with patch.object(new.subprocess,'run',side_effect=error):
                self.assertEqual(self.grader('a','b',1),(None,reason))
        with patch.object(new.subprocess,'run',return_value=SimpleNamespace(returncode=7,stdout=b'0.0')):
            self.assertEqual(self.grader('a','b',1),(None,'native_exit_7'))
        with patch.object(new.subprocess,'run',side_effect=AssertionError('argument must be rejected')):
            self.assertEqual(self.grader('a','x'*100000,1),(None,'native_argument_limit'))

    def test_script_drift_prevents_subprocess(self):
        self.grader.script.write_text('changed')
        with patch.object(new.subprocess,'run',side_effect=AssertionError('drift must be rejected')), self.assertRaisesRegex(ValueError,'source drift'):
            self.grader('a','b',1)


if __name__=='__main__':
    with patch.object(socket.socket,'connect',side_effect=AssertionError('network forbidden')):
        unittest.main(verbosity=2)
