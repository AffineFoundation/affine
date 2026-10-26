"""CPU-only TRAIN binding/admission controls; no model construction or forwards."""
import base64
import copy
from pathlib import Path
import sys
import unittest

from nacl.signing import SigningKey
from research.native_math_reuse_v2.broker import NativeInfrastructureError
from .native_adapter import NativeAdapter, bind_capability, canonical, sha


class FakeClient:
    def __init__(self):
        self.observations = [dict(stage='native_broker_ready', seconds=.2)]
        self.calls = []
        self.preflight_result = '1.0'
        self.grade_result = '1.0'
        self.closed = False

    def preflight(self, **kw):
        self.calls.append(('preflight', kw))
        self.observations.append(dict(stage='native_grade', mode='preflight', seconds=.1, child_seconds=.03))
        if isinstance(self.preflight_result, Exception):
            raise self.preflight_result
        return self.preflight_result

    def grade(self, **kw):
        self.calls.append(('grade', kw))
        self.observations.append(dict(stage='native_grade', mode='grade', seconds=.1, child_seconds=.03))
        if isinstance(self.grade_result, Exception):
            raise self.grade_result
        return self.grade_result

    def close(self):
        self.closed = True


class Tokenizer:
    def __init__(self, text):
        self.text = text
        self.tokens = None

    def decode(self, tokens, **_):
        self.tokens = tokens
        return self.text


class AdmissionControls(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.records = []
        self.guards = []
        self.task = dict(index=17, task_hash='a'*64, native_snapshot_row=dict(data=dict(answer='1')))
        self.adapter = NativeAdapter(self.client, guard=lambda:self.guards.append(True),
            record=self.records.append, final_box=lambda text:'1' if r'\boxed{' in text else None,
            eos_token_id=99)

    def test_eos_removed_before_grade_and_false_is_completed_wrong(self):
        tokenizer = Tokenizer(r'\boxed{2}')
        self.client.grade_result = '0.0'
        self.assertEqual(self.adapter.grade_completed(self.task, [1,2,99], tokenizer, update=3), 'wrong')
        self.assertEqual(tokenizer.tokens, [1,2])
        self.assertEqual(self.client.calls[-1][1], dict(index=17, task_hash='a'*64, reply=r'\boxed{2}'))
        self.assertEqual(self.records[-1]['update'], 3)
        self.assertEqual(self.records[-1]['child_seconds'], .03)

    def test_caps_and_unboxed_eos_never_grade_as_negative(self):
        for tokens, text in (([1,2], r'\boxed{2}'), ([1,2,99], 'unfinished'), ([], r'\boxed{2}')):
            with self.subTest(tokens=tokens, text=text):
                self.assertEqual(self.adapter.grade_completed(self.task, tokens, Tokenizer(text), update=0), 'unresolved')
        self.assertEqual(self.client.calls, [])

    def test_preflight_skip_preserves_whole_task_semantics(self):
        for result in ('0.0', 'reference_unparseable', '1.0'):
            self.client.preflight_result = result
            self.assertEqual(self.adapter.preflight(self.task, update=0), result)
        self.assertEqual([x[0] for x in self.client.calls], ['preflight']*3)
        self.assertEqual(self.client.calls[0][1]['answer'], '1')

    def test_infrastructure_error_is_not_wrong_and_timings_survive(self):
        self.client.grade_result = NativeInfrastructureError('timeout')
        with self.assertRaises(NativeInfrastructureError):
            self.adapter.grade_completed(self.task, [1,99], Tokenizer(r'\boxed{1}'), update=2)
        self.assertEqual(self.records[-1]['mode'], 'grade')
        self.adapter.close()
        self.assertTrue(self.client.closed)

    def test_no_result_cache_and_observation_logged_once(self):
        for _ in range(2):
            self.adapter.grade_completed(self.task, [1,99], Tokenizer(r'\boxed{1}'), update=0)
        self.assertEqual(len(self.client.calls), 2)
        self.adapter.flush()
        self.adapter.close()
        self.assertEqual(len(self.records), 3)

    def test_duplicate_eos_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            self.adapter.grade_completed(self.task, [1,99,2,99], Tokenizer(r'\boxed{1}'), update=0)
        self.assertEqual(self.client.calls, [])


class BindingControls(unittest.TestCase):
    def setUp(self):
        self.key = SigningKey.generate()
        self.authority = self.key.verify_key.encode().hex()
        self.scope = dict(workspace='/tmp/private-native-binding', run_id='research-job',
            native_environment_sha256='e'*64, source_files={'one.py':'1'*64},
            native_interpreter='/tmp/native/bin/python', created_at=100., expires_at=1000.,
            files={'TASKS.json':'b'*64})
        self.plan = dict(group_size=8)
        self.tasks = [dict(index=17, task_hash='a'*64, native_snapshot_row=dict(data=dict(answer='1')))]
        self.payload = dict(version='private-job-scoped-native-math-grader-v2', job_id=self.scope['run_id'],
            environment_sha256='e'*64, source_root=self.scope['workspace']+'/source',
            source_files=self.scope['source_files'], interpreter=self.scope['native_interpreter'],
            broker_interpreter=str(Path(sys.executable).resolve()), native_runtime_binding='c'*64,
            created_at=101., expires_at=999., tasks={'17':dict(task_hash='a'*64,row_sha256=sha(self.tasks[0]['native_snapshot_row']))},
            max_requests=9, snapshot_path=self.scope['workspace']+'/TASKS.json', snapshot_sha256='b'*64,
            asset_files={self.scope['workspace']+'/TASKS.json':'b'*64})

    def check(self, payload=None, tasks=None):
        p = payload or self.payload
        envelope = dict(payload=p, signer=self.authority,
            signature=base64.b64encode(self.key.sign(canonical(p)).signature).decode())
        return bind_capability(self.scope, self.plan, tasks or self.tasks, envelope,
            authority=self.authority, native_runtime_binding='c'*64)

    def test_exact_schedule_binds(self):
        self.assertEqual(self.check()['tasks']['17']['task_hash'], 'a'*64)

    def test_original_snapshot_in_source_inventory_needs_no_duplicate_file(self):
        path = self.scope['workspace']+'/source/assets/original-math7496.tasks.json'
        self.scope['source_files']['assets/original-math7496.tasks.json'] = 'b'*64
        self.scope['files'] = {}
        self.payload.update(snapshot_path=path, asset_files={path:'b'*64})
        self.assertEqual(self.check()['snapshot_path'], path)

    def test_wrong_task_extra_eval_row_or_reference_rejected(self):
        for mode in ('task', 'extra', 'reference'):
            p = copy.deepcopy(self.payload)
            if mode == 'task': p['tasks']['17']['task_hash'] = 'd'*64
            if mode == 'extra': p['tasks']['18'] = p['tasks']['17']
            if mode == 'reference': p['tasks']['17']['row_sha256'] = 'd'*64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.check(p)

    def test_extended_time_budget_or_retries_rejected(self):
        for key, value in (('created_at',99.), ('expires_at',1001.), ('max_requests',10)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(dict(self.payload, **{key:value}))

    def test_swapped_runtime_source_or_unpinned_snapshot_rejected(self):
        cases = [('native_runtime_binding','f'*64), ('interpreter','/tmp/other/bin/python'),
            ('source_files',{'one.py':'2'*64}), ('snapshot_sha256','f'*64),
            ('asset_files',{self.scope['workspace']+'/EXTRA.json':'f'*64})]
        for key, value in cases:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(dict(self.payload, **{key:value}))


if __name__ == '__main__':
    unittest.main()
