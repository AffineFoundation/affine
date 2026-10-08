"""Incomplete mathematical responses cannot manufacture failure quota or loss."""
import asyncio
import copy
from types import SimpleNamespace
import unittest

from subnet.math_completion import FIELD, VERSION, ENVIRONMENT_VERSION, enabled, final_box
from subnet.environments import EnvironmentSession, EnvironmentSpec
from ops.native_training_outcome_filter import VERSION as FILTER_VERSION, _filter_admitted_pairs


def spec():
    return dict(id='affine_math', adapter='prime_v1', version=ENVIRONMENT_VERSION,
                max_turns=1, config={FIELD: VERSION})


class CompletedAnswers(unittest.TestCase):
    def test_missing_empty_or_incomplete_latest_answer_unresolved(self):
        for text in ('unfinished reasoning', r'\boxed{', r'\boxed{}', r'\boxed{ }',
                     r'\boxed{7} later correction \boxed{8',
                     r'\boxed{\frac{1}{2}'):
            with self.subTest(text=text):
                self.assertIsNone(final_box(text))

    def test_nested_mathematical_answer_and_last_complete_answer(self):
        self.assertEqual(final_box(r'\boxed{\frac{1}{2}}'), r'\frac{1}{2}')
        self.assertEqual(final_box(r'\boxed{7} corrected \boxed{8}'), '8')
        self.assertEqual(final_box(r'\boxed{7}' + ' trailing text' * 2048), '7')

    def test_unknown_policy_wrong_environment_or_unversioned_marker_rejected(self):
        for field, value in ((FIELD, 'unknown'), ('id', 'affine_i3math'),
                             ('version', 'prime-v1-1'), ('max_turns', 2)):
            row = spec()
            if field == FIELD:
                row['config'][FIELD] = value
            else:
                row[field] = value
            with self.assertRaisesRegex(ValueError, 'contract'):
                enabled(row)
        self.assertFalse(enabled(dict(id='affine_math', config={})))


class NativeAdmission(unittest.TestCase):
    def setUp(self):
        self.policy = dict(version=FILTER_VERSION, workers=4, max_pairs=256,
                           per_grade_seconds=2, wall_seconds=10, max_reply_bytes=8192,
                           terminal_rule='max-or-eos-v1')
        self.definition = dict(env_id='affine_math', index=7, spec=spec())
        self.positive = dict(classification='positive', task_hash='trusted',
                             turns=[dict(output=[1, 0], text=r'\boxed{1}')])
        self.negative = dict(classification='negative', task_hash='trusted',
                             turns=[dict(output=[2, 2, 2], text=r'\boxed{2}')])

    def run_filter(self, decode):
        calls = []
        def grader(gold, reply, timeout):
            calls.append(reply)
            return (1 if final_box(reply) == '1' else 0), None
        accepted, report = _filter_admitted_pairs(
            [(self.definition, self.positive, self.negative)], self.policy,
            lambda *args: ('1', 'trusted', 3, {0}, 10), decode, grader)
        return accepted, report, calls

    def test_cap_without_final_answer_not_graded_or_used_as_negative(self):
        before = copy.deepcopy(self.negative)
        accepted, report, calls = self.run_filter(
            lambda output: r'\boxed{1}' if output == [1, 0] else 'unfinished reasoning')
        self.assertEqual(accepted, [])
        grade = report['rows'][0]['grades'][1]
        self.assertTrue(grade['non_eos_cap'])
        self.assertIsNone(grade['native_score'])
        self.assertEqual(grade['reason'], 'unresolved_math_answer')
        self.assertEqual(calls, [r'\boxed{1}'])
        self.assertEqual(before, self.negative)
        self.assertFalse(report['cheating_penalties'])

    def test_completed_wrong_answer_at_cap_still_a_negative(self):
        accepted, report, calls = self.run_filter(
            lambda output: r'\boxed{1}' if output == [1, 0] else r'\boxed{2}')
        self.assertEqual(len(accepted), 1)
        self.assertEqual(report['rows'][0]['grades'][1]['native_score'], 0)
        self.assertTrue(report['rows'][0]['grades'][1]['non_eos_cap'])

    def test_eos_without_answer_is_unresolved_too(self):
        self.negative['turns'][0]['output'] = [2, 0]
        accepted, report, calls = self.run_filter(
            lambda output: r'\boxed{1}' if output == [1, 0] else 'no answer')
        self.assertEqual(accepted, [])
        self.assertFalse(report['rows'][0]['grades'][1]['non_eos_cap'])
        self.assertIsNone(report['rows'][0]['grades'][1]['native_score'])

    def test_forged_complete_display_text_cannot_override_incomplete_tokens(self):
        self.negative['turns'][0]['text'] = r'\boxed{2}'
        accepted, report, calls = self.run_filter(
            lambda output: r'\boxed{1}' if output == [1, 0] else r'\boxed{')
        self.assertEqual(accepted, [])
        grade = report['rows'][0]['grades'][1]
        self.assertFalse(grade['submitted_text_matches_decoded'])
        self.assertIsNone(grade['native_score'])

    def test_historical_contract_retains_original_no_answer_zero(self):
        self.definition['spec']['config'].clear()
        accepted, report, calls = self.run_filter(
            lambda output: r'\boxed{1}' if output == [1, 0] else 'no answer')
        self.assertEqual(len(accepted), 1)
        self.assertEqual(report['rows'][0]['grades'][1]['native_score'], 0)


class SessionOutcome(unittest.TestCase):
    def run_session(self, text, *, historical=False):
        row = spec()
        if historical:
            row['config'].clear()
            row['version'] = 'prime-v1-1'
        session = object.__new__(EnvironmentSession)
        session.spec = EnvironmentSpec(**row, source_hash='test')
        session.turns = 0
        session.messages = []
        session.runtime = None
        session.trace = SimpleNamespace(state={}, rewards={}, reward=0.0)
        session._node = lambda *args, **kwargs: None
        calls = []
        async def finalize(trace, runtime):
            calls.append('finalize')
        async def score(trace, runtime):
            calls.append('score')
            trace.reward = 1.0 if final_box(text) == '1' else 0.0
            trace.rewards = {'correct': trace.reward}
        session.task = SimpleNamespace(data={}, hooks=lambda _: [], finalize=finalize, score=score)
        result = asyncio.run(session._step({'text': text}))
        return result, calls

    def test_incomplete_environment_finishes_neutral_without_grader(self):
        result, calls = self.run_session('unfinished reasoning')
        self.assertTrue(result['done'])
        self.assertEqual(result['classification'], 'neutral')
        self.assertEqual(calls, [])

    def test_correct_and_wrong_complete_answers_graded_normally(self):
        for text, expected in ((r'\boxed{1}', 'positive'), (r'\boxed{2}', 'negative')):
            result, calls = self.run_session(text)
            self.assertEqual(result['classification'], expected)
            self.assertEqual(calls, ['finalize', 'score'])

    def test_historical_session_preserves_original_negative(self):
        result, calls = self.run_session('unfinished reasoning', historical=True)
        self.assertEqual(result['classification'], 'negative')
        self.assertEqual(calls, ['finalize', 'score'])


if __name__ == '__main__':
    unittest.main()
