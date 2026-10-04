import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'subnet/vendor/legacy/rollouts/envs/affine_math_v1/affine_math_v1'
module_spec = importlib.util.spec_from_file_location('math_argument_control', PACKAGE/'taskset.py')
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)


class MathGraderArgumentTransportTests(unittest.TestCase):
    def run_grader(self, args):
        result = subprocess.run([sys.executable, '-I', '-B', str(PACKAGE/'verify.py'), *args],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return float(result.stdout.strip())

    def test_actual_os_failure_and_exact_null_byte_roundtrip(self):
        reply = 'reasoning\x00 and then \\boxed{2}'
        with self.assertRaisesRegex(ValueError, 'embedded null byte'):
            subprocess.run([sys.executable, '-B', str(PACKAGE/'verify.py'), '2', reply])
        args = module.verify_args('2', reply)
        self.assertTrue(all('\x00' not in value for value in args))
        self.assertEqual(json.loads(args[1]), ['2', reply])
        self.assertEqual(self.run_grader(args), 1.)

    def test_original_and_encoded_grading_match_without_text_changes(self):
        replies = ['\\boxed{2}', '\\boxed{3}', 'no answer',
                   '\\boxed{2} then \\boxed{3}', 'Unicode λ and "quotes": \\boxed{2}',
                   '\\boxed{\\frac{4}{2}}', '\\boxed{2']
        for reply in replies:
            with self.subTest(reply=reply):
                args = module.verify_args('2', reply)
                self.assertEqual(json.loads(args[1]), ['2', reply])
                self.assertEqual(self.run_grader(['2', reply]), self.run_grader(args))

    def test_null_text_is_graded_by_original_last_boxed_rule(self):
        for reply, expected in [('\x00no boxed answer', 0.),
                                ('\\boxed{2}\x00 then \\boxed{3}', 0.),
                                ('\x00\\boxed{\\frac{4}{2}}\x00', 1.)]:
            with self.subTest(reply=repr(reply)):
                self.assertEqual(self.run_grader(module.verify_args('2', reply)), expected)

    def test_non_string_or_wrong_shape_arguments_are_refused(self):
        for payload in ({'gold': '2'}, ['2'], ['2', 2], ['2', 'reply', 'extra']):
            result = subprocess.run([sys.executable, '-I', '-B', str(PACKAGE/'verify.py'),
                                     '--json-arguments', json.dumps(payload)],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
