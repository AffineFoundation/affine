"""Synthetic checks for a root-reviewable helper; no SSH, signatures or GPU."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HELPER = Path(__file__).resolve().parents[1] / 'state/root-audits/prepare_and_run_cp093_math200_baseline_20261004_v1.py'


@unittest.skipUnless(HELPER.is_file(), 'operator-local baseline helper is absent')
class CP093BaselineHelperReviewTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(HELPER.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'run_reviewed')
        assignment = next(node for node in function.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'program'
                                  for target in node.targets))
        self.program = ast.literal_eval(assignment.value.func.value)

    def execute_synthetic_preflight(self, *, model='NVIDIA H100 80GB HBM3',
                                    applications='', processes='', free=50*1024**3):
        with tempfile.TemporaryDirectory() as directory:
            program = self.program.replace('SOURCE', repr(directory))
            commands = []

            def output(command, **kwargs):
                commands.append(command)
                if command[0] == 'ps':
                    # A real ps listing contains the current remote code. The
                    # guard must not identify that code as an active runner.
                    return processes + '\npython -I -B -c ' + program
                if '--query-compute-apps=pid' in command:
                    return applications
                if '--query-gpu=name,uuid' in command:
                    return model + ', GPU-SYNTHETIC\n'
                raise AssertionError('unexpected synthetic command')

            stdout = io.StringIO()
            with patch('subprocess.check_output', side_effect=output), \
                    patch('os.statvfs', return_value=SimpleNamespace(f_bavail=free, f_frsize=1)), \
                    patch('importlib.metadata.version', side_effect=lambda name: 'synthetic-' + name), \
                    contextlib.redirect_stdout(stdout):
                exec(compile(program, '<synthetic preflight>', 'exec'), {})
            result = json.loads(stdout.getvalue())
            self.assertEqual(result['source_files'], {})
            self.assertTrue(result['GPU_idle'])
            self.assertEqual(len(commands), 3)
            return result

    def test_idle_h100_does_not_match_its_own_command(self):
        result = self.execute_synthetic_preflight()
        self.assertEqual(result['GPU_name'], 'NVIDIA H100 80GB HBM3')

    def test_busy_gpu_is_rejected_before_any_baseline_job(self):
        with self.assertRaises(AssertionError):
            self.execute_synthetic_preflight(applications='12345\n')

    def test_live_runner_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.execute_synthetic_preflight(processes='python -m subnet.remote_runner job.json')

    def test_wrong_gpu_and_insufficient_disk_are_rejected(self):
        with self.assertRaises(AssertionError):
            self.execute_synthetic_preflight(model='NVIDIA H200')
        with self.assertRaises(AssertionError):
            self.execute_synthetic_preflight(free=40*1024**3)


if __name__ == '__main__':
    unittest.main()
