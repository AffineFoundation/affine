import hashlib
import pathlib
import sys
import tempfile
import unittest
from ops import durable_audit_services as runner


class OptionalNumericalOverlay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        (self.root / 'subnet').mkdir()
        self.path = self.root / 'subnet/numerical_resolution.py'
        self.path.write_text('ORIGINAL_VALUE = 17\n')
        self.pin = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.previous = sys.modules.get('subnet.numerical_resolution')
        self.addCleanup(self.restore)

    def restore(self):
        sys.modules.pop('subnet.numerical_resolution', None)
        if self.previous is not None:
            sys.modules['subnet.numerical_resolution'] = self.previous

    def operator(self, pin=None):
        return {'overlay': {'root': str(self.root), 'files': {
            'subnet/numerical_resolution.py': self.pin if pin is None else pin}}}

    def test_exact_fourth_module_is_importable_from_its_pinned_origin(self):
        self.assertTrue(runner.load_optional_numerical_overlay(self.operator()))
        module = sys.modules['subnet.numerical_resolution']
        self.assertEqual(module.ORIGINAL_VALUE, 17)
        self.assertEqual(pathlib.Path(module.__file__), self.path)

    def test_legacy_three_module_policy_does_not_load_an_unpinned_file(self):
        self.assertFalse(runner.load_optional_numerical_overlay(
            {'overlay': {'root': str(self.root), 'files': {}}}))

    def test_wrong_pin_fails_before_module_execution(self):
        self.restore()
        with self.assertRaises(ValueError):
            runner.load_optional_numerical_overlay(self.operator('0' * 64))
        self.assertIs(sys.modules.get('subnet.numerical_resolution'), self.previous)

    def test_changed_module_fails_before_execution(self):
        self.path.write_text('raise AssertionError("must not execute")\n')
        with self.assertRaises(ValueError):
            runner.load_optional_numerical_overlay(self.operator())

    def test_failed_import_does_not_leave_a_partially_initialized_module(self):
        self.path.write_text('raise RuntimeError("bounded import failure")\n')
        pin = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaises(RuntimeError):
            runner.load_optional_numerical_overlay(self.operator(pin))
        self.assertNotIn('subnet.numerical_resolution', sys.modules)


if __name__ == '__main__':
    unittest.main()
