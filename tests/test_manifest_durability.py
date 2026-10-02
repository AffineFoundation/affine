import errno
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subnet.controller import save_manifest


class ManifestDurabilityTests(unittest.TestCase):
    def test_complete_replacement_and_private_permissions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'epoch-manifest.json'
            save_manifest(path, {'epoch': 'old'})
            value = {'epoch': 'new', 'payload': 'x' * 100000}
            save_manifest(path, value)
            self.assertEqual(json.loads(path.read_bytes()), value)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_disk_failure_never_exposes_partial_new_manifest(self):
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'epoch-manifest.json'
                if existing:
                    save_manifest(path, {'epoch': 'preserved'})
                with patch('subnet.controller.os.fsync', side_effect=OSError(errno.ENOSPC, 'full')):
                    with self.assertRaises(OSError):
                        save_manifest(path, {'epoch': 'new', 'payload': 'x' * 100000})
                if existing:
                    self.assertEqual(json.loads(path.read_bytes()), {'epoch': 'preserved'})
                else:
                    self.assertFalse(path.exists())
                self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_invalid_json_cannot_replace_previous_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'epoch-manifest.json'
            save_manifest(path, {'epoch': 'preserved'})
            with self.assertRaises(ValueError):
                save_manifest(path, {'reward': float('nan')})
            self.assertEqual(json.loads(path.read_bytes()), {'epoch': 'preserved'})


if __name__ == '__main__':
    unittest.main()
