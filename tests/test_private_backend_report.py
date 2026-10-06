import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from subnet.backend_jobs import canonical, write_private_report

class PrivateBackendReportTests(unittest.TestCase):
    def test_public_umask_cannot_make_report_public(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.json';mask=os.umask(0o022)
            try:write_private_report(path,{'success':True,'heldout':[{'reward':1}]})
            finally:os.umask(mask)
            self.assertEqual(path.stat().st_mode&0o777,0o600)
            self.assertEqual(path.read_bytes(),canonical({'success':True,'heldout':[{'reward':1}]}))
            self.assertEqual(list(Path(tmp).iterdir()),[path])

    def test_serialization_error_keeps_original_complete_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.json';write_private_report(path,{'success':True})
            original=path.read_bytes()
            with self.assertRaises(ValueError):write_private_report(path,{'reward':float('nan')})
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(list(Path(tmp).iterdir()),[path])

    def test_atomic_publish_failure_preserves_previous_bytes_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.json';write_private_report(path,{'old':True})
            with patch('subnet.backend_jobs.os.replace',side_effect=OSError('injected publish failure')):
                with self.assertRaises(OSError):write_private_report(path,{'new':True})
            self.assertEqual(json.loads(path.read_bytes()),{'old':True})
            self.assertEqual(path.stat().st_mode&0o777,0o600)
            self.assertEqual(list(Path(tmp).iterdir()),[path])
