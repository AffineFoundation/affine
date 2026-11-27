"""PIT host heartbeat (kept; discovery publisher cutover binding reverted)."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from ops.parallel_eval.pit.run_host import write_host_heartbeat


class HostHeartbeatTests(unittest.TestCase):
    def test_write_host_heartbeat_shape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            now = time.time()
            path = write_host_heartbeat(root, now=now)
            self.assertEqual(path, root / 'HOST-HEARTBEAT.json')
            doc = json.loads(path.read_text())
            self.assertEqual(doc['version'], 'affine-pit-host-heartbeat-v1')
            self.assertEqual(doc['at'], now)
            self.assertIn('pid', doc)


if __name__ == '__main__':
    unittest.main()
