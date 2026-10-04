import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from subnet.storage import canonical
from ops.controller_boundary_lease import lease, process


CHILD = '''import json,time
from pathlib import Path
root=Path(__import__('sys').argv[1]);state=root/'state/controller.json'
while True:
 if (root/'complete').exists() and not (root/'changed').exists():
  s=json.loads(state.read_text());s.update(round=7,active=None,training_steps=12,checkpoint=json.loads((root/'next.json').read_text()))
  t=state.with_suffix('.tmp');t.write_text(json.dumps(s));t.replace(state);(root/'changed').write_text('yes')
 (root/'heartbeat').write_text(str(time.time()))
 time.sleep(.02)
'''


class BoundaryLeaseControls(unittest.TestCase):
    def test_long_observation_does_not_allow_extending_the_hold_budget(self):
        with patch('ops.controller_boundary_lease.os.kill') as kill:
            for wait,hold in [(21601,1),(21600,901),(0,1)]:
                with self.assertRaisesRegex(ValueError,'lease budgets'):
                    lease('unused','unused','original-epoch','unused',wait,hold)
            kill.assert_not_called()

    def fixture(self, root):
        state = root / 'state'; state.mkdir()
        files = {'config.json': 'a' * 64, 'model.safetensors': 'b' * 64}
        cp = {'id': hashlib.sha256(canonical(files)).hexdigest(), 'files': files}
        (root / 'next.json').write_bytes(canonical(cp))
        (state / 'controller.json').write_bytes(canonical({'round': 6, 'training_steps': 9,
            'checkpoint': {'id': 'c' * 64}, 'active': {'epoch': 'original-epoch', 'phase': 'after'}}))
        config = root / 'config.json'; config.write_bytes(canonical({'state': str(state)})); config.chmod(0o600)
        child = subprocess.Popen([sys.executable, '-B', '-c', CHILD, str(root)], cwd=root)
        self.addCleanup(self.stop_child, child)
        end = time.monotonic() + 5
        while not (root / 'heartbeat').exists() and time.monotonic() < end: time.sleep(.01)
        self.assertTrue((root / 'heartbeat').exists())
        record = root / 'process.json'; record.write_bytes(canonical({'child_pid': child.pid,
            'child_ticks': process(child.pid)[1], 'cwd': str(root), 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest()})); record.chmod(0o600)
        return child, config, record

    @staticmethod
    def stop_child(child):
        if child.poll() is None:
            child.send_signal(__import__('signal').SIGCONT); child.terminate()
        child.wait(timeout=5)

    def test_real_hold_expires_and_resumes_same_original_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); child, config, record = self.fixture(root); output = root / 'lease'
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(lease, config, record, 'original-epoch', output, 5, 1)
                while not (output / 'authorization.private.json').exists(): time.sleep(.005)
                (root / 'complete').write_text('yes')
                end = time.monotonic() + 3
                while not (output / 'actual-held-boundary.private.json').exists() and time.monotonic() < end: time.sleep(.005)
                self.assertTrue((output / 'actual-held-boundary.private.json').exists()); self.assertEqual(process(child.pid)[0], 'T')
                heartbeat = (root / 'heartbeat').read_text(); time.sleep(.1)
                self.assertEqual((root / 'heartbeat').read_text(), heartbeat)
                self.assertEqual(future.result(timeout=3)['reason'], 'lease_expired')
            release = json.loads((output / 'release.private.json').read_text()); self.assertTrue(release['resumed_original'])
            self.assertEqual(release['jobs_restarted'], 0); time.sleep(.05)
            self.assertNotEqual((root / 'heartbeat').read_text(), heartbeat); self.assertIsNone(child.poll())

    def test_wait_timeout_never_stops_original_active_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); child, config, record = self.fixture(root)
            result = lease(config, record, 'original-epoch', root / 'lease', 1, 1)
            self.assertFalse(result['held']); self.assertNotEqual(process(child.pid)[0], 'T')
            self.assertEqual(json.loads((root / 'state/controller.json').read_text())['active']['epoch'], 'original-epoch')

    def test_changed_config_pid_ticks_or_target_refuses_without_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); child, config, record = self.fixture(root)
            old = json.loads(record.read_text()); record.write_bytes(canonical(dict(old, child_ticks='0')))
            with patch('ops.controller_boundary_lease.os.kill') as kill:
                with self.assertRaisesRegex(ValueError, 'running controller'): lease(config, record, 'original-epoch', root / 'one', 1, 1)
                record.write_bytes(canonical(old)); config.write_bytes(canonical({'state': str(root / 'other')}))
                with self.assertRaisesRegex(ValueError, 'config digest'): lease(config, record, 'original-epoch', root / 'two', 1, 1)
                config.write_bytes(canonical({'state': str(root / 'state')}))
                with self.assertRaisesRegex(ValueError, 'active epoch'): lease(config, record, 'wrong-epoch', root / 'three', 1, 1)
                kill.assert_not_called()

    def test_interruption_releases_only_our_observed_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); child, config, record = self.fixture(root); actual_sleep = time.sleep
            def sleep(seconds):
                if process(child.pid)[0] == 'T': raise InterruptedError('operator interrupted lease')
                (root / 'complete').write_text('yes'); actual_sleep(seconds)
            with patch('ops.controller_boundary_lease.time.sleep', side_effect=sleep):
                with self.assertRaises(InterruptedError): lease(config, record, 'original-epoch', root / 'lease', 5, 2)
            self.assertNotEqual(process(child.pid)[0], 'T'); self.assertIsNone(child.poll())
            release = json.loads((root / 'lease/release.private.json').read_text())
            self.assertTrue(release['resumed_original']); self.assertEqual(release['result']['error_type'], 'InterruptedError')

    def test_original_controller_exit_is_not_a_restart_instruction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); child, config, record = self.fixture(root)
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(lease, config, record, 'original-epoch', root / 'lease', 21600, 1)
                while not (root / 'lease/authorization.private.json').exists(): time.sleep(.005)
                child.terminate(); child.wait(timeout=3)
                result = future.result(timeout=3)
            self.assertEqual(result['reason'], 'original_controller_exited')
            self.assertFalse(json.loads((root / 'lease/release.private.json').read_text())['resumed_original'])
            self.assertEqual(json.loads((root / 'lease/authorization.private.json').read_text())['wait_seconds'],21600)

    def test_missed_boundary_never_stops_next_active_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); child, config, record = self.fixture(root)
            with concurrent.futures.ThreadPoolExecutor(1) as pool, patch('ops.controller_boundary_lease.os.kill') as kill:
                future = pool.submit(lease, config, record, 'original-epoch', root / 'lease', 5, 1)
                while not (root / 'lease/authorization.private.json').exists(): time.sleep(.005)
                p = root / 'state/controller.json'; status = json.loads(p.read_text())
                status.update(round=7, active={'epoch': 'new-epoch', 'phase': 'mine'})
                temp = p.with_suffix('.tmp'); temp.write_bytes(canonical(status)); temp.replace(p)
                result = future.result(timeout=3); kill.assert_not_called()
            self.assertEqual(result['reason'], 'next_epoch_already_active'); self.assertIsNone(child.poll())

    def test_existing_hold_by_another_owner_is_never_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); child, config, record = self.fixture(root)
            child.send_signal(__import__('signal').SIGSTOP)
            deadline = time.monotonic() + 2
            while process(child.pid)[0] != 'T' and time.monotonic() < deadline: time.sleep(.005)
            with patch('ops.controller_boundary_lease.os.kill') as kill:
                with self.assertRaisesRegex(ValueError, 'running controller'):
                    lease(config, record, 'original-epoch', root / 'lease', 1, 1)
                kill.assert_not_called()
            self.assertEqual(process(child.pid)[0], 'T')


if __name__ == '__main__': unittest.main()
