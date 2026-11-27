#!/usr/bin/env python3
"""Missing PE_STATE / PIT_STATE / gate roots must fail closed (no /home/const)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import durable, host_wiring, life_paths, queue_ingest, training_report
from .decision_emit import gate_epochs_root


class TestPathFailClosed(unittest.TestCase):
    def test_pe_state_root_raises_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as cm:
                life_paths.pe_state_root()
            self.assertIn('PE_STATE', str(cm.exception))
            self.assertNotIn('/home/const', str(cm.exception))

    def test_host_wiring_pe_state_no_host_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                host_wiring._pe_state_root()

    def test_pit_state_root_raises_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as cm:
                durable.pit_state_root()
            self.assertIn('fail closed', str(cm.exception).lower())

    def test_training_report_raises_without_pe_state(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                training_report._pe_state()

    def test_training_report_writes_only_under_pe_state(self):
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / 'report.json'
            report.write_text(
                '{"source_file_hashes":{},"batching":{},"tokens":{},'
                '"update_size":0,"gradient_norm":0}'
            )
            with mock.patch.dict(os.environ, {'PE_STATE': td}, clear=False):
                summary = training_report.copy_training_report_to_host(
                    report_path=str(report),
                    item_id='e42-cccccccc',
                    epoch=42,
                    candidate_id='c' * 64,
                )
            host_path = Path(summary['host_path'])
            self.assertTrue(str(host_path).startswith(td))
            self.assertIn('training-reports', str(host_path))
            self.assertTrue(host_path.is_file())

    def test_gate_and_k1_paths_fail_closed(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                gate_epochs_root()
            with self.assertRaises(RuntimeError):
                queue_ingest._gate_epochs_root()
            with self.assertRaises(RuntimeError):
                queue_ingest._k1_turns_path()

    def test_no_home_const_literals_in_path_helpers(self):
        roots = [
            Path(life_paths.__file__),
            Path(host_wiring.__file__),
            Path(durable.__file__),
            Path(training_report.__file__),
            Path(queue_ingest.__file__),
        ]
        for path in roots:
            src = path.read_text()
            self.assertNotIn(
                '/home/const', src,
                msg='%s still hardcodes /home/const' % path.name,
            )


if __name__ == '__main__':
    unittest.main()
