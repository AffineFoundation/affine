#!/usr/bin/env python3
"""Cutover enable/revert helpers + script dry logic (PR #96)."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from .cutover_ops import (
    TrainerBusyError,
    backup_pre_pit,
    clear_pit_claims,
    enable_cutover_on_flag,
    find_latest_pre_pit_backup,
    revert_cutover_on_flag,
    trainer_idle_check,
    warm_off_path,
    write_warm_off,
    remove_warm_off,
    write_flag_patched,
)

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent / 'scripts'


class TestCutoverOps(unittest.TestCase):
    def test_trainer_idle_refuses_mid_turn(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ep = root / 'epochs' / '7'
            ep.mkdir(parents=True)
            (ep / 'TRAINING-JOB.json').write_text('{}')
            with self.assertRaises(TrainerBusyError):
                trainer_idle_check(root)

    def test_trainer_idle_ok(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ep = root / 'epochs' / '8'
            ep.mkdir(parents=True)
            (ep / 'TRAINING-JOB.json').write_text('{}')
            (ep / 'TRAINING-RESULT.json').write_text('{}')
            self.assertEqual(trainer_idle_check(root), 8)

    def test_backup_and_enable_patch_preserves_flag_shape(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{\n  "parallel_eval": {\n    "pit": {"max_boxes": 3}\n  }\n}\n')
            flag = Path(f.name)
        try:
            backup_pre_pit(flag)
            found = find_latest_pre_pit_backup(flag)
            self.assertTrue(found and found.is_file())
            enable_cutover_on_flag(flag)
            doc = json.loads(flag.read_text())
            pit = doc['parallel_eval']['pit']
            self.assertTrue(pit['cutover'])
            self.assertEqual(pit['root_mode'], 'delegated_host')
            self.assertEqual(pit['max_boxes'], 3)
            raw = flag.read_text()
            self.assertIn('\n  ', raw)
        finally:
            flag.unlink(missing_ok=True)
            if found:
                found.unlink(missing_ok=True)

    def test_revert_sets_revert_flag(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'parallel_eval': {'enabled': True, 'pit': {'enabled': True, 'cutover': True, 'mode': 'live'}}}, f)
            flag = Path(f.name)
        try:
            revert_cutover_on_flag(flag)
            pit = json.loads(flag.read_text())['parallel_eval']['pit']
            self.assertTrue(pit['revert'])
            self.assertFalse(pit['enabled'])
            self.assertFalse(pit['cutover'])
        finally:
            flag.unlink(missing_ok=True)

    def test_warm_off_file(self):
        with tempfile.TemporaryDirectory() as td:
            st = Path(td)
            p = write_warm_off(st)
            self.assertEqual(p, warm_off_path(st))
            self.assertTrue(p.is_file())
            remove_warm_off(st)
            self.assertFalse(p.exists())

    def test_clear_pit_claims(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / 'a'
            b = Path(td) / 'b'
            a.mkdir()
            b.mkdir()
            (a / 'PIT-CLAIM-e1.json').write_text('{}')
            (b / 'PIT-CLAIM-e2.json').write_text('{}')
            removed = clear_pit_claims([a, b])
            self.assertEqual(len(removed), 2)

    def test_enable_shell_syntax(self):
        proc = subprocess.run(
            ['bash', '-n', str(SCRIPTS / 'cutover_enable.sh')],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


class TestCutoverRestore(unittest.TestCase):
    def test_restore_from_backup_overrides_cutover(self):
        with tempfile.TemporaryDirectory() as td:
            flag = Path(td) / 'PE_FLAG.json'
            flag.write_text(json.dumps({'parallel_eval': {'pit': {'enabled': False, 'mode': 'shadow'}}}, indent=2) + '\n')
            bak = backup_pre_pit(flag)
            write_flag_patched(flag, {'enabled': True, 'cutover': True, 'mode': 'live'}, pe_enabled=True)
            self.assertTrue(json.loads(flag.read_text())['parallel_eval']['pit']['cutover'])
            revert_cutover_on_flag(flag, restore_from=bak)
            pit = json.loads(flag.read_text())['parallel_eval']['pit']
            self.assertTrue(pit['revert'])
            self.assertFalse(pit.get('cutover'))
            self.assertFalse(pit['enabled'])


if __name__ == '__main__':
    unittest.main()
