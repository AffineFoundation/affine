#!/usr/bin/env python3
"""The live host scorer finds the PIT package from the code directory of a root audit.

On the host, ``pe_score.py`` is a file in ``<repo>/state/root-audits/<audit>/code``. That is one directory
deeper than ``<repo>/ops/parallel_eval/live``. A fixed number of parent directories gave ``<repo>/state``,
and the import of ``ops.parallel_eval.pit.shadow_compare`` failed.

The test copies the real ``pe_score.py`` and ``pe_common.py`` into that layout and imports the copy in a new
Python process. The repository root is not on the import path of that process.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LIVE = REPO / 'ops' / 'parallel_eval' / 'live'
LIVE_IMPORT_DIRS = (REPO / 'miner' / 'validator' / 'gate', REPO / 'ops' / 'gate_probe_error_guard', REPO / 'ops' / 'pods')
PROGRAM = (
    'import sys\n'
    'sys.path[:0] = %r\n'
    'import pe_score\n'
    'print(pe_score._repo_root())\n'
    'print(pe_score.evaluate_shadow_pass.__module__)\n'
    'print(sys.modules[pe_score.evaluate_shadow_pass.__module__].__file__)\n'
)


class HostLayout:
    """A host checkout in a temp directory, with the scorer in the code directory of a root audit."""

    def __init__(self, tmp: Path, *, with_pit: bool = True) -> None:
        self.tmp = tmp
        self.repo = tmp / 'host' / 'rewrite'
        self.code = self.repo / 'state' / 'root-audits' / 'audit' / 'code'
        self.code.mkdir(parents=True)
        self.home = tmp / 'home'
        self.home.mkdir()
        for name in ('pe_score.py', 'pe_common.py'):
            shutil.copy2(LIVE / name, self.code / name)
        if with_pit:
            (self.repo / 'ops').symlink_to(REPO / 'ops', target_is_directory=True)

    def import_scorer(self, **env: str) -> subprocess.CompletedProcess:
        paths = [str(self.code)] + [str(path) for path in LIVE_IMPORT_DIRS]
        base = {'HOME': str(self.home), 'PATH': os.environ.get('PATH', ''),
                'PODS_ROLES': str(REPO / 'ops' / 'pods' / 'roles.json')}
        return subprocess.run([sys.executable, '-B', '-c', PROGRAM % (paths,)], cwd=str(self.tmp), env=dict(base, **env),
                              capture_output=True, text=True, timeout=300)


class TestScorerImportFromTheRootAuditCodeDirectory(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()

    def test_scorer_in_the_code_directory_of_a_root_audit_imports_the_pit_package(self) -> None:
        host = HostLayout(self.tmp)
        self.assertEqual(host.code.parents[2], host.repo / 'state', 'the old rule (three directories up) gave state/')
        proc = host.import_scorer()
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        root, module, source = proc.stdout.strip().splitlines()[-3:]
        self.assertEqual(Path(root), host.repo)
        self.assertEqual(module, 'ops.parallel_eval.pit.shadow_compare')
        self.assertEqual(Path(source).resolve(), REPO / 'ops' / 'parallel_eval' / 'pit' / 'shadow_compare.py')

    def test_affine_repo_names_the_checkout_when_the_directories_above_have_no_pit_package(self) -> None:
        host = HostLayout(self.tmp, with_pit=False)
        proc = host.import_scorer(AFFINE_REPO=str(REPO))
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(Path(proc.stdout.strip().splitlines()[-3]), REPO)

    def test_no_pit_package_anywhere_is_an_import_error_that_names_the_places(self) -> None:
        host = HostLayout(self.tmp, with_pit=False)
        proc = host.import_scorer()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('ImportError', proc.stderr)
        self.assertIn('shadow_compare.py is not in AFFINE_REPO, not above', proc.stderr)

    def test_scorer_in_the_git_layout_gives_the_repository(self) -> None:
        paths = [str(LIVE)] + [str(path) for path in LIVE_IMPORT_DIRS]
        home = self.tmp / 'home'
        home.mkdir()
        env = {'HOME': str(home), 'PATH': os.environ.get('PATH', ''), 'PODS_ROLES': str(REPO / 'ops' / 'pods' / 'roles.json')}
        proc = subprocess.run([sys.executable, '-B', '-c', PROGRAM % (paths,)], cwd=str(self.tmp), env=env,
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(Path(proc.stdout.strip().splitlines()[-3]), REPO)


if __name__ == '__main__':
    unittest.main()
