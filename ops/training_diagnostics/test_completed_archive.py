"""Synthetic completed-baseline/latest capture regression; no network or model calls."""
import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest

import train_archive as archive
import train_contract as contract
from test_training_eval import fixture, sign
import test_training_lifecycle as lifecycle


class CompletedPairArchiveTests(unittest.TestCase):
    def closed_pair(self, root):
        monitor = lifecycle.LifecycleTests().make_capture(root)
        archive.capture_and_archive(monitor, 'step-0-test')
        baseline = root / 'jobs/step-0-test'
        self.assertTrue((baseline / 'captured/task-exposure-metadata.json').exists())
        (root / 'state.json').write_text(json.dumps({'baseline_job': baseline.name}))
        plan, rows = fixture()
        plan = copy.deepcopy(plan)
        plan.update(phase='latest', optimizer_step=4)
        plan['checkpoint'] = {'files': {'config.json': contract.digest('new-config')}}
        plan['checkpoint']['id'] = contract.digest(plan['checkpoint']['files'])
        for row in rows:
            row['checkpoint'] = plan['checkpoint']['id']
        envelope = sign(plan)
        files = {'task-' + str(row['index']) + '.json': contract.canonical(row) for row in rows}
        result = dict(version=plan['version'], plan_sha256=contract.digest(envelope),
                      checkpoint=plan['checkpoint']['id'], tasks=128,
                      cohort_sha256=plan['cohort_sha256'], profile_sha256=plan['profile_sha256'],
                      seeds_sha256=plan['seeds_sha256'],
                      raw_artifacts={name: archive.sha(raw) for name, raw in files.items()},
                      correct=96, errors=0, unresolved=0, accuracy=96/128, completed_at=1791792001.)
        files['result.json'] = contract.canonical(result)
        latest = root / 'jobs/step-4-test'
        latest.mkdir()
        (latest / 'plan.json').write_bytes(contract.canonical(envelope))
        for name in ('evaluate.py', 'train_contract.py'):
            (latest / name).write_bytes(b'# synthetic source\n')
        monitor.ssh_code = lambda _: dict(files={name: base64.b64encode(raw).decode()
                                                 for name, raw in files.items()})
        return monitor, baseline, latest

    def test_latest_after_archived_baseline_keeps_exact128(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            monitor, _, latest = self.closed_pair(root)
            ack, aggregate = archive.capture_and_archive(monitor, latest.name)
            self.assertTrue(aggregate['comparability_passed'])
            self.assertEqual(aggregate['requested'], 128)
            self.assertEqual(len(ack['payload']['archive']), 134)
            self.assertEqual(len(list((root / 'public').glob('*.signed.json'))), 2)

    def test_unexpected_numbered_baseline_record_is_not_silently_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            monitor, baseline, latest = self.closed_pair(root)
            (baseline / 'captured/task-999999.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'exact baseline raw task identities'):
                archive.capture_and_archive(monitor, latest.name)
            self.assertEqual(len(list((root / 'public').glob('*.signed.json'))), 1)

    def test_missing_baseline_record_prevents_latest_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            monitor, baseline, latest = self.closed_pair(root)
            (baseline / 'captured/task-0.json').unlink()
            with self.assertRaisesRegex(ValueError, 'exact baseline raw task identities'):
                archive.capture_and_archive(monitor, latest.name)
            self.assertEqual(len(list((root / 'public').glob('*.signed.json'))), 1)


if __name__ == '__main__':
    unittest.main()
