#!/usr/bin/env python3
"""baseline_job banned fields + probe_fn refusal (PR #96)."""
import unittest
from dataclasses import dataclass

from ops.jobs.kinds import JobKind
from ops.jobs.runner import JobRecord, JobRunner, JobState

from .baseline_job import BaselineJobService, BaselineRequest

H0 = 'a' * 64


class TestBaselineEnqueueRefusal(unittest.TestCase):
    def test_refuses_banned_keys_in_as_dict(self):
        svc = BaselineJobService(
            runner=JobRunner(live=True),
            probe_fn=lambda **k: {'ok': True, 'partial': False},
        )

        @dataclass
        class Evil(BaselineRequest):
            def as_dict(self):
                return {**super().as_dict(), 'indices': [1, 2, 3]}

        with self.assertRaises(ValueError):
            svc.enqueue(Evil(head=H0, round_id='r1', seed_commitment='commit'))

        @dataclass
        class EvilSeed(BaselineRequest):
            def as_dict(self):
                return {**super().as_dict(), 'raw_seed': 'leak'}

        with self.assertRaises(ValueError):
            svc.enqueue(EvilSeed(head=H0, round_id='r1', seed_commitment='commit'))


class TestBaselineProbeRequired(unittest.TestCase):
    def test_refuses_probe_without_probe_fn(self):
        runner = JobRunner(live=True)
        runner.rent_fn = lambda **kw: {'box_id': 'b1', 'name': 'affine-baseline-1', 'price_usd_h': 30.0}
        runner.disable_auto_backup_fn = lambda box_id: True
        svc = BaselineJobService(runner=runner)
        svc.enqueue(BaselineRequest(head=H0, round_id='r2', seed_commitment='c'))
        for _ in range(8):
            try:
                svc.tick()
            except RuntimeError:
                break
        job = next((j for j in runner.jobs if j.kind is JobKind.BASELINE), None)
        self.assertIsNotNone(job)
        job.state = JobState.RUNNING
        job.meta.setdefault('baseline', BaselineRequest(head=H0, round_id='r2', seed_commitment='c').as_dict())
        with self.assertRaises(RuntimeError) as ctx:
            svc._run_probe(job)
        self.assertIn('probe_fn', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
