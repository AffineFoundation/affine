#!/usr/bin/env python3
"""Baseline / bench / prior jobs + shared runner (SPEC §3–§4)."""
import tempfile
import unittest
from pathlib import Path

from ops.jobs.kinds import JobKind, counts_toward_item_cap, kind_cap
from ops.jobs.runner import PRICE_CEILING_USD_H, FailClosedError, JobRunner, JobState

from .baseline_job import BaselineJobService, BaselineRequest
from .bench_box import MemBenchBoxStore
from .bench_job import BenchJobService, BenchRequest, coalesce_bench_queue
from .prior_job import PriorJobService


H0 = 'a' * 64
H1 = 'b' * 64
H2 = 'c' * 64
H3 = 'd' * 64


class TestJobCaps(unittest.TestCase):
    def test_non_item_outside_item_cap(self):
        self.assertTrue(counts_toward_item_cap(JobKind.ITEM))
        for k in (JobKind.BASELINE, JobKind.BENCH, JobKind.PRIOR):
            self.assertFalse(counts_toward_item_cap(k))
        self.assertEqual(kind_cap(JobKind.ITEM, live=True), 5)
        self.assertEqual(kind_cap(JobKind.ITEM, live=False), 2)
        self.assertEqual(kind_cap(JobKind.BENCH, live=True), 1)

    def test_price_ceiling(self):
        self.assertEqual(PRICE_CEILING_USD_H, 56.0)
        runner = JobRunner(live=True, price_ceiling_usd_h=56.0)

        def rent(**kw):
            return {'box_id': 'x', 'name': kw['name'], 'price_usd_h': 60.0}

        runner.rent_fn = rent
        job = runner.request(JobKind.BENCH, subject=H0)
        out = runner.tick()
        # Over-ceiling offer: wait (do not burn attempt), stay REQUESTED.
        self.assertTrue(
            any(a.get('note') == 'wait-offer' for a in out['actions'])
            or job.state is JobState.REQUESTED
        )
        self.assertEqual(job.attempt, 0)
        self.assertIsNone(job.box_id)

    def test_auto_backup_must_disable(self):
        runner = JobRunner(live=True)
        runner.disable_auto_backup_fn = lambda box_id: False
        runner.request(JobKind.PRIOR, subject=H0, gpu='H100')
        out = runner.tick()
        self.assertTrue(any(a.get('note') == 'fail-closed' for a in out['actions']))

    def test_item_and_baseline_pinned_b300(self):
        runner = JobRunner(live=True)
        with self.assertRaises(ValueError):
            runner.request(JobKind.ITEM, subject='1', gpu='B200')
        with self.assertRaises(ValueError):
            runner.request(JobKind.BASELINE, subject='r1', gpu='H200')


class TestBaselineJob(unittest.TestCase):
    def test_baseline_does_not_count_toward_item_cap(self):
        runner = JobRunner(live=True)
        # Fill item cap conceptually — baseline still starts.
        for i in range(5):
            runner.request(JobKind.ITEM, subject=str(i))
        self.assertEqual(runner.item_cap_occupancy(), 0)  # not yet rented
        # Request items into active by ticking once.
        runner.tick()
        self.assertEqual(runner.item_cap_occupancy(), 5)
        svc = BaselineJobService(
            runner=runner,
            draw_provider=lambda rid: {'indices': list(range(10))},
            probe_fn=lambda **k: {'ok': True, 'partial': False, 'cells': [1]},
            store_fn=lambda **k: {'ok': True},
        )
        svc.enqueue(BaselineRequest(head=H0, round_id='round-1', seed_commitment='commit'))
        out = svc.tick()
        self.assertFalse(out['counts_toward_item_cap'])
        self.assertEqual(runner.active_count(JobKind.BASELINE), 1)
        self.assertEqual(runner.item_cap_occupancy(), 5)

    def test_draw_not_stored_on_job(self):
        runner = JobRunner(live=True)
        leaked = []

        def store(**k):
            leaked.append('indices' in k and k.get('indices') is not None)
            return {'ok': True}

        svc = BaselineJobService(
            runner=runner,
            draw_provider=lambda rid: {'indices': [1, 2, 3]},
            probe_fn=lambda **k: {'ok': True, 'partial': False, 'cells': [0]},
            store_fn=store,
        )
        svc.enqueue(BaselineRequest(head=H0, round_id='r', seed_commitment='c'))
        for _ in range(12):
            svc.tick()
        for job in runner.jobs:
            blob = str(job.as_dict())
            self.assertNotIn('indices', blob)


class TestBenchJob(unittest.TestCase):
    def test_coalesce_keeps_newest_on_backlog(self):
        now = 10_000.0
        q = [
            BenchRequest(head=H0, queued_at=100, held_since=now - 7200),  # held ≥1h
            BenchRequest(head=H1, queued_at=200, held_since=now - 60),    # fresh
            BenchRequest(head=H2, queued_at=300, held_since=now - 120),   # fresh
            BenchRequest(head=H3, queued_at=400, held_since=now - 30),    # newest
        ]
        kept = coalesce_bench_queue(q, now=now)
        heads = {r.head for r in kept}
        self.assertIn(H3, heads)  # newest
        self.assertIn(H0, heads)  # held ≥1h
        self.assertNotIn(H1, heads)
        self.assertNotIn(H2, heads)

    def test_bench_every_green_and_merge_only(self):
        merges = []
        files = {'model.safetensors': 'c' * 64}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = MemBenchBoxStore(default_ticks=1)
            runner = JobRunner(live=True)
            svc = BenchJobService(
                runner=runner,
                state_root=root,
                box_store_factory=lambda _j: store,
                publish_fn=lambda **k: merges.append(k) or {
                    'ok': True, 'merge_only': True, 'landed': True,
                },
                results_dir=str(root / 'bench-results-out'),
                resolve_files_fn=lambda head, files=None, **_k: dict(files or {}),
                presign_fn=lambda head, files: {
                    n: 'https://example.invalid/%s' % n for n in files
                },
            )
            svc.wire_rent()
            svc.on_green_fresh(head=H0, epoch=1, files=files)
            for _ in range(40):
                store.advance()
                svc.tick()
                if svc.completed:
                    break
            self.assertEqual(len(svc.completed), 1)
            self.assertEqual(svc.completed[0]['profile'], 'qwen3.5-family-pp0')
            self.assertTrue(svc.completed[0]['merged'])
            self.assertTrue(merges)


class TestPriorJob(unittest.TestCase):
    def test_prior_per_head(self):
        runner = JobRunner(live=True)
        svc = PriorJobService(
            runner=runner,
            run_fn=lambda **k: {'ok': True, 'grades': {'n': 1}},
            store_grades_fn=lambda **k: {'ok': True},
        )
        svc.on_green_fresh(head=H0, epoch=1)
        for _ in range(15):
            svc.tick()
        self.assertEqual(len(svc.completed), 1)
        for job in runner.jobs:
            self.assertNotIn('grades', job.meta.get('result') or {})


if __name__ == '__main__':
    unittest.main()
