#!/usr/bin/env python3
"""On-demand PIT GREEN bench: enqueue, idempotency, restart, rent-wait, release."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from ops.jobs.kinds import JobKind
from ops.jobs.runner import JobRunner, JobState, WaitForOffer

from . import BENCH_PROFILE
from .bench_job import (
    BENCH_MAX_ATTEMPTS,
    BenchJobService,
    build_bench_rent_service,
    build_bench_service,
)
from .config import PitConfig, load_pit_config
from .pe_rent import RentError


H0 = 'a' * 64
H1 = 'b' * 64


class _Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _svc(
    root: Path,
    *,
    rent_fn=None,
    release_fn=None,
    run_fn=None,
    merge_fn=None,
    log_tail_fn=None,
    offer_precheck_fn=None,
    clock: Optional[_Clock] = None,
) -> BenchJobService:
    now = clock or _Clock()
    released: List[tuple] = []

    def _release(*, box_id: str, reason: str = 'done', **_k: Any) -> Dict[str, Any]:
        released.append((box_id, reason))
        return dict(ok=True, box_id=box_id, reason=reason)

    rent = build_bench_rent_service(
        price_ceiling_usd_h=56.0,
        offer_precheck_fn=offer_precheck_fn,
        host_rent=None,
        host_release=None,
    )
    runner = JobRunner(live=True, price_ceiling_usd_h=56.0, now_fn=now)
    if rent_fn is not None:
        runner.rent_fn = rent_fn
    if release_fn is not None:
        runner.release_fn = release_fn
    else:
        runner.release_fn = _release
    svc = BenchJobService(
        runner=runner,
        state_root=root,
        rent=rent if rent_fn is None else None,
        run_fn=run_fn or (lambda **k: {
            'ok': True,
            'profile': BENCH_PROFILE,
            'presence_penalty': 0.0,
            'benchmarks': 10,
        }),
        merge_fn=merge_fn or (lambda **k: {'ok': True, 'merge_only': True}),
        log_tail_fn=log_tail_fn,
        now_fn=now,
    )
    svc._released = released  # type: ignore[attr-defined]
    if rent_fn is None:
        svc.wire_rent()
        # Capture releases through rent.ledger path when using built-in rent_fn.
        orig = svc.runner.release_fn

        def _wrap(*, box_id: str, reason: str = 'done', **kw: Any) -> Dict[str, Any]:
            released.append((box_id, reason))
            return orig(box_id=box_id, reason=reason, **kw)

        svc.runner.release_fn = _wrap
    return svc


def _drive(svc: BenchJobService, n: int = 20) -> None:
    for _ in range(n):
        svc.tick()


class TestBenchConfigFlag(unittest.TestCase):
    def test_default_on(self):
        cfg = PitConfig(enabled=True, mode='live', max_boxes=5)
        self.assertTrue(cfg.bench_on_green)
        loaded = load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'max_boxes': 5}})
        self.assertTrue(loaded.bench_on_green)

    def test_can_disable(self):
        loaded = load_pit_config({
            'pit': {'enabled': True, 'mode': 'live', 'max_boxes': 5, 'bench_on_green': False},
        })
        self.assertFalse(loaded.bench_on_green)


class TestGreenEnqueue(unittest.TestCase):
    def test_green_enqueues_exactly_one_bench(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            svc = _svc(root)
            req = svc.on_green_fresh(head=H0, epoch=7)
            self.assertIsNotNone(req)
            self.assertEqual(len(svc.queue), 1)
            self.assertTrue((root / 'bench' / 'enqueued' / ('%s.json' % H0)).is_file())
            _drive(svc)
            self.assertEqual(len(svc.completed), 1)
            self.assertEqual(svc.completed[0]['head'], H0)
            self.assertEqual(svc.completed[0]['profile'], BENCH_PROFILE)
            self.assertTrue(svc.completed[0]['ok'])
            self.assertTrue((root / 'bench' / 'done' / ('%s.json' % H0)).is_file())
            # Exactly one bench job was started.
            bench_jobs = [j for j in svc.runner.jobs if j.kind is JobKind.BENCH]
            self.assertEqual(len(bench_jobs), 1)

    def test_duplicate_apply_no_second_bench(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            svc = _svc(root)
            self.assertIsNotNone(svc.on_green_fresh(head=H0, epoch=1))
            self.assertIsNone(svc.on_green_fresh(head=H0, epoch=1))  # duplicate
            self.assertEqual(len(svc.queue), 1)
            _drive(svc)
            self.assertEqual(len(svc.completed), 1)
            # Post-done duplicate still refused.
            self.assertIsNone(svc.on_green_fresh(head=H0, epoch=2))
            self.assertEqual(len(svc.queue), 0)
            self.assertEqual(len([j for j in svc.runner.jobs if j.kind is JobKind.BENCH]), 1)


class TestRestartMidBench(unittest.TestCase):
    def test_restart_mid_bench_resumes_no_duplicate_rent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rents: List[str] = []

            def rent(**kw: Any) -> Dict[str, Any]:
                rents.append(kw['name'])
                return {
                    'box_id': 'box-resume-1',
                    'name': kw['name'],
                    'price_usd_h': 35.0,
                    'gpu': 'B300',
                    'auto_backup_disabled': True,
                }

            run_calls = {'n': 0}

            def run_fn(**kw: Any) -> Dict[str, Any]:
                run_calls['n'] += 1
                if run_calls['n'] == 1:
                    # Simulate crash mid-run: leave job RUNNING with box.
                    raise RuntimeError('simulated mid-bench crash')
                return {
                    'ok': True,
                    'profile': BENCH_PROFILE,
                    'presence_penalty': 0.0,
                    'benchmarks': 10,
                }

            svc = _svc(root, rent_fn=rent, run_fn=run_fn, log_tail_fn=lambda **k: 'tail-line\n' * 5)
            svc.on_green_fresh(head=H0, epoch=1)
            # Advance to RUNNING then fail once (releases box, FAILED → retry).
            for _ in range(8):
                out = svc.tick()
                if any(a.get('op') == 'bench_failed' for a in out.get('bench_actions') or []):
                    break
            # Persist active / enqueued markers, then rebuild service (host restart).
            svc2 = _svc(root, rent_fn=rent, run_fn=run_fn)
            svc2.ensure_loaded()
            # Same head must not re-enqueue.
            self.assertIsNone(svc2.on_green_fresh(head=H0, epoch=1))
            _drive(svc2, 25)
            # At most one successful completion; rents only for attempts (≤2).
            self.assertLessEqual(len(rents), BENCH_MAX_ATTEMPTS)
            done = (root / 'bench' / 'done' / ('%s.json' % H0))
            self.assertTrue(done.is_file())


class TestRentOverCeilingWaits(unittest.TestCase):
    def test_rent_over_ceiling_waits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clock = _Clock()
            offers = {'price': 90.0}
            rents = {'n': 0}

            def precheck(**_k: Any) -> Dict[str, Any]:
                if offers['price'] > 56.0:
                    raise RentError('no eligible B300 offer under ceiling')
                return {'gpu': 'B300', 'price_usd_h': offers['price']}

            def rent(**kw: Any) -> Dict[str, Any]:
                if offers['price'] > float(kw['price_ceiling_usd_h']):
                    raise WaitForOffer('offer above ceiling')
                rents['n'] += 1
                return {
                    'box_id': 'box-ok',
                    'name': kw['name'],
                    'price_usd_h': offers['price'],
                    'gpu': 'B300',
                    'auto_backup_disabled': True,
                }

            svc = _svc(
                root,
                rent_fn=rent,
                offer_precheck_fn=precheck,
                clock=clock,
            )
            # When rent_fn is injected, BenchJobService.rent is None — WaitForOffer from rent_fn.
            svc.on_green_fresh(head=H0, epoch=1)
            wait_notes = []
            for _ in range(5):
                out = svc.tick()
                wait_notes.extend(
                    a for a in out.get('actions') or [] if a.get('note') == 'wait-offer'
                )
                clock.advance(30)
            self.assertTrue(wait_notes)
            self.assertEqual(rents['n'], 0)
            # Job still REQUESTED, attempt not burned.
            jobs = [j for j in svc.runner.jobs if j.kind is JobKind.BENCH]
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].state, JobState.REQUESTED)
            self.assertEqual(jobs[0].attempt, 0)
            # Offer drops under ceiling → rents and completes.
            offers['price'] = 40.0
            _drive(svc, 20)
            self.assertEqual(rents['n'], 1)
            self.assertEqual(len(svc.completed), 1)
            self.assertTrue(svc.completed[0]['ok'])


class TestBenchFailureReleasesBox(unittest.TestCase):
    def test_bench_failure_releases_box(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            released: List[tuple] = []

            def rent(**kw: Any) -> Dict[str, Any]:
                return {
                    'box_id': 'box-fail-%d' % kw['attempt'],
                    'name': kw['name'],
                    'price_usd_h': 35.0,
                    'gpu': 'B300',
                    'auto_backup_disabled': True,
                }

            def release(*, box_id: str, reason: str = 'done', **_k: Any) -> Dict[str, Any]:
                released.append((box_id, reason))
                return dict(ok=True, box_id=box_id, reason=reason)

            def run_fn(**_k: Any) -> Dict[str, Any]:
                raise RuntimeError('bench boom')

            tails: List[str] = []

            def log_tail(**_k: Any) -> str:
                t = 'job.log line A\njob.log line B\nFATAL bench boom\n'
                tails.append(t)
                return t

            svc = _svc(
                root,
                rent_fn=rent,
                release_fn=release,
                run_fn=run_fn,
                log_tail_fn=log_tail,
            )
            svc.on_green_fresh(head=H0, epoch=1)
            _drive(svc, 30)
            # Initial + one retry → both boxes released.
            self.assertGreaterEqual(len(released), 2)
            self.assertTrue(any('bench-run-failed' in r[1] or r[1] in (
                'retry', 'no_decision', 'verified', 'bench-run-failed',
            ) for r in released))
            # Terminal done marker (failed) so we do not re-enqueue forever.
            self.assertTrue(svc.is_done(H0))
            self.assertTrue(any('FATAL bench boom' in t for t in tails))
            # Events carry log tail.
            fail_events = [e for e in svc.events if e.get('event') == 'bench_run_failed']
            self.assertTrue(fail_events)
            self.assertIn('FATAL', fail_events[0].get('job_log_tail') or '')


class TestBenchOutsideItemCap(unittest.TestCase):
    def test_bench_cap_separate_from_max_boxes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runner = JobRunner(live=True, price_ceiling_usd_h=56.0)
            # Fill item cap.
            for i in range(5):
                runner.request(JobKind.ITEM, subject=str(i))
            runner.tick()
            self.assertEqual(runner.item_cap_occupancy(), 5)
            svc = BenchJobService(
                runner=runner,
                state_root=root,
                run_fn=lambda **k: {
                    'ok': True, 'profile': BENCH_PROFILE, 'presence_penalty': 0.0,
                },
                merge_fn=lambda **k: {'ok': True, 'merge_only': True},
            )
            svc.on_green_fresh(head=H0, epoch=1)
            out = svc.tick()
            self.assertFalse(out['counts_toward_item_cap'])
            self.assertEqual(runner.active_count(JobKind.BENCH), 1)
            self.assertEqual(runner.item_cap_occupancy(), 5)


class TestBuildHelpers(unittest.TestCase):
    def test_build_bench_service_loads_markers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            svc = build_bench_service(
                state_root=root,
                run_fn=lambda **k: {
                    'ok': True, 'profile': BENCH_PROFILE, 'presence_penalty': 0.0,
                },
                merge_fn=lambda **k: {'ok': True, 'merge_only': True},
            )
            svc.on_green_fresh(head=H1, epoch=3)
            self.assertTrue(svc.is_enqueued(H1))
            svc2 = build_bench_service(
                state_root=root,
                run_fn=lambda **k: {
                    'ok': True, 'profile': BENCH_PROFILE, 'presence_penalty': 0.0,
                },
                merge_fn=lambda **k: {'ok': True, 'merge_only': True},
            )
            # Queue restored from disk; duplicate enqueue refused.
            self.assertTrue(svc2.is_enqueued(H1))
            self.assertIsNone(svc2.on_green_fresh(head=H1, epoch=3))


if __name__ == '__main__':
    unittest.main()
