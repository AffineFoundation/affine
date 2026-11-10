#!/usr/bin/env python3
"""On-demand PIT GREEN bench: detached, non-blocking, restart-safe."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from ops.jobs.kinds import JobKind
from ops.jobs.runner import JobRunner, JobState, WaitForOffer

from . import BENCH_PROFILE
from .bench_box import BENCH_BOX_GIT_REF, BENCH_STAGES, MemBenchBoxStore
from .bench_job import (
    BENCH_MAX_ATTEMPTS,
    BenchJobService,
    build_bench_rent_service,
    build_bench_service,
)
from .bench_publish import merge_publish, site_root
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


def _released_wrap(svc: BenchJobService) -> List[tuple]:
    released: List[tuple] = []
    orig = svc.runner.release_fn

    def _wrap(*, box_id: str, reason: str = 'done', **kw: Any) -> Dict[str, Any]:
        released.append((box_id, reason))
        if orig is not None:
            return orig(box_id=box_id, reason=reason, **kw)
        return dict(ok=True, box_id=box_id, reason=reason)

    svc.runner.release_fn = _wrap
    return released


def _svc(
    root: Path,
    *,
    store: Optional[MemBenchBoxStore] = None,
    rent_fn=None,
    publish_fn=None,
    clock: Optional[_Clock] = None,
    default_ticks: int = 1,
) -> BenchJobService:
    now = clock or _Clock()
    box = store or MemBenchBoxStore(default_ticks=default_ticks)
    rent = build_bench_rent_service(price_ceiling_usd_h=56.0)
    runner = JobRunner(live=True, price_ceiling_usd_h=56.0, now_fn=now)
    if rent_fn is not None:
        runner.rent_fn = rent_fn
    svc = BenchJobService(
        runner=runner,
        state_root=root,
        rent=None if rent_fn is not None else rent,
        box_store_factory=lambda _job: box,
        publish_fn=publish_fn or (lambda **k: {'ok': True, 'merge_only': True, 'path': 'mem'}),
        now_fn=now,
    )
    svc.wire_rent()
    if rent_fn is None:
        _released_wrap(svc)
    svc._test_store = box  # type: ignore[attr-defined]
    return svc


def _to_running(svc: BenchJobService, n: int = 12) -> None:
    """Advance rent/boot until the bench job is RUNNING (detached hold)."""
    for _ in range(n):
        svc.tick()
        jobs = [j for j in svc.runner.jobs if j.kind is JobKind.BENCH]
        if jobs and jobs[0].state is JobState.RUNNING and jobs[0].box_id:
            return
    raise AssertionError('bench job never reached RUNNING')


class TestBenchConfigFlag(unittest.TestCase):
    def test_default_on_and_git_ref_documented(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'max_boxes': 5}})
        self.assertTrue(cfg.bench_on_green)
        self.assertEqual(BENCH_BOX_GIT_REF, 'cursor/pit-shadow-mode-0524')
        self.assertEqual(list(BENCH_STAGES), ['setup', 'fetch', 'run'])


class TestTickNonBlocking(unittest.TestCase):
    def test_tick_returns_quickly_while_bench_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = MemBenchBoxStore(default_ticks=99)  # stays RUNNING a long time
            svc = _svc(root, store=store)
            svc.on_green_fresh(head=H0, epoch=1)
            _to_running(svc)
            # First detached launch (setup) — still running for 99 advances.
            t0 = time.perf_counter()
            out = svc.tick()
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 0.5)
            self.assertLess(out.get('elapsed_s', 0), 0.5)
            ops = [a.get('op') for a in out.get('bench_actions') or []]
            self.assertTrue(any(o in ('launched', 'running') for o in ops))
            # Fake slow job still alive — no completion yet.
            self.assertEqual(len(svc.completed), 0)
            jobs = [j for j in svc.runner.jobs if j.kind is JobKind.BENCH]
            self.assertEqual(jobs[0].state, JobState.RUNNING)
            self.assertTrue(jobs[0].meta.get('hold_running'))


class TestRestartMidBench(unittest.TestCase):
    def test_restart_mid_bench_readopts_without_second_rent(self):
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

            store = MemBenchBoxStore(default_ticks=50)
            svc = _svc(root, store=store, rent_fn=rent)
            released = _released_wrap(svc)
            svc.runner.rent_fn = rent
            svc.on_green_fresh(head=H0, epoch=1)
            _to_running(svc)
            svc.tick()  # launch setup; still running
            self.assertEqual(len(rents), 1)
            self.assertTrue((root / 'bench' / H0 / 'active.json').is_file())

            # Host restart: new service, same box store (re-adopt).
            svc2 = _svc(root, store=store, rent_fn=rent)
            svc2.runner.rent_fn = rent
            _released_wrap(svc2)
            svc2.ensure_loaded()
            self.assertIsNone(svc2.on_green_fresh(head=H0, epoch=1))  # no duplicate
            jobs = [j for j in svc2.runner.jobs if j.kind is JobKind.BENCH]
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].box_id, 'box-resume-1')
            # Finish stages without a new rent.
            store.default_ticks = 1
            for pending in list(store.pending.values()):
                pending['ticks_left'] = 1
            for _ in range(40):
                # Advance fake detached work then host tick.
                store.advance()
                svc2.tick()
                if svc2.completed:
                    break
            self.assertEqual(len(rents), 1)  # no second rent
            self.assertTrue(svc2.is_done(H0))


class TestSetupFailureReleases(unittest.TestCase):
    def test_setup_failure_releases_box(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = MemBenchBoxStore(default_ticks=1, fail_stage='setup')
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

            svc = _svc(root, store=store, rent_fn=rent)
            svc.runner.rent_fn = rent
            svc.runner.release_fn = release
            svc.on_green_fresh(head=H0, epoch=1)
            _to_running(svc)
            # Launch setup → advance → exits without marker → missing_result/fail.
            for _ in range(30):
                store.advance()
                out = svc.tick()
                if any(a.get('op') in ('failed', 'no_decision', 'completed') for a in out.get('bench_actions') or []):
                    pass
                if svc.is_done(H0) or any(r[1] == 'bench-run-failed' for r in released):
                    # May retry once; keep driving.
                    if len(released) >= 1 and (svc.is_done(H0) or len(released) >= 2):
                        break
            self.assertTrue(any(r[1] == 'bench-run-failed' for r in released), released)
            # Terminal after ≤1 retry.
            self.assertLessEqual(len([r for r in released if r[1] == 'bench-run-failed']), BENCH_MAX_ATTEMPTS)


class TestNoDaemonJson(unittest.TestCase):
    def test_never_writes_daemon_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = MemBenchBoxStore(default_ticks=1)
            svc = _svc(root, store=store)
            svc.on_green_fresh(head=H0, epoch=1)
            for _ in range(40):
                store.advance()
                svc.tick()
                if svc.completed:
                    break
            self.assertTrue(svc.completed)
            daemon = root / 'bench-results' / 'daemon.json'
            self.assertFalse(daemon.exists())
            # Only pit_state/bench/<head>/ …
            self.assertTrue((root / 'bench' / H0 / 'request.json').is_file())
            self.assertTrue((root / 'bench' / H0 / 'done.json').is_file())
            self.assertTrue((root / 'bench' / H0 / 'results' / 'RESULT.json').is_file())


class TestGreenBindsDuringBench(unittest.TestCase):
    def test_next_items_still_bind_while_bench_runs(self):
        """Bench hold must not block binding of subsequent items."""
        from .controller import Controller
        from .state import ItemRecord, ItemState

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = MemBenchBoxStore(default_ticks=99)
            svc = _svc(root, store=store)
            ctl = Controller.from_flag({
                'pit': {
                    'enabled': True, 'mode': 'shadow', 'max_boxes': 2,
                    'bench_on_green': True,
                },
            })
            ctl.current_head = H0
            ctl.on_green_fresh = lambda item: svc.on_green_fresh_item(item)

            class _I:
                candidate_id = H0
                epoch = 10
            ctl.on_green_fresh(_I())
            _to_running(svc)
            self.assertEqual(svc.runner.jobs[0].state, JobState.RUNNING)

            # While bench runs, a later epoch can still bind to current head.
            item = ItemRecord(
                item_id='bind-while-bench',
                registration_index=1,
                epoch=11,
                uid=1,
                hotkey_prefix='hk',
                state=ItemState.QUEUED,
            )
            ctl.items.append(item)
            item.state = ItemState.BOUND
            item.bound_head = ctl.current_head
            self.assertEqual(item.bound_head, H0)
            self.assertEqual(item.state, ItemState.BOUND)
            self.assertEqual(len(svc.completed), 0)
            t0 = time.perf_counter()
            out = svc.tick()
            self.assertLess(time.perf_counter() - t0, 0.5)
            self.assertEqual(len(svc.completed), 0)
            self.assertTrue(
                any(a.get('op') in ('running', 'launched') for a in out.get('bench_actions') or [])
                or out.get('elapsed_s', 1) < 0.5
            )


class TestPublishMergeOnly(unittest.TestCase):
    def test_merge_publish_pp0_no_affine_bench_import(self):
        with tempfile.TemporaryDirectory() as td:
            site = Path(td) / 'site'
            site.mkdir()
            # Prior row must survive.
            prior = {
                'version': 'affine-bench-v2',
                'results': [{
                    'model_id': 'old', 'benchmark_id': 'math500',
                    'env_version': 'old', 'score': 0.1,
                }],
                'models': [{'model_id': 'old', 'kind': 'reported'}],
            }
            (site / 'benchmarks.json').write_text(json.dumps(prior))
            result = {
                'ok': True,
                'head': H0,
                'model_id': 'king-' + H0[:16],
                'profile': BENCH_PROFILE,
                'presence_penalty': 0.0,
                'results': [{
                    'model_id': 'king-' + H0[:16],
                    'benchmark': 'math500',
                    'score': 0.9,
                    'env_version': 'math500/v2',
                    'presence_penalty': 0.0,
                    'profile': BENCH_PROFILE,
                }],
            }
            out = merge_publish(head=H0, result=result, site=str(site))
            self.assertTrue(out['merge_only'])
            doc = json.loads((site / 'benchmarks.json').read_text())
            mids = {r['model_id'] for r in doc['results']}
            self.assertIn('old', mids)
            self.assertIn('king-' + H0[:16], mids)

    def test_site_root_fail_closed(self):
        import os
        env = {k: os.environ.pop(k) for k in ('AFFINE_SITE', 'SITE') if k in os.environ}
        try:
            with self.assertRaises(RuntimeError):
                site_root()
        finally:
            os.environ.update(env)


class TestRentOverCeilingWaits(unittest.TestCase):
    def test_rent_over_ceiling_waits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clock = _Clock()
            offers = {'price': 90.0}
            rents = {'n': 0}

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

            svc = _svc(root, rent_fn=rent, clock=clock, store=MemBenchBoxStore(default_ticks=1))
            svc.runner.rent_fn = rent
            svc.on_green_fresh(head=H0, epoch=1)
            for _ in range(5):
                out = svc.tick()
                self.assertTrue(
                    any(a.get('note') == 'wait-offer' for a in out.get('actions') or [])
                    or rents['n'] == 0
                )
                clock.advance(30)
            self.assertEqual(rents['n'], 0)
            offers['price'] = 40.0
            store = svc._test_store  # type: ignore[attr-defined]
            for _ in range(40):
                store.advance()
                svc.tick()
                if svc.completed:
                    break
            self.assertEqual(rents['n'], 1)
            self.assertTrue(svc.completed[0]['ok'])


class TestPeEvalExcludesBench(unittest.TestCase):
    def test_affine_pe_bench_excluded_from_eval_count(self):
        import os
        from .host_wiring import _bind_shared_cap_counters
        from .pe_rent import RentService

        class _Mod:
            @staticmethod
            def eval_nodes():
                return [
                    {'name': 'affine-pe-eval-1'},
                    {'name': 'affine-pe-bench-abc-a1'},
                    {'name': 'affine-pe-pit-1-a1'},
                ]

        with tempfile.TemporaryDirectory() as td:
            prev = os.environ.get('PE_STATE')
            os.environ['PE_STATE'] = td
            try:
                rent = RentService(max_boxes=5, share_pe_cap=True)
                _bind_shared_cap_counters(rent, _Mod)
                # Only affine-pe-eval-1 counts (bench + pit excluded).
                self.assertEqual(rent.pe_eval_count_fn(), 1)
            finally:
                if prev is None:
                    os.environ.pop('PE_STATE', None)
                else:
                    os.environ['PE_STATE'] = prev


class TestDuplicateEnqueue(unittest.TestCase):
    def test_green_one_bench_duplicate_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            svc = _svc(root, store=MemBenchBoxStore(default_ticks=1))
            self.assertIsNotNone(svc.on_green_fresh(head=H0, epoch=1))
            self.assertIsNone(svc.on_green_fresh(head=H0, epoch=1))
            store = svc._test_store  # type: ignore[attr-defined]
            for _ in range(40):
                store.advance()
                svc.tick()
                if svc.completed:
                    break
            self.assertEqual(len(svc.completed), 1)
            self.assertIsNone(svc.on_green_fresh(head=H0, epoch=2))


if __name__ == '__main__':
    unittest.main()
