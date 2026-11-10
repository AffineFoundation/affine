#!/usr/bin/env python3
"""On-demand PIT GREEN bench: detached, non-blocking, restart-safe."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ops.jobs.kinds import JobKind
from ops.jobs.runner import JobRunner, JobState, WaitForOffer

from . import BENCH_PROFILE
from .bench_box import (
    BENCH_BOX_GIT_REF,
    BENCH_STAGES,
    MemBenchBoxStore,
    job_dir_for,
    setup_command,
)
from .bench_fetch_inputs import push_fetch_inputs
from .bench_job import (
    BENCH_MAX_ATTEMPTS,
    BenchJobService,
    build_bench_rent_service,
    build_bench_service,
)
from .bench_publish import results_root, write_results
from .config import load_pit_config


H0 = 'a' * 64
H1 = 'b' * 64
FILES = {'model.safetensors': 'c' * 64, 'config.json': 'd' * 64}
FAKE_URLS = {
    'model.safetensors': 'https://example.invalid/model.safetensors?X-Amz-Signature=deadbeef',
    'config.json': 'https://example.invalid/config.json?X-Amz-Signature=cafebabe',
}


class _Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _resolve(head: str, files=None, **_k: Any) -> Dict[str, str]:
    if files:
        return {str(k): (v if isinstance(v, str) else v['sha256']) for k, v in files.items()}
    return dict(FILES)


def _presign(head: str, files: Mapping[str, Any]) -> Dict[str, str]:
    return {name: FAKE_URLS[name] for name in files}


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
    results_dir: Optional[str] = None,
    clock: Optional[_Clock] = None,
    default_ticks: int = 1,
) -> BenchJobService:
    now = clock or _Clock()
    box = store or MemBenchBoxStore(default_ticks=default_ticks)
    rent = build_bench_rent_service(price_ceiling_usd_h=56.0)
    runner = JobRunner(live=True, price_ceiling_usd_h=56.0, now_fn=now)
    if rent_fn is not None:
        runner.rent_fn = rent_fn
    if results_dir is None:
        results_dir = str(root / 'bench-results-out')
    svc = BenchJobService(
        runner=runner,
        state_root=root,
        rent=None if rent_fn is not None else rent,
        box_store_factory=lambda _job: box,
        publish_fn=publish_fn,
        results_dir=results_dir,
        resolve_files_fn=_resolve,
        presign_fn=_presign,
        now_fn=now,
    )
    svc.wire_rent()
    if rent_fn is None:
        _released_wrap(svc)
    svc._test_store = box  # type: ignore[attr-defined]
    return svc


def _enqueue(svc: BenchJobService, head: str = H0, epoch: int = 1) -> None:
    svc.on_green_fresh(head=head, epoch=epoch, files=FILES)


def _to_running(svc: BenchJobService, n: int = 12) -> None:
    """Advance rent/boot until the bench job is RUNNING (detached hold)."""
    for _ in range(n):
        svc.tick()
        jobs = [j for j in svc.runner.jobs if j.kind is JobKind.BENCH]
        if jobs and jobs[0].state is JobState.RUNNING and jobs[0].box_id:
            return
    raise AssertionError('bench job never reached RUNNING')


def _drive_done(svc: BenchJobService, store: MemBenchBoxStore, n: int = 40) -> None:
    for _ in range(n):
        store.advance()
        svc.tick()
        if svc.completed:
            return
    raise AssertionError('bench never completed')


class TestBenchConfigFlag(unittest.TestCase):
    def test_default_on_and_git_ref_documented(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'max_boxes': 5}})
        self.assertTrue(cfg.bench_on_green)
        self.assertIsNone(cfg.bench_results_dir)
        self.assertEqual(BENCH_BOX_GIT_REF, 'cursor/pit-shadow-mode-0524')
        self.assertEqual(list(BENCH_STAGES), ['setup', 'fetch', 'run'])

    def test_bench_results_dir_from_config(self):
        cfg = load_pit_config({
            'pit': {
                'enabled': True, 'mode': 'shadow',
                'bench_results_dir': '$BENCH_RESULTS_DIR',
            },
        })
        self.assertEqual(cfg.bench_results_dir, '$BENCH_RESULTS_DIR')


class TestShaSetupCommand(unittest.TestCase):
    def test_sha_ref_uses_init_fetch_depth_1(self):
        sha = '0123456789abcdef0123456789abcdef01234567'
        cmd = setup_command(git_ref=sha)
        self.assertIn('git -C', cmd)
        self.assertIn('init', cmd)
        self.assertIn('fetch --depth 1 origin', cmd)
        self.assertIn(sha, cmd)
        self.assertNotIn('git clone --depth 1 --branch', cmd)
        # Full clone fallback must not appear on the SHA path.
        self.assertNotIn('git clone https://', cmd.replace('git -C', ''))

    def test_branch_ref_still_shallow_clones(self):
        cmd = setup_command(git_ref='cursor/pit-shadow-mode-0524')
        self.assertIn('git clone --depth 1 --branch', cmd)
        self.assertNotIn('git init', cmd)


class TestFetchInputsPushed(unittest.TestCase):
    def test_host_pushes_descriptor_and_urls_before_fetch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = MemBenchBoxStore(default_ticks=1)
            svc = _svc(root, store=store)
            _enqueue(svc)
            _to_running(svc)
            # Drive setup → launch fetch (prepare_fetch pushes inputs).
            saw_fetch = False
            for _ in range(20):
                store.advance()
                out = svc.tick()
                for a in out.get('bench_actions') or []:
                    if a.get('op') == 'launched' and a.get('stage') == 'fetch':
                        saw_fetch = True
                        jd = job_dir_for(H0)
                        desc = store.read_text('%s/descriptor.json' % jd)
                        urls = store.read_text('%s/urls.private.json' % jd)
                        self.assertIsNotNone(desc)
                        self.assertIsNotNone(urls)
                        d = json.loads(desc)
                        u = json.loads(urls)
                        self.assertEqual(d['id'], H0)
                        self.assertEqual(d['files'], FILES)
                        self.assertEqual(set(u), set(FILES))
                        # Tick note must not carry URL material.
                        self.assertNotIn('urls', a.get('fetch_prep') or {})
                        blob = json.dumps(a)
                        self.assertNotIn('example.invalid', blob)
                        self.assertNotIn('X-Amz-Signature', blob)
                        # Host state keeps descriptor, never urls.private.json.
                        host_desc = root / 'bench' / H0 / 'descriptor.json'
                        self.assertTrue(host_desc.is_file())
                        self.assertFalse((root / 'bench' / H0 / 'urls.private.json').exists())
                        break
                if saw_fetch:
                    break
            self.assertTrue(saw_fetch)
            # After fetch completes, box deletes urls.
            for _ in range(10):
                store.advance()
                svc.tick()
                jd = job_dir_for(H0)
                if store.read_text('%s/markers/fetch.done' % jd):
                    self.assertIsNone(store.read_text('%s/urls.private.json' % jd))
                    break
            # Events must not log URLs.
            events = (root / 'bench' / H0 / 'events.jsonl').read_text()
            self.assertIn('bench_fetch_inputs', events)
            self.assertNotIn('example.invalid', events)
            self.assertNotIn('X-Amz-Signature', events)

    def test_push_fetch_inputs_unit(self):
        store = MemBenchBoxStore()
        jd = '/data/bench/jobs/testhead'
        receipt = push_fetch_inputs(
            store, jd, head=H0, files=FILES,
            resolve_fn=_resolve, presign_fn=_presign,
        )
        self.assertTrue(receipt['ok'])
        self.assertEqual(receipt['n_files'], 2)
        self.assertNotIn('urls', receipt)
        self.assertIsNotNone(store.read_text('%s/descriptor.json' % jd))
        self.assertIsNotNone(store.read_text('%s/urls.private.json' % jd))
        # Receipt must not contain URL strings.
        self.assertNotIn('http', json.dumps(receipt))


class TestTickNonBlocking(unittest.TestCase):
    def test_tick_returns_quickly_while_bench_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = MemBenchBoxStore(default_ticks=99)  # stays RUNNING a long time
            svc = _svc(root, store=store)
            _enqueue(svc)
            _to_running(svc)
            t0 = time.perf_counter()
            out = svc.tick()
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 0.5)
            self.assertLess(out.get('elapsed_s', 0), 0.5)
            ops = [a.get('op') for a in out.get('bench_actions') or []]
            self.assertTrue(any(o in ('launched', 'running') for o in ops))
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
            _released_wrap(svc)
            svc.runner.rent_fn = rent
            _enqueue(svc)
            _to_running(svc)
            svc.tick()  # launch setup; still running
            self.assertEqual(len(rents), 1)
            self.assertTrue((root / 'bench' / H0 / 'active.json').is_file())

            svc2 = _svc(root, store=store, rent_fn=rent)
            svc2.runner.rent_fn = rent
            _released_wrap(svc2)
            svc2.ensure_loaded()
            self.assertIsNone(svc2.on_green_fresh(head=H0, epoch=1, files=FILES))
            jobs = [j for j in svc2.runner.jobs if j.kind is JobKind.BENCH]
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].box_id, 'box-resume-1')
            store.default_ticks = 1
            for pending in list(store.pending.values()):
                pending['ticks_left'] = 1
            _drive_done(svc2, store)
            self.assertEqual(len(rents), 1)
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
            _enqueue(svc)
            _to_running(svc)
            for _ in range(30):
                store.advance()
                svc.tick()
                if svc.is_done(H0) or any(r[1] == 'bench-run-failed' for r in released):
                    if len(released) >= 1 and (svc.is_done(H0) or len(released) >= 2):
                        break
            self.assertTrue(any(r[1] == 'bench-run-failed' for r in released), released)
            self.assertLessEqual(
                len([r for r in released if r[1] == 'bench-run-failed']), BENCH_MAX_ATTEMPTS,
            )


class TestNoDaemonJson(unittest.TestCase):
    def test_never_writes_daemon_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = MemBenchBoxStore(default_ticks=1)
            svc = _svc(root, store=store)
            _enqueue(svc)
            _drive_done(svc, store)
            self.assertTrue(svc.completed)
            daemon = root / 'bench-results' / 'daemon.json'
            self.assertFalse(daemon.exists())
            self.assertTrue((root / 'bench' / H0 / 'request.json').is_file())
            self.assertTrue((root / 'bench' / H0 / 'done.json').is_file())
            self.assertTrue((root / 'bench' / H0 / 'results' / 'RESULT.json').is_file())


class TestGreenBindsDuringBench(unittest.TestCase):
    def test_next_items_still_bind_while_bench_runs(self):
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
                meta = {'candidate_files': FILES}

            ctl.on_green_fresh(_I())
            _to_running(svc)
            self.assertEqual(svc.runner.jobs[0].state, JobState.RUNNING)

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


class TestWriteResultsForPublishFleet(unittest.TestCase):
    def test_lands_results_dir_not_benchmarks_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results = root / 'bench-results-out'
            site = root / 'site'
            site.mkdir()
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
            out = write_results(
                head=H0, result=result, results_dir_path=str(results), now=1.0,
            )
            self.assertTrue(out['merge_only'])
            self.assertTrue(out['landed'])
            mid = 'king-' + H0[:16]
            landed = results / 'results' / mid / 'math500.json'
            self.assertTrue(landed.is_file())
            doc = json.loads(landed.read_text())
            self.assertEqual(doc['model_id'], mid)
            self.assertEqual(doc['benchmark'], 'math500')
            self.assertEqual(doc['presence_penalty'], 0.0)
            self.assertEqual(doc['score'], 0.9)
            # website/benchmarks.json untouched — publish_fleet owns the merge.
            untouched = json.loads((site / 'benchmarks.json').read_text())
            self.assertEqual(untouched['results'][0]['model_id'], 'old')
            self.assertEqual(len(untouched['results']), 1)

    def test_results_root_fail_closed(self):
        env = {
            k: os.environ.pop(k)
            for k in ('BENCH_RESULTS_DIR', 'AFFINE_BENCH_RESULTS')
            if k in os.environ
        }
        try:
            with self.assertRaises(RuntimeError):
                results_root()
        finally:
            os.environ.update(env)

    def test_end_to_end_lands_into_results_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results = root / 'pub-results'
            store = MemBenchBoxStore(default_ticks=1)
            svc = _svc(root, store=store, results_dir=str(results))
            _enqueue(svc)
            _drive_done(svc, store)
            mid = 'king-' + H0[:16]
            landed = results / 'results' / mid / 'math500.json'
            self.assertTrue(landed.is_file(), list((results / 'results').rglob('*')))
            row = json.loads(landed.read_text())
            self.assertEqual(row['model_id'], mid)
            self.assertEqual(row['benchmark'], 'math500')
            self.assertEqual(float(row['presence_penalty']), 0.0)
            # Never wrote website/benchmarks.json from PIT.
            self.assertFalse((root / 'website' / 'benchmarks.json').exists())
            pub = json.loads((root / 'bench' / H0 / 'publish.json').read_text())
            self.assertTrue(pub.get('landed'))
            self.assertNotIn('urls', pub)


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
            _enqueue(svc)
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
            _drive_done(svc, store)
            self.assertEqual(rents['n'], 1)
            self.assertTrue(svc.completed[0]['ok'])


class TestPeEvalExcludesBench(unittest.TestCase):
    def test_affine_pe_bench_excluded_from_eval_count(self):
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
            self.assertIsNotNone(svc.on_green_fresh(head=H0, epoch=1, files=FILES))
            self.assertIsNone(svc.on_green_fresh(head=H0, epoch=1, files=FILES))
            store = svc._test_store  # type: ignore[attr-defined]
            _drive_done(svc, store)
            self.assertEqual(len(svc.completed), 1)
            self.assertIsNone(svc.on_green_fresh(head=H0, epoch=2, files=FILES))


if __name__ == '__main__':
    unittest.main()
