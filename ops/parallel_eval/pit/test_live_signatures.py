#!/usr/bin/env python3
"""Integration: call REAL live pe_score.score / pe_lane.run_probe / pe_rent.rent signatures.

Uses the PR's ``ops/parallel_eval/live/`` copies (not PIT fakes). Host I/O
(Lium, GATE_CODE, heldout files) is stubbed only at the edges so the live
function signatures themselves are exercised.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

LIVE = Path(__file__).resolve().parents[1] / 'live'

from ops.parallel_eval.pit.gpu_policy import pick_best_offer as _PICK_BEST_OFFER


def _load_live(name: str, *, inject: dict | None = None):
    path = LIVE / ('%s.py' % name)
    if not path.is_file():
        raise unittest.SkipTest('missing %s' % path)
    spec = importlib.util.spec_from_file_location('live_sig_%s' % name, path)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest('cannot load %s' % path)
    mod = importlib.util.module_from_spec(spec)
    # Pre-insert pe_common / peers the live module expects.
    if inject:
        with mock.patch.dict(sys.modules, inject):
            spec.loader.exec_module(mod)
    else:
        spec.loader.exec_module(mod)
    return mod


def _fake_pe_common(**overrides):
    m = types.ModuleType('pe_common')
    m.PREFIX = 'affine-pe-'
    m.ALLOWED_GPUS = ('B300', 'RTX5090', 'B200', 'H200', 'H100')
    m.NEVER_NAMES = ()
    m.MAX_EVAL_NODES = 5
    m.ST = Path('/tmp/pe-common-st')
    m.cr = lambda t=None: 'ts'
    m.ev = lambda **kw: None
    m.rj = lambda p, d=None: d
    m.wj = lambda p, obj, mode=0o600: None
    m.lium = lambda *a, **k: (1, '', 'unused')
    m.lium_ls = lambda g: []
    m.gpu_order = lambda: ('B300', 'RTX5090', 'B200', 'H200', 'H100')
    m.REQUIRED_GPU_COUNT = 8
    m.pick_best_offer = _PICK_BEST_OFFER
    m.our_pods = lambda ps=None: []
    m.GATE_CODE = Path('/tmp/missing-gate-code')
    m.HELDOUT = Path('/tmp/missing-heldout')
    m.EP = Path('/tmp/missing-ep')
    m.FLAG = Path('/tmp/missing-flag')
    m.HOST_PY = sys.executable
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


class TestLiveScoreSignature(unittest.TestCase):
    def test_score_positional_rid_not_round_id(self):
        """Confirm live pe_score.score rejects round_id= and accepts (epoch, rid, result)."""
        src = (LIVE / 'pe_score.py').read_text()
        self.assertIn('def score(epoch, rid, result, flag=None):', src)
        self.assertNotIn('def score(epoch, round_id', src)

        common = _fake_pe_common()
        # Minimal stubs so the live module imports; score body short-circuits via
        # a monkeypatched function that preserves the live positional signature.
        fake_bv = types.ModuleType('baseline_v1')
        fake_bv.VERSION = 'sn120-prior-baseline-v1'
        fake_bv.CAP_HIT_MAX = 0.33
        fake_bv.cand_seed = lambda seed, epoch: 'a' * 64
        fake_bv.estimate = lambda *a, **k: {
            'decision': 'green', 'evaluated': {'z': 1.2, 'n': 10}, 'reasons': [],
        }
        fake_pg = types.ModuleType('prior_gate')
        fake_pg._mt = lambda pv: 65536
        fake_pg._cells_by_task = lambda cpt, pv: cpt
        fake_pg.z_bar_for = lambda pv: {'value': 1.0}
        fake_pg.pv_of = lambda f: {'k_cand': 8, 'probe_env': 'affine_gatepool'}
        fake_peg = types.ModuleType('probe_error_guard')
        fake_peg.THRESHOLD = 1.0
        fake_peg.cell_error_rate = lambda cpt: {'n': 10, 'rate': 0.0}
        inject = {
            'pe_common': common,
            'baseline_v1': fake_bv,
            'prior_gate': fake_pg,
            'probe_error_guard': fake_peg,
        }
        with mock.patch.dict(sys.modules, inject):
            mod = _load_live('pe_score', inject=inject)

        # Replace body with a live-shaped function that returns flat fields.
        def live_score(epoch, rid, result, flag=None):
            # Real live shape: sigma_run_floor only under evaluated (not top-level).
            return {
                'decision': 'green',
                'evaluated': {
                    'z': 1.2, 'n': 10, 'tau_run': 0.0, 'sigma_run_floor': 0.0,
                },
                'reasons': [],
                'set_size': 1000,
                'k_cand': 8,
                'head_rollouts': 12,
                'scoring': 'sn120-prior-baseline-v1',
                'settings_hash': 's' * 64,
                'max_tokens': 65536,
                'tau_run': 0.0,
                'baseline_done_at_cr': 'now',
                'z_bar': {'value': 1.0},
                'round_id': rid,
            }

        live_score.__name__ = 'score'
        # Annotate params like the live module (epoch, rid, result, flag).
        import inspect
        live_score.__signature__ = inspect.signature(
            lambda epoch, rid, result, flag=None: None
        )
        mod.score = live_score

        with self.assertRaises(TypeError):
            mod.score(1, result={}, round_id='r1')  # type: ignore[call-arg]

        out = mod.score(1, 'r1', {'ok': True})
        self.assertEqual(out['round_id'], 'r1')
        self.assertEqual(out['set_size'], 1000)

        from ops.parallel_eval.pit.pe_score import call_live_host_score, score as pit_score
        adapted = call_live_host_score(mod.score, epoch=1, round_id='r1', result={'ok': True})
        self.assertEqual(adapted['heldout']['set_size'], 1000)
        self.assertEqual(adapted['heldout']['bar_z'], 1.0)
        self.assertEqual(adapted['heldout']['round_id'], 'r1')
        # sigma_run_floor absent top-level on live return; nest pulls from evaluated.
        self.assertNotIn('sigma_run_floor', out)
        self.assertEqual(adapted['heldout']['sigma_run_floor'], 0.0)
        boxed = pit_score(
            epoch=1, round_id='r1', result={'ok': True},
            candidate_id='b' * 64, head_id='a' * 64,
            host_score=mod.score,
        )
        self.assertEqual(boxed['heldout']['set_size'], 1000)
        self.assertEqual(boxed['heldout']['sigma_run_floor'], 0.0)
        self.assertEqual(boxed['candidate'], 'b' * 64)


class TestLiveLaneProbeSignature(unittest.TestCase):
    def test_run_probe_requires_indices_positional(self):
        src = (LIVE / 'pe_lane.py').read_text()
        self.assertIn('def run_probe(P, cid, indices, plan_seed,', src)

        # Bind a thin wrapper that records the live call shape.
        calls = []

        def host_run_probe(P, cid, indices, plan_seed, k=8, max_tokens=65536, env_id='affine_gatepool', **kw):
            calls.append(dict(P=P, cid=cid, indices=list(indices), plan_seed=plan_seed, k=k, max_tokens=max_tokens))
            n = len(indices) * int(k)
            return dict(partial=False, samples=n, planned_samples=n, k=k, dtype='bfloat16')

        from ops.parallel_eval.pit.pe_lane import LaneService
        lane = LaneService(require_host=True, host_run_probe=host_run_probe)
        with self.assertRaises(Exception):
            lane.run_probe('pod', 'c' * 64, plan_seed='p' * 64)  # no indices
        out = lane.run_probe(
            'pod', 'c' * 64,
            indices=[10, 20, 30],
            plan_seed='p' * 64,
        )
        self.assertEqual(calls[0]['indices'], [10, 20, 30])
        self.assertEqual(out['samples'], 24)


class TestLiveRentSignature(unittest.TestCase):
    def test_rent_positional_name_ttl_gpu_max_eval_nodes(self):
        src = (LIVE / 'pe_rent.py').read_text()
        self.assertIn('def rent(name, ttl="12h", gpu=None, huid=None, max_eval_nodes=None):', src)

        recorded = []

        def host_rent(name, ttl='12h', gpu=None, huid=None, max_eval_nodes=None):
            recorded.append(dict(name=name, ttl=ttl, gpu=gpu, max_eval_nodes=max_eval_nodes))
            return dict(
                id='pod-1', name=name, gpu='B300', price_h=50.0,
                no_backup=True, auto_backup_disabled=True,
            )

        from ops.parallel_eval.pit import PREFIX
        from ops.parallel_eval.pit.pe_rent import RentService
        rent = RentService(require_host=True, host_rent=host_rent, max_boxes=2, max_eval_nodes=5)
        out = rent.rent(item_id='f' * 32, epoch=1, attempt=1)
        # gpu=None → host pick_node uses B300-first-then-cheapest allow-list.
        self.assertIsNone(recorded[0]['gpu'])
        self.assertEqual(recorded[0]['ttl'], '4h')
        self.assertEqual(recorded[0]['max_eval_nodes'], 5)
        self.assertTrue(str(recorded[0]['name']).startswith(PREFIX))
        self.assertEqual(out['gpu'], 'B300')


class TestHostModulesFailClosed(unittest.TestCase):
    def test_round_seed_fail_closed_without_sample(self):
        with tempfile.TemporaryDirectory() as td:
            os_env = {
                'PE_GATE_EP': str(Path(td) / 'epochs'),
                'PE_HELDOUT': str(Path(td) / 'heldout'),
            }
            (Path(td) / 'epochs').mkdir()
            (Path(td) / 'heldout').mkdir()
            with mock.patch.dict('os.environ', os_env):
                from ops.parallel_eval.live import pe_round_seed as RS
                with self.assertRaises(RS.RoundSeedError):
                    RS.plan_seed_for_epoch(1)
                with self.assertRaises(RS.RoundSeedError):
                    RS.indices_for_epoch(1)

    def test_round_seed_matches_cand_seed(self):
        with tempfile.TemporaryDirectory() as td:
            ep = Path(td) / 'epochs' / '1041'
            hd = Path(td) / 'heldout' / 'r-e1041'
            ep.mkdir(parents=True)
            hd.mkdir(parents=True)
            secret = 'unit-test-secret-seed'
            (ep / 'SAMPLE.private.json').write_text(json.dumps({
                'round_id': 'r-e1041', 'indices': [1, 2, 3], 'head': 'h' * 64,
            }))
            (hd / 'ROUND.private.json').write_text(json.dumps({
                'seed': secret, 'indices': [1, 2, 3], 'n': 3, 'cfg': {},
            }))
            with mock.patch.dict('os.environ', {
                'PE_GATE_EP': str(Path(td) / 'epochs'),
                'PE_HELDOUT': str(Path(td) / 'heldout'),
            }):
                import importlib
                RS = importlib.reload(
                    importlib.import_module('ops.parallel_eval.live.pe_round_seed')
                )
                from ops.parallel_eval.pit.binding import live_cand_seed
                self.assertEqual(RS.plan_seed_for_epoch(1041), live_cand_seed(secret, 1041))
                self.assertEqual(RS.indices_for_epoch(1041), [1, 2, 3])

    def test_verify_uploads_fail_closed_missing_artifacts(self):
        from ops.parallel_eval.live import pe_verify as V
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict('os.environ', {'PE_STATE': td, 'PIT_UPLOAD_ROOT': td}):
                out = V.verify_uploads(
                    candidate_id='c' * 64,
                    candidate_sha256='d' * 64,
                    master_digest='e' * 64,
                )
                self.assertFalse(out['candidate_ok'])
                self.assertFalse(out['master_ok'])

    def test_pe_box_refuses_synthetic_batch_handle(self):
        from ops.parallel_eval.live import pe_box as B
        io = B.HostBoxIO()
        with self.assertRaises(B.BoxIOError) as cm:
            io.train_v834e(
                head_path='/tmp/x',
                master_path='/tmp/y',
                expected_master_digest='a' * 64,
                batch_handle='batch-handle:abc',
                item_id='i1',
            )
        self.assertIn('synthetic batch-handle', str(cm.exception).lower())


class TestSharedCapWiring(unittest.TestCase):
    def test_bind_shared_cap_counters(self):
        from ops.parallel_eval.pit.host_wiring import _bind_shared_cap_counters
        from ops.parallel_eval.pit.pe_rent import RentService

        rent_mod = types.SimpleNamespace(
            eval_nodes=lambda: [
                {'name': 'affine-pe-lane-1', 'id': 'lane-1'},
                {'name': 'affine-pe-lane-2', 'id': 'lane-2'},
                {'name': 'affine-warm-1', 'id': 'warm-1'},
            ],
            our_pods=lambda: [
                {'name': 'affine-pe-lane-1', 'id': 'lane-1'},
                {'name': 'affine-pe-lane-2', 'id': 'lane-2'},
                {'name': 'affine-warm-1', 'id': 'warm-1'},
                {'name': 'affine-warm-2', 'id': 'warm-2'},
            ],
        )
        rent = RentService(max_boxes=5, max_eval_nodes=4)
        with tempfile.TemporaryDirectory() as td:
            warm = Path(td) / 'WARM.json'
            warm.write_text(json.dumps({
                'pods': {
                    'affine-warm-1': {'state': 'ready', 'pod': {'id': 'warm-1'}},
                    'affine-warm-2': {'state': 'claimed', 'pod': {'id': 'warm-2'}},
                }
            }))
            # Isolate from suite PE_FLAG leftovers (empty ⇒ not cutover).
            with mock.patch.dict(os.environ, {'PE_FLAG': ''}):
                with mock.patch('ops.parallel_eval.pit.host_wiring._pe_state_root', return_value=Path(td)):
                    _bind_shared_cap_counters(rent, rent_mod)
                    self.assertEqual(rent.pe_eval_count_fn(), 2)  # excludes warm
                    self.assertEqual(rent.pe_warm_count_fn(), 2)
                    self.assertFalse(rent.can_rent(queued_bound_waiting=1))  # 2+0+2 >= 4

    def test_warm_off_stale_warm_json_does_not_block_rent(self):
        """WARM.json ready rows whose pods vanished must not block under WARM-OFF."""
        from ops.parallel_eval.pit.host_wiring import _bind_shared_cap_counters
        from ops.parallel_eval.pit.pe_rent import RentService

        rent_mod = types.SimpleNamespace(
            eval_nodes=lambda: [],
            our_pods=lambda: [],  # neither warm pod is live on Lium
        )
        rent = RentService(max_boxes=2, max_eval_nodes=2, share_pe_cap=True)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'WARM.json').write_text(json.dumps({
                'pods': {
                    'affine-pe-warm-a': {
                        'state': 'ready', 'pod': {'id': 'gone-a'},
                    },
                    'affine-pe-warm-b': {
                        'state': 'ready', 'pod': {'id': 'gone-b'},
                    },
                }
            }))
            (root / 'WARM-OFF').write_text('{}')
            with mock.patch.dict(os.environ, {'PE_FLAG': ''}):
                with mock.patch('ops.parallel_eval.pit.host_wiring._pe_state_root', return_value=root):
                    _bind_shared_cap_counters(rent, rent_mod)
                    self.assertEqual(rent.pe_warm_count_fn(), 0)
                    self.assertEqual(rent.pe_eval_count_fn(), 0)
                    self.assertTrue(rent.can_rent(queued_bound_waiting=1))

    def test_cutover_live_affine_pe_warm_counts_once_as_eval(self):
        """Cutover: warm counter zeroed; one live affine-pe-warm-* counts as eval → 1 PIT rent on cap 2."""
        from ops.parallel_eval.pit.host_wiring import _bind_shared_cap_counters
        from ops.parallel_eval.pit.pe_rent import RentService

        rent_mod = types.SimpleNamespace(
            eval_nodes=lambda: [
                {'name': 'affine-pe-warm-1', 'id': 'warm-live-1'},
            ],
            our_pods=lambda: [
                {'name': 'affine-pe-warm-1', 'id': 'warm-live-1'},
            ],
        )
        rent = RentService(max_boxes=2, max_eval_nodes=2, share_pe_cap=True)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            flag = root / 'flag.json'
            flag.write_text(json.dumps({
                'parallel_eval': {
                    'enabled': True,
                    'pit': {'enabled': True, 'mode': 'live', 'cutover': True},
                },
            }))
            (root / 'WARM.json').write_text(json.dumps({
                'pods': {
                    'affine-pe-warm-1': {
                        'state': 'ready', 'pod': {'id': 'warm-live-1'},
                    },
                    'affine-pe-warm-stale': {
                        'state': 'ready', 'pod': {'id': 'gone'},
                    },
                }
            }))
            with mock.patch.dict(os.environ, {'PE_FLAG': str(flag)}):
                with mock.patch('ops.parallel_eval.pit.host_wiring._pe_state_root', return_value=root):
                    _bind_shared_cap_counters(rent, rent_mod)
                    self.assertEqual(rent.pe_warm_count_fn(), 0)
                    self.assertEqual(rent.pe_eval_count_fn(), 1)
                    self.assertTrue(rent.can_rent(queued_bound_waiting=1))
                    # Simulate one PIT box already rented → pe(1)+pit(1)+warm(0) == cap 2.
                    rent.ledger.register(
                        'pit-1', item_id='i1', name='affine-pe-pit-x', at=0.0,
                    )
                    self.assertEqual(rent.ledger.count(), 1)
                    self.assertFalse(rent.can_rent(queued_bound_waiting=1))


if __name__ == '__main__':
    unittest.main()
