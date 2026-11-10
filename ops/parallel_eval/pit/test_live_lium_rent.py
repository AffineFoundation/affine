#!/usr/bin/env python3
"""Live Lium rent seam: adapter invokes host pe_rent with ttl 4h, --no-backup."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from . import PREFIX
from .gpu_policy import pick_best_offer
from .pe_lane import LaneError, LaneService
from .pe_rent import RentError, RentService


class TestLiveLiumRent(unittest.TestCase):
    def test_pit_adapter_passes_ttl_and_max_eval_nodes(self):
        calls = []
        lium_args = []

        def host_rent(name, ttl, *, gpu=None, max_eval_nodes=None):
            calls.append(dict(name=name, ttl=ttl, gpu=gpu, max_eval_nodes=max_eval_nodes))
            lium_args.append('--no-backup')
            return dict(
                id='live-pod-1', name=name, gpu='B300', price_h=55.0,
                no_backup=True, auto_backup_disabled=True,
            )

        rent = RentService(require_host=True, host_rent=host_rent, max_eval_nodes=5, max_boxes=2)
        out = rent.rent(item_id='f' * 32, epoch=7, attempt=2)
        self.assertEqual(out['gpu'], 'B300')
        # gpu=None lets host pick_node use B300-first-then-cheapest.
        self.assertIsNone(calls[0]['gpu'])
        self.assertEqual(calls[0]['ttl'], '4h')
        self.assertEqual(calls[0]['max_eval_nodes'], 5)
        self.assertIn('--no-backup', lium_args)

    def test_live_pe_rent_rent_with_fake_lium(self):
        """Monkeypatched ops.parallel_eval.live.pe_rent records lium argv."""
        import importlib.util

        live_pe_rent_path = Path(__file__).resolve().parent.parent / 'live' / 'pe_rent.py'
        spec = importlib.util.spec_from_file_location('pe_live_rent', live_pe_rent_path)
        if spec is None or spec.loader is None:
            self.skipTest('live pe_rent path missing')
        live_pe_rent = importlib.util.module_from_spec(spec)

        recorded = []

        def fake_lium(*args, **kwargs):
            recorded.append(list(args))
            return (
                0,
                json.dumps(dict(pod=dict(
                    id='lium-id-1',
                    name='%s99-a1' % PREFIX,
                    gpu_model='B300',
                    price_per_hour=50.0,
                    ip='127.0.0.1',
                    ssh_port=22,
                ))),
                '',
            )

        def fake_pick_node(prefer=None, fallback=True, price_ceiling=None):
            return 'B300', dict(huid='exec-b300', price_per_hour=50.0, country_code='US', tier='t1')

        def fake_our_pods(ps=None):
            return []

        fake_common = mock.MagicMock()
        fake_common.PREFIX = PREFIX
        fake_common.ALLOWED_GPUS = ('B300', 'RTX5090', 'B200', 'H200', 'H100')
        fake_common.REQUIRED_GPU_COUNT = 8
        fake_common.pick_best_offer = pick_best_offer
        fake_common.cr = lambda t=None: 'ts'
        fake_common.ev = lambda **kw: None
        fake_common.lium = fake_lium
        fake_common.lium_ls = lambda g: []
        fake_common.gpu_order = lambda: ('B300', 'RTX5090', 'B200', 'H200', 'H100')
        fake_common.our_pods = fake_our_pods
        fake_common.NEVER_NAMES = ()

        with tempfile.TemporaryDirectory() as td:
            st = Path(td) / 'state'
            st.mkdir(parents=True, exist_ok=True)
            fake_common.ST = st
            fake_common.rj = lambda p, d=None: d
            fake_common.wj = lambda p, obj, mode=0o600: None
            replace_pod = mock.MagicMock(ssh_ok=lambda *a, **k: True)
            with mock.patch.dict(sys.modules, {
                'pe_common': fake_common,
                'replace_pod': replace_pod,
            }), mock.patch.object(time, 'sleep', lambda _s: None):
                spec.loader.exec_module(live_pe_rent)
                live_pe_rent.lium = fake_lium
                live_pe_rent.pick_node = fake_pick_node
                live_pe_rent.our_pods = fake_our_pods
                live_pe_rent.PREFIX = PREFIX
                live_pe_rent.PJ = st / 'pods.json'
                live_pe_rent.load_pods = lambda: {}
                live_pe_rent.save_pods = lambda d: None
                live_pe_rent.R = replace_pod
                raw = live_pe_rent.rent('%s99-a1' % PREFIX, ttl='4h', gpu='B300', max_eval_nodes=5)

        self.assertIsNotNone(raw)
        flat = sum(recorded, [])
        self.assertIn('--no-backup', flat)

        def bound_rent(name, ttl, *, gpu, max_eval_nodes):
            return dict(
                id='lium-id-1', name=name, gpu='B300', price_h=50.0,
                no_backup=True, auto_backup_disabled=True,
            )

        rent = RentService(require_host=True, host_rent=bound_rent, max_boxes=2)
        out = rent.rent(item_id='g' * 32, epoch=99, attempt=1)
        self.assertEqual(out['id'], 'lium-id-1')

    def test_lane_require_host_refuses_synthetic_probe(self):
        lane = LaneService(require_host=True)
        with self.assertRaises(LaneError):
            lane.run_probe(
                'box',
                'cand',
                indices=[1, 2, 3],
                plan_seed='seed-handle',
            )


if __name__ == '__main__':
    unittest.main()
