#!/usr/bin/env python3
"""Rental safety: require_host, caps, ledger durability, reconcile."""
import tempfile
import unittest
from pathlib import Path

from . import PREFIX
from .pe_rent import RentError, RentService, RentalLedger


class TestRentalSafety(unittest.TestCase):
    def test_require_host_refuses_without_module(self):
        rent = RentService(require_host=True, host_rent=None)
        with self.assertRaises(RentError):
            rent.rent(item_id='x' * 32, epoch=1, attempt=1)

    def test_synthetic_when_require_host_false(self):
        rent = RentService(require_host=False)
        out = rent.rent(item_id='a' * 32, epoch=1, attempt=1)
        self.assertTrue(out.get('synthetic'))
        self.assertEqual(rent.active_count(), 1)

    def test_global_eval_node_cap(self):
        rent = RentService(max_boxes=5, max_eval_nodes=5, pe_eval_count_fn=lambda: 3, pe_warm_count_fn=lambda: 2)
        self.assertFalse(rent.can_rent(queued_bound_waiting=1))
        rent.pe_warm_count_fn = lambda: 1
        rent.pe_eval_count_fn = lambda: 2
        self.assertTrue(rent.can_rent(queued_bound_waiting=1))

    def test_host_rent_adapter_args(self):
        calls = []

        def host_rent(name, ttl, *, gpu, max_eval_nodes):
            calls.append(dict(name=name, ttl=ttl, gpu=gpu, max_eval_nodes=max_eval_nodes))
            return dict(
                id='pod-1', name=name, gpu='B300', price_h=40.0,
                auto_backup_disabled=True,
            )

        rent = RentService(host_rent=host_rent, max_boxes=2)
        rent.rent(item_id='b' * 32, epoch=2, attempt=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['gpu'], 'B300')
        self.assertEqual(calls[0]['ttl'], '4h')
        self.assertEqual(calls[0]['name'], '%s2-a1' % PREFIX)

    def test_price_ceiling_releases_before_ledger(self):
        released = []

        def host_rent(name, ttl, *, gpu, max_eval_nodes):
            return dict(id='p99', name=name, gpu='B300', price_h=80.0, no_backup=True)

        def host_release(name_or_id, reason='done'):
            released.append((name_or_id, reason))
            return True

        rent = RentService(host_rent=host_rent, host_release=host_release)
        with self.assertRaises(RentError):
            rent.rent(item_id='c' * 32, epoch=1, attempt=1)
        self.assertEqual(rent.active_count(), 0)
        self.assertEqual(released, [('p99', 'rent-price-ceiling')])

    def test_wrong_gpu_releases_before_ledger(self):
        released = []

        def host_rent(name, ttl, *, gpu, max_eval_nodes):
            return dict(id='h200-1', name=name, gpu='H200', price_h=30.0, no_backup=True)

        def host_release(name_or_id, reason='done'):
            released.append((name_or_id, reason))
            return True

        rent = RentService(host_rent=host_rent, host_release=host_release)
        with self.assertRaises(RentError):
            rent.rent(item_id='d' * 32, epoch=1, attempt=1)
        self.assertEqual(rent.active_count(), 0)
        self.assertEqual(released[0][1], 'rent-gpu-mismatch')

    def test_no_backup_required(self):
        released = []

        def host_rent(name, ttl, *, gpu, max_eval_nodes):
            return dict(id='nb-1', name=name, gpu='B300', price_h=30.0)

        def host_release(name_or_id, reason='done'):
            released.append((name_or_id, reason))
            return True

        rent = RentService(
            host_rent=host_rent,
            host_release=host_release,
            disable_auto_backup_fn=lambda _bid: False,
        )
        with self.assertRaises(RentError):
            rent.rent(item_id='e' * 32, epoch=1, attempt=1)
        self.assertEqual(released[0][1], 'auto-backup-still-on')

    def test_ledger_save_load(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / 'ledger.json')
            led = RentalLedger()
            led.register('box-a', item_id='it', name='%se1-a1' % PREFIX, at=1.0)
            led.save(path)
            loaded = RentalLedger.load(path)
            self.assertEqual(loaded.count(), 1)
            self.assertIn('box-a', loaded.active)

    def test_reconcile_adopt_and_release_orphan(self):
        released = []
        name_ok = '%se3-a1' % PREFIX
        name_orphan = '%se9-a9' % PREFIX

        def host_release(name_or_id, reason='done'):
            released.append((name_or_id, reason))
            return True

        rent = RentService(host_release=host_release)
        listings = [
            dict(id='owned-1', name=name_ok, gpu='B300'),
            dict(id='orphan-9', name=name_orphan, gpu='B300'),
        ]
        out = rent.reconcile(listings, item_box_ids={'owned-1'})
        self.assertIn('owned-1', out['adopted'])
        self.assertIn('orphan-9', out['released'])
        self.assertEqual(released, [('orphan-9', 'reconcile-orphan')])


if __name__ == '__main__':
    unittest.main()
