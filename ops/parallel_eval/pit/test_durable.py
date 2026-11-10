#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .config import PitConfig
from .controller import Controller
from .durable import DurableStore, item_from_dict, item_to_dict
from .pe_lane import LaneService
from .pe_rent import RentService
from .queue_ingest import sync as queue_ingest_sync
from .state import ItemRecord, ItemState


class TestDurable(unittest.TestCase):
    def test_item_roundtrip(self):
        item = ItemRecord(
            item_id='abc', registration_index=1, epoch=10,
            state=ItemState.BOUND, bound_head='a' * 64, shadow=False,
        )
        restored = item_from_dict(item_to_dict(item))
        self.assertEqual(restored.item_id, 'abc')
        self.assertEqual(restored.state, ItemState.BOUND)
        self.assertEqual(restored.bound_head, 'a' * 64)

    def test_store_persist_and_claim(self):
        with tempfile.TemporaryDirectory() as td:
            store = DurableStore(root=Path(td))
            item = ItemRecord(
                item_id='i1', registration_index=0, epoch=42,
                candidate_id='c' * 64, bound_head='h' * 64,
                state=ItemState.TRAINING, shadow=False,
            )
            store.save_item(item)
            store.enqueue('i1')
            store.write_claim(42, candidate='c' * 64, head='h' * 64, item_id='i1')
            loaded = store.load_item('i1')
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.epoch, 42)
            self.assertEqual(store.queue(), ['i1'])
            claim = Path(td) / 'claims' / 'PIT-CLAIM-e42.json'
            self.assertTrue(claim.is_file())
            # Also mirrored at store root for pe_live lookup.
            self.assertTrue((Path(td) / 'PIT-CLAIM-e42.json').is_file())

    def test_list_items_skips_worker_result_sidecars(self):
        """Live recovery writes <id>.worker-result.json next to items; must not load as items."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pe = root / 'pe'
            life = root / 'life'
            pit = pe / 'pit'
            for p in (pit / 'items', life / 'epochs'):
                p.mkdir(parents=True)
            (life / 'CURRENT.json').write_text(json.dumps({
                'epoch_index': 1056, 'checkpoint': {'id': 'a' * 64},
            }))
            (pe / 'CURRENT.json').write_text(json.dumps({
                'epoch_index': 1056, 'checkpoint': {'id': 'a' * 64},
            }))
            (pe / 'HEAD-CHAIN.json').write_text(json.dumps({
                'genesis': 'a' * 64, 'applied': [],
            }))
            store = DurableStore(root=pit)
            item = ItemRecord(
                item_id='e1056' + '0' * 27, registration_index=1056, epoch=1056,
                state=ItemState.TRAINING, bound_head='a' * 64, candidate_id='b' * 64,
                box_id='pod-1056', shadow=False,
            )
            store.save_item(item)
            wr = pit / 'items' / ('%s.worker-result.json' % item.item_id)
            wr.write_text(json.dumps({
                'ok': True,
                'note': 'box-complete',
                'item': item_to_dict(item),
            }) + '\n')
            # Orphan sidecar with no top-level item_id (the live failure shape).
            (pit / 'items' / 'orphan.worker-result.json').write_text(
                json.dumps({'ok': True, 'error': 'none'}) + '\n'
            )
            listed = store.list_items()
            self.assertEqual([i.item_id for i in listed], [item.item_id])

            env = {
                'LIFE_ROOT': str(life),
                'PE_LIFE': str(life),
                'PE_STATE': str(pe),
                'PIT_STATE': str(pit),
                'K1_TURNS': str(root / 'K1-TURNS.jsonl'),
            }
            (root / 'K1-TURNS.jsonl').write_text('')
            ctl = Controller(
                config=PitConfig(enabled=True, mode='shadow', max_boxes=2),
                rent=RentService(max_boxes=2),
                lane=LaneService(),
                box_io_factory=lambda: None,
            )
            ctl.items.extend(listed)
            with mock.patch.dict(os.environ, env, clear=False):
                # Must not raise KeyError/'item_id' from parsing worker-result as item.
                out = queue_ingest_sync(ctl, store=store)
            self.assertIsInstance(out, dict)


if __name__ == '__main__':
    unittest.main()
