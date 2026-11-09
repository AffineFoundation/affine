#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

from .durable import DurableStore, item_from_dict, item_to_dict
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


if __name__ == '__main__':
    unittest.main()
