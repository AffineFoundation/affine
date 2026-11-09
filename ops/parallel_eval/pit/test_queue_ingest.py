#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from pathlib import Path

from .config import PitConfig
from .controller import Controller
from .durable import DurableStore
from .pe_lane import LaneService
from .pe_rent import RentService
from .queue_ingest import sync


class TestQueueIngest(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.pe_state = self.td / 'pe'
        self.life = self.td / 'life'
        self.pending = self.life / 'pending'
        self.pending.mkdir(parents=True)
        self.pe_state.mkdir(parents=True)
        os.environ['PE_STATE'] = str(self.pe_state)
        os.environ['PE_LIFE_STATE'] = str(self.life)
        (self.pe_state / 'HEAD-CHAIN.json').write_text(json.dumps({
            'genesis': 'g' * 64,
            'applied': [],
        }))
        (self.pe_state / 'CURRENT.json').write_text(json.dumps({
            'epoch_index': 42,
            'checkpoint': {'id': 'g' * 64},
        }))
        (self.pe_state / 'MASTER-INDEX.json').write_text(json.dumps({'g' * 64: 'm' * 64}))

    def tearDown(self):
        os.environ.pop('PE_STATE', None)
        os.environ.pop('PE_LIFE_STATE', None)
        os.environ.pop('PIT_STATE', None)

    def _ctl(self):
        return Controller(
            config=PitConfig(enabled=True, mode='live', max_boxes=2),
            rent=RentService(max_boxes=2),
            lane=LaneService(),
            box_io_factory=lambda: None,  # not used in ingest-only test
        )

    def test_enqueue_from_pending_and_set_head(self):
        (self.pending / 'e101.json').write_text(json.dumps({
            'epoch': 101,
            'training_head': 'g' * 64,
            'candidate': 'c' * 64,
        }))
        (self.life / 'epochs' / '101').mkdir(parents=True)
        (self.life / 'epochs' / '101' / 'K1-TURN.json').write_text(json.dumps({
            'status': 'turn',
            'uid': 7,
            'hotkey': '5HotkeyPrefixxx',
            'queue_position': 3,
        }))
        ctl = self._ctl()
        store = DurableStore(root=self.td / 'pit')
        out = sync(ctl, store=store)
        self.assertEqual(out['enqueued'], [101])
        self.assertTrue(out['head_set'])
        self.assertEqual(ctl.current_head, 'g' * 64)
        self.assertEqual(ctl.current_master_digest, 'm' * 64)
        self.assertEqual(len(ctl.items), 1)
        self.assertEqual(ctl.items[0].epoch, 101)
        self.assertEqual(ctl.items[0].uid, 7)
        self.assertEqual(ctl.items[0].registration_index, 3)

        out2 = sync(ctl, store=store)
        self.assertEqual(out2['enqueued'], [])

    def test_inbox_batch_preferred_over_pending(self):
        inbox = self.td / 'pit' / 'inbox'
        inbox.mkdir(parents=True)
        os.environ['PIT_STATE'] = str(self.td / 'pit')
        (inbox / 'batch-55.json').write_text(json.dumps({
            'epoch': 55,
            'uid': 9,
            'hotkey_prefix': '5InboxHotkeyxx',
            'registration_index': 1,
        }))
        (self.pending / 'e55.json').write_text(json.dumps({
            'epoch': 55,
            'uid': 1,
            'hotkey_prefix': '5PendingHotkey',
            'pit_claimed': False,
        }))
        ctl = self._ctl()
        out = sync(ctl, store=DurableStore(root=self.td / 'pit'))
        self.assertEqual(out['enqueued'], [55])
        self.assertEqual(ctl.items[0].uid, 9)
        self.assertEqual(ctl.items[0].meta.get('ingest_source'), 'inbox')

    def test_ignores_pit_claimed_pending_stubs(self):
        (self.pending / 'e77.json').write_text(json.dumps({
            'epoch': 77,
            'pit_claimed': True,
            'uid': 3,
        }))
        ctl = self._ctl()
        out = sync(ctl, store=DurableStore(root=self.td / 'pit'))
        self.assertEqual(out['enqueued'], [])


if __name__ == '__main__':
    unittest.main()
