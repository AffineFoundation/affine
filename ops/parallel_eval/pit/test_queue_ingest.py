#!/usr/bin/env python3
"""queue_ingest: real-shaped K1-TURN (string epoch, int epoch_index) + CURRENT floor."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

from .binding import dequeue_and_bind
from .config import PitConfig
from .controller import Controller
from .durable import DurableStore
from .pe_lane import LaneService
from .pe_rent import RentService
from .queue_ingest import IngestError, sync, write_ingest_alert
from .run_host import tick_once
from .state import ItemState


HEAD = 'a' * 64
MASTER = 'b' * 64
STALE_HEAD = 'c' * 64


class TestQueueIngest(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.pe_state = self.td / 'pe'
        self.life = self.td / 'life'
        self.pending = self.life / 'pending'
        self.pending.mkdir(parents=True)
        self.pe_state.mkdir(parents=True)
        (self.pe_state / 'pit').mkdir()
        # Hermetic: every path from tempdir — never $HOME / /home/const defaults.
        self.k1_turns = self.td / 'K1-TURNS.jsonl'
        self.k1_turns.write_text('')
        self.fp32_idx = self.td / 'fp32-master'
        self.fp32_idx.mkdir()
        os.environ['PE_STATE'] = str(self.pe_state)
        os.environ['PE_LIFE_STATE'] = str(self.life)
        os.environ['PE_LIFE'] = str(self.life)
        os.environ['PIT_STATE'] = str(self.pe_state / 'pit')
        os.environ['K1_TURNS'] = str(self.k1_turns)
        os.environ['PIT_FP32_MASTER_INDEX'] = str(self.fp32_idx)
        (self.pe_state / 'HEAD-CHAIN.json').write_text(json.dumps({
            'genesis': HEAD,
            'applied': [],
        }))
        # Real CURRENT: integer epoch_index + checkpoint.
        (self.life / 'CURRENT.json').write_text(json.dumps({
            'epoch_index': 1050,
            'checkpoint': {'id': HEAD},
            'round_id': 'round-test',
        }))
        (self.pe_state / 'CURRENT.json').write_text(json.dumps({
            'epoch_index': 1050,
            'checkpoint': {'id': HEAD},
        }))
        (self.pe_state / 'MASTER-INDEX.json').write_text(json.dumps({HEAD: MASTER}))

    def tearDown(self):
        for k in (
            'PE_STATE', 'PE_LIFE_STATE', 'PE_LIFE', 'PIT_STATE', 'K1_TURNS',
            'PIT_FP32_MASTER_INDEX', 'AFFINE_FP32_MASTER_INDEX',
        ):
            os.environ.pop(k, None)

    def _ctl(self):
        return Controller(
            config=PitConfig(enabled=True, mode='live', max_boxes=2),
            rent=RentService(max_boxes=2),
            lane=LaneService(),
            box_io_factory=lambda: None,
        )

    def _write_epoch(
        self,
        epoch_index: int,
        *,
        head: str = HEAD,
        with_job: bool = True,
        with_dispatch: bool = True,
        with_result: bool = False,
        with_finalized: bool = False,
        k1: bool = True,
        run_id: Optional[str] = None,
        uid: int = 7,
    ) -> Path:
        ed = self.life / 'epochs' / str(epoch_index)
        ed.mkdir(parents=True, exist_ok=True)
        if k1:
            rid = run_id or ('affine-mimo9b-grpo-20261013-e%06d' % epoch_index)
            (ed / 'K1-TURN.json').write_text(json.dumps({
                'status': 'turn',
                'epoch': rid,  # real shape: run-id STRING
                'epoch_index': epoch_index,
                'uid': uid,
                'hotkey': '5HotkeyPrefixxx',
                'queue_position': 999,  # must NOT become registration_index
            }))
        if with_job:
            (ed / 'TRAINING-JOB.json').write_text(json.dumps({
                'payload': {
                    'manifest': {
                        'payload': {
                            'checkpoint': {'id': head, 'files': {'model.safetensors': '1' * 64}},
                        }
                    },
                    'training': {'optimizer_step': 1},
                }
            }))
        if with_dispatch:
            (ed / 'TRAIN-DISPATCH-FF.json').write_text(json.dumps({
                'signature': 'sig', 'signer': 'ROOT', 'payload': {'selected': []},
            }))
        if with_result:
            (ed / 'TRAINING-RESULT.json').write_text('{}')
        if with_finalized:
            (ed / 'FINALIZED.json').write_text('{}')
        return ed

    def test_real_k1_turn_epoch_index_not_string_epoch(self):
        """int(turn['epoch']) would crash on run-id; use epoch_index."""
        self._write_epoch(1050)
        self._write_epoch(1051)
        # Backlog below CURRENT floor must not enqueue.
        self._write_epoch(1040)
        # Missing JOB/DISPATCH skipped.
        self._write_epoch(1052, with_job=False)
        # Completed epochs skipped.
        self._write_epoch(1053, with_result=True)
        self._write_epoch(1054, with_finalized=True)

        ctl = self._ctl()
        store = DurableStore(root=self.pe_state / 'pit')
        out = sync(ctl, store=store)
        self.assertEqual(out['current_epoch_floor'], 1050)
        self.assertEqual(sorted(out['enqueued']), [1050, 1051])
        self.assertTrue(out['head_set'])
        self.assertEqual(ctl.current_head, HEAD)
        epochs = sorted(i.epoch for i in ctl.items)
        self.assertEqual(epochs, [1050, 1051])
        for item in ctl.items:
            # registration_index = epoch_index, not queue_position=999
            self.assertEqual(item.registration_index, item.epoch)
            self.assertEqual(item.meta.get('ingest_source'), 'k1-turn')
            self.assertTrue(str(item.meta.get('batch_path')).endswith(str(item.epoch)))

        out2 = sync(ctl, store=store)
        self.assertEqual(out2['enqueued'], [])

    def test_k1_jsonl_skips_rows_without_epoch_index(self):
        turns = self.td / 'K1-TURNS.jsonl'
        os.environ['K1_TURNS'] = str(turns)
        turns.write_text('\n'.join([
            json.dumps({'epoch': 'affine-mimo9b-grpo-20261013-e001040', 'uid': 1}),
            json.dumps({
                'epoch': 'affine-mimo9b-grpo-20261013-e001050',
                'epoch_index': 1050,
                'uid': 2,
                'hotkey': '5JsonlHotkeyxxxx',
            }),
        ]) + '\n')
        self._write_epoch(1050, k1=False)
        ctl = self._ctl()
        out = sync(ctl, store=DurableStore(root=self.pe_state / 'pit'))
        self.assertEqual(out['enqueued'], [1050])
        self.assertEqual(ctl.items[0].uid, 2)

    def test_inbox_respects_current_floor(self):
        pit_root = self.pe_state / 'pit'
        inbox = pit_root / 'inbox'
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / 'old.json').write_text(json.dumps({'epoch': 1000, 'uid': 1}))
        (inbox / 'new.json').write_text(json.dumps({
            'epoch': 1055, 'uid': 9, 'hotkey_prefix': '5InboxHotkeyxx',
        }))
        # Epoch dir must be ready for pending-like paths; inbox alone is enough.
        ctl = self._ctl()
        out = sync(ctl, store=DurableStore(root=pit_root))
        self.assertEqual(out['enqueued'], [1055])
        self.assertEqual(ctl.items[0].uid, 9)
        self.assertEqual(ctl.items[0].registration_index, 1055)

    def test_stale_head_skips_rent(self):
        """Before renting: TRAINING-JOB checkpoint != current head → no rental."""
        self._write_epoch(1050, head=STALE_HEAD)
        ctl = self._ctl()
        store = DurableStore(root=self.pe_state / 'pit')
        sync(ctl, store=store)
        self.assertEqual(len(ctl.items), 1)
        # Drive BOUND then rent attempt.
        item = dequeue_and_bind(
            ctl.items[0],
            head=HEAD,
            head_epoch=1050,
            round_id='r',
            master_digest=MASTER,
            dequeued_at_mono=1.0,
            round_seed_commitment='d' * 64,
        )
        self.assertIsNotNone(item)
        ctl.items[0] = item
        advanced, note = ctl._rent_on_demand(item)
        self.assertEqual(note, 'stale-head-skip')
        self.assertEqual(advanced.state, ItemState.NO_DECISION)
        self.assertTrue((advanced.meta or {}).get('stale_head_skip'))
        self.assertIsNone(advanced.box_id)

    def test_ingest_failure_writes_alert_and_tick_raises(self):
        ctl = self._ctl()
        store = DurableStore(root=self.pe_state / 'pit')
        with mock.patch(
            'ops.parallel_eval.pit.run_host.queue_ingest_sync',
            side_effect=RuntimeError('boom-ingest'),
        ):
            with self.assertRaises(IngestError):
                tick_once(ctl, store, dry_run=True)
        alert = self.pe_state / 'pit' / 'INGEST-ALERT.json'
        health = self.pe_state / 'pit' / 'INGEST-HEALTH.json'
        self.assertTrue(alert.is_file())
        self.assertTrue(health.is_file())
        doc = json.loads(alert.read_text())
        self.assertFalse(doc['ok'])
        self.assertIn('boom-ingest', doc['error'])

    def test_write_ingest_alert_direct(self):
        path = write_ingest_alert(RuntimeError('direct'), root=self.pe_state / 'pit')
        self.assertTrue(path.is_file())

    def test_master_from_pit_fp32_index_without_master_index_json(self):
        """Prod has no MASTER-INDEX.json — fall back to PIT_FP32_MASTER_INDEX/<head>.json."""
        (self.pe_state / 'MASTER-INDEX.json').unlink()
        # Published website index shape (same source pe_box verifies on the box).
        (self.fp32_idx / (HEAD + '.json')).write_text(json.dumps({
            'bf16_sha256': 'd' * 64,
            'master_digest': MASTER,
        }))
        self._write_epoch(1050)
        ctl = self._ctl()
        out = sync(ctl, store=DurableStore(root=self.pe_state / 'pit'))
        self.assertTrue(out['head_set'], msg=out)
        self.assertEqual(out.get('master'), MASTER)
        self.assertEqual(ctl.current_head, HEAD)
        self.assertEqual(ctl.current_master_digest, MASTER)
        self.assertFalse(out.get('master_missing'))

    def test_master_missing_fail_closed_head_set_false(self):
        (self.pe_state / 'MASTER-INDEX.json').unlink()
        # No PIT_FP32_MASTER_INDEX entry either.
        self._write_epoch(1050)
        ctl = self._ctl()
        out = sync(ctl, store=DurableStore(root=self.pe_state / 'pit'))
        self.assertFalse(out['head_set'])
        self.assertTrue(out.get('master_missing'))
        self.assertIsNone(ctl.current_head)


if __name__ == '__main__':
    unittest.main()
