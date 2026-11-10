"""Missing SAMPLE must not block the bind queue; terminal no_sample frees pending.

e1068 never got SAMPLE (gate only samples the current lifecycle epoch). PIT binds
oldest-first; a refused bind must skip that item and continue FIFO among bindable
items so e1069+ can bind. When an epoch can never get a sample (epoch <
lifecycle_epoch - 1, or SAMPLE missing ≥30m), terminate as no_sample: DECISION +
PIT-RESULT, settlement pending_operator, paid=False, no rounds turn credit.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .binding import select_next_queued
from .config import load_pit_config
from .controller import NO_SAMPLE_WAIT_S, Controller
from .decision_emit import (
    TERMINAL_SETTLEMENT_DECISIONS,
    build_decision_doc,
    is_terminal_settlement,
    map_pit_decision,
)
from .decisions import DecisionLog
from .durable import DurableStore
from .gate_writer import RecordingGateWriter
from .pe_lane import LaneService
from .pe_rent import RentService
from .run_host import _gate_emit_ready, _item_needs_box, tick_once
from .state import ItemRecord, ItemState


H = 'a' * 64
C = 'c' * 64
M = 'd' * 64


class TestSelectSkipIds(unittest.TestCase):
    def test_skip_oldest_selects_next_fifo(self):
        e1068 = ItemRecord(
            item_id='e1068', registration_index=1068, epoch=1068,
            state=ItemState.QUEUED,
        )
        e1069 = ItemRecord(
            item_id='e1069', registration_index=1069, epoch=1069,
            state=ItemState.QUEUED,
        )
        self.assertEqual(
            select_next_queued([e1068, e1069], rented=0, max_boxes=2).item_id,
            'e1068',
        )
        nxt = select_next_queued(
            [e1068, e1069], rented=0, max_boxes=2, skip_ids={'e1068'},
        )
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt.item_id, 'e1069')


class TestNoSampleDecisionMapping(unittest.TestCase):
    def test_map_and_terminal(self):
        self.assertEqual(map_pit_decision('no_sample'), 'no_sample')
        self.assertTrue(is_terminal_settlement('no_sample'))
        self.assertIn('no_sample', TERMINAL_SETTLEMENT_DECISIONS)

    def test_build_decision_doc(self):
        item = ItemRecord(
            item_id='e1068', registration_index=1, epoch=1068,
            state=ItemState.NO_DECISION, bound_head='', candidate_id='',
            decision='no_sample', paid=False,
            meta={'no_sample_reason': 'epoch 1068 < lifecycle_epoch 1070 - 1'},
        )
        doc = build_decision_doc(
            item,
            scored={
                'decision': 'no_sample',
                'evaluated': {'z': None},
                'reasons': ['no_sample'],
                'heldout': {},
                'probe': {},
            },
        )
        self.assertEqual(doc['decision'], 'no_sample')
        self.assertEqual(doc['settlement'], 'pending_operator')
        self.assertFalse(doc['paid'])
        self.assertEqual(doc['chart_status'], 'no_sample')
        self.assertEqual(
            (doc.get('parallel_eval') or {}).get('no_sample_reason')[:20],
            'epoch 1068 < lifecyc',
        )

    def test_decision_log_rejects_paid_no_sample(self):
        log = DecisionLog()
        with self.assertRaises(ValueError):
            log.append(
                epoch=1068, uid=1, hotkey_prefix='5A', binding_H='',
                binding_round_id='', candidate='', decision='no_sample',
                paid=True, decided_at=1.0,
            )
        e = log.append(
            epoch=1068, uid=1, hotkey_prefix='5A', binding_H='',
            binding_round_id='', candidate='', decision='no_sample',
            paid=False, decided_at=1.0,
        )
        self.assertEqual(e.decision, 'no_sample')
        self.assertFalse(e.paid)


class TestRefuseBindSkipsToNext(unittest.TestCase):
    def test_e1068_missing_sample_e1069_binds(self):
        """Oldest (e1068) refused for missing SAMPLE; e1069 still binds same tick."""
        samples = {1069: 'r-e1069'}

        def round_id_fn(epoch: int) -> str:
            rid = samples.get(int(epoch))
            if not rid:
                raise ValueError('SAMPLE missing for epoch %s' % epoch)
            return rid

        ctl = Controller(
            config=load_pit_config({
                'pit': {'enabled': True, 'mode': 'shadow', 'max_boxes': 2},
            }),
            rent=RentService(max_boxes=2),
            lane=LaneService(),
            box_io_factory=lambda: None,
            round_id_fn=round_id_fn,
            plan_seed_fn=lambda e: 'p' * 64,
            lifecycle_epoch_fn=lambda: 1069,  # e1068 == life-1 → not yet terminal
        )
        ctl.set_head(H, epoch=1068, master_digest=M)
        ctl.enqueue(epoch=1068, registration_index=1068)
        ctl.enqueue(epoch=1069, registration_index=1069)
        out = ctl.tick()
        ops = [a.get('op') for a in out.get('actions') or []]
        self.assertIn('refuse-bind-round-id', ops)
        self.assertIn('bound', ops)
        by_ep = {it.epoch: it for it in ctl.items}
        self.assertEqual(by_ep[1068].state, ItemState.QUEUED)
        self.assertIsNotNone((by_ep[1068].meta or {}).get('sample_wait_since'))
        # Same tick may advance BOUND → rent/provision; binding itself must succeed.
        self.assertNotEqual(by_ep[1069].state, ItemState.QUEUED)
        self.assertEqual(by_ep[1069].bound_round_id, 'r-e1069')
        # FIFO among bindable: refuse e1068 then bind e1069 (not break).
        refuse_i = ops.index('refuse-bind-round-id')
        bound_i = ops.index('bound')
        self.assertLess(refuse_i, bound_i)

    def test_refuse_does_not_break_queue_order_among_bindable(self):
        samples = {1070: 'r-e1070', 1071: 'r-e1071'}

        def round_id_fn(epoch: int) -> str:
            rid = samples.get(int(epoch))
            if not rid:
                raise ValueError('SAMPLE missing')
            return rid

        ctl = Controller(
            config=load_pit_config({
                'pit': {'enabled': True, 'mode': 'shadow', 'max_boxes': 3},
            }),
            rent=RentService(max_boxes=3),
            lane=LaneService(),
            box_io_factory=lambda: None,
            round_id_fn=round_id_fn,
            plan_seed_fn=lambda e: 'p' * 64,
            # life=1070 → e1069 == life-1 (not yet terminal); skip + continue.
            lifecycle_epoch_fn=lambda: 1070,
        )
        ctl.set_head(H, epoch=1069, master_digest=M)
        for ep in (1069, 1070, 1071):
            ctl.enqueue(epoch=ep, registration_index=ep)
        out = ctl.tick()
        bound_epochs = []
        for a in out.get('actions') or []:
            if a.get('op') != 'bound':
                continue
            it = next(x for x in ctl.items if x.item_id == a['item_id'])
            bound_epochs.append(it.epoch)
        self.assertEqual(bound_epochs, [1070, 1071])
        by_ep = {it.epoch: it for it in ctl.items}
        self.assertEqual(by_ep[1069].state, ItemState.QUEUED)
        self.assertIsNotNone((by_ep[1069].meta or {}).get('sample_wait_since'))


class TestTerminalNoSample(unittest.TestCase):
    def test_stale_vs_lifecycle_terminates(self):
        clock = {'t': 1000.0}

        def round_id_fn(epoch: int) -> str:
            raise ValueError('SAMPLE missing for epoch %s' % epoch)

        ctl = Controller(
            config=load_pit_config({
                'pit': {'enabled': True, 'mode': 'shadow', 'max_boxes': 2},
            }),
            rent=RentService(max_boxes=2),
            lane=LaneService(),
            box_io_factory=lambda: None,
            round_id_fn=round_id_fn,
            now_fn=lambda: clock['t'],
            lifecycle_epoch_fn=lambda: 1070,  # 1068 < 1069 → terminal
        )
        ctl.set_head(H, epoch=1068, master_digest=M)
        ctl.enqueue(epoch=1068, registration_index=1068)
        ctl.enqueue(epoch=1069, registration_index=1069)
        out = ctl.tick()
        ops = [a.get('op') for a in out.get('actions') or []]
        self.assertIn('no_sample', ops)
        by_ep = {it.epoch: it for it in ctl.items}
        # Same tick advances NO_DECISION → RELEASED (no box to destroy).
        self.assertIn(by_ep[1068].state, (ItemState.NO_DECISION, ItemState.RELEASED))
        self.assertEqual(by_ep[1068].decision, 'no_sample')
        self.assertFalse(by_ep[1068].paid)
        self.assertTrue((by_ep[1068].meta or {}).get('gate_emit_ready'))
        self.assertEqual((by_ep[1068].meta or {}).get('settlement'), 'pending_operator')
        self.assertTrue((by_ep[1068].meta or {}).get('no_sample_alerted'))
        # e1069 also missing sample and == life-1 → not terminal yet; stays QUEUED.
        self.assertEqual(by_ep[1069].state, ItemState.QUEUED)

    def test_thirty_minute_timeout_terminates(self):
        clock = {'t': 1_000.0}

        ctl = Controller(
            config=load_pit_config({
                'pit': {'enabled': True, 'mode': 'shadow', 'max_boxes': 1},
            }),
            rent=RentService(max_boxes=1),
            lane=LaneService(),
            box_io_factory=lambda: None,
            round_id_fn=lambda e: (_ for _ in ()).throw(ValueError('SAMPLE missing')),
            now_fn=lambda: clock['t'],
            lifecycle_epoch_fn=lambda: 1068,  # equal → not stale-vs-life
        )
        ctl.set_head(H, epoch=1068, master_digest=M)
        ctl.enqueue(epoch=1068, registration_index=1068)
        out1 = ctl.tick()
        self.assertIn('refuse-bind-round-id', [a.get('op') for a in out1['actions']])
        self.assertEqual(ctl.items[0].state, ItemState.QUEUED)
        since = float((ctl.items[0].meta or {})['sample_wait_since'])

        clock['t'] = since + NO_SAMPLE_WAIT_S + 1.0
        out2 = ctl.tick()
        self.assertIn('no_sample', [a.get('op') for a in out2['actions']])
        self.assertEqual(ctl.items[0].decision, 'no_sample')
        self.assertIn(
            ctl.items[0].state, (ItemState.NO_DECISION, ItemState.RELEASED),
        )


class TestNoSampleEmit(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.pe = Path(self.td) / 'pe'
        self.life = Path(self.td) / 'life'
        self.gate = Path(self.td) / 'gate'
        (self.pe / 'pit').mkdir(parents=True)
        (self.life / 'epochs' / '1068').mkdir(parents=True)
        (self.gate / 'epochs').mkdir(parents=True)
        self.prev = {
            k: os.environ.get(k)
            for k in (
                'PE_STATE', 'PIT_STATE', 'LIFE_ROOT', 'PE_LIFE',
                'HG_GATE_STATE', 'PE_GATE_EP', 'PE_HELDOUT', 'PE_GATE_STATE',
            )
        }
        os.environ['PE_STATE'] = str(self.pe)
        os.environ['PIT_STATE'] = str(self.pe / 'pit')
        os.environ['LIFE_ROOT'] = str(self.life)
        os.environ['PE_LIFE'] = str(self.life)
        os.environ['HG_GATE_STATE'] = str(self.gate)
        os.environ['PE_GATE_STATE'] = str(self.gate)
        os.environ['PE_GATE_EP'] = str(self.gate / 'epochs')
        os.environ['PE_HELDOUT'] = str(self.gate / 'heldout')
        (self.gate / 'heldout').mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        for k, v in self.prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_recording_write_no_sample_no_turn(self):
        gw = RecordingGateWriter(epochs_root=self.gate / 'epochs')
        path = gw.write_no_sample(
            epoch=1068, head='', candidate='', why='no SAMPLE',
        )
        doc = json.loads(path.read_text())
        self.assertEqual(doc['decision'], 'no_sample')
        self.assertEqual(doc['settlement'], 'pending_operator')
        self.assertFalse(doc['paid'])
        self.assertEqual(gw.turn_adds, [])
        self.assertEqual(gw.closes, [])

    def test_tick_emits_decision_and_pit_result_no_turn_credit(self):
        (self.life / 'CURRENT.json').write_text(json.dumps({
            'epoch_index': 1070, 'checkpoint': {'id': H},
        }) + '\n')
        (self.life / 'epochs' / '1068' / 'PIT-DISPATCHED.json').write_text(
            json.dumps({'at': 1.0}) + '\n'
        )
        cfg = load_pit_config({
            'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 2},
        })
        store = DurableStore(root=self.pe / 'pit')
        ctl = Controller(
            config=cfg,
            rent=RentService(max_boxes=2),
            lane=LaneService(),
            box_io_factory=lambda: None,
            durable_save_fn=store.save_item,
        )
        ctl.current_head = H
        item = ItemRecord(
            item_id='e1068', registration_index=1068, epoch=1068,
            state=ItemState.NO_DECISION, decision='no_sample', paid=False,
            result_stored=True,
            meta={
                'gate_emit_ready': True,
                'settlement': 'pending_operator',
                'no_sample_reason': 'epoch 1068 < lifecycle_epoch 1070 - 1',
                'scored': {
                    'decision': 'no_sample',
                    'evaluated': {'z': None},
                    'reasons': ['no_sample'],
                    'heldout': {},
                    'probe': {},
                },
                'epoch_dir': str(self.life / 'epochs' / '1068'),
            },
        )
        ctl.items = [item]
        store.save_item(item)
        self.assertTrue(_gate_emit_ready(item))
        self.assertFalse(_item_needs_box(item))

        applies = []
        benches = []
        ctl.apply_hook = lambda it: applies.append(it.item_id)
        ctl.on_green_fresh = lambda it: benches.append(it.item_id)

        with mock.patch(
            'ops.parallel_eval.pit.run_host.queue_ingest_sync',
            return_value={'ok': True, 'enqueued': 0},
        ), mock.patch(
            'ops.parallel_eval.pit.cutover_apply.drive_cutover_apply',
            return_value={'ok': True},
        ), mock.patch(
            'ops.parallel_eval.pit.run_host.build_host_rounds',
            return_value=None,
        ):
            summary = tick_once(ctl, store)

        final = ctl.items[0]
        self.assertTrue((final.meta or {}).get('decision_emitted'), summary)
        self.assertEqual(final.decision, 'no_sample')

        dec_path = self.gate / 'epochs' / '1068' / 'DECISION.json'
        self.assertTrue(dec_path.is_file(), 'DECISION.json must be written')
        dec = json.loads(dec_path.read_text())
        self.assertEqual(dec['decision'], 'no_sample')
        self.assertEqual(dec.get('settlement'), 'pending_operator')
        self.assertFalse(dec.get('paid'))

        pit_result = self.life / 'epochs' / '1068' / 'PIT-RESULT.json'
        self.assertTrue(pit_result.is_file(), 'PIT-RESULT must free pending')
        pr = json.loads(pit_result.read_text())
        self.assertEqual(pr['decision'], 'no_sample')
        self.assertEqual(pr.get('settlement'), 'pending_operator')
        self.assertFalse(pr.get('paid'))

        self.assertEqual(applies, [])
        self.assertEqual(benches, [])
        # No heldout round / no TURNS credit for no_sample.
        self.assertFalse(any((self.gate / 'heldout').glob('**/TURNS.private.jsonl')))


if __name__ == '__main__':
    unittest.main()
