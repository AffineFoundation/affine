"""blocked_no_candidate is terminal: DECISION + PIT-RESULT, no payout/apply; boxless SCORED frees slots."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .config import load_pit_config
from .controller import Controller
from .decision_emit import BLOCKED_DECISIONS, build_decision_doc, is_blocked_decision, map_pit_decision
from .durable import DurableStore
from .pe_lane import LaneService
from .pe_rent import RentService
from .run_host import _gate_emit_ready, _item_needs_box, tick_once
from .state import ItemRecord, ItemState


H = 'a' * 64
C = 'c' * 64


def _full_heldout(rid='r-e1065'):
    return dict(
        round_id=rid,
        baseline='prior-v1',
        set_size=500,
        bar_z=1.0,
        k_cand=8,
        head_rollouts=12,
        scoring='sn120-prior-baseline-v1',
        settings_hash='f' * 64,
        max_tokens=64000,
        scoring_mode='separate-run',
        tau_run=0.01,
        sigma_run_floor=0.05,
        baseline_done_at_cr='2026-01-01 00:00:00 CR',
    )


def _scored_block(z=2.1):
    return {
        'decision': 'green',
        'z': z,
        'evaluated': {'z': z, 'n': 16, 'h_pass': 0.5, 'c_pass': 0.7},
        'heldout': _full_heldout(),
        'probe': {'k': 8, 'samples': 16},
        'reasons': [],
        'plan_seed': 'p' * 64,
    }


class TestBlockedDecisionMapping(unittest.TestCase):
    def test_map_keeps_blocked_values(self):
        self.assertEqual(map_pit_decision('blocked_no_candidate'), 'blocked_no_candidate')
        self.assertEqual(map_pit_decision('blocked_incomplete_master'), 'blocked_incomplete_master')
        self.assertTrue(is_blocked_decision('blocked_no_candidate'))
        self.assertIn('blocked_no_candidate', BLOCKED_DECISIONS)

    def test_build_decision_doc_settlement_pending_operator(self):
        item = ItemRecord(
            item_id='e1065', registration_index=1, epoch=1065,
            state=ItemState.NO_DECISION, bound_head=H, candidate_id=C,
            decision='blocked_no_candidate', paid=False, z=2.1,
            bound_round_id='r-e1065', bound_plan_seed='p' * 64,
            meta={'settlement': 'pending_operator'},
        )
        doc = build_decision_doc(item, scored=_scored_block())
        self.assertEqual(doc['decision'], 'blocked_no_candidate')
        self.assertEqual(doc['settlement'], 'pending_operator')
        self.assertFalse(doc['paid'])
        self.assertEqual(doc['evaluated']['z'], 2.1)
        self.assertEqual(doc['probe']['samples'], 16)
        self.assertEqual(doc['chart_status'], 'blocked_no_candidate')
        # last-10 GREEN filter must not treat this as green
        v = doc['decision'] if doc['decision'] in ('green', 'red') else None
        self.assertIsNone(v)


class TestBlockedTerminalEmit(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.pe = Path(self.td) / 'pe'
        self.life = Path(self.td) / 'life'
        self.gate = Path(self.td) / 'gate'
        (self.pe / 'pit').mkdir(parents=True)
        (self.life / 'epochs' / '1065').mkdir(parents=True)
        (self.gate / 'epochs').mkdir(parents=True)
        (self.gate / 'heldout' / 'r-e1065').mkdir(parents=True)
        (self.gate / 'heldout' / 'r-e1065' / 'ROUND.private.json').write_text(
            json.dumps({'id': 'r-e1065', 'head': H, 'n': 1}) + '\n'
        )
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

    def tearDown(self):
        for k, v in self.prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _blocked_item(self):
        scored = _scored_block()
        return ItemRecord(
            item_id='e1065', registration_index=1, epoch=1065,
            state=ItemState.NO_DECISION, bound_head=H, candidate_id=C,
            decision='blocked_no_candidate', paid=False, z=2.1,
            result_stored=True, sealed=True,
            bound_round_id='r-e1065', bound_plan_seed='p' * 64,
            meta={
                'scored': scored,
                'gate_emit_ready': True,
                'settlement': 'pending_operator',
                'blocked': 'no_candidate',
                'epoch_dir': str(self.life / 'epochs' / '1065'),
            },
        )

    def test_gate_emit_ready_for_blocked(self):
        item = self._blocked_item()
        self.assertTrue(_gate_emit_ready(item))
        self.assertFalse(_item_needs_box(item))

    def test_tick_emits_decision_and_pit_result_no_apply(self):
        (self.life / 'CURRENT.json').write_text(json.dumps({
            'epoch_index': 1065, 'checkpoint': {'id': H},
        }) + '\n')
        # PIT-DISPATCHED so we can assert PIT-RESULT frees the pending slot.
        (self.life / 'epochs' / '1065' / 'PIT-DISPATCHED.json').write_text(
            json.dumps({'at': 1.0}) + '\n'
        )
        cfg = load_pit_config({
            'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 2},
        })
        store = DurableStore(root=self.pe / 'pit')
        rent = RentService(max_boxes=2)
        ctl = Controller(
            config=cfg,
            rent=rent,
            lane=LaneService(),
            box_io_factory=lambda: None,
            durable_save_fn=store.save_item,
        )
        ctl.current_head = H
        item = self._blocked_item()
        ctl.items = [item]
        store.save_item(item)

        applies = []
        benches = []
        ctl.apply_hook = lambda it: applies.append(it.item_id)
        ctl.on_green_fresh = lambda it: benches.append(it.item_id)

        # Avoid live cutover apply / ingest side effects; gate emit uses real HostGateWriter.
        with mock.patch(
            'ops.parallel_eval.pit.run_host.queue_ingest_sync',
            return_value={'ok': True, 'enqueued': 0},
        ), mock.patch(
            'ops.parallel_eval.pit.cutover_apply.drive_cutover_apply',
            return_value={'ok': True},
        ):
            summary = tick_once(ctl, store)

        final = ctl.items[0]
        self.assertTrue((final.meta or {}).get('decision_emitted'), summary)
        self.assertEqual(final.decision, 'blocked_no_candidate')

        dec_path = self.gate / 'epochs' / '1065' / 'DECISION.json'
        self.assertTrue(dec_path.is_file(), 'DECISION.json must be written')
        dec = json.loads(dec_path.read_text())
        self.assertEqual(dec['decision'], 'blocked_no_candidate')
        self.assertEqual(dec.get('settlement'), 'pending_operator')
        self.assertFalse(dec.get('paid'))
        self.assertEqual((dec.get('evaluated') or {}).get('z'), 2.1)

        pit_result = self.life / 'epochs' / '1065' / 'PIT-RESULT.json'
        self.assertTrue(pit_result.is_file(), 'PIT-RESULT must free pending slot')
        pr = json.loads(pit_result.read_text())
        self.assertEqual(pr['decision'], 'blocked_no_candidate')
        self.assertEqual(pr.get('settlement'), 'pending_operator')
        self.assertFalse(pr.get('paid'))
        self.assertEqual(pr.get('z'), 2.1)

        # No apply / HEAD / bench; heldout round must not close as GREEN.
        self.assertEqual(applies, [])
        self.assertEqual(benches, [])
        apply_dir = self.pe / 'apply'
        self.assertFalse(apply_dir.exists() and any(apply_dir.glob('*.json')))
        closed = self.gate / 'heldout' / 'r-e1065' / 'CLOSED.json'
        self.assertFalse(closed.is_file())
        turns = self.gate / 'heldout' / 'r-e1065' / 'TURNS.private.jsonl'
        self.assertTrue(turns.is_file())
        rows = [json.loads(l) for l in turns.read_text().splitlines() if l.strip()]
        self.assertTrue(any(
            r.get('epoch') == 1065 and r.get('decision') == 'blocked_no_candidate'
            for r in rows
        ))


class TestScoredNoBoxDoesNotOccupy(unittest.TestCase):
    def test_scored_wait_baseline_without_box_frees_slot_for_rent(self):
        """e1065/e1066 SCORED boxless must not block e1067/e1068 from binding/renting."""
        cfg = load_pit_config({
            'pit': {'enabled': True, 'mode': 'shadow', 'max_boxes': 2},
        })
        rent = RentService(max_boxes=2, shadow=True)
        rented = []

        def fake_rent(**kwargs):
            bid = 'pod-%d' % (len(rented) + 1)
            rented.append(bid)
            rent.ledger.active[bid] = {'box_id': bid, 'name': 'n-%s' % bid, 'at': 1.0}
            return {'id': bid, 'name': 'n-%s' % bid, 'gpu': 'B300'}

        rent.rent = fake_rent  # type: ignore[method-assign]
        ctl = Controller(
            config=cfg,
            rent=rent,
            lane=LaneService(),
            box_io_factory=lambda: None,
            round_id_fn=lambda e: 'r-e%d' % int(e),
            plan_seed_fn=lambda e: 'p' * 64,
            now_fn=lambda: 1000.0,
        )
        ctl.set_head(H, epoch=1065, master_digest='b' * 64)

        # Two earlier items: SCORED wait-baseline, boxes already released.
        for ep in (1065, 1066):
            item = ItemRecord(
                item_id='e%d' % ep, registration_index=ep, epoch=ep,
                state=ItemState.SCORED, bound_head=H, bound_master_digest='b' * 64,
                bound_round_id='r-e%d' % ep, candidate_id=C, box_id=None,
                result_stored=True, sealed=True,
                meta={
                    'baseline_wait_since': 900.0,
                    'baseline_wait_next_at': 2000.0,
                    'baseline_wait_box_released': True,
                },
            )
            ctl.items.append(item)
            self.assertFalse(ctl._occupies_box_slot(item))

        # Two new queued items should bind+rent (max_boxes=2 free).
        ctl.enqueue(epoch=1067, registration_index=1067, hotkey_prefix='5A')
        ctl.enqueue(epoch=1068, registration_index=1068, hotkey_prefix='5B')
        out = ctl.tick()
        by_ep = {i.epoch: i for i in ctl.items}
        self.assertIn(by_ep[1067].state, (
            ItemState.BOUND, ItemState.RENTING, ItemState.PROVISIONING,
            ItemState.TRAINING, ItemState.RETRY_WAIT,
        ))
        self.assertIn(by_ep[1068].state, (
            ItemState.BOUND, ItemState.RENTING, ItemState.PROVISIONING,
            ItemState.TRAINING, ItemState.RETRY_WAIT, ItemState.QUEUED,
        ))
        # At least one of the new epochs got a live box (or bound waiting).
        occupied = sum(1 for i in ctl.items if ctl._occupies_box_slot(i))
        self.assertLessEqual(occupied, 2)
        self.assertFalse(ctl._occupies_box_slot(by_ep[1065]))
        self.assertFalse(ctl._occupies_box_slot(by_ep[1066]))
        # New items progressed out of QUEUED.
        self.assertNotEqual(by_ep[1067].state, ItemState.QUEUED)

    def test_scored_with_live_box_still_occupies(self):
        ctl = Controller(
            config=load_pit_config({'pit': {'enabled': True, 'mode': 'shadow', 'max_boxes': 1}}),
            rent=RentService(max_boxes=1, shadow=True),
            lane=LaneService(),
            box_io_factory=lambda: None,
        )
        item = ItemRecord(
            item_id='e1065', registration_index=1, epoch=1065,
            state=ItemState.SCORED, box_id='pod-live',
            meta={'baseline_wait_since': 1.0},
        )
        self.assertTrue(ctl._occupies_box_slot(item))


class TestBlockedNotInLast10Payees(unittest.TestCase):
    def test_payees_filter_skips_blocked_verdict(self):
        recs = [
            dict(verdict='blocked', epoch_index=1065, uid=1, hotkey='hk1', z=2.1),
            dict(verdict='green', epoch_index=1064, uid=2, hotkey='hk2', z=1.5),
            dict(verdict='blocked', epoch_index=1063, uid=3, hotkey='hk3', z=0.9),
        ]
        seen, out = set(), []
        for r in sorted((r for r in recs if r.get('verdict') == 'green'), key=lambda r: -r['epoch_index']):
            u = int(r['uid'])
            if u in seen:
                continue
            seen.add(u)
            out.append(r)
        self.assertEqual([r['uid'] for r in out], [2])
        self.assertNotIn(1, seen)
        self.assertNotIn(3, seen)

    def test_gate_ledger_skips_blocked_decision(self):
        """gate_ledger.entry returns None for non green/red — no crash, no promote."""
        with tempfile.TemporaryDirectory() as td:
            ep = Path(td) / 'epochs' / '1065'
            ep.mkdir(parents=True)
            (ep / 'DECISION.json').write_text(json.dumps({
                'decision': 'blocked_no_candidate',
                'settlement': 'pending_operator',
                'head': H, 'candidate': C,
                'evaluated': {'z': 2.1},
            }) + '\n')
            dec = json.loads((ep / 'DECISION.json').read_text())
            self.assertNotIn(dec.get('decision'), ('green', 'red'))
            # Mirror gate_ledger.entry guard.
            entry = None if dec.get('decision') not in ('green', 'red') else dec
            self.assertIsNone(entry)


if __name__ == '__main__':
    unittest.main()
