#!/usr/bin/env python3
"""Sanctioned PIT GREEN rerun: refusals, archive, frozen indices, ledger, GPU."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from .bench_publish import format_node_gpu
from .binding import select_next_queued
from .controller import Controller
from .durable import DurableStore
from .gate_ledger_reconcile import reconcile_rows
from .gate_writer import GateWriteError, HostGateWriter
from .queue_ingest import _epoch_dir_ready
from .rerun import (
    RERUN_ATTEMPT,
    RERUN_PRIORITY,
    RerunError,
    archive_attempt1,
    consume_rerun_request,
    preflight,
    rerun_cand_seed,
    write_fresh_sample,
    write_rerun_request,
)
from .state import ItemRecord, ItemState


HEAD = 'a' * 64
CAND = 'b' * 64
OTHER = 'c' * 64
HOTKEY = '5TestHotkeyXXXX'
ROUND_INDICES = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Env:
    def __init__(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.life = self.root / 'life'
        self.gate = self.root / 'gate' / 'epochs'
        self.heldout = self.root / 'gate' / 'heldout'
        self.pe = self.root / 'pe'
        self.pit = self.root / 'pit'
        self.k1_joins = self.root / 'k1_joins'
        for p in (self.life / 'epochs', self.gate, self.heldout, self.pe, self.pit, self.k1_joins):
            p.mkdir(parents=True, exist_ok=True)
        self.env = {
            'LIFE_ROOT': str(self.life),
            'PE_LIFE': str(self.life),
            'PE_GATE_EP': str(self.gate),
            'HG_GATE_STATE': str(self.root / 'gate'),
            'PE_HELDOUT': str(self.heldout),
            'PE_STATE': str(self.pe),
            'PIT_STATE': str(self.pit),
            'K1_JOINS': str(self.k1_joins),
        }
        self._prev = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        (self.pe / 'HEAD-CHAIN.json').write_text(json.dumps({
            'head': HEAD, 'applied': [{'candidate': HEAD, 'new_head': HEAD}],
        }))
        rid = 'r20261110T065516Z-22a385fd'
        round_dir = self.heldout / rid
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / 'ROUND.private.json').write_text(json.dumps({
            'id': rid, 'head': HEAD, 'seed': 'r' * 64,
            'indices': list(ROUND_INDICES), 'n': len(ROUND_INDICES),
            'ids': ['task-%d' % i for i in ROUND_INDICES],
            'cfg': {'probe_env': 'math'},
        }))
        (self.heldout / 'CURRENT.json').write_text(json.dumps({'id': rid}))
        self.round_id = rid
        self.round_seed = 'r' * 64
        self.store = DurableStore(self.pit)

    def close(self) -> None:
        for k, v in self._prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.td.cleanup()

    def epoch_dir(self, e: int) -> Path:
        d = self.life / 'epochs' / str(e)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def gate_dir(self, e: int) -> Path:
        d = self.gate / str(e)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_turn_and_batch(self, e: int, *, file_sha: Any = None) -> str:
        raw = b'{"format":"affine-batch-v2","groups":[]}\n'
        digest = file_sha or _sha(raw)
        ed = self.epoch_dir(e)
        ff = ed / 'ff'
        ff.mkdir(parents=True, exist_ok=True)
        (ff / ('%s.json' % HOTKEY)).write_bytes(raw)
        (ed / 'K1-TURN.json').write_text(json.dumps({
            'status': 'turn', 'epoch_index': e, 'epoch': 'run-e%d' % e,
            'uid': 7, 'hotkey': HOTKEY, 'file_sha256': digest, 'file_bytes': len(raw),
            'head': HEAD,
        }))
        return digest

    def write_sample_decision(self, e: int, *, decision: str = 'green', paid: bool = False) -> None:
        gd = self.gate_dir(e)
        (gd / 'SAMPLE.private.json').write_text(json.dumps({
            'epoch': e, 'head': HEAD, 'round_id': 'old-round',
            'indices': [1, 2, 3], 'seed': 's' * 64, 'attempt': 1,
        }))
        (gd / 'COMMIT.json').write_text(json.dumps({'epoch': e, 'head': HEAD}))
        (gd / 'EVALUATED.private.json').write_text(json.dumps({'evaluated': {'z': 2.0}}))
        (gd / 'DECISION.json').write_text(json.dumps({
            'epoch': e, 'head': HEAD, 'candidate': CAND, 'decision': decision,
            'paid': paid, 'evaluated': {'z': 2.0},
        }))

    def make_item(self, e: int, *, state: ItemState = ItemState.RELEASED, paid: bool = False) -> ItemRecord:
        item = ItemRecord(
            item_id='item-e%d' % e, registration_index=e, epoch=e,
            uid=7, hotkey_prefix=HOTKEY[:16], state=state,
            bound_head=HEAD, candidate_id=CAND, attempt=1, paid=paid,
        )
        self.store.save_item(item)
        return item


class TestRerunRefusals(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = _Env()

    def tearDown(self) -> None:
        self.fx.close()

    def test_refuse_live_box(self):
        e = 1060
        self.fx.write_turn_and_batch(e)
        self.fx.write_sample_decision(e)
        item = self.fx.make_item(e, state=ItemState.PROBING)
        item.box_id = 'box-live'
        self.fx.store.save_item(item)
        check = preflight(e, store=self.fx.store)
        self.assertFalse(check.ok)
        self.assertTrue(any('live box' in r for r in check.reasons))

    def test_refuse_when_applied_in_head_chain(self):
        e = 1062
        self.fx.write_turn_and_batch(e)
        self.fx.write_sample_decision(e)
        self.fx.make_item(e, state=ItemState.RELEASED)
        (self.fx.pe / 'HEAD-CHAIN.json').write_text(json.dumps({
            'head': CAND,
            'applied': [{'candidate': CAND, 'new_head': CAND}],
        }))
        check = preflight(e, store=self.fx.store)
        self.assertFalse(check.ok)
        self.assertTrue(any('HEAD-CHAIN' in r for r in check.reasons))

    def test_refuse_when_paid_settled(self):
        e = 1065
        self.fx.write_turn_and_batch(e)
        self.fx.write_sample_decision(e, paid=True)
        self.fx.make_item(e, state=ItemState.RELEASED, paid=False)
        (self.fx.epoch_dir(e) / 'FINALIZED.json').write_text('{}')
        check = preflight(e, store=self.fx.store)
        self.assertFalse(check.ok)
        self.assertTrue(any('paid' in r.lower() or 'FINALIZED' in r for r in check.reasons))

    def test_refuse_item_paid_flag(self):
        e = 1065
        self.fx.write_turn_and_batch(e)
        self.fx.write_sample_decision(e, paid=False)
        self.fx.make_item(e, state=ItemState.RELEASED, paid=True)
        check = preflight(e, store=self.fx.store)
        self.assertFalse(check.ok)
        self.assertTrue(any('item.paid' in r for r in check.reasons))

    def test_head_moved_still_allowed(self):
        """Jacob: reruns bind CURRENT head — head move is not a refuse."""
        e = 1060
        self.fx.write_turn_and_batch(e)
        self.fx.write_sample_decision(e)
        self.fx.make_item(e, state=ItemState.RELEASED)
        (self.fx.pe / 'HEAD-CHAIN.json').write_text(json.dumps({
            'head': OTHER, 'applied': [{'candidate': OTHER, 'new_head': OTHER}],
        }))
        # Round also tracks current head.
        rd = self.fx.heldout / self.fx.round_id / 'ROUND.private.json'
        doc = json.loads(rd.read_text())
        doc['head'] = OTHER
        rd.write_text(json.dumps(doc))
        check = preflight(e, store=self.fx.store)
        self.assertTrue(check.ok, check.reasons)

    def test_refuse_batch_sha_mismatch(self):
        e = 1060
        self.fx.write_turn_and_batch(e, file_sha='0' * 64)
        self.fx.write_sample_decision(e)
        self.fx.make_item(e, state=ItemState.RELEASED)
        check = preflight(e, store=self.fx.store)
        self.assertFalse(check.ok)
        self.assertTrue(any('sha256' in r for r in check.reasons))

    def test_refuse_missing_batch_file(self):
        e = 1060
        ed = self.fx.epoch_dir(e)
        (ed / 'K1-TURN.json').write_text(json.dumps({
            'status': 'turn', 'epoch_index': e, 'hotkey': HOTKEY,
            'file_sha256': '1' * 64, 'head': HEAD,
        }))
        self.fx.write_sample_decision(e)
        self.fx.make_item(e, state=ItemState.RELEASED)
        check = preflight(e, store=self.fx.store)
        self.assertFalse(check.ok)
        self.assertTrue(any('batch file missing' in r for r in check.reasons))


class TestRerunHappyPath(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = _Env()

    def tearDown(self) -> None:
        self.fx.close()

    def _ok_epoch(self, e: int) -> ItemRecord:
        self.fx.write_turn_and_batch(e)
        self.fx.write_sample_decision(e, paid=False)
        (self.fx.epoch_dir(e) / 'PIT-RESULT.json').write_text(json.dumps({
            'status': 'pit-result', 'decision': 'green_fresh',
        }))
        (self.fx.epoch_dir(e) / 'PIT-DISPATCHED.json').write_text(json.dumps({
            'epoch': e, 'status': 'pit-dispatched',
        }))
        return self.fx.make_item(e, state=ItemState.RELEASED, paid=False)

    def test_cli_writes_request_not_items(self):
        e = 1060
        item = self._ok_epoch(e)
        doc = write_rerun_request(
            epoch=e, operator='jacob', reason='lost green', store_root=self.fx.pit,
        )
        self.assertEqual(doc['status'], 'requested')
        req = self.fx.pit / 'rerun' / ('RERUN-REQUEST-e%d.json' % e)
        self.assertTrue(req.is_file())
        self.assertTrue((self.fx.pit / 'items' / ('%s.json' % item.item_id)).is_file())
        self.assertTrue((self.fx.gate / str(e) / 'DECISION.json').is_file())

    def test_rerun_indices_match_current_round_frozen_list(self):
        """BLOCKER 1: SAMPLE indices == CURRENT round frozen list; seed ≠ attempt 1."""
        e = 1060
        self._ok_epoch(e)
        # Archive first so write_fresh_sample can place SAMPLE.
        store = self.fx.store
        item = store.list_items()[0]
        archive_attempt1(e, store=store, item=item)
        sample = write_fresh_sample(e, attempt=2)
        self.assertEqual(sample['indices'], ROUND_INDICES)
        self.assertEqual(sample['round_id'], self.fx.round_id)
        self.assertEqual(sample['seed'], self.fx.round_seed)
        classic = hashlib.sha256(
            ('%s:prior-baseline:cand:%d' % (self.fx.round_seed, e)).encode()
        ).hexdigest()
        self.assertEqual(sample['plan_seed'], rerun_cand_seed(self.fx.round_seed, e, 2))
        self.assertNotEqual(sample['plan_seed'], classic)

    def test_archive_and_consume(self):
        e = 1065
        old = self._ok_epoch(e)
        write_rerun_request(
            epoch=e, operator='jacob', reason='lost green e1065', store_root=self.fx.pit,
        )
        ctl = Controller.from_flag({'pit': {'enabled': True, 'mode': 'shadow', 'max_boxes': 2}})
        ctl.items = [old]
        out = consume_rerun_request(e, store=self.fx.store, ctl=ctl)
        self.assertEqual(out['status'], 'consumed')
        gate_arch = self.fx.gate / str(e) / 'attempt-1'
        for name in ('SAMPLE.private.json', 'DECISION.json', 'EVALUATED.private.json', 'COMMIT.json'):
            self.assertTrue((gate_arch / name).is_file(), name)
        self.assertTrue((self.fx.life / 'epochs' / str(e) / 'attempt-1' / 'PIT-RESULT.json').is_file())
        self.assertTrue((self.fx.pit / 'rerun' / ('e%d' % e) / 'attempt-1' / ('%s.json' % old.item_id)).is_file())
        self.assertFalse((self.fx.gate / str(e) / 'DECISION.json').exists())
        sm = json.loads((self.fx.gate / str(e) / 'SAMPLE.private.json').read_text())
        self.assertEqual(sm['attempt'], 2)
        self.assertEqual(sm['indices'], ROUND_INDICES)
        self.assertEqual(sm['plan_seed'], rerun_cand_seed(self.fx.round_seed, e, 2))
        # RERUN-ACTIVE present; PIT-RESULT absent (backpressure still counts).
        ed = self.fx.life / 'epochs' / str(e)
        self.assertTrue((ed / 'RERUN-ACTIVE.json').is_file())
        self.assertFalse((ed / 'PIT-RESULT.json').exists())
        self.assertFalse(_epoch_dir_ready(ed))  # ingest skips via RERUN-ACTIVE
        new_items = [i for i in ctl.items if i.item_id != old.item_id]
        self.assertEqual(len(new_items), 1)
        nxt = new_items[0]
        self.assertEqual(nxt.attempt, 2)
        self.assertEqual(nxt.priority, RERUN_PRIORITY)
        self.assertEqual(nxt.rerun_of, old.item_id)

    def test_refuse_rewrite_when_archive_exists(self):
        e = 1062
        old = self._ok_epoch(e)
        archive_attempt1(e, store=self.fx.store, item=old)
        with self.assertRaises(RerunError) as ctx:
            archive_attempt1(e, store=self.fx.store, item=None)
        self.assertIn('refuse rewrite', str(ctx.exception))
        # Hashes preserved.
        rr = json.loads((self.fx.pit / 'rerun' / ('e%d' % e) / 'RERUN.json').read_text())
        self.assertIn('DECISION.json', rr['hashes']['gate'])

    def test_resume_after_archive_without_rewriting_hashes(self):
        e = 1060
        old = self._ok_epoch(e)
        write_rerun_request(
            epoch=e, operator='jacob', reason='resume', store_root=self.fx.pit,
        )
        ctl = Controller.from_flag({'pit': {'enabled': True, 'mode': 'shadow', 'max_boxes': 2}})
        ctl.items = [old]
        # Archive only, simulate draw failure before SAMPLE.
        archive = archive_attempt1(e, store=self.fx.store, item=old)
        ctl.items = []
        hashes_before = json.dumps(archive['hashes'], sort_keys=True)
        out = consume_rerun_request(e, store=self.fx.store, ctl=ctl)
        self.assertTrue(out.get('resumed'))
        hashes_after = json.dumps(
            json.loads((self.fx.pit / 'rerun' / ('e%d' % e) / 'RERUN.json').read_text())['hashes'],
            sort_keys=True,
        )
        self.assertEqual(hashes_before, hashes_after)
        sm = json.loads((self.fx.gate / str(e) / 'SAMPLE.private.json').read_text())
        self.assertEqual(sm['indices'], ROUND_INDICES)

    def test_priority_binds_ahead_of_registration_index(self):
        older = ItemRecord(
            item_id='old', registration_index=1, epoch=1, state=ItemState.QUEUED, priority=0,
        )
        rerun = ItemRecord(
            item_id='rerun', registration_index=9999, epoch=1060,
            state=ItemState.QUEUED, priority=RERUN_PRIORITY, attempt=2,
        )
        nxt = select_next_queued([older, rerun], rented=0, max_boxes=2)
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt.item_id, 'rerun')

    def test_gate_writer_accepts_fresh_decision_after_archive(self):
        e = 1060
        self._ok_epoch(e)
        store = self.fx.store
        item = store.list_items()[0]
        archive_attempt1(e, store=store, item=item)
        self.assertFalse((self.fx.gate / str(e) / 'DECISION.json').exists())

        class _Rounds:
            def __init__(self):
                self._turns = []
                self._closed = False

            def get(self, rid):
                return {'id': rid, 'closed': self._closed}

            def turns(self, rid):
                return list(self._turns)

            def turn_add(self, rid, row):
                self._turns.append(row)

            def closed(self, rid):
                return self._closed

            def close(self, *a, **k):
                self._closed = True

        class _BV:
            @staticmethod
            def assert_public(doc, open_round):
                return None

        heldout = {
            'round_id': self.fx.round_id, 'set_size': 10, 'bar_z': 1.0,
            'k_cand': 4, 'head_rollouts': 4, 'scoring': 'prior-v1',
            'settings_hash': 'h' * 64, 'baseline': 'prior-v1',
            'max_tokens': 1024, 'scoring_mode': 'prior', 'tau_run': 0.0,
            'sigma_run_floor': 0.0, 'baseline_done_at_cr': 'now',
        }
        gw = HostGateWriter(
            epochs_root=self.fx.gate, shadow=False, rounds=_Rounds(),
            baseline_v1=_BV(),
        )
        path = gw.write_decision(
            epoch=e, head=HEAD, candidate=CAND, decision='green',
            evaluated={'z': 3.0, 'n': 10},
            reasons=['ok'], heldout=heldout,
            plan_seed='p' * 64, parallel_eval_meta={'source': 'pit'},
        )
        self.assertTrue(path.is_file())
        (self.fx.gate / str(e) / 'DECISION.json').write_text(json.dumps({
            'epoch': e, 'head': HEAD, 'candidate': CAND, 'decision': 'red',
            'no_decision': False, 'merge_blocked': False, 'evaluated': {'z': 0},
        }))
        with self.assertRaises(GateWriteError) as ctx:
            gw.write_decision(
                epoch=e, head=HEAD, candidate=CAND, decision='green',
                evaluated={'z': 3.0, 'n': 10},
                reasons=['ok'], heldout=heldout,
                plan_seed='p' * 64, parallel_eval_meta={'source': 'pit'},
            )
        self.assertIn('conflicting DECISION', str(ctx.exception))

    def test_no_stale_decision_reused(self):
        e = 1062
        self._ok_epoch(e)
        old_dec = json.loads((self.fx.gate / str(e) / 'DECISION.json').read_text())
        write_rerun_request(epoch=e, operator='jacob', reason='x', store_root=self.fx.pit)
        ctl = Controller.from_flag({'pit': {'enabled': True, 'mode': 'shadow', 'max_boxes': 2}})
        ctl.items = list(self.fx.store.list_items())
        consume_rerun_request(e, store=self.fx.store, ctl=ctl)
        self.assertFalse((self.fx.gate / str(e) / 'DECISION.json').exists())
        archived = json.loads(
            (self.fx.gate / str(e) / 'attempt-1' / 'DECISION.json').read_text()
        )
        self.assertEqual(archived['candidate'], old_dec['candidate'])
        sm = json.loads((self.fx.gate / str(e) / 'SAMPLE.private.json').read_text())
        self.assertEqual(sm['attempt'], 2)
        self.assertEqual(sm['indices'], ROUND_INDICES)


class TestLedgerReconcile(unittest.TestCase):
    def test_pit_green_without_head_chain_rejected(self):
        rows = reconcile_rows(
            [{'epoch': 1062, 'candidate': CAND, 'decision': 'green', 'verdict': 'promoted'}],
            chain_candidates=set(),
            archived_epochs=set(),
            pit_epochs={1062},
        )
        self.assertEqual(rows[0]['verdict'], 'rejected')
        self.assertEqual(rows[0]['reconcile'], 'green-without-head-chain')

    def test_pit_green_with_head_chain_promoted(self):
        rows = reconcile_rows(
            [{'epoch': 1062, 'candidate': CAND, 'decision': 'green', 'verdict': 'promoted'}],
            chain_candidates={CAND},
            archived_epochs=set(),
            pit_epochs={1062},
        )
        self.assertEqual(rows[0]['verdict'], 'promoted')

    def test_archived_pit_attempt_superseded(self):
        rows = reconcile_rows(
            [{'epoch': 1060, 'candidate': CAND, 'decision': 'green', 'verdict': 'promoted'}],
            chain_candidates={CAND},
            archived_epochs={1060},
            pit_epochs={1060},
        )
        self.assertEqual(rows[0]['verdict'], 'superseded')
        self.assertTrue(rows[0]['superseded'])

    def test_pre_pe_promoted_rows_untouched(self):
        """BLOCKER 2: pre-PE promoted history must not flip without PIT-DISPATCHED."""
        pre = [
            {'epoch': 100, 'candidate': 'd' * 64, 'decision': 'green', 'verdict': 'promoted'},
            {'epoch': 200, 'candidate': 'e' * 64, 'decision': 'green', 'verdict': 'promoted'},
            {'epoch': 271, 'candidate': 'f' * 64, 'decision': 'green', 'verdict': 'promoted'},
        ]
        rows = reconcile_rows(
            pre + [{'epoch': 1062, 'candidate': CAND, 'decision': 'green', 'verdict': 'promoted'}],
            chain_candidates=set(),  # empty HEAD-CHAIN
            archived_epochs=set(),
            pit_epochs={1062},  # only e1062 is PIT
        )
        self.assertEqual(rows[0]['verdict'], 'promoted')
        self.assertEqual(rows[1]['verdict'], 'promoted')
        self.assertEqual(rows[2]['verdict'], 'promoted')
        self.assertNotIn('reconcile', rows[0])
        self.assertEqual(rows[3]['verdict'], 'rejected')


class TestGpuFallback(unittest.TestCase):
    def test_format_node_gpu_unknown_is_question_mark(self):
        self.assertEqual(format_node_gpu(None), '?')
        self.assertEqual(format_node_gpu(''), '?')
        self.assertEqual(format_node_gpu('B300', gpu_count=8), '8xB300')


if __name__ == '__main__':
    unittest.main()
