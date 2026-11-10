"""Live pe_score.Pending must keep the box; scored-no-decision retries; GREEN w/o C blocks."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .apply_live import LiveApplicator
from .config import load_pit_config
from .controller import BASELINE_WAIT_ALERT_S, BASELINE_WAIT_RELEASE_S, Controller
from .pe_lane import LaneService
from .pe_rent import RentService
from .pe_score import BaselinePending, call_live_host_score, is_live_pending_exc, score
from .state import ItemRecord, ItemState


# Real live exception shape (ops.parallel_eval.live.pe_score.Pending) — class name
# + message; import may fail hermetically, so mirror the live class exactly.
class Pending(Exception):
    """Mirror of live pe_score.Pending."""


H = 'a' * 64
C = 'c' * 64
PROBE = {
    'k': 8, 'samples': 16, 'partial': False, 'dtype': 'bf16',
    'planned_samples': 16, 'score': {'decision': 'green', 'z': 2.0},
}


def _bundle_meta():
    return {
        'probe_bundle': {
            'item_id': 'e1065',
            'bound_head': H,
            'bound_master_digest': 'b' * 64,
            'candidate_id': C,
            'probe_result': dict(PROBE),
            'master_upload': {},
            'train_receipt': {
                'candidate_id': C,
                'candidate_path': 'box:/ckpt/' + C,
                'candidate_master_path': '/workspace/master',
                'defer_candidate_upload': True,
                'candidate_on_box': True,
            },
            'box_public_key_hex': 'ab' * 32,
            'training_report': {},
            'train_gpu': 'B300',
            'eval_gpu': 'B300',
        },
        'candidate_on_box': True,
        'defer_candidate_upload': True,
        'train_gpu': 'B300',
        'eval_gpu': 'B300',
    }


class TestLivePendingShape(unittest.TestCase):
    def test_is_live_pending_by_class_name_and_message(self):
        exc = Pending('baseline for round r-abc not ready')
        self.assertTrue(is_live_pending_exc(exc))
        self.assertTrue(is_live_pending_exc(BaselinePending('baseline for round r-x not ready')))
        self.assertFalse(is_live_pending_exc(RuntimeError('boom')))

    def test_call_live_host_score_converts_pending_to_decision(self):
        def host(epoch, rid, result):
            raise Pending('baseline for round %s not ready' % rid)

        out = call_live_host_score(host, epoch=1065, round_id='r-test', result=PROBE)
        self.assertEqual(out.get('decision'), 'pending')
        self.assertIn('baseline for round r-test not ready', out.get('pending_reason', ''))

    def test_pit_score_returns_pending_not_raise(self):
        def host(epoch, rid, result):
            raise Pending('baseline for round %s not ready' % rid)

        out = score(
            epoch=1065, round_id='r-test', result=PROBE,
            candidate_id=C, head_id=H, host_score=host, require_host=True,
        )
        self.assertEqual(out['decision'], 'pending')
        self.assertEqual(out['candidate'], C)


class TestBaselinePendingKeepBox(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        pe = Path(self.td) / 'pe'
        (pe / 'pit').mkdir(parents=True)
        self.prev = {k: os.environ.get(k) for k in ('PE_STATE', 'PIT_STATE')}
        os.environ['PE_STATE'] = str(pe)
        os.environ['PIT_STATE'] = str(pe / 'pit')
        self.clock = {'t': 1_000.0}

    def tearDown(self):
        for k, v in self.prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _ctl(self, score_fn):
        cfg = load_pit_config({
            'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 2},
        })
        rent = RentService(max_boxes=2)
        rent.ledger.active['pod-1065'] = {
            'box_id': 'pod-1065', 'name': 'affine-pe-pit-1065-a0', 'at': 1.0,
        }
        ctl = Controller(
            config=cfg,
            rent=rent,
            lane=LaneService(),
            box_io_factory=lambda: None,
            now_fn=lambda: self.clock['t'],
            score_fn=score_fn,
        )
        ctl.current_head = H
        ctl.current_master_digest = 'b' * 64
        ctl.current_round_id = 'r-test'
        ctl.rent.list_pods = lambda: [{'id': 'pod-1065', 'name': 'affine-pe-pit-1065-a0'}]
        return ctl

    def _scored_item(self, *, box_id='pod-1065'):
        return ItemRecord(
            item_id='e1065', registration_index=1, epoch=1065,
            state=ItemState.SCORED, bound_head=H, bound_master_digest='b' * 64,
            bound_round_id='r-test', candidate_id=C, box_id=box_id,
            box_name='affine-pe-pit-1065-a0' if box_id else None,
            result_stored=True, sealed=True,
            meta=_bundle_meta(),
        )

    def test_decide_pending_keeps_box_and_retries(self):
        calls = {'n': 0}

        def host(epoch, rid, result):
            calls['n'] += 1
            raise Pending('baseline for round %s not ready' % rid)

        ctl = self._ctl(host)
        item = self._scored_item()
        ctl.items.append(item)

        updated, note = ctl._decide(item)
        self.assertEqual(note, 'wait-baseline')
        self.assertEqual(updated.state, ItemState.SCORED)
        self.assertEqual(updated.box_id, 'pod-1065')
        self.assertIn('pod-1065', ctl.rent.ledger.active)
        self.assertIsNone(updated.decision)
        self.assertIn('baseline_wait_since', updated.meta or {})
        first_calls = calls['n']
        self.assertGreaterEqual(first_calls, 1)

        # Within backoff: no re-score.
        self.clock['t'] += 1.0
        updated2, note2 = ctl._decide(updated)
        self.assertEqual(note2, 'wait-baseline')
        self.assertEqual(calls['n'], first_calls)
        self.assertEqual(updated2.box_id, 'pod-1065')

        # Past backoff: score again, still pending, still keep.
        self.clock['t'] = float((updated2.meta or {}).get('baseline_wait_next_at') or 0) + 1.0
        updated3, note3 = ctl._decide(updated2)
        self.assertEqual(note3, 'wait-baseline')
        self.assertGreater(calls['n'], first_calls)
        self.assertEqual(updated3.box_id, 'pod-1065')

    def test_pending_alert_after_two_hours_still_keeps_box(self):
        """2h alert + no upload path: refuse release (never drop unuploaded C), keep box."""
        def host(epoch, rid, result):
            raise Pending('baseline for round %s not ready' % rid)

        ctl = self._ctl(host)
        # No live_applicator → 45m cap cannot upload → must keep the box.
        item = self._scored_item()
        updated, _ = ctl._decide(item)
        self.clock['t'] = float((updated.meta or {})['baseline_wait_since']) + float(BASELINE_WAIT_ALERT_S) + 1.0
        # Clear backoff so _decide re-enters wait path with alert.
        meta = dict(updated.meta or {})
        meta['baseline_wait_next_at'] = self.clock['t'] - 1.0
        updated.meta = meta
        updated2, note = ctl._decide(updated)
        self.assertEqual(note, 'wait-baseline-upload-failed')
        self.assertTrue((updated2.meta or {}).get('baseline_wait_alerted'))
        self.assertTrue((Path(os.environ['PIT_STATE']) / 'BASELINE-WAIT-ALERT.json').is_file())
        self.assertEqual(updated2.box_id, 'pod-1065')
        self.assertIn('pod-1065', ctl.rent.ledger.active)

    def test_tick_pending_does_not_fail_and_release(self):
        def host(epoch, rid, result):
            raise Pending('baseline for round %s not ready' % rid)

        ctl = self._ctl(host)
        item = self._scored_item()
        ctl.items = [item]
        out = ctl.tick()
        notes = [a.get('note') for a in out.get('actions') or [] if a.get('item_id') == 'e1065']
        self.assertTrue(
            any(n == 'wait-baseline' for n in notes)
            or any(a.get('op') == 'keep-after-result' for a in out.get('actions') or []),
            out.get('actions'),
        )
        kept = ctl.items[0]
        self.assertEqual(kept.box_id, 'pod-1065')
        self.assertEqual(kept.state, ItemState.SCORED)
        self.assertIn('pod-1065', ctl.rent.ledger.active)

    def test_fail_and_release_keeps_scored_unuploaded(self):
        ctl = self._ctl(lambda *a, **k: {'decision': 'red', 'z': -1})
        item = self._scored_item()
        out = ctl._fail_and_release(item, 'simulated-score-boom')
        self.assertEqual(out.box_id, 'pod-1065')
        self.assertEqual(out.state, ItemState.SCORED)
        self.assertIn('pod-1065', ctl.rent.ledger.active)
        self.assertTrue((out.meta or {}).get('post_result_keep'))

    def test_scored_no_decision_retries_then_records(self):
        """e1065-style scored item (box gone) retries; RED records a decision."""
        calls = {'n': 0}

        def host(epoch, rid, result):
            calls['n'] += 1
            if calls['n'] < 2:
                raise Pending('baseline for round %s not ready' % rid)
            return {'decision': 'red', 'z': -0.5, 'evaluated': {'z': -0.5}}

        ctl = self._ctl(host)
        item = self._scored_item(box_id=None)
        item.meta = dict(item.meta or {})
        # No box — still must retry scoring.
        updated, note = ctl._decide(item)
        self.assertEqual(note, 'wait-baseline')
        self.assertIsNone(updated.decision)

        self.clock['t'] = float((updated.meta or {}).get('baseline_wait_next_at') or 0) + 1.0
        updated2, note2 = ctl._decide(updated)
        self.assertEqual(updated2.state, ItemState.DECIDED)
        self.assertEqual(updated2.decision, 'red')
        self.assertIn(note2, ('red', 'decided'))
        self.assertFalse(updated2.paid)

    def test_green_without_candidate_blocks_terminal(self):
        def host(epoch, rid, result):
            return {'decision': 'green', 'z': 2.1, 'evaluated': {'z': 2.1}}

        ctl = self._ctl(host)
        item = self._scored_item(box_id=None)  # box already released
        updated, note = ctl._decide(item)
        self.assertEqual(note, 'blocked-no-candidate')
        self.assertEqual(updated.state, ItemState.NO_DECISION)
        self.assertEqual(updated.decision, 'blocked_no_candidate')
        self.assertFalse(updated.paid)
        self.assertEqual((updated.meta or {}).get('blocked'), 'no_candidate')

    def test_green_stale_box_id_blocks_no_apply_gate_or_bench(self):
        """e1065-like: box_id present but released → blocked_no_candidate; no side effects."""
        applies: list = []
        benches: list = []
        accepts: list = []

        def host(epoch, rid, result):
            return {'decision': 'green', 'z': 2.1, 'evaluated': {'z': 2.1}}

        ctl = self._ctl(host)
        # Ledger no longer holds the released box (e1065/e1066 class).
        ctl.rent.ledger.active.clear()
        ctl.rent.list_pods = lambda: []
        ctl.apply_hook = lambda it: applies.append(it.item_id)
        ctl.on_green_fresh = lambda it: benches.append(it.item_id)
        cfg = ctl.config
        ctl.live_applicator = LiveApplicator(
            config=cfg,
            verify_uploads=lambda **k: {'candidate_ok': True, 'master_ok': True},
            accept_pit_decision=lambda payload: accepts.append(dict(payload)),
            upload_candidate=lambda **k: {'ok': True},
            upload_master=lambda **k: {'ok': True, 'complete': True, 'shard_count': 2},
            checkpoint_signer=object(),
        )
        stale = '0f07be4e' + 'a' * 56
        item = self._scored_item(box_id=stale)
        item.box_name = 'affine-pe-pit-1065-gone'
        ctl.items = [item]

        updated, note = ctl._decide(item)
        self.assertEqual(note, 'blocked-no-candidate')
        self.assertEqual(updated.state, ItemState.NO_DECISION)
        self.assertEqual(updated.decision, 'blocked_no_candidate')
        self.assertFalse(updated.paid)
        self.assertFalse((updated.meta or {}).get('gate_emit_ready'))
        self.assertFalse((updated.meta or {}).get('green_apply_pending'))
        self.assertEqual(applies, [])
        self.assertEqual(benches, [])
        self.assertEqual(accepts, [])
        # No APPLY ready file under PE_STATE.
        pe = Path(os.environ['PE_STATE'])
        apply_files = list(pe.rglob('apply/*.json')) + list(pe.rglob('**/APPLY*.json'))
        self.assertEqual(apply_files, [])

    def test_migrate_stale_box_ids_moves_to_released_box_id(self):
        ctl = self._ctl(lambda *a, **k: {'decision': 'red', 'z': -1})
        ctl.rent.ledger.active.clear()
        stale = '639bf826' + 'b' * 56
        item = self._scored_item(box_id=stale)
        item.box_name = 'affine-pe-pit-1066-gone'
        ctl.items = [item]
        moved = ctl.migrate_stale_box_ids()
        self.assertEqual(moved, [stale])
        self.assertIsNone(item.box_id)
        self.assertIsNone(item.box_name)
        self.assertEqual((item.meta or {}).get('released_box_id'), stale)
        self.assertEqual((item.meta or {}).get('released_box_name'), 'affine-pe-pit-1066-gone')

    def test_release_stamps_released_box_id(self):
        ctl = self._ctl(lambda *a, **k: {'decision': 'red', 'z': -1})
        item = self._scored_item()
        # Mark uploaded so fail/release path is allowed for a RED decided item.
        item.state = ItemState.DECIDED
        item.decision = 'red'
        item.meta = dict(item.meta or {}, live_apply={
            'candidate_verified': True, 'master_verified': True, 'master_complete': True,
        })
        with mock.patch(
            'ops.parallel_eval.pit.controller.master_index_is_complete', return_value=True,
        ):
            released, note = ctl._release(item, reason='test-release')
        self.assertIn(note, ('released', 'already-released', 'release-noop'))
        self.assertIsNone(released.box_id)
        self.assertEqual((released.meta or {}).get('released_box_id'), 'pod-1065')

    def test_baseline_wait_cap_releases_when_already_uploaded(self):
        def host(epoch, rid, result):
            raise Pending('baseline for round %s not ready' % rid)

        ctl = self._ctl(host)
        item = self._scored_item()
        item.meta = dict(item.meta or {}, live_apply={
            'candidate_verified': True, 'master_verified': True, 'master_complete': True,
        })
        with mock.patch(
            'ops.parallel_eval.pit.controller.master_index_is_complete', return_value=True,
        ):
            updated, _ = ctl._decide(item)
            self.clock['t'] = (
                float((updated.meta or {})['baseline_wait_since'])
                + float(BASELINE_WAIT_RELEASE_S) + 1.0
            )
            meta = dict(updated.meta or {})
            meta['baseline_wait_next_at'] = self.clock['t'] - 1.0
            updated.meta = meta
            updated2, note = ctl._decide(updated)
        self.assertEqual(note, 'wait-baseline-box-released')
        self.assertIsNone(updated2.box_id)
        self.assertEqual(updated2.state, ItemState.SCORED)
        self.assertNotIn('pod-1065', ctl.rent.ledger.active)
        self.assertEqual((updated2.meta or {}).get('released_box_id'), 'pod-1065')
        self.assertTrue((updated2.meta or {}).get('baseline_wait_box_released'))

    def test_baseline_wait_cap_uploads_then_releases(self):
        uploads = {'cand': 0, 'master': 0}

        def host(epoch, rid, result):
            raise Pending('baseline for round %s not ready' % rid)

        ctl = self._ctl(host)
        item = self._scored_item()
        meta = dict(item.meta or {})
        meta.update({
            'candidate_sha256': C,
            'candidate_master_digest': 'd' * 64,
            'candidate_path': 'box:/ckpt/' + C,
            'candidate_master_path': '/workspace/master',
            'candidate_on_box': True,
            'defer_candidate_upload': True,
        })
        item.meta = meta

        def upload_cand(**k):
            uploads['cand'] += 1
            return {'ok': True}

        def upload_master(**k):
            uploads['master'] += 1
            return {'ok': True, 'complete': True, 'shard_count': 4}

        ctl.live_applicator = LiveApplicator(
            config=ctl.config,
            verify_uploads=lambda **k: {
                'candidate_ok': True, 'master_ok': True, 'master_complete': True,
            },
            upload_candidate=upload_cand,
            upload_master=upload_master,
            checkpoint_signer=object(),
        )
        with mock.patch(
            'ops.parallel_eval.pit.controller.master_index_is_complete', return_value=True,
        ), mock.patch(
            'ops.parallel_eval.pit.apply_live.master_index_is_complete', return_value=True,
        ):
            updated, _ = ctl._decide(item)
            self.clock['t'] = (
                float((updated.meta or {})['baseline_wait_since'])
                + float(BASELINE_WAIT_RELEASE_S) + 1.0
            )
            m = dict(updated.meta or {})
            m['baseline_wait_next_at'] = self.clock['t'] - 1.0
            updated.meta = m
            updated2, note = ctl._decide(updated)
        self.assertEqual(note, 'wait-baseline-box-released')
        self.assertEqual(uploads['cand'], 1)
        self.assertEqual(uploads['master'], 1)
        self.assertIsNone(updated2.box_id)
        self.assertEqual(updated2.state, ItemState.SCORED)
        self.assertTrue(
            (updated2.meta or {}).get('live_apply', {}).get('candidate_verified')
        )
        self.assertTrue(
            (updated2.meta or {}).get('live_apply', {}).get('master_verified')
        )

    def test_baseline_wait_cap_never_releases_unuploaded(self):
        def host(epoch, rid, result):
            raise Pending('baseline for round %s not ready' % rid)

        ctl = self._ctl(host)
        item = self._scored_item()
        ctl.live_applicator = LiveApplicator(
            config=ctl.config,
            verify_uploads=lambda **k: {'candidate_ok': False, 'master_ok': False},
            upload_candidate=lambda **k: (_ for _ in ()).throw(RuntimeError('upload boom')),
            upload_master=lambda **k: {'ok': True, 'complete': True, 'shard_count': 1},
            checkpoint_signer=object(),
        )
        updated, _ = ctl._decide(item)
        self.clock['t'] = (
            float((updated.meta or {})['baseline_wait_since'])
            + float(BASELINE_WAIT_RELEASE_S) + 1.0
        )
        m = dict(updated.meta or {})
        m['baseline_wait_next_at'] = self.clock['t'] - 1.0
        updated.meta = m
        updated2, note = ctl._decide(updated)
        self.assertEqual(note, 'wait-baseline-upload-failed')
        self.assertEqual(updated2.box_id, 'pod-1065')
        self.assertIn('pod-1065', ctl.rent.ledger.active)
        self.assertEqual(updated2.state, ItemState.SCORED)


if __name__ == '__main__':
    unittest.main()
