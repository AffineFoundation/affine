#!/usr/bin/env python3
"""GREEN must never lose its box before upload+verify (live e1060 regression)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ops.parallel_eval.live import pe_box

from .apply_live import ApplyLiveError, LiveApplicator, PromoteRefused, refuse_if_unpromotable
from .config import load_pit_config
from .controller import Controller
from .durable import DurableStore, item_to_dict
from .pe_lane import LaneService
from .pe_rent import RentService
from .run_host import _bound_box_ids, _item_needs_box, _start_rent_reconcile, tick_once
from .signer import FixtureRootSigner
from .state import ItemRecord, ItemState


H = 'a' * 64
C = 'b' * 64
M = 'c' * 64


def _heldout(rid='r-e1060'):
    return dict(
        round_id=rid, set_size=500, bar_z=1.0, k_cand=8, head_rollouts=12,
        scoring='ivw-v2', settings_hash='f' * 64, baseline='prior-v1',
        max_tokens=64000, scoring_mode='separate-run', tau_run=0.01,
        sigma_run_floor=0.0, baseline_done_at_cr='2026-01-01 00:00:00 CR',
    )


def _green_item(**kw) -> ItemRecord:
    base = dict(
        item_id='e1060' + '0' * 27,
        registration_index=1060,
        epoch=1060,
        state=ItemState.DECIDED,
        decision='green_fresh',
        paid=True,
        fresh=True,
        box_id='pod-1060',
        box_name='affine-pe-pit-1060-a0',
        bound_head=H,
        bound_master_digest=M,
        bound_round_id='r-e1060',
        bound_plan_seed='p' * 64,
        candidate_id=C,
        meta=dict(
            pod=dict(id='pod-1060', name='affine-pe-pit-1060-a0'),
            candidate_path='box:/ckpt/' + C,
            candidate_on_box=True,
            defer_candidate_upload=True,
            box_candidate_dir='/ckpt/' + C,
            candidate_files={'model.safetensors': 'd' * 64},
            candidate_sha256=C,
            candidate_master_digest=M,
            scored=dict(
                decision='green', z=1.7, evaluated=dict(z=1.7),
                heldout=_heldout(), plan_seed='p' * 64, candidate=C,
            ),
        ),
    )
    base.update(kw)
    return ItemRecord(**base)


class TestPeBoxUploadCandidateSignature(unittest.TestCase):
    def test_apply_fresh_real_pe_box_upload_candidate_no_duplicate_kw(self):
        """apply_live passes path+candidate_path; pe_box.upload_candidate must accept both."""
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'cutover': True}})
        with tempfile.TemporaryDirectory() as td:
            cand_dir = Path(td) / 'cand'
            cand_dir.mkdir()
            (cand_dir / 'model.safetensors').write_bytes(b'weights')
            uploads = Path(td) / 'uploads'
            uploads.mkdir()
            item = _green_item(
                meta=dict(
                    candidate_path=str(cand_dir),
                    candidate_sha256=C,
                    candidate_master_digest=M,
                    candidate_files={'model.safetensors': 'd' * 64},
                    scored=dict(decision='green', z=1.7, heldout=_heldout(), candidate=C),
                ),
            )
            # Real module seam (not a mocked signature).
            calls = []

            def wrap_upload(**kwargs):
                calls.append(dict(kwargs))
                return pe_box.upload_candidate(**kwargs)

            applicator = LiveApplicator(
                config=cfg,
                verify_uploads=lambda **k: {'candidate_ok': True, 'master_ok': True},
                upload_candidate=wrap_upload,
                upload_master=lambda **k: {'ok': True},
                accept_pit_decision=lambda payload: True,
                checkpoint_signer=FixtureRootSigner(),
            )
            env = {
                'PIT_UPLOAD_ROOT': str(uploads),
                'PE_LIFE_STATE': str(Path(td) / 'life'),
            }
            (Path(td) / 'life' / 'pending').mkdir(parents=True)
            with mock.patch.dict(os.environ, env, clear=False):
                # Must not raise TypeError: multiple values for candidate_path.
                result = applicator._apply_fresh(
                    item, scored=item.meta['scored'],
                )
            self.assertTrue(result.applied)
            self.assertEqual(len(calls), 1)
            self.assertIn('candidate_path', calls[0])
            self.assertIn('path', calls[0])


class TestReconcileKeepsGreenBox(unittest.TestCase):
    def test_startup_reconcile_readopts_green_does_not_orphan_release(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'cutover': True}})
        released = []
        listings = [dict(id='pod-1060', name='affine-pe-pit-1060-a0', gpu='B300')]

        rent = RentService(
            max_boxes=2,
            host_release=lambda name_or_id, reason='done': released.append((name_or_id, reason)),
            list_pods=lambda: list(listings),
        )
        ctl = Controller(
            config=cfg,
            rent=rent,
            lane=LaneService(),
            box_io_factory=lambda: None,
            durable_save_fn=lambda item: None,
        )
        item = _green_item()
        # Simulate crash mid-apply: box_id cleared incorrectly but pod meta remains.
        item.box_id = None
        ctl.items.append(item)
        self.assertTrue(_item_needs_box(item))
        rent.ledger_path = None  # skip disk
        out = _start_rent_reconcile(ctl)
        self.assertEqual(released, [])
        self.assertEqual(item.box_id, 'pod-1060')
        self.assertIn('pod-1060', out.get('adopted') or out.get('restored_box_ids') or [])
        self.assertIn('pod-1060', _bound_box_ids(ctl))
        # True unbound orphan is still reaped.
        listings.append(dict(id='orphan-9', name='affine-pe-pit-9999-a0', gpu='B300'))
        out2 = _start_rent_reconcile(ctl)
        self.assertIn('orphan-9', out2['released'])


class TestGreenApplyRetry(unittest.TestCase):
    def test_upload_failure_keeps_box_and_retries(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'cutover': True}})
        with tempfile.TemporaryDirectory() as td:
            pit = Path(td) / 'pit'
            pit.mkdir()
            env = {'PIT_STATE': str(pit), 'PE_STATE': str(Path(td))}
            destroyed = []

            def boom_upload(**kwargs):
                raise ApplyLiveError('got multiple values for keyword argument candidate_path')

            applicator = LiveApplicator(
                config=cfg,
                verify_uploads=lambda **k: {'candidate_ok': True, 'master_ok': True},
                upload_candidate=boom_upload,
                destroy_state=lambda **k: destroyed.append(dict(k)) or {'ok': True},
                checkpoint_signer=FixtureRootSigner(),
            )
            rent = RentService(max_boxes=1)
            rent.ledger.register('pod-1060', item_id='e1060', name='affine-pe-pit-1060-a0', at=1.0)
            clock = {'t': 1000.0}

            def now():
                return clock['t']

            with mock.patch.dict(os.environ, env, clear=False):
                ctl = Controller(
                    config=cfg,
                    rent=rent,
                    lane=LaneService(),
                    box_io_factory=lambda: None,
                    live_applicator=applicator,
                    destroy_box_state_fn=lambda **k: destroyed.append(dict(k)) or {'ok': True},
                    now_fn=now,
                    durable_save_fn=lambda item: None,
                )
                item = _green_item()
                ctl.items.append(item)
                updated, note = ctl._finish_decision(item)
                self.assertEqual(note, 'green-apply-retry')
                self.assertEqual(updated.state, ItemState.DECIDED)
                self.assertEqual(updated.decision, 'green_fresh')
                self.assertEqual(updated.box_id, 'pod-1060')
                self.assertIn('pod-1060', rent.ledger.active)
                self.assertEqual(destroyed, [])
                self.assertTrue((pit / 'GREEN-APPLY-ALERT.json').is_file())
                self.assertGreater(float(updated.meta['apply_retry_at']), clock['t'])
                # Fail path must also keep the box.
                kept = ctl._fail_and_release(updated, 'simulated-tick-exc')
                self.assertEqual(kept.box_id, 'pod-1060')
                self.assertEqual(kept.state, ItemState.DECIDED)
                self.assertEqual(destroyed, [])

    def test_never_promote_on_timeout(self):
        with self.assertRaises(PromoteRefused):
            refuse_if_unpromotable({'decision': 'green', 'timeout': True})


class TestGreenRestartE2E(unittest.TestCase):
    def test_green_worker_result_restart_keeps_box_uploads_and_pit_result(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'cutover': True}})
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            life = root / 'life'
            ep = life / 'epochs' / '1060'
            ep.mkdir(parents=True)
            gate = root / 'gate'
            epochs = gate / 'epochs' / '1060'
            heldout = gate / 'heldout' / 'r-e1060'
            heldout.mkdir(parents=True)
            epochs.mkdir(parents=True)
            (heldout / 'ROUND.private.json').write_text(json.dumps({
                'id': 'r-e1060', 'head': H, 'n': 1, 'seed': 's' * 64, 'indices': [0],
            }))
            pe = root / 'pe'
            pit = pe / 'pit'
            pit.mkdir(parents=True)
            (life / 'CURRENT.json').write_text(json.dumps({
                'epoch_index': 1060, 'checkpoint': {'id': H},
            }))
            (pe / 'CURRENT.json').write_text(json.dumps({
                'epoch_index': 1060, 'checkpoint': {'id': H},
            }))
            (pe / 'HEAD-CHAIN.json').write_text(json.dumps({'genesis': H, 'applied': []}))
            (pe / 'MASTER-INDEX.json').write_text(json.dumps({H: M}))
            (root / 'life-state' / 'pending').mkdir(parents=True)
            cand_dir = root / 'cand'
            cand_dir.mkdir()
            (cand_dir / 'model.safetensors').write_bytes(b'weights')
            uploads = root / 'uploads'
            uploads.mkdir()

            listings = [dict(id='pod-1060', name='affine-pe-pit-1060-a0', gpu='B300')]
            released = []
            uploaded = []

            def realish_upload(**kwargs):
                uploaded.append(dict(kwargs))
                return pe_box.upload_candidate(**kwargs)

            env = {
                'LIFE_ROOT': str(life), 'PE_LIFE': str(life),
                'PE_LIFE_STATE': str(root / 'life-state'),
                'PE_STATE': str(pe), 'PIT_STATE': str(pit),
                'HG_GATE_STATE': str(gate), 'PE_GATE_EP': str(gate / 'epochs'),
                'PE_HELDOUT': str(gate / 'heldout'),
                'K1_TURNS': str(root / 'K1-TURNS.jsonl'),
                'PIT_UPLOAD_ROOT': str(uploads),
            }
            (root / 'K1-TURNS.jsonl').write_text('')

            with mock.patch.dict(os.environ, env, clear=False):
                store = DurableStore(root=pit)
                item = _green_item(
                    meta=dict(
                        pod=dict(id='pod-1060', name='affine-pe-pit-1060-a0'),
                        candidate_path=str(cand_dir),
                        candidate_sha256=C,
                        candidate_master_digest=M,
                        candidate_files={'model.safetensors': 'd' * 64},
                        epoch_dir=str(ep),
                        scored=dict(
                            decision='green', z=1.7, evaluated=dict(z=1.7),
                            heldout=_heldout(), plan_seed='p' * 64, candidate=C,
                        ),
                    ),
                )
                store.save_item(item)
                # Durable worker-result present (restart recovery).
                wr = pit / 'items' / ('%s.worker-result.json' % item.item_id)
                wr.write_text(json.dumps({
                    'ok': True, 'note': 'box-complete', 'item': item_to_dict(item),
                }) + '\n')

                rent = RentService(
                    max_boxes=1,
                    host_release=lambda name_or_id, reason='done': released.append((name_or_id, reason)),
                    list_pods=lambda: list(listings),
                )
                rent.ledger_path = str(pit / 'rent-ledger.json')
                applicator = LiveApplicator(
                    config=cfg,
                    verify_uploads=lambda **k: {'candidate_ok': True, 'master_ok': True},
                    upload_candidate=realish_upload,
                    upload_master=lambda **k: {'ok': True, 'master_digest': M},
                    accept_pit_decision=lambda payload: True,
                    destroy_state=lambda **k: None,
                    checkpoint_signer=FixtureRootSigner(),
                )
                ctl = Controller(
                    config=cfg,
                    rent=rent,
                    lane=LaneService(),
                    box_io_factory=lambda: None,
                    live_applicator=applicator,
                    score_fn=lambda **k: {
                        'decision': 'green', 'z': 1.7, 'candidate': C,
                        'heldout': _heldout(), 'evaluated': {'z': 1.7},
                    },
                    durable_save_fn=store.save_item,
                    destroy_box_state_fn=lambda **k: None,
                    round_id_fn=lambda e: 'r-e1060',
                    plan_seed_fn=lambda e: 'p' * 64,
                )
                # Host restart: load items then reconcile (build_controller order).
                ctl.items = store.list_items()
                self.assertEqual(len(ctl.items), 1)
                # Clear box_id to simulate lost in-memory binding.
                ctl.items[0].box_id = None
                _start_rent_reconcile(ctl)
                self.assertEqual(released, [])
                self.assertEqual(ctl.items[0].box_id, 'pod-1060')
                ctl.set_head(H, epoch=1060, master_digest=M)

                emitted = False
                out = {}
                for _ in range(8):
                    out = tick_once(ctl, store, dry_run=False)
                    if (ctl.items[0].meta or {}).get('decision_emitted'):
                        emitted = True
                        break
                self.assertTrue(emitted, ctl.items[0].meta)
                self.assertTrue(uploaded)
                final = ctl.items[0]
                self.assertEqual(final.decision, 'green_fresh')
                self.assertTrue((final.meta or {}).get('live_apply', {}).get('candidate_verified'))
                self.assertTrue((ep / 'PIT-RESULT.json').is_file())
                self.assertTrue((epochs / 'DECISION.json').is_file())
                # After verified apply the box is released; orphan cleanup never ran first.
                self.assertNotIn(
                    ('pod-1060', 'reconcile-orphan'), released,
                )
                self.assertTrue(
                    final.box_id is None
                    or any(r[0] == 'pod-1060' for r in released)
                )


if __name__ == '__main__':
    unittest.main()
