"""DECIDED fail/replace, per-item isolation, worker-result restart, e1055 recovery."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from .apply_live import LiveApplicator, destroy_box_state as apply_destroy
from .config import load_pit_config
from .controller import Controller
from .durable import DurableStore, item_to_dict
from .box import ScriptedBoxIO
from .pe_lane import LaneService
from .pe_rent import RentService
from .run_host import _gate_emit_ready, tick_once
from .state import ItemRecord, ItemState, apply_transition


def _ctl(**kwargs):
    kw = dict(
        rent=RentService(max_boxes=2),
        lane=LaneService(),
        box_io_factory=lambda: ScriptedBoxIO(),
    )
    kw.update(kwargs)
    return Controller(**kw)


H = 'a' * 64
C = 'b' * 64
M = 'c' * 64


def _heldout(rid='r-1'):
    return dict(
        round_id=rid, set_size=500, bar_z=1.0, k_cand=8, head_rollouts=12,
        scoring='ivw-v2', settings_hash='f' * 64, baseline='prior-v1',
        max_tokens=64000, scoring_mode='separate-run', tau_run=0.01,
        sigma_run_floor=0.0, baseline_done_at_cr='2026-01-01 00:00:00 CR',
    )


class TestDecidedFailIsolation(unittest.TestCase):
    def test_destroy_box_state_ignores_extra_kwargs(self):
        def host_destroy(*, box_id, reason='done'):
            return dict(ok=True, box_id=box_id, reason=reason)

        out = apply_destroy(
            host_destroy, box_id='pod-1', candidate_id=C, reason='red',
        )
        self.assertTrue(out['ok'])

        def pe_box_style(*, box_id, reason='done', **_ignored):
            return dict(ok=True, box_id=box_id)

        out2 = apply_destroy(
            pe_box_style, box_id='pod-2', candidate_id=C, reason='red',
        )
        self.assertEqual(out2['box_id'], 'pod-2')

    def test_fail_and_release_decided_returns_item_not_tuple(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'cutover': True}})
        rent = RentService(max_boxes=2)
        rent.ledger.register('pod-x', item_id='i1', name='affine-pe-pit-x', at=0.0)
        ctl = _ctl(config=cfg, rent=rent)
        item = ItemRecord(
            item_id='i1', registration_index=0, epoch=1055, state=ItemState.DECIDED,
            decision='red', paid=False, box_id='pod-x', candidate_id=C, bound_head=H,
            meta=dict(scored=dict(decision='red', heldout=_heldout(), evaluated=dict(z=-1))),
        )
        ctl.items.append(item)
        out = ctl._fail_and_release(item, 'destroy-typeerror')
        self.assertIsInstance(out, ItemRecord)
        self.assertEqual(out.state, ItemState.RELEASED)
        self.assertTrue((out.meta or {}).get('gate_emit_ready'))
        # Idempotent: second release does not grow history.
        n = len(rent.ledger.history)
        ctl._release(out, reason='again')
        self.assertEqual(len(rent.ledger.history), n)

    def test_tick_continues_after_one_item_fails(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'cutover': True}})
        ctl = _ctl(config=cfg)
        bad = ItemRecord(
            item_id='bad', registration_index=0, epoch=1, state=ItemState.DECIDED,
            decision='red', box_id=None, candidate_id=C, bound_head=H,
        )
        good = ItemRecord(
            item_id='good', registration_index=1, epoch=2, state=ItemState.QUEUED,
        )
        ctl.items.extend([bad, good])

        def boom(item):
            if item.item_id == 'bad':
                raise RuntimeError('simulated')
            return item, 'noop'

        with mock.patch.object(ctl, '_advance', side_effect=boom):
            out = ctl.tick()
        ops = [a['op'] for a in out['actions'] if a.get('item_id') == 'bad']
        self.assertIn('fail', ops)
        # good item still present (tick did not abort)
        self.assertTrue(any(i.item_id == 'good' for i in ctl.items))

    def test_worker_result_on_disk_survives_restart_no_rerun(self):
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 1}})
        with tempfile.TemporaryDirectory() as td:
            pit = Path(td) / 'pit'
            pit.mkdir()
            env = {'PIT_STATE': str(pit), 'PE_STATE': str(Path(td))}
            with mock.patch.dict(os.environ, env, clear=False):
                store = DurableStore(root=pit)
                ctl = _ctl(config=cfg, rent=RentService(max_boxes=1), durable_save_fn=store.save_item)
                crashed = ItemRecord(
                    item_id='w1', registration_index=0, epoch=7,
                    state=ItemState.TRAINING, box_id='pod-1', bound_head=H,
                    candidate_id=C, result_stored=False,
                    meta=dict(pod=dict(id='pod-1', name='affine-pe-pit-1')),
                )
                finished = ItemRecord(
                    item_id='w1', registration_index=0, epoch=7,
                    state=ItemState.RESULT, box_id='pod-1', bound_head=H,
                    candidate_id=C, result_stored=True,
                    meta=dict(
                        pod=dict(id='pod-1', name='affine-pe-pit-1'),
                        scored=dict(decision='red', z=-0.2, heldout=_heldout(), evaluated=dict(z=-0.2)),
                        probe_bundle=dict(ok=True),
                    ),
                )
                ctl._save_worker_result('w1', dict(ok=True, item=finished, note='box-complete'))
                # Simulate restart: new controller, no in-memory worker.
                ctl2 = _ctl(config=cfg, rent=RentService(max_boxes=1), durable_save_fn=store.save_item)
                ctl2.items.append(crashed)
                ran = []

                def no_rerun(*a, **k):
                    ran.append(1)
                    raise AssertionError('must not re-run box')

                with mock.patch.object(ctl2, '_run_box_async', side_effect=no_rerun):
                    advanced, note = ctl2._poll_box_worker(crashed)
                self.assertEqual(ran, [])
                self.assertEqual(advanced.state, ItemState.RESULT)
                self.assertTrue(advanced.result_stored)
                self.assertIn('box-complete', note)

    def test_e1055_recovery_finishes_emit_without_box(self):
        """decided + DECISION(RED) on disk + no box → turns + PIT-RESULT, decision unchanged."""
        cfg = load_pit_config({'pit': {'enabled': True, 'mode': 'live', 'cutover': True}})
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            life = root / 'life'
            ep = life / 'epochs' / '1055'
            ep.mkdir(parents=True)
            gate = root / 'gate'
            epochs = gate / 'epochs'
            heldout = gate / 'heldout' / 'r-e1055'
            heldout.mkdir(parents=True)
            epochs.mkdir(parents=True)
            (heldout / 'ROUND.private.json').write_text(json.dumps({
                'id': 'r-e1055', 'head': H, 'n': 1,
            }))
            # Pre-written RED decision (live failure left this behind).
            (epochs / '1055').mkdir(parents=True)
            dec = dict(
                version='sn120-promotion-gate-v1', epoch=1055, head=H, candidate=C,
                mode='live', decision='red', evaluated=dict(z=-0.4), reasons=['below'],
                heldout=_heldout('r-e1055'), no_decision=False, merge_blocked=False,
                source='pit', red_credit='zero', chart_status='red',
            )
            (epochs / '1055' / 'DECISION.json').write_text(json.dumps(dec, indent=1) + '\n')
            pe = root / 'pe'
            pit = pe / 'pit'
            pit.mkdir(parents=True)
            (life / 'CURRENT.json').write_text(json.dumps({
                'epoch_index': 1055, 'checkpoint': {'id': H},
            }))
            env = {
                'LIFE_ROOT': str(life), 'PE_LIFE': str(life),
                'PE_STATE': str(pe), 'PIT_STATE': str(pit),
                'HG_GATE_STATE': str(gate), 'PE_GATE_EP': str(epochs),
                'PE_HELDOUT': str(gate / 'heldout'),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                store = DurableStore(root=pit)
                ctl = _ctl(config=cfg, rent=RentService(max_boxes=1), durable_save_fn=store.save_item)
                item = ItemRecord(
                    item_id='e1055', registration_index=0, epoch=1055,
                    state=ItemState.DECIDED, decision='red', paid=False,
                    box_id=None, candidate_id=C, bound_head=H,
                    bound_round_id='r-e1055', bound_plan_seed='p' * 64,
                    last_error='host heldout rounds() unavailable',
                    meta=dict(
                        scored=dict(
                            decision='red', evaluated=dict(z=-0.4),
                            heldout=_heldout('r-e1055'), plan_seed='p' * 64,
                        ),
                        epoch_dir=str(ep),
                    ),
                )
                ctl.items.append(item)
                self.assertTrue(_gate_emit_ready(item) or item.state.value == 'decided')
                # Finish decision → released + gate_emit_ready, decision stays red.
                advanced, note = ctl._finish_decision(item)
                ctl._replace(advanced)
                self.assertEqual(advanced.decision, 'red')
                self.assertEqual(advanced.state, ItemState.RELEASED)
                self.assertTrue((advanced.meta or {}).get('gate_emit_ready'))
                self.assertTrue(_gate_emit_ready(advanced))
                out = tick_once(ctl, store, dry_run=False)
                self.assertTrue((ep / 'PIT-RESULT.json').exists())
                final = ctl.items[0]
                self.assertEqual(final.decision, 'red')
                self.assertTrue((final.meta or {}).get('decision_emitted'))
                turns = (gate / 'heldout' / 'r-e1055' / 'TURNS.private.jsonl')
                self.assertTrue(turns.exists())
                rows = [json.loads(l) for l in turns.read_text().splitlines() if l.strip()]
                self.assertTrue(any(r.get('epoch') == 1055 and r.get('decision') == 'red' for r in rows))
                self.assertTrue(any(a.get('epoch') == 1055 for a in (out.get('advanced') or [])))

    def test_worker_result_on_disk_ticks_to_decision_emit_and_release(self):
        """Finished worker-result + no worker → decide, DECISION/PIT-RESULT, release box."""
        cfg = load_pit_config({
            'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 1},
        })
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            life = root / 'life'
            ep = life / 'epochs' / '1056'
            ep.mkdir(parents=True)
            gate = root / 'gate'
            epochs = gate / 'epochs'
            heldout = gate / 'heldout' / 'r-e1056'
            heldout.mkdir(parents=True)
            epochs.mkdir(parents=True)
            (heldout / 'ROUND.private.json').write_text(json.dumps({
                'id': 'r-e1056', 'head': H, 'n': 1, 'seed': 's' * 64, 'indices': [0],
            }))
            pe = root / 'pe'
            pit = pe / 'pit'
            pit.mkdir(parents=True)
            (life / 'CURRENT.json').write_text(json.dumps({
                'epoch_index': 1056, 'checkpoint': {'id': H},
            }))
            (pe / 'CURRENT.json').write_text(json.dumps({
                'epoch_index': 1056, 'checkpoint': {'id': H},
            }))
            (pe / 'HEAD-CHAIN.json').write_text(json.dumps({'genesis': H, 'applied': []}))
            (pe / 'MASTER-INDEX.json').write_text(json.dumps({H: M}))
            env = {
                'LIFE_ROOT': str(life), 'PE_LIFE': str(life),
                'PE_STATE': str(pe), 'PIT_STATE': str(pit),
                'HG_GATE_STATE': str(gate), 'PE_GATE_EP': str(epochs),
                'PE_HELDOUT': str(gate / 'heldout'),
                'K1_TURNS': str(root / 'K1-TURNS.jsonl'),
            }
            (root / 'K1-TURNS.jsonl').write_text('')
            destroyed = []

            def score_fn(**_k):
                return {
                    'decision': 'red', 'z': -0.3, 'candidate': C,
                    'evaluated': {'z': -0.3}, 'heldout': _heldout('r-e1056'),
                    'plan_seed': 'p' * 64,
                }

            with mock.patch.dict(os.environ, env, clear=False):
                store = DurableStore(root=pit)
                rent = RentService(max_boxes=1)
                rent.ledger.register('pod-1056', item_id='e1056', name='l1056-a0', at=1.0)
                applicator = LiveApplicator(
                    config=cfg,
                    destroy_state=lambda **k: destroyed.append(dict(k)) or {'ok': True},
                )
                ctl = _ctl(
                    config=cfg, rent=rent, durable_save_fn=store.save_item,
                    score_fn=score_fn, live_applicator=applicator,
                    destroy_box_state_fn=lambda **k: destroyed.append(dict(k)) or {'ok': True},
                    round_id_fn=lambda e: 'r-e1056',
                    plan_seed_fn=lambda e: 'p' * 64,
                )
                ctl.set_head(H, epoch=1056, master_digest=M)
                crashed = ItemRecord(
                    item_id='e1056' + '0' * 27, registration_index=1056, epoch=1056,
                    state=ItemState.TRAINING, box_id='pod-1056', box_name='l1056-a0',
                    bound_head=H, bound_master_digest=M, bound_round_id='r-e1056',
                    bound_plan_seed='p' * 64, candidate_id=C, result_stored=False,
                    meta=dict(
                        pod=dict(id='pod-1056', name='l1056-a0'),
                        epoch_dir=str(ep),
                    ),
                )
                finished = ItemRecord(
                    item_id=crashed.item_id, registration_index=1056, epoch=1056,
                    state=ItemState.RESULT, box_id='pod-1056', box_name='l1056-a0',
                    bound_head=H, bound_master_digest=M, bound_round_id='r-e1056',
                    bound_plan_seed='p' * 64, candidate_id=C, result_stored=True,
                    meta=dict(
                        pod=dict(id='pod-1056', name='l1056-a0'),
                        epoch_dir=str(ep),
                        probe_bundle=dict(
                            item_id=crashed.item_id, candidate_id=C,
                            bound_head=H, bound_master_digest=M,
                            probe_result=dict(score=dict(decision='red', z=-0.3)),
                            master_upload={}, train_receipt={},
                            box_public_key_hex='00', training_report={},
                        ),
                    ),
                )
                store.save_item(crashed)
                ctl._save_worker_result(
                    crashed.item_id, dict(ok=True, item=finished, note='box-complete'),
                )
                # Sidecar must not break list_items / rebuild.
                ctl.items = store.list_items()
                self.assertEqual(len(ctl.items), 1)
                self.assertEqual(ctl.items[0].state, ItemState.TRAINING)

                emitted = False
                for _ in range(12):
                    out = tick_once(ctl, store, dry_run=False)
                    item = ctl.items[0]
                    if (item.meta or {}).get('decision_emitted'):
                        emitted = True
                        break
                self.assertTrue(emitted, 'decision never emitted: state=%s decision=%s meta=%s' % (
                    ctl.items[0].state, ctl.items[0].decision, ctl.items[0].meta,
                ))
                final = ctl.items[0]
                self.assertEqual(final.decision, 'red')
                self.assertFalse(final.paid)
                self.assertEqual(final.state, ItemState.RELEASED)
                self.assertIsNone(final.box_id)
                self.assertNotIn('pod-1056', rent.ledger.active)
                self.assertTrue((epochs / '1056' / 'DECISION.json').is_file())
                self.assertTrue((ep / 'PIT-RESULT.json').is_file())
                self.assertTrue(any(a.get('epoch') == 1056 for a in (out.get('advanced') or [])))


if __name__ == '__main__':
    unittest.main()

