#!/usr/bin/env python3
"""Cutover post-GREEN path: pending → apply ready → lifecycle consume → record_applied."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

HEAD = 'a' * 64
CAND = 'b' * 64
AUTH = {'input': HEAD, 'id': CAND, 'payload': {'checkpoint_id': CAND, 'input_head': HEAD}}


class TestCutoverApplyReconcile(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.life = self.td / 'life'
        self.pe = self.td / 'pe'
        self.gate = self.td / 'gate' / 'epochs'
        for p in (self.life / 'pending', self.life / 'apply', self.life / 'epochs', self.pe, self.gate):
            p.mkdir(parents=True)
        self.flag = self.td / 'flag.json'
        self.flag.write_text(json.dumps({
            'parallel_eval': {
                'enabled': True,
                'pit': {'enabled': True, 'mode': 'live', 'cutover': True},
            },
        }))
        (self.pe / 'HEAD-CHAIN.json').write_text(json.dumps({'genesis': HEAD, 'applied': []}))
        (self.pe / 'GREENS.json').write_text('[]')
        (self.life / 'CURRENT.json').write_text(json.dumps({
            'epoch_index': 50,
            'checkpoint': {'id': HEAD},
            'optimizer_step': 1,
            'checkpoint_authorization': {'input': 'genesis', 'id': HEAD},
        }))
        self.env = {
            'PE_LIFE_STATE': str(self.life),
            'LIFE_ROOT': str(self.life),
            'PE_LIFE': str(self.life),
            'PE_STATE': str(self.pe),
            'PE_FLAG': str(self.flag),
            'PE_ST': str(self.pe),
            'PE_GATE_EP': str(self.gate),
            'HG_GATE_STATE': str(self.td / 'gate'),
        }
        self._prev = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        live = Path(__file__).resolve().parents[1] / 'live'
        if str(live) not in sys.path:
            sys.path.insert(0, str(live))
        self._live_mod_names = (
            'replace_pod', 'pe_common', 'pe_apply', 'pe_live', 'pe_lifecycle',
        )
        self._prev_mods = {n: sys.modules.get(n) for n in self._live_mod_names}

    def tearDown(self):
        for k, v in self._prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        # Restore sys.modules so a stub replace_pod cannot leak into later suites.
        for name, prev in self._prev_mods.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev

    def test_pit_green_reaches_record_applied_under_cutover(self):
        # Stub host-only deps before importing live pe_* (scoped; tearDown restores).
        rp = types.ModuleType('replace_pod')

        class _Pod:
            pass

        rp.Pod = _Pod
        with mock.patch.dict(sys.modules, {'replace_pod': rp}):
            for name in ('pe_common', 'pe_apply', 'pe_live', 'pe_lifecycle'):
                sys.modules.pop(name, None)
            import pe_apply as PA  # type: ignore
            import pe_live as PL  # type: ignore
            import pe_lifecycle as Life  # type: ignore
            import pe_common as PC  # type: ignore

            # Point gate epochs + lifecycle CURRENT at our temp trees.
            PL.EP = self.gate
            PC.LIFE = self.life
            PL.LIFE = self.life

            recorded = []
            real_record = PA.record_applied

            def _spy(item, kind, s, new_head, base_head, merge_info=None):
                recorded.append(dict(candidate=item['candidate'], new_head=new_head, base_head=base_head))
                return real_record(item, kind, s, new_head, base_head, merge_info=merge_info)

            # 1) PIT GREEN → pending JSON + GREENS row (accept_pit_decision).
            pend = self.life / 'pending' / 'e51-test.json'
            pend.write_text(json.dumps({
                'epoch': 51,
                'candidate': CAND,
                'training_head': HEAD,
                'checkpoint': {'id': CAND, 'files': {'x': '1' * 64}},
                'optimizer_step': 2,
                'checkpoint_authorization': AUTH,
                'pe_keep': True,
                'source': 'pit',
            }))
            self.assertTrue(PA.accept_pit_decision({
                'source': 'pit',
                'epoch': 51,
                'candidate': CAND,
                'training_head': HEAD,
                'green_at': time.time(),
                'decision': 'green_fresh',
                'pending': str(pend),
            }))

            # 2) Gate DECISION so apply_tick can match (and for k1 last-10 shape).
            (self.gate / '51').mkdir(parents=True)
            (self.gate / '51' / 'DECISION.json').write_text(json.dumps({
                'decision': 'green',
                'candidate': CAND,
                'source': 'pit',
                'evaluated': {'z': 2.0, 'n': 100},
                'decided_at': time.time(),
            }))

            # 3) Under cutover skip_dispatch=True — apply still runs → ready.
            out = PL.tick(lambda *a, **k: None, skip_dispatch=True)
            self.assertEqual(out.get('dispatch'), dict(skipped='warm-off-or-pit-cutover'))
            ready = self.life / 'apply' / 'e51.json'
            self.assertTrue(ready.is_file(), msg='apply_tick must write ready under cutover: %s' % out)
            self.assertEqual(json.loads(ready.read_text())['state'], 'ready')

            # 4) Lifecycle consumes ready → offered → updated.
            class LC:
                def _training(self, directory, opening, dispatch, current, now):
                    return None

            def v829i(Lifecycle):
                prior = Lifecycle._training

                def _training(self, *a):
                    return prior(self, *a)

                Lifecycle._training = _training

            v829i(LC)
            Life.install_pe(LC)
            ep = self.life / 'epochs' / '51'
            ep.mkdir(parents=True)
            cur = json.loads((self.life / 'CURRENT.json').read_text())
            r = LC()._training(ep, None, None, cur, 0)
            self.assertEqual(r['status'], 'updated')
            self.assertEqual(r['checkpoint']['id'], CAND)
            offered = json.loads(ready.read_text())
            self.assertEqual(offered['state'], 'offered')

            # 5) Durable CURRENT + FINALIZED so confirm_applied → consumed.
            # confirm_applied requires CURRENT.epoch_index > offered_epoch.
            ckpt = {'id': CAND, 'files': {'x': '1' * 64}}
            (self.life / 'CURRENT.json').write_text(json.dumps({
                'epoch_index': 52,
                'checkpoint': ckpt,
                'optimizer_step': 2,
                'checkpoint_authorization': AUTH,
            }))
            (ep / 'FINALIZED.json').write_text(json.dumps({
                'payload': {
                    'input_checkpoint': HEAD,
                    'result': {
                        'status': 'updated',
                        'checkpoint': ckpt,
                        'optimizer_step': 2,
                        'checkpoint_authorization': AUTH,
                    },
                },
            }))
            # Offered apply row must carry the same checkpoint object the FINALIZED used.
            offered = json.loads(ready.read_text())
            offered['checkpoint'] = ckpt
            offered['checkpoint_authorization'] = AUTH
            ready.write_text(json.dumps(offered))
            confirmed = Life.confirm_applied(self.life, json.loads((self.life / 'CURRENT.json').read_text()))
            self.assertTrue(confirmed, msg='confirm_applied must consume offered apply')
            self.assertEqual(json.loads(ready.read_text())['state'], 'consumed')

            # 6) reconcile_tick → record_applied (HEAD-CHAIN + hash-sync hook).
            with mock.patch.object(PA, 'record_applied', side_effect=_spy):
                rec = PL.reconcile_tick()
            self.assertIn(51, rec)
            self.assertEqual(recorded[0]['new_head'], CAND)
            self.assertEqual(recorded[0]['base_head'], HEAD)
            chain = json.loads((self.pe / 'HEAD-CHAIN.json').read_text())
            self.assertTrue(any(a.get('new_head') == CAND for a in chain.get('applied') or []))


class TestPitGreenLast10Credit(unittest.TestCase):
    """PIT DECISION source=pit counts as GREEN for k1 last-10 payees."""

    def test_pit_source_green_is_k1_verdict_green(self):
        from .decision_emit import build_decision_doc, map_pit_decision
        from .state import ItemRecord, ItemState

        item = ItemRecord(
            item_id='it1', registration_index=1, epoch=77,
            state=ItemState.DECIDED, bound_head=HEAD, candidate_id=CAND,
            decision='green_fresh', fresh=True, paid=True, uid=3,
            hotkey_prefix='5Miner', bound_round_id='r-77',
        )
        doc = build_decision_doc(
            item,
            scored={
                'evaluated': {'z': 1.5, 'n': 500},
                'heldout': {'round_id': 'r-77', 'set_size': 500, 'bar_z': 1.0,
                            'k_cand': 8, 'head_rollouts': 8, 'scoring': 'prior-v1',
                            'settings_hash': 's' * 64, 'baseline': 'prior-v1',
                            'max_tokens': 65536, 'scoring_mode': 'separate',
                            'tau_run': 0.0, 'sigma_run_floor': 0.0,
                            'baseline_done_at_cr': 'x'},
                'reasons': [],
            },
        )
        self.assertEqual(doc['source'], 'pit')
        self.assertEqual(doc['decision'], 'green')
        self.assertEqual(map_pit_decision('green_fresh'), 'green')
        # k1_service.verdicts: v = dec.get('decision') if in ('green','red') — no source filter.
        v = doc['decision'] if doc['decision'] in ('green', 'red') else None
        self.assertEqual(v, 'green')
        # last-10 payees filter: verdict == 'green' (distinct UIDs).
        recs = [
            dict(verdict='green', epoch_index=77, uid=3, hotkey='5MinerCreditKey', z=1.5),
            dict(verdict='red', epoch_index=76, uid=2, hotkey='5Other', z=0.1),
        ]
        seen, out = set(), []
        for r in sorted((r for r in recs if r.get('verdict') == 'green'), key=lambda r: -r['epoch_index']):
            u = int(r['uid'])
            if u in seen:
                continue
            seen.add(u)
            out.append(r)
        self.assertEqual([r['uid'] for r in out], [3])
        self.assertEqual(doc['parallel_eval']['source'], 'pit')


if __name__ == '__main__':
    unittest.main()
