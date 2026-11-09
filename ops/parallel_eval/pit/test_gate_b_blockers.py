#!/usr/bin/env python3
"""Gate B blockers (c2accd87 follow-up): real live return shapes.

1. pe_lane.boot returns None — adapters must coerce.
2. round_id from SAMPLE.private.json (refuse fixture default).
3. shadow prefix affine-pe-pit-shadow-* releases via live pe_rent assert;
   release verifies lium ps and alerts if still present.
4. Lifecycle cutover writes JOB/DISPATCH/K1 + PIT-DISPATCHED, then queue_ingest
   picks FINALIZED+PIT-DISPATCHED epochs; decision writes PIT-RESULT; next epoch
   proceeds. No fixture defaults.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from unittest import mock

from .box import ScriptedBoxIO
from .config import PitConfig
from .controller import Controller
from .durable import DurableStore
from .pe_lane import LaneService, _as_host_dict
from .pe_rent import PREFIX_LIVE, PREFIX_SHADOW, RentError, RentService, is_live_pit_name, is_shadow_pit_name
from .queue_ingest import sync, write_pit_result
from .state import ItemState


HEAD = 'a' * 64
MASTER = 'b' * 64
CAND = 'd' * 64


class TestBootNoneCoerce(unittest.TestCase):
    def test_as_host_dict_none_default(self):
        self.assertEqual(_as_host_dict(None, default={'ok': True}), {'ok': True})
        self.assertEqual(_as_host_dict({'a': 1}), {'a': 1})

    def test_as_host_dict_unwraps_run_probe_tuple(self):
        """Live pe_lane.run_probe returns (res, meta); unwrap mapping payload."""
        res = {'partial': False, 'samples': 8456, 'k': 8, 'dtype': 'bfloat16'}
        meta = {'run_name': 'r', 'wall_s': 1.0, 'plan_seed': 'p' * 64, 'k': 8, 'n': 1057}
        self.assertEqual(_as_host_dict((res, meta)), res)
        with self.assertRaises(Exception):
            _as_host_dict(('not-a-mapping', meta))

    def test_lane_boot_tolerates_live_none(self):
        """Real live pe_lane.boot returns None (side-effect only)."""
        live_path = Path(__file__).resolve().parents[1] / 'live'
        src = (live_path / 'pe_lane.py').read_text()
        self.assertIn('def boot(P):', src)
        # Live boot body has no explicit return — Python yields None.
        boot_body = src.split('def boot(P):', 1)[1].split('def push_bundle')[0]
        self.assertNotIn('return ', boot_body)

        calls = []

        def host_boot(pod, **kw):
            calls.append(pod)
            return None  # live shape

        lane = LaneService(require_host=True, host_boot=host_boot)
        out = lane.boot({'id': 'box-1', 'name': 'affine-pe-pit-1-a1'})
        self.assertEqual(out.get('ok'), True)
        self.assertEqual(len(calls), 1)

        # Serve None also coerced.
        lane.host_serve = lambda *a, **k: None
        out2 = lane.serve({'id': 'box-1'}, CAND)
        self.assertEqual(out2.get('ok'), True)

    def test_lane_run_probe_unwraps_live_tuple(self):
        """Fake host run_probe returns live (res, meta); RESULT shape matches e1053."""
        planned = 8456

        def host_run_probe(P, cid, indices, plan_seed, k=8, max_tokens=65536, env_id='affine_gatepool', **kw):
            # Live shape: (result_mapping, meta_dict). samples == len(indices)*k.
            n = len(indices) * int(k)
            self.assertEqual(n, planned)
            res = {
                'partial': False,  # bool
                'samples': n,      # int
                'k': k,
                'dtype': 'bfloat16',
                'planned_samples': n,
            }
            meta = {
                'run_name': 'probe-e1053',
                'wall_s': 12.0,
                'plan_seed': plan_seed,
                'k': k,
                'n': len(indices),
            }
            return res, meta

        lane = LaneService(require_host=True, host_run_probe=host_run_probe)
        # 1057 indices * k=8 = 8456 (live RESULT after full e1053 rollouts).
        indices = list(range(1057))
        out = lane.run_probe(
            {'id': 'box-1'}, CAND,
            indices=indices,
            plan_seed='p' * 64,
            k=8,
        )
        self.assertIs(out.get('partial'), False)
        self.assertEqual(out.get('samples'), 8456)
        self.assertIsInstance(out.get('partial'), bool)
        self.assertIsInstance(out.get('samples'), int)


class TestRoundIdFromSample(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.gate_ep = self.td / 'gate' / 'epochs'
        self.gate_ep.mkdir(parents=True)
        os.environ['PE_GATE_EP'] = str(self.gate_ep)
        os.environ['PE_STATE'] = str(self.td / 'pe')
        os.environ['PE_LIFE'] = str(self.td / 'life')
        os.environ['PIT_STATE'] = str(self.td / 'pe' / 'pit')
        (self.td / 'pe' / 'pit').mkdir(parents=True)
        (self.td / 'life').mkdir(parents=True)

    def tearDown(self):
        for k in ('PE_GATE_EP', 'PE_STATE', 'PE_LIFE', 'PIT_STATE'):
            os.environ.pop(k, None)

    def test_bind_uses_sample_round_id_not_fixture(self):
        (self.gate_ep / '42').mkdir()
        (self.gate_ep / '42' / 'SAMPLE.private.json').write_text(json.dumps({
            'round_id': 'r-live-42',
            'indices': [1, 2, 3],
        }))

        # Use the REAL live pe_round_seed.round_id_for_epoch.
        live_path = Path(__file__).resolve().parents[1] / 'live'
        if str(live_path) not in sys.path:
            sys.path.insert(0, str(live_path))
        import pe_round_seed as seed  # type: ignore

        rid = seed.round_id_for_epoch(42)
        self.assertEqual(rid, 'r-live-42')

        ctl = Controller(
            config=PitConfig(enabled=True, mode='shadow', max_boxes=1),
            rent=RentService(max_boxes=1),
            lane=LaneService(),
            box_io_factory=ScriptedBoxIO,
            round_id_fn=seed.round_id_for_epoch,
            plan_seed_fn=lambda e: 'p' * 64,
        )
        # No current_round_id — must come from SAMPLE via round_id_fn.
        self.assertEqual(ctl.current_round_id, '')
        ctl.set_head(HEAD, epoch=40, master_digest=MASTER)
        ctl.enqueue(epoch=42, registration_index=42)
        out = ctl.tick()
        ops = [a.get('op') for a in out.get('actions') or []]
        self.assertIn('bound', ops)
        self.assertEqual(ctl.items[0].bound_round_id, 'r-live-42')

    def test_refuse_bind_when_sample_missing(self):
        ctl = Controller(
            config=PitConfig(enabled=True, mode='shadow', max_boxes=1),
            rent=RentService(max_boxes=1),
            lane=LaneService(),
            box_io_factory=ScriptedBoxIO,
            # round_id_fn raises like live pe_round_seed when SAMPLE missing.
            round_id_fn=lambda e: (_ for _ in ()).throw(ValueError('SAMPLE missing')),
        )
        ctl.set_head(HEAD, epoch=1, master_digest=MASTER)
        ctl.enqueue(epoch=99, registration_index=99)
        out = ctl.tick()
        ops = [a.get('op') for a in out.get('actions') or []]
        self.assertIn('refuse-bind-round-id', ops)
        self.assertEqual(ctl.items[0].state, ItemState.QUEUED)

    def test_fixture_default_removed_from_controller(self):
        ctl = Controller(
            config=PitConfig(enabled=True, mode='shadow', max_boxes=1),
            rent=RentService(max_boxes=1),
            lane=LaneService(),
            box_io_factory=ScriptedBoxIO,
        )
        self.assertNotEqual(ctl.current_round_id, 'round-fixture')
        self.assertEqual(ctl.current_round_id, '')
        self.assertNotEqual(ctl.round_seed_commitment, 'seed-commitment-fixture')


class TestShadowReleasePrefix(unittest.TestCase):
    def test_shadow_prefix_satisfies_live_affine_pe(self):
        from ops.parallel_eval.pit import PREFIX_SHADOW as SH
        self.assertTrue(SH.startswith('affine-pe-'))
        self.assertTrue(SH.startswith('affine-pe-pit-shadow-'))
        self.assertTrue(is_shadow_pit_name(SH + '10-a1'))
        self.assertFalse(is_live_pit_name(SH + '10-a1'))
        self.assertTrue(is_live_pit_name(PREFIX_LIVE + '10-a1'))

    def test_release_verifies_lium_ps_and_alerts(self):
        """Live pe_rent.release asserts affine-pe-*; we verify pod is gone after."""
        live_ps = [{'id': 'pod-1', 'name': PREFIX_SHADOW + '7-a1'}]
        released = []

        def host_release(name_or_id, reason='done'):
            # Mimic live pe_rent.release prefix assert.
            assert str(name_or_id).startswith('affine-pe-') or any(
                (p.get('name') or '').startswith('affine-pe-') for p in live_ps
            ), 'live pe_rent.release prefix assert'
            released.append((name_or_id, reason))
            # Bug shape: ledger thinks released but pod still in ps.
            # (We leave live_ps populated so verify fails.)

        def list_pods():
            return list(live_ps)

        rent = RentService(
            max_boxes=2,
            shadow=True,
            name_prefix=PREFIX_SHADOW,
            share_pe_cap=False,
            require_host=True,
            host_rent=lambda name, ttl, **kw: {
                'id': 'pod-1', 'name': name, 'gpu': 'B300',
                'price_usd_h': 40.0, 'auto_backup_disabled': True,
            },
            host_release=host_release,
            list_pods=list_pods,
        )
        out = rent.rent(item_id='it1', epoch=7, attempt=1)
        self.assertTrue(out['name'].startswith('affine-pe-pit-shadow-'))
        with self.assertRaises(RentError) as cm:
            rent.release(out['box_id'], reason='done')
        self.assertIn('release verify failed', str(cm.exception))
        self.assertEqual(len(released), 1)

        # After ps clears, release succeeds.
        live_ps.clear()
        rent2 = RentService(
            max_boxes=2,
            shadow=True,
            name_prefix=PREFIX_SHADOW,
            share_pe_cap=False,
            require_host=True,
            host_rent=lambda name, ttl, **kw: {
                'id': 'pod-2', 'name': name, 'gpu': 'B300',
                'price_usd_h': 40.0, 'auto_backup_disabled': True,
            },
            host_release=lambda *a, **k: None,
            list_pods=lambda: [],
        )
        out2 = rent2.rent(item_id='it2', epoch=8, attempt=1)
        ok = rent2.release(out2['box_id'], reason='done')
        self.assertTrue(ok['ok'])


class TestCutoverLifecycleIngestE2E(unittest.TestCase):
    """lifecycle tick (cutover) → queue_ingest → fake box → decision/credit → next epoch.

    Isolation: pe_lifecycle.flag_path()/pe_state_path() re-read PE_FLAG /
    PE_LIFE_STATE on every call so a prior suite import cannot freeze stale
    temp paths (the full-suite flake vs alone).
    """

    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.life = self.td / 'life'
        self.pe = self.td / 'pe'
        self.gate_ep = self.td / 'gate' / 'epochs'
        self.flag = self.td / 'flag.json'
        for p in (self.life, self.pe / 'pit', self.gate_ep, self.pe / 'apply'):
            p.mkdir(parents=True)
        self.fp32 = self.td / 'fp32'
        self.fp32.mkdir()
        (self.fp32 / ('%s.json' % HEAD)).write_text(json.dumps({
            'master_digest': MASTER, 'bf16_sha256': HEAD,
        }))
        self.env = {
            'PE_LIFE_STATE': str(self.life),
            'PE_LIFE': str(self.life),
            'PE_STATE': str(self.pe),
            'PIT_STATE': str(self.pe / 'pit'),
            'PE_FLAG': str(self.flag),
            'PE_GATE_EP': str(self.gate_ep),
            'PIT_FP32_MASTER_INDEX': str(self.fp32),
            'K1_TURNS': str(self.td / 'K1-TURNS.jsonl'),
            # Hash-sync hook: hermetic fake (never a host script).
            'PE_SYNC_EXPECTED_HASH': str(self.td / 'fake_sync_expected_hash.py'),
            'PE_SYNC_EXPECTED_HASH_PYTHON': sys.executable,
            'PE_SYNC_EXPECTED_HASH_LOG': str(self.td / 'hook-sync.log'),
        }
        (self.td / 'fake_sync_expected_hash.py').write_text('import sys\n')
        (self.td / 'K1-TURNS.jsonl').write_text('')
        self._prev = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        # Cutover flag shape (real).
        self.flag.write_text(json.dumps({
            'parallel_eval': {
                'enabled': True,
                'pit': {
                    'enabled': True,
                    'mode': 'live',
                    'cutover': True,
                    'max_boxes': 2,
                    'master_upload': 'after_green',
                    'root_mode': 'delegated_host',
                },
            },
        }))
        (self.life / 'CURRENT.json').write_text(json.dumps({
            'epoch_index': 200,
            'checkpoint': {'id': HEAD},
            'optimizer_step': 10,
            'checkpoint_authorization': {'input': 'genesis', 'id': HEAD},
        }))
        (self.pe / 'HEAD-CHAIN.json').write_text(json.dumps({
            'genesis': HEAD, 'applied': [],
        }))
        (self.pe / 'MASTER-INDEX.json').write_text(json.dumps({HEAD: MASTER}))

    def tearDown(self):
        for k, v in self._prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_cutover_writes_job_then_ingest_then_credit_then_next(self):
        live_dir = Path(__file__).resolve().parents[1] / 'live'
        if str(live_dir) not in sys.path:
            sys.path.insert(0, str(live_dir))
        import pe_lifecycle as PL  # type: ignore

        calls: List[str] = []

        class Life:
            def _training(self, directory, opening, dispatch, current, now):
                # Base training (ff/v829 stand-in): writes JOB+DISPATCH+K1, returns
                # None while waiting for standing trainer.
                calls.append(Path(directory).name)
                d = Path(directory)
                if not (d / 'TRAINING-JOB.json').exists():
                    (d / 'TRAINING-JOB.json').write_text(json.dumps({
                        'payload': {
                            'manifest': {
                                'payload': {
                                    'checkpoint': {'id': HEAD, 'files': {}},
                                }
                            },
                            'training': {'optimizer_step': 10},
                        }
                    }))
                    (d / 'TRAIN-DISPATCH-FF.json').write_text(json.dumps({
                        'signature': 'sig', 'signer': 'ROOT',
                        'payload': {'selected': [{'batch_sha256': 'x'}]},
                    }))
                    (d / 'K1-TURN.json').write_text(json.dumps({
                        'status': 'turn',
                        'epoch_index': int(d.name),
                        'epoch': 'affine-mimo-e%06d' % int(d.name),
                        'uid': 3,
                        'hotkey': '5MinerCreditKey',
                        'miner': '5MinerCreditKey',
                        'tasks': 1,
                        'rollouts': 1,
                        'queue_position': 0,
                        'head': HEAD,
                        'at': 1.0,
                    }))
                    return None  # standing-trainer wait shape
                return dict(
                    status='updated',
                    checkpoint=dict(id=CAND),
                    optimizer_step=current['optimizer_step'] + 1,
                    checkpoint_authorization=dict(input=HEAD, id=CAND),
                )

        # Match live install_pe expectation: v829i-style wrapper exposes
        # closure var ``prior`` (see live/test_pe_lifecycle.py).
        def v829i(Lifecycle):
            prior = Lifecycle._training

            def _training(self, *a):
                r = prior(self, *a)
                return None if (r and r.get('status') == 'updated') else r

            Lifecycle._training = _training

        v829i(Life)
        PL.install_pe(Life)
        L = Life()
        cur = json.loads((self.life / 'CURRENT.json').read_text())

        # --- Epoch 200: cutover must write artifacts + PIT-DISPATCHED + no-update ---
        ep200 = self.life / 'epochs' / '200'
        ep200.mkdir(parents=True)
        r = L._training(ep200, None, None, cur, 0)
        self.assertEqual(r['status'], 'no-update')
        self.assertTrue(str(r.get('reason', '')).startswith('pit-decoupled-e200'))
        self.assertTrue((ep200 / 'TRAINING-JOB.json').is_file())
        self.assertTrue((ep200 / 'TRAIN-DISPATCH-FF.json').is_file())
        self.assertTrue((ep200 / 'K1-TURN.json').is_file())
        self.assertTrue((ep200 / 'PIT-DISPATCHED.json').is_file())
        # Standing trainer result must NOT have been applied as HEAD.
        self.assertNotEqual(r.get('checkpoint', {}).get('id'), CAND)

        # Lifecycle finalizes (serial path writes FINALIZED).
        (ep200 / 'FINALIZED.json').write_text(json.dumps({
            'payload': {'input_checkpoint': HEAD, 'result': r},
        }))

        # SAMPLE for round_id (real live source).
        (self.gate_ep / '200').mkdir(parents=True)
        (self.gate_ep / '200' / 'SAMPLE.private.json').write_text(json.dumps({
            'round_id': 'r-e200', 'indices': [10, 11, 12],
        }))

        # --- queue_ingest picks FINALIZED+PIT-DISPATCHED ---
        import pe_round_seed as seed  # type: ignore

        ctl = Controller(
            config=PitConfig(
                enabled=True, mode='live', cutover=True, max_boxes=1,
                root_mode='delegated_host',
            ),
            rent=RentService(max_boxes=1),
            lane=LaneService(),
            box_io_factory=lambda: ScriptedBoxIO(
                on_train=lambda **k: {
                    'candidate_id': CAND,
                    'candidate_path': 'box:/c',
                    'candidate_master_path': 'box:/m',
                    'used_fp32_bootstrap_override': False,
                    'report': str(self.td / 'report.json'),
                    'defer_candidate_upload': True,
                    'candidate_on_box': True,
                    'artifact_transport': 'on-box-until-green',
                    'fetch': 'r2-box-side',
                },
            ),
            round_id_fn=seed.round_id_for_epoch,
            plan_seed_fn=lambda e: 'p' * 64,
            score_fn=lambda **k: {
                'decision': 'green',
                'z': 2.0,
                'evaluated': {'z': 2.0, 'sigma_run_floor': 0.0},
                'heldout': {
                    'round_id': 'r-e200',
                    'set_size': 3,
                    'plan_seed': 'p' * 64,
                    'sigma_run_floor': 0.0,
                },
            },
        )
        # Write a minimal report so ScriptedBoxIO path can proceed if needed.
        (self.td / 'report.json').write_text(json.dumps({
            'status': 'ok',
            'optimizer': {'sgd': {'fp32_master': {
                'source': 'disk',
                'input_master_digest': MASTER,
                'output_master_digest': '5' * 64,
            }}},
        }))
        store = DurableStore(root=self.pe / 'pit')
        out = sync(ctl, store=store)
        self.assertIn(200, out['enqueued'], msg='FINALIZED+PIT-DISPATCHED must enqueue: %s' % out)
        self.assertTrue(out['head_set'])
        self.assertEqual(ctl.items[0].epoch, 200)
        self.assertEqual(ctl.items[0].meta.get('epoch_dir'), str(ep200))

        # Bind uses SAMPLE round_id (not fixture).
        tick = ctl.tick()
        self.assertTrue(any(a.get('op') == 'bound' for a in tick.get('actions') or []))
        self.assertEqual(ctl.items[0].bound_round_id, 'r-e200')

        # --- Fake decision/credit: write DECISION + PIT-RESULT (miner credit path) ---
        write_pit_result(
            ep200,
            decision='green_fresh',
            item_id=ctl.items[0].item_id,
            extra={'paid': True, 'uid': 3, 'hotkey': '5MinerCreditKey'},
        )
        (self.gate_ep / '200' / 'DECISION.json').write_text(json.dumps({
            'decision': 'green',
            'epoch': 200,
            'uid': 3,
            'hotkey': '5MinerCreditKey',
            'paid': True,
            'z': 2.0,
        }))
        # Re-ingest must skip (PIT-RESULT present).
        out2 = sync(ctl, store=store)
        self.assertNotIn(200, out2['enqueued'])

        # --- Next epoch proceeds under cutover ---
        ep201 = self.life / 'epochs' / '201'
        ep201.mkdir(parents=True)
        r2 = L._training(ep201, None, None, cur, 0)
        self.assertEqual(r2['status'], 'no-update')
        self.assertTrue((ep201 / 'PIT-DISPATCHED.json').is_file())
        self.assertTrue((ep201 / 'TRAINING-JOB.json').is_file())
        self.assertTrue((ep201 / 'K1-TURN.json').is_file())
        (ep201 / 'FINALIZED.json').write_text(json.dumps({
            'payload': {'input_checkpoint': HEAD, 'result': r2},
        }))
        (self.gate_ep / '201').mkdir(parents=True)
        (self.gate_ep / '201' / 'SAMPLE.private.json').write_text(json.dumps({
            'round_id': 'r-e201', 'indices': [1, 2],
        }))
        out3 = sync(ctl, store=store)
        self.assertIn(201, out3['enqueued'])
        # Inner was invoked for both epochs (job write), never waited on trainer.
        self.assertEqual(calls, ['200', '201'])


class TestSyncExpectedHashInjectable(unittest.TestCase):
    """pe_apply hash-sync hook must never hardcode a host script path."""

    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.pe = self.td / 'pe'
        self.pe.mkdir()
        self.prev = {
            k: os.environ.get(k)
            for k in (
                'PE_STATE',
                'PE_SYNC_EXPECTED_HASH',
                'PE_SYNC_EXPECTED_HASH_PYTHON',
                'PE_SYNC_EXPECTED_HASH_LOG',
                'PE_SYNC_EXPECTED_HASH_ALERT',
            )
        }
        os.environ['PE_STATE'] = str(self.pe)
        # Point at a hermetic fake — never the live host script.
        self.fake = self.td / 'fake_sync_expected_hash.py'
        self.log = self.td / 'hook-sync.log'
        self.fake.write_text(
            'import sys\n'
            'open(%r, "a").write(" ".join(sys.argv) + "\\n")\n'
            % str(self.log)
        )
        os.environ['PE_SYNC_EXPECTED_HASH'] = str(self.fake)
        os.environ['PE_SYNC_EXPECTED_HASH_PYTHON'] = sys.executable
        os.environ['PE_SYNC_EXPECTED_HASH_LOG'] = str(self.td / 'sync.log')
        live_dir = Path(__file__).resolve().parents[1] / 'live'
        if str(live_dir) not in sys.path:
            sys.path.insert(0, str(live_dir))

    def tearDown(self):
        for k, v in self.prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_record_applied_invokes_env_script_not_host_default(self):
        import pe_apply as PA  # type: ignore

        src = Path(PA.__file__).read_text()
        self.assertNotIn('/home/', src)
        self.assertIn('PE_SYNC_EXPECTED_HASH', src)
        cfg = PA._sync_expected_hash_config()
        self.assertEqual(cfg['sync'], str(self.fake))

        PA.set_genesis(HEAD)
        PA.add_green(1, CAND, HEAD, green_at=1.0)
        it = PA.next_green()
        PA.record_applied(it, 'fresh', 0, CAND, HEAD)
        self.assertTrue(self.log.is_file(), 'fake sync script must have been executed')
        self.assertIn('--head=%s' % CAND, self.log.read_text())

    def test_unset_sync_env_is_noop(self):
        import pe_apply as PA  # type: ignore

        os.environ.pop('PE_SYNC_EXPECTED_HASH', None)
        os.environ.pop('PIT_SYNC_EXPECTED_HASH', None)
        self.assertIsNone(PA._sync_expected_hash_config())
        self.assertIsNone(PA._sync_expected_hash_after_head_apply(CAND))


if __name__ == '__main__':
    unittest.main()
