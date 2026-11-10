"""Cutover HEAD apply + complete fp32 master (e1062 class).

Tests go through the REAL pe_live import path (bare ``import pe_apply`` bound
as ``pe_live.PA``), including the dual-name package load that previously made
monkeypatched ``next_green`` a no-op.
"""
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

from .apply_live import ApplyLiveError, LiveApplicator
from .config import load_pit_config
from .box import ScriptedBoxIO
from .controller import Controller
from .cutover_apply import (
    confirm_head_advanced,
    cutover_apply_alert,
    drive_cutover_apply,
    master_index_is_complete,
    normalize_pe_modules,
)
from .pe_lane import LaneService
from .pe_rent import RentService
from .signer import FixtureRootSigner
from .state import ItemRecord, ItemState


HEAD = 'a' * 64
CAND = 'b' * 64
MASTER = 'd' * 64
AUTH = {'input': HEAD, 'id': CAND, 'payload': {'checkpoint_id': CAND, 'input_head': HEAD}}

_PKG_APPLY = 'ops.parallel_eval.live.pe_apply'
_PKG_LIVE = 'ops.parallel_eval.live.pe_live'
_PKG_LIFE = 'ops.parallel_eval.live.pe_lifecycle'
_PKG_COMMON = 'ops.parallel_eval.live.pe_common'


def _live_cfg():
    return load_pit_config({
        'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 5},
    })


def _write_complete_master_index(idx_dir: Path, candidate_id: str, digest: str = MASTER) -> Path:
    idx_dir.mkdir(parents=True, exist_ok=True)
    path = idx_dir / ('%s.json' % candidate_id)
    path.write_text(json.dumps({
        'bf16_sha256': candidate_id,
        'master_digest': digest,
        'files': {
            'model-00001-of-00002.safetensors': {'sha256': '1' * 64, 'bytes': 100},
            'model-00002-of-00002.safetensors': {'sha256': '2' * 64, 'bytes': 100},
            'meta.json': {'sha256': '3' * 64, 'bytes': 40},
        },
    }) + '\n')
    return path


def _write_digest_only_index(idx_dir: Path, candidate_id: str, digest: str = MASTER) -> Path:
    idx_dir.mkdir(parents=True, exist_ok=True)
    path = idx_dir / ('%s.json' % candidate_id)
    path.write_text(json.dumps({
        'bf16_sha256': candidate_id,
        'master_digest': digest,
        'files': {'master_digest.txt': {'sha256': '9' * 64, 'bytes': 65}},
    }) + '\n')
    return path


class TestMasterIndexComplete(unittest.TestCase):
    def test_digest_only_is_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            idx = Path(td)
            _write_digest_only_index(idx, CAND)
            self.assertFalse(master_index_is_complete(CAND, index_dir=idx))

    def test_shards_are_complete(self):
        with tempfile.TemporaryDirectory() as td:
            idx = Path(td)
            _write_complete_master_index(idx, CAND)
            self.assertTrue(master_index_is_complete(CAND, index_dir=idx))


class TestIncompleteMasterBlocksApply(unittest.TestCase):
    def test_applicator_refuses_incomplete_master_keeps_signal(self):
        with tempfile.TemporaryDirectory() as td:
            life = Path(td) / 'life'
            (life / 'pending').mkdir(parents=True)
            idx = Path(td) / 'fp32'
            _write_digest_only_index(idx, CAND)
            pe = Path(td) / 'pe'
            pe.mkdir()
            env = {
                'PE_LIFE_STATE': str(life),
                'PE_STATE': str(pe),
                'PIT_FP32_MASTER_INDEX': str(idx),
            }
            with mock.patch.dict(os.environ, env):
                app = LiveApplicator(
                    config=_live_cfg(),
                    verify_uploads=lambda **k: {
                        'candidate_ok': True, 'master_ok': True, 'master_complete': False,
                    },
                    accept_pit_decision=lambda p: True,
                    checkpoint_signer=FixtureRootSigner(),
                    upload_master=lambda **k: {
                        'ok': True, 'published': True, 'complete': False, 'shard_count': 0,
                    },
                )
                item = ItemRecord(
                    item_id='i1', registration_index=1, epoch=1062,
                    state=ItemState.DECIDED, bound_head=HEAD, candidate_id=CAND,
                    decision='green_fresh', fresh=True, paid=True, shadow=False,
                    box_id='pod-1062',
                    meta={
                        'candidate_sha256': CAND,
                        'candidate_master_digest': MASTER,
                        'candidate_on_box': True,
                    },
                )
                with self.assertRaises(ApplyLiveError) as cm:
                    app.apply(item, scored={'decision': 'green', 'evaluated': {'z': 2.0}})
                self.assertIn('incomplete', str(cm.exception).lower())


class TestCutoverHeadApplyE2E(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.life = self.td / 'life'
        self.pe = self.td / 'pe'
        self.gate = self.td / 'gate' / 'epochs'
        self.idx = self.td / 'fp32-idx'
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
            'epoch_index': 1061,
            'checkpoint': {'id': HEAD},
            'optimizer_step': 1,
            'checkpoint_authorization': {'input': 'genesis', 'id': HEAD},
        }))
        self.sync_log = self.td / 'sync.log'
        self.sync_script = self.td / 'sync_expected_hash.py'
        self.sync_script.write_text(
            'import sys\nopen(%r,"a").write("synced:"+sys.argv[-1]+"\\n")\n' % str(self.sync_log)
        )
        self.env = {
            'PE_LIFE_STATE': str(self.life),
            'LIFE_ROOT': str(self.life),
            'PE_LIFE': str(self.life),
            'PE_STATE': str(self.pe),
            'PE_FLAG': str(self.flag),
            'PE_ST': str(self.pe),
            'PE_GATE_EP': str(self.gate),
            'HG_GATE_STATE': str(self.td / 'gate'),
            'PIT_FP32_MASTER_INDEX': str(self.idx),
            'PIT_STATE': str(self.pe / 'pit'),
            'PE_SYNC_EXPECTED_HASH': str(self.sync_script),
            'PE_SYNC_EXPECTED_HASH_PYTHON': sys.executable,
            'PE_SYNC_EXPECTED_HASH_LOG': str(self.td / 'sync-run.log'),
        }
        (self.pe / 'pit').mkdir(parents=True, exist_ok=True)
        self._prev = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        live = Path(__file__).resolve().parents[1] / 'live'
        if str(live) not in sys.path:
            sys.path.insert(0, str(live))
        self._live_mod_names = (
            'replace_pod', 'pe_common', 'pe_apply', 'pe_live', 'pe_lifecycle',
            _PKG_APPLY, _PKG_LIVE, _PKG_LIFE, _PKG_COMMON,
        )
        self._prev_mods = {n: sys.modules.get(n) for n in self._live_mod_names}

    def tearDown(self):
        for k, v in self._prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for name, prev in self._prev_mods.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev

    def _import_live_real_path(self, *, dual_package_first: bool = False):
        """Import via the same path pe_live uses in production.

        When dual_package_first=True, deliberately load the package name first
        (the historical dual-module trap), then import pe_live — normalize must
        still yield one shared pe_apply object.
        """
        rp = types.ModuleType('replace_pod')

        class _Pod:
            pass

        rp.Pod = _Pod
        with mock.patch.dict(sys.modules, {'replace_pod': rp}):
            for name in self._live_mod_names:
                if name != 'replace_pod':
                    sys.modules.pop(name, None)
            if dual_package_first:
                # Force the package module into sys.modules first.
                importlib = __import__('importlib')
                importlib.import_module(_PKG_APPLY)
            import pe_live as PL  # type: ignore  # noqa: E402
            import pe_lifecycle as Life  # type: ignore  # noqa: E402
            import pe_common as PC  # type: ignore  # noqa: E402
            PA = PL.PA
            # Identity must hold after normalize (and pe_live's own aliasing).
            PA2, PL2 = normalize_pe_modules(pe_live=PL)
            self.assertIs(PA2, PL.PA)
            self.assertIs(PA2, sys.modules.get('pe_apply'))
            self.assertIs(PA2, sys.modules.get(_PKG_APPLY))
            PL.EP = self.gate
            PC.LIFE = self.life
            PL.LIFE = self.life
            return PA, PL, Life, PC

    def _queue_pending_green(self, PA, *, epoch: int = 1062, pending_name: str = 'e1062.json'):
        pend = self.life / 'pending' / pending_name
        pend.write_text(json.dumps({
            'epoch': epoch,
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
            'epoch': epoch,
            'candidate': CAND,
            'training_head': HEAD,
            'green_at': time.time(),
            'decision': 'green_fresh',
            'pending': str(pend),
        }))
        (self.gate / str(epoch)).mkdir(parents=True)
        (self.gate / str(epoch) / 'DECISION.json').write_text(json.dumps({
            'decision': 'green',
            'candidate': CAND,
            'source': 'pit',
            'evaluated': {'z': 2.0, 'n': 100},
            'decided_at': time.time(),
        }))
        return pend

    def test_e2e_green_reaches_head_chain_hash_sync_and_bench(self):
        PA, PL, Life, PC = self._import_live_real_path(dual_package_first=True)
        _write_complete_master_index(self.idx, CAND)
        benches = []
        self._queue_pending_green(PA)

        # Drive via REAL pe_live path — do NOT pass a separate pe_apply object.
        out = drive_cutover_apply(pe_live=PL, alert_fn=cutover_apply_alert)
        self.assertEqual(out.get('pe_apply_id'), out.get('pe_live_pa_id'))
        ready = self.life / 'apply' / 'e1062.json'
        self.assertTrue(ready.is_file(), msg=out)
        self.assertEqual(json.loads(ready.read_text())['state'], 'ready')

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
        ep = self.life / 'epochs' / '1062'
        ep.mkdir(parents=True)
        cur = json.loads((self.life / 'CURRENT.json').read_text())
        r = LC()._training(ep, None, None, cur, 0)
        self.assertEqual(r['status'], 'updated')
        self.assertEqual(r['checkpoint']['id'], CAND)

        ckpt = {'id': CAND, 'files': {'x': '1' * 64}}
        (self.life / 'CURRENT.json').write_text(json.dumps({
            'epoch_index': 1063,
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
        offered = json.loads(ready.read_text())
        offered['checkpoint'] = ckpt
        offered['checkpoint_authorization'] = AUTH
        ready.write_text(json.dumps(offered))
        self.assertTrue(Life.confirm_applied(
            self.life, json.loads((self.life / 'CURRENT.json').read_text()),
        ))

        rec = drive_cutover_apply(pe_live=PL, alert_fn=cutover_apply_alert)
        self.assertIn(1062, rec.get('recorded_epochs') or [])
        self.assertTrue(confirm_head_advanced(CAND, pe_apply=PL.PA))
        chain = json.loads((self.pe / 'HEAD-CHAIN.json').read_text())
        self.assertTrue(any(a.get('new_head') == CAND for a in chain.get('applied') or []))
        self.assertTrue(self.sync_log.is_file() or (self.td / 'sync-run.log').is_file())

        ctl = Controller(
            config=_live_cfg(),
            rent=RentService(max_boxes=1),
            lane=LaneService(),
            box_io_factory=lambda: ScriptedBoxIO(),
            on_green_fresh=lambda item: benches.append(item.candidate_id),
        )
        item = ItemRecord(
            item_id='e1062', registration_index=1, epoch=1062,
            state=ItemState.DECIDED, bound_head=HEAD, candidate_id=CAND,
            decision='green_fresh', fresh=True, paid=True, shadow=False,
            box_id='pod-1062',
            meta={
                'green_apply_pending': True,
                'live_apply': {
                    'pe_apply_accepted': True,
                    'candidate_verified': True,
                    'master_verified': True,
                    'head_advanced': False,
                },
                'gate_emit_ready': True,
            },
        )
        ctl.items.append(item)
        updated, note = ctl._drive_pending_head(item)
        self.assertIn(note, ('released', 'green-fresh-applied', 'already-released'))
        self.assertEqual(benches, [CAND])
        self.assertTrue((updated.meta or {}).get('head_advanced'))
        self.assertEqual(updated.state, ItemState.RELEASED)

    def test_restart_with_pending_green_resumes(self):
        PA, PL, Life, PC = self._import_live_real_path()
        _write_complete_master_index(self.idx, CAND)
        self._queue_pending_green(PA, pending_name='e1062-restart.json')
        drive_cutover_apply(pe_live=PL, alert_fn=cutover_apply_alert)
        self.assertTrue((self.life / 'apply' / 'e1062.json').is_file())

    def test_incomplete_master_no_apply_file_single_alert_real_pe_live(self):
        """REAL pe_live path: incomplete master ⇒ no apply/, no HEAD, one alert."""
        PA, PL, Life, PC = self._import_live_real_path(dual_package_first=True)
        _write_digest_only_index(self.idx, CAND)
        self._queue_pending_green(PA, pending_name='e1062-bad.json')

        alerts = []

        def _alert(doc):
            alerts.append(dict(doc))
            return cutover_apply_alert(doc)

        out1 = drive_cutover_apply(pe_live=PL, alert_fn=_alert)
        self.assertIn(CAND, out1.get('refused_incomplete_master') or [])
        apply_path = self.life / 'apply' / 'e1062.json'
        self.assertFalse(apply_path.exists(), msg='must not write APPLY ready: %s' % out1)
        chain = json.loads((self.pe / 'HEAD-CHAIN.json').read_text())
        self.assertEqual(chain.get('applied') or [], [])
        self.assertEqual(len(alerts), 1)

        # Second tick: still no apply file, still no HEAD, no re-alert / no re-mark churn.
        greens_before = (self.pe / 'GREENS.json').read_text()
        out2 = drive_cutover_apply(pe_live=PL, alert_fn=_alert)
        self.assertEqual(out2.get('refused_incomplete_master') or [], [])
        self.assertFalse(apply_path.exists())
        self.assertEqual(len(alerts), 1)
        greens_after = (self.pe / 'GREENS.json').read_text()
        self.assertEqual(greens_before, greens_after)
        row = next(g for g in json.loads(greens_after) if g.get('candidate') == CAND)
        self.assertEqual(row.get('apply_blocked'), 'incomplete-fp32-master')
        self.assertTrue(row.get('apply_blocked_alerted'))
        self.assertIs(PL.PA.next_green(), None)

        # Controller keeps the box.
        ctl = Controller(
            config=_live_cfg(),
            rent=RentService(max_boxes=1),
            lane=LaneService(),
            box_io_factory=lambda: ScriptedBoxIO(),
        )
        item = ItemRecord(
            item_id='e1062', registration_index=1, epoch=1062,
            state=ItemState.DECIDED, bound_head=HEAD, candidate_id=CAND,
            decision='green_fresh', fresh=True, paid=True, shadow=False,
            box_id='pod-1062',
            meta={
                'green_apply_pending': True,
                'live_apply': {
                    'pe_apply_accepted': True,
                    'candidate_verified': True,
                    'master_verified': False,
                    'head_advanced': False,
                },
                'gate_emit_ready': True,
            },
        )
        ctl.items.append(item)
        updated, note = ctl._drive_pending_head(item)
        self.assertEqual(note, 'green-master-incomplete')
        self.assertEqual(updated.box_id, 'pod-1062')
        self.assertEqual(updated.state, ItemState.DECIDED)
        from .durable import pit_state_root
        self.assertTrue((pit_state_root() / 'GREEN-APPLY-ALERT.json').is_file())


if __name__ == '__main__':
    unittest.main()
