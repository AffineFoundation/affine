#!/usr/bin/env python3
"""Jacob 2026-11-09 / Gate B: PIT cutover writes JOB for ingest but never waits on trainer."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

_TD = Path(tempfile.mkdtemp())
_LIFE = _TD / 'life'
_EP = _TD / 'epochs'
os.environ['PE_LIFE_STATE'] = str(_LIFE)
os.environ['PE_GATE_EP'] = str(_EP)
_FLAG = _TD / 'flag.json'
os.environ['PE_FLAG'] = str(_FLAG)
_LIFE.mkdir(parents=True)

_PE_LIVE = Path(__file__).resolve().parents[1] / 'live'
sys.path.insert(0, str(_PE_LIVE))
import pe_lifecycle as PL  # noqa: E402


def _flag(**pit_kw):
    doc = dict(
        parallel_eval=dict(
            enabled=True,
            pit=dict(enabled=True, mode='live', cutover=True, **pit_kw),
        ),
    )
    _FLAG.write_text(json.dumps(doc))


_TRAINER_CALLS = []


class Life:
    def _training(self, directory, opening, dispatch, current, now):
        # ff/v829 stand-in: write JOB+DISPATCH+K1 then return None (standing-trainer wait).
        # Cutover must call this for the write, then mark PIT-DISPATCHED and finalize
        # no-update — never wait on / apply a standing-trainer result.
        _TRAINER_CALLS.append(Path(directory).name)
        d = Path(directory)
        if not (d / 'TRAINING-JOB.json').exists():
            (d / 'TRAINING-JOB.json').write_text('{"payload":{"training":{}}}')
            (d / 'TRAIN-DISPATCH-FF.json').write_text('{"signature":"s","payload":{}}')
            (d / 'K1-TURN.json').write_text(json.dumps({
                'status': 'turn', 'epoch_index': int(d.name), 'uid': 1,
            }))
            return None
        # Would be trainer completion — cutover must not apply this as HEAD.
        return dict(
            status='updated',
            checkpoint=dict(id='T' * 64),
            optimizer_step=current['optimizer_step'] + 1,
            checkpoint_authorization=dict(input=current['checkpoint']['id'], id='T' * 64),
        )


def _v829i(Lifecycle):
    prior = Lifecycle._training

    def _training(self, *a, **k):
        return prior(self, *a, **k)

    Lifecycle._training = _training


class TestPitLifecycleDecoupled(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _flag()
        _v829i(Life)
        PL.install_pe(Life)

    def setUp(self):
        os.environ['PE_LIFE_STATE'] = str(_LIFE)
        os.environ['PE_GATE_EP'] = str(_EP)
        os.environ['PE_FLAG'] = str(_FLAG)
        _flag()
        _TRAINER_CALLS.clear()
        for p in _LIFE.glob('**/*'):
            if p.is_file():
                p.unlink()
        for p in _EP.glob('**/*'):
            if p.is_file():
                p.unlink()
        (_LIFE / 'apply').mkdir(parents=True, exist_ok=True)
        _EP.mkdir(parents=True, exist_ok=True)

    def _cur(self, head='H' * 64):
        return dict(checkpoint=dict(id=head), optimizer_step=10, checkpoint_authorization=None)

    def _ep(self, e):
        d = _EP / str(e)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_no_ready_green_returns_decoupled_no_update(self):
        ep = self._ep(1)
        r = Life()._training(ep, None, None, self._cur(), time.time())
        self.assertIsNotNone(r)
        self.assertEqual(r['status'], 'no-update')
        self.assertTrue(str(r['reason']).startswith('pit-decoupled-e1-kept-'))
        # ff path ran once to write JOB; standing-trainer result not applied.
        self.assertEqual(_TRAINER_CALLS, ['1'])
        self.assertTrue((ep / 'TRAINING-JOB.json').is_file())
        self.assertTrue((ep / 'TRAIN-DISPATCH-FF.json').is_file())
        self.assertTrue((ep / 'K1-TURN.json').is_file())
        self.assertTrue((ep / 'PIT-DISPATCHED.json').is_file())
        self.assertNotEqual(r.get('checkpoint', {}).get('id'), 'T' * 64)
        # Must never write lifecycle PIT-CLAIM wait tokens.
        self.assertFalse(any(_LIFE.glob('PIT-CLAIM-e*.json')))

    def test_successive_epochs_never_hold_none(self):
        for e in (1, 2, 3, 4, 5, 6):
            ep = self._ep(e)
            r = Life()._training(ep, None, None, self._cur(), time.time())
            self.assertIsNotNone(r, msg='epoch %s held' % e)
            self.assertEqual(r['status'], 'no-update')
            self.assertIn('pit-decoupled', r['reason'])
            self.assertTrue((ep / 'PIT-DISPATCHED.json').is_file())
        self.assertEqual(_TRAINER_CALLS, ['1', '2', '3', '4', '5', '6'])

    def test_ready_fresh_green_offers_apply(self):
        head = 'H' * 64
        cand = 'C' * 64
        auth = dict(input=head, id=cand)
        (_LIFE / 'apply' / 'C.json').write_text(json.dumps(dict(
            state='ready',
            base_head=head,
            green_at=1.0,
            checkpoint=dict(id=cand),
            optimizer_step=11,
            checkpoint_authorization=auth,
        )))
        r = Life()._training(self._ep(10), None, None, self._cur(head), time.time())
        self.assertEqual(r['status'], 'updated')
        self.assertEqual(r['checkpoint']['id'], cand)
        self.assertEqual(r['checkpoint_authorization'], auth)
        offered = json.loads((_LIFE / 'apply' / 'C.json').read_text())
        self.assertEqual(offered['state'], 'offered')
        self.assertEqual(offered['offered_epoch'], 10)
        # ready_apply short-circuits before ff/job write.
        self.assertEqual(_TRAINER_CALLS, [])

    def test_stranded_merge_blocked_row_ignored_by_ready_apply(self):
        """Stale/stranded GREENS land as merge_blocked and must never be offered."""
        head = 'H' * 64
        (_LIFE / 'apply' / 'stale.json').write_text(json.dumps(dict(
            state='merge_blocked',
            base_head=head,
            green_at=1.0,
            checkpoint=dict(id='S' * 64),
            optimizer_step=1,
            checkpoint_authorization={},
            merge_blocked=True,
            stranded=True,
        )))
        ep = self._ep(11)
        r = Life()._training(ep, None, None, self._cur(head), time.time())
        self.assertEqual(r['status'], 'no-update')
        self.assertIn('pit-decoupled', r['reason'])
        self.assertEqual(r['checkpoint']['id'], head)
        self.assertTrue((ep / 'PIT-DISPATCHED.json').is_file())


if __name__ == '__main__':
    unittest.main()
