#!/usr/bin/env python3
"""PR #96: PIT cutover lifecycle must not stall the serial gate."""
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
os.environ['PIT_DECISION_TIMEOUT_S'] = '2'
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


class Life:
    def _training(self, directory, opening, dispatch, current, now):
        return None


def _v829i(Lifecycle):
    prior = Lifecycle._training
    def _training(self, *a, **k):
        return prior(self, *a, **k)
    Lifecycle._training = _training


class TestPitLifecycleInflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _flag()
        _v829i(Life)
        PL.install_pe(Life)

    def setUp(self):
        os.environ['PE_LIFE_STATE'] = str(_LIFE)
        os.environ['PE_GATE_EP'] = str(_EP)
        os.environ['PE_FLAG'] = str(_FLAG)
        os.environ['PIT_DECISION_TIMEOUT_S'] = '2'
        _flag()
        for p in _LIFE.glob('**/*'):
            if p.is_file():
                p.unlink()
        for p in _EP.glob('**/*'):
            if p.is_file():
                p.unlink()
        _EP.mkdir(parents=True, exist_ok=True)

    def _cur(self, head='H' * 64):
        return dict(checkpoint=dict(id=head), optimizer_step=10, checkpoint_authorization=None)

    def _ep(self, e):
        d = _EP / str(e)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_claim_returns_no_update_not_none(self):
        r = Life()._training(self._ep(1), None, None, self._cur(), time.time())
        self.assertIsNotNone(r)
        self.assertEqual(r['status'], 'no-update')
        self.assertTrue(str(r['reason']).startswith('pit-claimed-e1'))
        self.assertTrue((_LIFE / 'PIT-CLAIM-e1.json').is_file())

    def test_second_epoch_proceeds_after_claim(self):
        Life()._training(self._ep(1), None, None, self._cur(), time.time())
        r2 = Life()._training(self._ep(2), None, None, self._cur(), time.time())
        self.assertIsNotNone(r2)
        self.assertEqual(r2['status'], 'no-update')
        self.assertTrue(str(r2['reason']).startswith('pit-claimed-e2'))

    def test_merge_blocked_no_update(self):
        e = 3
        d = self._ep(e)
        (_LIFE / ('PIT-CLAIM-e%d.json' % e)).write_text(json.dumps(dict(epoch=e, at=time.time())))
        dec = dict(decision='green', merge_blocked=True, candidate='C' * 64)
        (d / 'DECISION.json').write_text(json.dumps(dec))
        r = Life()._training(d, None, None, self._cur(), time.time())
        self.assertEqual(r['status'], 'no-update')
        self.assertIn('pit-merge-blocked', r['reason'])
        self.assertTrue((d / 'PROMOTION-GATE.json').is_file())

    def test_timeout_no_decision_never_promote(self):
        e = 4
        d = self._ep(e)
        (_LIFE / ('PIT-CLAIM-e%d.json' % e)).write_text(
            json.dumps(dict(epoch=e, at=time.time() - 10))
        )
        r = Life()._training(d, None, None, self._cur(), time.time())
        self.assertEqual(r['status'], 'no-update')
        self.assertIn('no-decision', r['reason'])
        self.assertEqual(r['checkpoint']['id'], self._cur()['checkpoint']['id'])
        marker = json.loads((d / 'PROMOTION-GATE.json').read_text())
        self.assertTrue(marker.get('no_decision'))


if __name__ == '__main__':
    unittest.main()
