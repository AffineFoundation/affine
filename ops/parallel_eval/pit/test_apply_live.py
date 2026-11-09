#!/usr/bin/env python3
"""apply_live fail-closed (CUTOVER-PLAN blocker 5)."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .apply_live import ApplyLiveError, LiveApplicator, PromoteRefused
from .config import load_pit_config
from .signer import FixtureRootSigner
from .state import ItemRecord, ItemState


def _live_cfg():
    return load_pit_config({
        'pit': {'enabled': True, 'mode': 'live', 'cutover': True, 'max_boxes': 5},
    })


def _fresh_item(**kw):
    base = dict(
        item_id='i', registration_index=0, epoch=1, state=ItemState.DECIDED,
        bound_head='a' * 64, candidate_id='b' * 64, decision='green_fresh',
        fresh=True, merge_blocked=False, paid=True, shadow=False,
        meta={'candidate_sha256': 'c' * 64, 'candidate_master_digest': 'd' * 64},
    )
    base.update(kw)
    return ItemRecord(**base)


class TestApplyLive(unittest.TestCase):
    def test_missing_verify_uploads_fail_closed(self):
        app = LiveApplicator(config=_live_cfg(), verify_uploads=None)
        with self.assertRaises(ApplyLiveError) as cm:
            app.apply(_fresh_item(), scored={'decision': 'green', 'evaluated': {'z': 2}})
        self.assertIn('verify_uploads', str(cm.exception))

    def test_hash_fail_fail_closed(self):
        app = LiveApplicator(
            config=_live_cfg(),
            verify_uploads=lambda **k: {'candidate_ok': False, 'master_ok': True},
        )
        with self.assertRaises(ApplyLiveError):
            app.apply(_fresh_item(), scored={'decision': 'green', 'evaluated': {'z': 2}})

    def test_timeout_never_promotes(self):
        app = LiveApplicator(
            config=_live_cfg(),
            verify_uploads=lambda **k: {'candidate_ok': True, 'master_ok': True},
        )
        with self.assertRaises(PromoteRefused):
            app.apply(
                _fresh_item(),
                scored={'decision': 'green', 'timeout': True, 'evaluated': {'z': 2}},
            )

    def test_fresh_green_ok(self):
        accepted = []
        with tempfile.TemporaryDirectory() as td:
            life = Path(td) / 'life'
            (life / 'pending').mkdir(parents=True)
            with mock.patch.dict(os.environ, {'PE_LIFE_STATE': str(life)}):
                app = LiveApplicator(
                    config=_live_cfg(),
                    verify_uploads=lambda **k: {'candidate_ok': True, 'master_ok': True},
                    accept_pit_decision=lambda p: accepted.append(p) or True,
                    checkpoint_signer=FixtureRootSigner(),
                    upload_master=lambda **k: {'ok': True, 'published': True},
                )
                out = app.apply(_fresh_item(), scored={'decision': 'green', 'evaluated': {'z': 2}})
                self.assertTrue(out.applied)
                self.assertTrue(out.master_verified)
                self.assertEqual(accepted[0]['source'], 'pit')
                self.assertTrue(list((life / 'pending').glob('e*.json')))

    def test_fresh_green_refuses_missing_signer(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {'PE_LIFE_STATE': td}):
                app = LiveApplicator(
                    config=_live_cfg(),
                    verify_uploads=lambda **k: {'candidate_ok': True, 'master_ok': True},
                    upload_master=lambda **k: {'ok': True},
                )
                with self.assertRaises(ApplyLiveError) as cm:
                    app.apply(_fresh_item(), scored={'decision': 'green', 'evaluated': {'z': 2}})
                self.assertIn('checkpoint_signer', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
