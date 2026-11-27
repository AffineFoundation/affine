#!/usr/bin/env python3
"""apply_live fail-closed (CUTOVER-PLAN blocker 5)."""
import json
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
            idx = Path(td) / 'fp32-idx'
            idx.mkdir()
            cand = 'b' * 64
            (idx / ('%s.json' % cand)).write_text(json.dumps({
                'bf16_sha256': cand,
                'master_digest': 'd' * 64,
                'files': {
                    'model-00001-of-00002.safetensors': {'sha256': '1' * 64, 'bytes': 10},
                    'model-00002-of-00002.safetensors': {'sha256': '2' * 64, 'bytes': 10},
                    'meta.json': {'sha256': '3' * 64, 'bytes': 20},
                },
            }))
            env = {
                'PE_LIFE_STATE': str(life),
                'PE_STATE': str(Path(td) / 'pe'),
                'PIT_FP32_MASTER_INDEX': str(idx),
            }
            (Path(td) / 'pe').mkdir()
            with mock.patch.dict(os.environ, env):
                with mock.patch(
                    'ops.parallel_eval.pit.apply_live.drive_cutover_apply',
                    return_value={'ok': True},
                ):
                    with mock.patch(
                        'ops.parallel_eval.pit.apply_live.confirm_head_advanced',
                        return_value=False,
                    ):
                        with mock.patch(
                            'ops.parallel_eval.pit.apply_live.master_index_is_complete',
                            return_value=True,
                        ):
                            app = LiveApplicator(
                                config=_live_cfg(),
                                verify_uploads=lambda **k: {
                                    'candidate_ok': True, 'master_ok': True, 'master_complete': True,
                                },
                                accept_pit_decision=lambda p: accepted.append(p) or True,
                                checkpoint_signer=FixtureRootSigner(),
                                upload_master=lambda **k: {
                                    'ok': True, 'published': True, 'complete': True, 'shard_count': 2,
                                },
                            )
                            out = app.apply(
                                _fresh_item(), scored={'decision': 'green', 'evaluated': {'z': 2}},
                            )
                self.assertTrue(out.applied)
                self.assertTrue(out.master_verified)
                # HEAD advances only after HEAD-CHAIN append — not at queue time.
                self.assertFalse(out.head_advanced)
                self.assertEqual(accepted[0]['source'], 'pit')
                self.assertTrue(list((life / 'pending').glob('e*.json')))

    def test_fresh_green_refuses_missing_signer(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {'PE_LIFE_STATE': td}):
                with mock.patch(
                    'ops.parallel_eval.pit.apply_live.master_index_is_complete',
                    return_value=True,
                ):
                    app = LiveApplicator(
                        config=_live_cfg(),
                        verify_uploads=lambda **k: {
                            'candidate_ok': True, 'master_ok': True, 'master_complete': True,
                        },
                        upload_master=lambda **k: {
                            'ok': True, 'complete': True, 'shard_count': 2,
                        },
                    )
                    with self.assertRaises(ApplyLiveError) as cm:
                        app.apply(
                            _fresh_item(), scored={'decision': 'green', 'evaluated': {'z': 2}},
                        )
                self.assertIn('checkpoint_signer', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
