#!/usr/bin/env python3
"""PR #96: accept_pit_decision must carry pending JSON for apply_tick."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_TD = Path(tempfile.mkdtemp())
_LIFE = _TD / 'life'
_PENDING = _LIFE / 'pending'
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
_PE_LIVE = Path(__file__).resolve().parents[1] / 'live'
sys.path.insert(0, str(_PE_LIVE))
import pe_apply as PA  # noqa: E402
from ops.parallel_eval.pit.checkpoint_auth import (  # noqa: E402
    build_checkpoint_auth_payload,
    sign_checkpoint_authorization,
    verify_checkpoint_authorization,
)
from ops.parallel_eval.pit.signer import FixtureRootSigner  # noqa: E402


class TestApplyPending(unittest.TestCase):
    def setUp(self):
        _LIFE.mkdir(parents=True, exist_ok=True)
        _PENDING.mkdir(parents=True, exist_ok=True)
        os.environ['PE_LIFE_STATE'] = str(_LIFE)
        os.environ['PE_STATE'] = str(_TD / 'pe' / str(id(self)))
        Path(os.environ['PE_STATE']).mkdir(parents=True, exist_ok=True)
        greens = _TD / 'pe' / 'GREENS.json'
        if greens.exists():
            greens.unlink()

    def test_accept_pit_decision_row_has_pending(self):
        H, C = 'a' * 64, 'b' * 64
        auth = sign_checkpoint_authorization(
            build_checkpoint_auth_payload(checkpoint_id=C, input_head=H, optimizer_step=3),
            signer=FixtureRootSigner(),
        )
        pend_path = _PENDING / 'e7-test.json'
        pend_doc = dict(
            epoch=7,
            candidate=C,
            training_head=H,
            checkpoint=dict(id=C, files=dict(x='sha')),
            optimizer_step=3,
            checkpoint_authorization=auth,
            pe_keep=True,
            source='pit',
        )
        pend_path.write_text(json.dumps(pend_doc))
        item = dict(
            source='pit',
            epoch=7,
            candidate=C,
            training_head=H,
            green_at=1.0,
            decision='green_fresh',
            merge_blocked=False,
            pending=str(pend_path),
        )
        self.assertTrue(PA.accept_pit_decision(item))
        row = PA.greens()[0]
        self.assertEqual(row.get('pending'), str(pend_path))
        loaded = json.loads(Path(row['pending']).read_text())
        verify_checkpoint_authorization(loaded['checkpoint_authorization'], expected_input_head=H, expected_checkpoint_id=C)
        # apply_tick-style access
        pend = PA.rj(row['pending'])
        self.assertIn('checkpoint_authorization', pend)
        self.assertTrue(pend.get('pe_keep'))


if __name__ == '__main__':
    unittest.main()
