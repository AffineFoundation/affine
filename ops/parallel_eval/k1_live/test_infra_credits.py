#!/usr/bin/env python3
"""K1 infra credits — match-all identity + z>1 → credited_infra_loss (pay final)."""
from __future__ import annotations

import unittest
from pathlib import Path

from ops.parallel_eval.k1_live.infra_credits import (
    apply_credits_to_verdicts,
    apply_infra_credit,
    example_doc,
    load_infra_credits,
    match_credit,
)


HK = '5TestHotkeyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
C = 'a' * 64
E = 'b' * 64


class TestInfraCredits(unittest.TestCase):
    def test_example_format(self):
        doc = example_doc()
        self.assertTrue(doc['enabled'])
        self.assertEqual(doc['version'], 1)
        self.assertTrue(doc['credits'])

    def test_match_requires_all_fields_and_z(self):
        credit = {
            'epoch': 1062, 'uid': 847, 'hotkey': HK, 'reg_block': 100,
            'candidate': C, 'evidence_hash': E, 'z': 1.2,
        }
        ok, reason = match_credit(
            credit, uid=847, hotkey=HK, reg_block=100,
            candidate=C, evidence_hash=E, z=1.2,
        )
        self.assertTrue(ok, reason)
        ok, reason = match_credit(
            credit, uid=848, hotkey=HK, reg_block=100,
            candidate=C, evidence_hash=E, z=1.2,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, 'uid-mismatch')
        ok, reason = match_credit(
            credit, uid=847, hotkey=HK, reg_block=100,
            candidate=C, evidence_hash=E, z=0.9,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, 'z-not-gt-1')

    def test_apply_forces_green_and_is_final(self):
        credit = {
            'epoch': 1062, 'uid': 847, 'hotkey': HK, 'reg_block': 100,
            'candidate': C, 'evidence_hash': E, 'z': 1.5, 'note': 'e1062',
        }
        logs = []
        rec = apply_infra_credit(
            {'epoch_index': 1062, 'verdict': 'blocked', 'uid': 847, 'hotkey': HK},
            turn={'uid': 847, 'hotkey': HK, 'reg_block': 100, 'file_sha256': E},
            decision={'candidate': C, 'evaluated': {'z': 1.5}},
            credit=credit,
            log_fn=lambda **kw: logs.append(kw),
        )
        self.assertEqual(rec['verdict'], 'green')
        self.assertTrue(rec['credited_infra_loss'])
        self.assertTrue(rec['credit_final'])
        self.assertTrue(any(l.get('event') == 'infra_credit_accepted' for l in logs))

    def test_mismatch_logged_ignored(self):
        credit = {
            'epoch': 1062, 'uid': 847, 'hotkey': HK, 'reg_block': 100,
            'candidate': C, 'evidence_hash': E, 'z': 1.5,
        }
        logs = []
        rec = apply_infra_credit(
            {'epoch_index': 1062, 'verdict': 'red', 'uid': 847},
            turn={'uid': 847, 'hotkey': 'WRONG', 'reg_block': 100, 'file_sha256': E},
            decision={'candidate': C, 'evaluated': {'z': 1.5}},
            credit=credit,
            log_fn=lambda **kw: logs.append(kw),
        )
        self.assertEqual(rec['verdict'], 'red')
        self.assertFalse(rec.get('credited_infra_loss'))
        self.assertTrue(any(l.get('event') == 'infra_credit_mismatch' for l in logs))

    def test_credit_final_survives_rerun_verdict_list(self):
        docs = {
            'enabled': True,
            'credits': [{
                'epoch': 1062, 'uid': 847, 'hotkey': HK, 'reg_block': 100,
                'candidate': C, 'evidence_hash': E, 'z': 1.5,
            }],
        }
        recs = [{
            'epoch_index': 1062, 'verdict': 'green', 'uid': 847, 'hotkey': HK,
            'credited_infra_loss': True, 'credit_final': True, 'candidate': C, 'z': 1.5,
        }]
        out = apply_credits_to_verdicts(
            recs, credits_doc=docs,
            turns_by_epoch={}, decisions_by_epoch={},
        )
        self.assertTrue(out[0]['credit_final'])
        self.assertEqual(out[0]['verdict'], 'green')

    def test_load_missing_soft(self):
        doc = load_infra_credits(Path('/tmp/no-such-k1-infra-credits.json'))
        self.assertFalse(doc['enabled'])


if __name__ == '__main__':
    unittest.main()
