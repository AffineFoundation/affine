#!/usr/bin/env python3
"""score_epoch_shadow must record the box candidate, not the live epoch candidate."""
import unittest

from .pe_score import ScoreError, score, score_epoch_shadow


BOX_C = 'b' * 64
LIVE_C = 'l' * 64
HEAD = 'a' * 64


class TestPeScore(unittest.TestCase):
    def test_score_records_box_candidate(self):
        out = score(
            epoch=10,
            round_id='r1',
            result={'score': {'decision': 'green', 'z': 1.2}},
            candidate_id=BOX_C,
            head_id=HEAD,
        )
        self.assertEqual(out['candidate'], BOX_C)
        self.assertEqual(out['candidate_id'], BOX_C)

    def test_score_epoch_shadow_ignores_live_candidate(self):
        live = {
            'decision': 'green',
            'candidate': LIVE_C,
            'z': 1.1,
            'evaluated': {'z': 1.1},
        }
        out = score_epoch_shadow(
            epoch=10,
            round_id='r1',
            box_result={'score': {'decision': 'green', 'z': 1.3}},
            box_candidate_id=BOX_C,
            box_head_id=HEAD,
            live_decision=live,
        )
        self.assertEqual(out['candidate'], BOX_C)
        self.assertEqual(out['scored']['candidate'], BOX_C)
        self.assertNotEqual(out['candidate'], LIVE_C)
        self.assertTrue(out['comparison']['candidates_differ'])
        self.assertEqual(out['comparison']['box_candidate'], BOX_C)
        self.assertEqual(out['comparison']['live_candidate'], LIVE_C)

    def test_host_score_cannot_override_candidate(self):
        def host_score(**kwargs):
            # Malicious/confused host scorer returns the live candidate.
            return {'decision': 'green', 'z': 2.0, 'candidate': LIVE_C}

        out = score(
            epoch=1, round_id='r', result={},
            candidate_id=BOX_C, head_id=HEAD, host_score=host_score,
        )
        self.assertEqual(out['candidate'], BOX_C)

    def test_require_host_refuses_embedded_score(self):
        with self.assertRaises(ScoreError) as cm:
            score(
                epoch=1, round_id='r',
                result={'score': {'decision': 'green', 'z': 1.0}},
                candidate_id=BOX_C, head_id=HEAD,
                require_host=True,
            )
        self.assertIn('embedded', str(cm.exception).lower())

    def test_rejects_bad_candidate(self):
        with self.assertRaises(ScoreError):
            score(
                epoch=1, round_id='r',
                result={'score': {'decision': 'green', 'z': 1}},
                candidate_id='nope', head_id=HEAD,
            )


if __name__ == '__main__':
    unittest.main()
