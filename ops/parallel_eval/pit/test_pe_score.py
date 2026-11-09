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
            'train_gpu': 'H200',
            'eval_gpu': 'H200',
            'training_report': {
                'source_file_hashes': {'batch': 'b' * 64},
                'batching': {'tasks': 125, 'rollouts': 1000},
                'tokens': {'loss_tokens': 10},
                'update_size': 1.0,
                'gradient_norm': 1.0,
            },
            'heldout': {'round_id': 'r1', 'plan_seed': 'ps'},
            'plan_seed': 'ps',
            'per_task': {
                't0': {'c': 1, 'n': 2},
                't1': {'c': 2, 'n': 2},
            },
        }
        box_result = {
            'score': {'decision': 'green', 'z': 1.3},
            'train_gpu': 'B300',
            'eval_gpu': 'B300',
            'envs': {'affine_gatepool': {'per_task': {
                't0': {'c': 2, 'n': 2},
                't1': {'c': 1, 'n': 2},
            }}},
        }
        out = score_epoch_shadow(
            epoch=10,
            round_id='r1',
            box_result=box_result,
            box_candidate_id=BOX_C,
            box_head_id=HEAD,
            live_decision=live,
            box_report=live['training_report'],
            live_report=live['training_report'],
            box_meta={'round_id': 'r1', 'plan_seed': 'ps'},
            live_meta={'round_id': 'r1', 'plan_seed': 'ps', 'train_gpu': 'H200'},
            train_gpu='B300',
            eval_gpu='B300',
        )
        self.assertEqual(out['candidate'], BOX_C)
        self.assertEqual(out['scored']['candidate'], BOX_C)
        self.assertNotEqual(out['candidate'], LIVE_C)
        self.assertTrue(out['comparison']['candidates_differ'])
        self.assertEqual(out['comparison']['box_candidate'], BOX_C)
        self.assertEqual(out['comparison']['live_candidate'], LIVE_C)
        self.assertEqual(out['comparison']['train_gpu'], 'B300')
        self.assertEqual(out['comparison']['live_train_gpu'], 'H200')
        self.assertIn('shadow_pass', out)
        self.assertEqual(out['shadow_pass']['pass_criterion'], 'e1041-noise-tolerant-v1')
        self.assertFalse(out['shadow_pass']['require_equal_candidate_hashes'])
        # agreement tracks noise-tolerant pass, not equal candidate hashes
        self.assertEqual(out['comparison']['agreement'], out['shadow_pass']['passed'])

    def test_host_score_cannot_override_candidate(self):
        def host_score(*, epoch, round_id, result):
            # Malicious/confused host scorer returns the live candidate.
            return {'decision': 'green', 'z': 2.0, 'candidate': LIVE_C}

        out = score(
            epoch=1, round_id='r', result={},
            candidate_id=BOX_C, head_id=HEAD, host_score=host_score,
        )
        self.assertEqual(out['candidate'], BOX_C)

    def test_live_positional_score_nests_heldout(self):
        """Live pe_score.score(epoch, rid, result) — positional, flat fields."""
        def live_score(epoch, rid, result, flag=None):
            self.assertEqual(epoch, 42)
            self.assertEqual(rid, 'round-xyz')
            return {
                'decision': 'green',
                'evaluated': {'z': 1.4, 'n': 100},
                'reasons': [],
                'set_size': 1000,
                'k_cand': 8,
                'head_rollouts': 12,
                'scoring': 'sn120-prior-baseline-v1',
                'settings_hash': 'f' * 64,
                'max_tokens': 65536,
                'tau_run': 0.0,
                'sigma_run_floor': 0.0,
                'baseline_done_at_cr': '2026-11-09 00:00:00 CR',
                'z_bar': {'value': 1.0},
                'round_id': rid,
            }

        out = score(
            epoch=42, round_id='round-xyz', result={'ok': True},
            candidate_id=BOX_C, head_id=HEAD, host_score=live_score,
        )
        self.assertEqual(out['decision'], 'green')
        self.assertEqual(out['heldout']['set_size'], 1000)
        self.assertEqual(out['heldout']['bar_z'], 1.0)
        self.assertEqual(out['heldout']['round_id'], 'round-xyz')
        self.assertEqual(out['heldout']['baseline'], 'prior-v1')
        self.assertEqual(out['z'], 1.4)

    def test_live_decision_none_maps_to_no_decision(self):
        def live_score(epoch, rid, result, flag=None):
            return {'decision': None, 'guard': 'candidate-probe-errors', 'set_size': 10}

        out = score(
            epoch=1, round_id='r', result={},
            candidate_id=BOX_C, head_id=HEAD, host_score=live_score,
        )
        self.assertEqual(out['decision'], 'no_decision')
        self.assertEqual(out['heldout']['set_size'], 10)

    def test_round_id_kwarg_never_passed_to_live(self):
        seen = {}

        def live_score(epoch, rid, result, flag=None):
            seen['ok'] = True
            return {'decision': 'red', 'evaluated': {'z': 0.1}, 'set_size': 5}

        # Would TypeError if adapter passed round_id=
        score(
            epoch=1, round_id='rid-1', result={},
            candidate_id=BOX_C, head_id=HEAD, host_score=live_score,
        )
        self.assertTrue(seen.get('ok'))

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
