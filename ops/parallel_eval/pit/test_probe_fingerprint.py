"""A candidate is scored only against a baseline of the same hardware class and probe layout.

The serve records are made from the real probe layouts (``probe_layout``), in the shape that ``probe_serve.py``
puts on a probe result. The scoring seam is the real ``pe_score.call_live_host_score``. The scorer is a recorder:
the tests prove that it does not run on a mismatch.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from ops.parallel_eval.pit import probe_fingerprint as PF
from ops.parallel_eval.pit import probe_layout as PL
from ops.parallel_eval.pit.pe_score import call_live_host_score
from ops.parallel_eval.pit.pe_score import score as pit_score
from ops.parallel_eval.pit.round_fixture import EPOCH, RID, Round
from ops.parallel_eval.pit.round_fixture import probe_result as worker_result

SAMPLING = {'harness': 'live-math-v3', 'sampling_contract': {'temperature': 1.0, 'top_p': 1.0, 'max_tokens': 65536}}
GPU_NAME = {'RTX5090': 'NVIDIA GeForce RTX 5090', 'B300': 'NVIDIA B300', 'H100': 'NVIDIA H100 80GB HBM3'}


def serve_record(gpu: str, layout: PL.ProbeLayout, sampling=SAMPLING) -> dict:
    """The ``serve_layout`` record of an 8-GPU box of this class that serves this layout."""
    groups = layout.groups(8)
    return dict(
        layout=layout.as_dict(), launcher=layout.launcher(), gpu_groups=[list(g) for g in groups], serve_s=100.0,
        replicas=[dict(replica=i, pid=100 + i, gpus=list(g), port=PL.BASE_PORT + i, vllm_version=layout.vllm_version)
                  for i, g in enumerate(groups)],
        concurrency_per_replica=layout.concurrency(), runtime=dict(vllm_version=layout.vllm_version),
        gpus=[GPU_NAME[gpu]] * 8, sampling=copy.deepcopy(sampling),
    )


def probe_result(gpu: str = 'RTX5090', layout: PL.ProbeLayout = None, **kw) -> dict:
    layout = layout or PL.shadow_plan(gpu).primary
    return {'envs': {'affine_gatepool': {'per_task': {'1': {'n': 8}}}}, 'serve_layout': serve_record(gpu, layout, **kw)}


def baseline_of_probe(result: dict) -> dict:
    """A round baseline that the baseline writer made from a HEAD probe on a box: rollouts and the stamp."""
    return PF.stamp({'per_task': {'1': {'n': 64}}, 'max_tokens': 65536}, result)


class Scorer:
    def __init__(self):
        self.calls = []

    def __call__(self, epoch, rid, result, flag=None):
        self.calls.append((epoch, rid))
        return {'decision': 'green', 'evaluated': {'z': 2.5}, 'round_id': rid}


class TestFingerprint(unittest.TestCase):
    def test_two_boxes_of_one_class_and_layout_have_one_fingerprint(self):
        a, b = PF.of(probe_result('RTX5090')), PF.of(probe_result('RTX5090'))
        self.assertEqual(a, b)
        self.assertEqual(a['version'], PF.VERSION)
        self.assertEqual((a['fields']['gpu_class'], a['fields']['gpu_count']), ('RTX5090', 8))
        self.assertEqual((a['fields']['tensor_parallel'], a['fields']['replicas']), (2, 4))
        self.assertEqual(a['fields']['vllm_version'], PL.shadow_plan('RTX5090').primary.vllm_version)
        self.assertEqual((a['fields']['dtype'], a['fields']['max_model_len']), ('bfloat16', 69632))
        self.assertEqual(a['fields']['sampling'], SAMPLING)
        self.assertIsNone(PF.refusal(baseline_of_probe(probe_result('RTX5090')), probe_result('RTX5090')))

    def test_each_fact_of_the_hardware_and_layout_changes_the_fingerprint(self):
        tp2 = PL.shadow_plan('RTX5090').primary
        one_gpu = PL.ProbeLayout.from_mapping(dict(tp2.as_dict(), gpus_per_replica=1, name='one-gpu'))
        cases = {
            'gpu_class': probe_result('H100', tp2),
            'tensor_parallel': probe_result('RTX5090', one_gpu),
            'vllm_version': probe_result('RTX5090', PL.ProbeLayout.from_mapping(dict(tp2.as_dict(), vllm_version='0.34.0'))),
            'max_num_seqs': probe_result('RTX5090', PL.ProbeLayout.from_mapping(dict(tp2.as_dict(), max_num_seqs=64))),
            'prefix_caching': probe_result('RTX5090', PL.ProbeLayout.from_mapping(dict(tp2.as_dict(), prefix_caching=True))),
            'sampling': probe_result('RTX5090', tp2, sampling=dict(SAMPLING, sampling_contract={'temperature': 0.7})),
        }
        baseline = baseline_of_probe(probe_result('RTX5090', tp2))
        for field, result in cases.items():
            refused = PF.refusal(baseline, result)
            self.assertIsNotNone(refused, field)
            self.assertEqual((refused['decision'], refused['guard']), ('no_decision', PF.GUARD), field)
            self.assertTrue(any(text.startswith(field + ':') for text in refused['probe_fingerprint']['differences']), field)
            self.assertEqual(refused['rebaseline'], dict(required=True, reason=PF.GUARD, fingerprint=PF.of(result)), field)

    def test_a_missing_or_not_valid_fingerprint_on_one_side_fails_closed(self):
        result = probe_result('RTX5090')
        ready_without = {'per_task': {'1': {'n': 64}}, 'fingerprint': 'f' * 64, 'settings_hash': 'e' * 64}
        self.assertIn('baseline: no hardware and layout fingerprint', PF.refusal(ready_without, result)['reasons'][0])
        self.assertIn('baseline: no hardware and layout fingerprint', PF.refusal(None, result)['reasons'][0])
        baseline = baseline_of_probe(result)
        self.assertIn('candidate: no hardware and layout fingerprint', PF.refusal(baseline, {'envs': {}})['reasons'][0])
        changed = copy.deepcopy(baseline)
        changed[PF.KEY]['fields']['gpu_class'] = 'B300'            # the stamp no longer agrees with its hash
        self.assertIn('does not agree with its hash', PF.refusal(changed, result)['reasons'][0])
        for broken in ({'gpus': []}, {'gpus': ['NVIDIA B300'] * 4 + ['NVIDIA H100'] * 4}, 'text'):
            bad = dict(result, serve_layout=broken if not isinstance(broken, dict) else dict(result['serve_layout'], **broken))
            self.assertEqual(PF.refusal(baseline, bad)['decision'], 'no_decision', broken)
        no_sampling = probe_result('RTX5090', sampling=None)      # the box bundle has no manifest
        self.assertIn("missing ['sampling']", PF.refusal(baseline, no_sampling)['reasons'][0])
        with self.assertRaises(PF.FingerprintError):
            PF.stamp({'per_task': {}}, {'envs': {}})

    def test_no_side_declares_its_hardware_and_layout_no_decision(self):
        """A probe with the live serve.sh against a baseline of the serial gate: nothing proves one hardware."""
        gate_baseline = {'per_task': {'1': {}}, 'fingerprint': 'f' * 64, 'settings_hash': 'e' * 64,
                         'settings': {'runtime': ['0123456789abcdef  serve.sh', '      8 NVIDIA B300']}}
        refused = PF.refusal(gate_baseline, {'envs': {}, 'eval_gpu': 'RTX5090'})
        self.assertEqual((refused['decision'], refused['guard']), ('no_decision', PF.GUARD))
        self.assertEqual(refused['reasons'], [
            PF.GUARD + ': baseline: no hardware and layout fingerprint',
            PF.GUARD + ': candidate: no hardware and layout fingerprint'])
        self.assertEqual(refused['rebaseline'], dict(required=True, reason=PF.GUARD, fingerprint=None))
        self.assertEqual(refused['probe_fingerprint']['hardware_note'], dict(baseline='8 NVIDIA B300', candidate='RTX5090'))
        for baseline, result in ((None, None), ({}, {}), (gate_baseline, {'envs': {}}), ('text', 7)):
            self.assertEqual(PF.refusal(baseline, result)['decision'], 'no_decision', (baseline, result))


class TestScoringSeam(unittest.TestCase):
    """The mismatch path of the real scoring seam: no decision, and the scorer does not run."""

    def score(self, baseline, result, scorer=None):
        scorer = scorer or Scorer()
        out = call_live_host_score(scorer, epoch=1094, round_id='r7', result=result,
                                   baseline_of=lambda epoch, rid: baseline)
        return out, scorer

    def test_a_5090_candidate_against_a_5090_baseline_of_the_same_layout_is_scored(self):
        out, scorer = self.score(baseline_of_probe(probe_result('RTX5090')), probe_result('RTX5090'))
        self.assertEqual(out['decision'], 'green')
        self.assertEqual(scorer.calls, [(1094, 'r7')])

    def test_a_5090_tp2_candidate_against_a_b300_live_layout_baseline_gets_no_decision(self):
        b300_live = probe_result('B300', PL.LIVE)
        out, scorer = self.score(baseline_of_probe(b300_live), probe_result('RTX5090'))
        self.assertEqual(scorer.calls, [])                          # never a comparison across hardware
        self.assertEqual((out['decision'], out['guard']), ('no_decision', PF.GUARD))
        named = {text.split(':')[0] for text in out['probe_fingerprint']['differences']}
        self.assertLessEqual({'gpu_class', 'tensor_parallel', 'replicas', 'vllm_version'}, named)
        self.assertTrue(out['rebaseline']['required'])
        self.assertEqual(out['rebaseline']['fingerprint']['fields']['gpu_class'], 'RTX5090')
        self.assertEqual(out['heldout']['round_id'], 'r7')
        self.assertIsNone(out.get('z'))

    def test_a_baseline_without_a_fingerprint_gets_no_decision(self):
        out, scorer = self.score({'per_task': {'1': {'n': 64}}, 'fingerprint': 'f' * 64}, probe_result('RTX5090'))
        self.assertEqual((out['decision'], scorer.calls), ('no_decision', []))

    def test_a_baseline_that_is_not_ready_stays_pending(self):
        class Pending(Exception):
            pass

        def scorer(epoch, rid, result, flag=None):
            raise Pending('baseline for round %s not ready' % rid)

        out, _ = self.score({}, probe_result('RTX5090'), scorer)
        self.assertEqual(out['decision'], 'pending')

    def test_no_side_declares_it_the_scorer_does_not_run(self):
        out, scorer = self.score({'per_task': {'1': {'n': 64}}, 'fingerprint': 'f' * 64}, {'envs': {}, 'eval_gpu': 'B300'})
        self.assertEqual((out['decision'], out['guard'], scorer.calls), ('no_decision', PF.GUARD, []))

    def test_a_score_against_a_baseline_that_the_check_did_not_see_is_not_used(self):
        """The check reads no ready baseline, but the scorer finds one and gives GREEN: no fingerprint was compared."""
        for seen in ({}, None, {'per_task': {}}):
            out, scorer = self.score(seen, probe_result('RTX5090'))
            self.assertEqual(scorer.calls, [(1094, 'r7')], seen)
            self.assertEqual((out['decision'], out['guard']), ('no_decision', PF.GUARD), seen)
            self.assertIn('did not see', out['reasons'][0])
            self.assertIsNone(out.get('z'))

        def undecided(epoch, rid, result, flag=None):
            return {'decision': None, 'guard': 'candidate-probe-errors'}

        out, _ = self.score({}, probe_result('RTX5090'), undecided)
        self.assertEqual((out['decision'], out['guard']), ('no_decision', 'candidate-probe-errors'))

    def test_the_seam_gives_the_epoch_and_the_round_to_the_baseline_source(self):
        asked = []

        def baseline_of(epoch, rid):
            asked.append((epoch, rid))
            return baseline_of_probe(probe_result('RTX5090'))

        out = call_live_host_score(Scorer(), epoch=1094, round_id='r7', result=probe_result('RTX5090'), baseline_of=baseline_of)
        self.assertEqual((out['decision'], asked), ('green', [(1094, 'r7')]))

    def test_the_pit_score_of_a_mismatch_is_no_decision_and_never_green(self):
        baseline, scorer = baseline_of_probe(probe_result('B300', PL.LIVE)), Scorer()

        def score_fn(**kw):      # the adapter of host_wiring: the live scorer and the baseline of the round
            return call_live_host_score(scorer, epoch=kw['epoch'], round_id=kw['round_id'], result=kw['result'],
                                        baseline_of=lambda epoch, rid: baseline)

        out = pit_score(epoch=1094, round_id='r7', result=probe_result('RTX5090'), candidate_id='c' * 64,
                        head_id='a' * 64, host_score=score_fn, require_host=True)
        self.assertEqual((out['decision'], out['guard'], scorer.calls), ('no_decision', PF.GUARD, []))
        self.assertEqual(out['candidate'], 'c' * 64)


class TestLiveScorerSeam(unittest.TestCase):
    """The real live scorer (``live/pe_score.py``) on real round files, behind the real seam.

    The candidate is far above the head: the live scorer gives GREEN for it. The seam must not use that GREEN
    unless the baseline that the scorer reads has the fingerprint of the candidate probe.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.round = Round(Path(tmp.name) / 'gate')
        self.live = self.round.scorer
        self.cells = self.round.candidate(0.25)

    def candidate(self, gpu=None, layout=None):
        result = worker_result(self.cells, planned_tasks=len(self.round.ids))
        if gpu:
            result['serve_layout'] = serve_record(gpu, layout or PL.shadow_plan(gpu).primary)
        return result

    def write_baseline(self, head_probe=None, name='BASELINE.private.json'):
        doc = dict(self.round.baseline)
        if head_probe is not None:
            PF.stamp(doc, head_probe)
        (self.round.dir / name).write_text(json.dumps(doc))

    def score(self, result):
        return call_live_host_score(self.live.score, epoch=EPOCH, round_id=RID, result=result,
                                    baseline_of=self.live.scoring_baseline)

    def test_the_live_scorer_gives_green_but_a_baseline_with_no_stamp_gives_no_decision(self):
        self.assertEqual(self.live.score(EPOCH, RID, self.candidate())['decision'], 'green')
        for result in (self.candidate(), self.candidate('RTX5090'), self.candidate('B300', PL.LIVE)):
            out = self.score(result)
            self.assertEqual((out['decision'], out['guard']), ('no_decision', PF.GUARD))
            self.assertIn('baseline: no hardware and layout fingerprint', out['reasons'][0])
            self.assertIsNone(out.get('z'))

    def test_a_stamped_baseline_scores_its_own_class_and_layout_only(self):
        self.write_baseline(self.candidate('RTX5090'))
        out = self.score(self.candidate('RTX5090'))
        self.assertEqual(out['decision'], 'green')
        self.assertEqual(out['evaluated']['z'], self.live.score(EPOCH, RID, self.candidate('RTX5090'))['evaluated']['z'])
        for result in (self.candidate('B300', PL.LIVE), self.candidate('H100', PL.shadow_plan('RTX5090').primary),
                       self.candidate()):
            out = self.score(result)
            self.assertEqual((out['decision'], out['guard']), ('no_decision', PF.GUARD))

    def test_the_baseline_of_the_turn_is_the_one_that_is_checked(self):
        """A round that scores each turn against its own head run: the scorer reads BASELINE-e<epoch>."""
        doc = json.loads((self.round.dir / 'ROUND.private.json').read_text())
        doc['cfg']['scoring_mode'] = 'same-run'
        (self.round.dir / 'ROUND.private.json').write_text(json.dumps(doc))
        turn = 'BASELINE-e%d.private.json' % EPOCH
        self.write_baseline(self.candidate('RTX5090'))                              # the round baseline: 5090
        self.write_baseline(self.candidate('B300', PL.LIVE), name=turn)             # the baseline of the turn: B300
        self.assertEqual(self.live.scoring_baseline(EPOCH, RID)[PF.KEY]['fields']['gpu_class'], 'B300')
        self.assertEqual(self.live.score(EPOCH, RID, self.candidate('RTX5090'))['decision'], 'green')
        out = self.score(self.candidate('RTX5090'))
        self.assertEqual((out['decision'], out['guard']), ('no_decision', PF.GUARD))
        self.write_baseline(self.candidate('RTX5090'), name=turn)
        self.assertEqual(self.score(self.candidate('RTX5090'))['decision'], 'green')

    def test_no_round_or_no_baseline_stays_pending(self):
        (self.round.dir / 'BASELINE.private.json').unlink()
        self.assertEqual(self.live.scoring_baseline(EPOCH, RID), {})
        self.assertEqual(self.score(self.candidate('RTX5090'))['decision'], 'pending')
        self.assertEqual(self.live.scoring_baseline(EPOCH, 'no-such-round'), {})


if __name__ == '__main__':
    unittest.main()
