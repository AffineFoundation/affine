"""Head-vs-head parity harness: the statistics, the verdicts, and two real sides on a local box."""
from __future__ import annotations

import contextlib
import io
import json
import math
import random
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ops.parallel_eval.pit import probe_layout as PL
from ops.parallel_eval.pit import probe_parity as PP
from ops.parallel_eval.pit import probe_serve as PS
from ops.parallel_eval.pit.probe_fixture import ENV_ID, HARNESS, BoxCase
from ops.parallel_eval.pit.training_report import copy_training_report_to_host

CID = 'c' * 64


def sample(rng, probs, k=8, *, budget=0.0):
    """Cells of one side: task i passes each of its k rollouts with probability probs[i]."""
    cells = {}
    for index, p in enumerate(probs):
        c = sum(rng.random() < p for _ in range(k))
        b = sum(rng.random() < budget for _ in range(k))
        cells[str(index)] = dict(n=k, c=c, w=k - c, u=0, b=b, x=0, tok=100 * k, L=[100] * k)
    return cells


def result(cells, *, cid=CID, seed='s1', partial=False, wall=1000.0):
    samples = sum(cell['n'] for cell in cells.values())
    return dict(version='sn120-env-probe-worker-v1', checkpoint_id=cid, seed=seed, partial=partial, samples=samples,
                planned_samples=samples, wall_seconds=wall, rollouts_per_min=60.0 * samples / wall, replicas=4,
                envs={ENV_ID: dict(per_task=cells)})


def tasks(rng, count):
    return [rng.uniform(0.05, 0.95) for _ in range(count)]


class TestExactStatistics(unittest.TestCase):
    def test_fisher_p_of_known_tables(self):
        self.assertAlmostEqual(PP.fisher_p(8, 8, 0, 8), 2 / math.comb(16, 8), places=15)   # the strongest 8 v 8 result
        self.assertAlmostEqual(PP.fisher_p(3, 3, 0, 3), 0.1, places=12)
        self.assertEqual(PP.fisher_p(4, 8, 4, 8), 1.0)
        self.assertEqual(PP.fisher_p(0, 8, 0, 8), 1.0)
        self.assertEqual(PP.fisher_p(1, 2, 0, 2), 1.0)
        self.assertAlmostEqual(PP.fisher_p(7, 8, 2, 8), PP.fisher_p(2, 8, 7, 8), places=15)
        # 8 v 8 has no single-task result that survives a correction over 1000 tasks.
        self.assertGreater(PP.fisher_p(8, 8, 0, 8), 0.05 / 1000)

    def test_the_null_spread_of_one_task_is_the_hypergeometric_one(self):
        for n_a, n_b, m in ((8, 8, 8), (8, 8, 1), (8, 12, 5), (3, 7, 9), (8, 8, 0), (8, 8, 16)):
            pmf, var, var_sq = PP.null_task(n_a, n_b, m)
            total = n_a + n_b
            self.assertAlmostEqual(sum(pmf), 1.0, places=12)
            self.assertAlmostEqual(var, m * (total - m) / (n_a * n_b * (total - 1)), places=12)
            self.assertGreaterEqual(var_sq, 0.0)

    def test_the_chance_of_a_count_of_rare_tasks_is_exact(self):
        def binomial_tail(n, q, seen):
            return sum(math.comb(n, j) * q ** j * (1 - q) ** (n - j) for j in range(seen, n + 1))

        for n, q, seen in ((972, 0.0004, 3), (200, 0.01, 1), (50, 0.3, 20), (10, 0.5, 10)):
            self.assertAlmostEqual(PP._count_tail([q] * n, seen), binomial_tail(n, q, seen), places=12)
        self.assertEqual(PP._count_tail([0.2, 0.3], 0), 1.0)
        self.assertEqual(PP._count_tail([0.2, 0.3], 3), 0.0)
        self.assertAlmostEqual(PP._count_tail([0.2, 0.3, 0.0], 2), 0.06, places=12)
        self.assertAlmostEqual(PP._count_tail([0.2, 0.3], 1), 1 - 0.8 * 0.7, places=12)

    def test_the_spread_statistic_is_calibrated_in_its_right_tail(self):
        # A chi-square with 400 degrees of freedom is beyond 490.206 with chance 0.00135, the tail of z = 3.
        # The plain (stat - mean) / sd puts that point at z 3.19, so it calls too many equal pairs different.
        self.assertAlmostEqual(PP._sum_of_squares_z(490.206, 400, 800), 3.0, delta=0.005)
        self.assertAlmostEqual(PP._sum_of_squares_z(400.0, 400, 800), 0.0, delta=0.03)
        self.assertIsNone(PP._sum_of_squares_z(1.0, 0, 0))

    def test_equal_layouts_give_standard_normal_statistics(self):
        rng = random.Random(1)
        shift, spread, verdicts = [], [], []
        for _ in range(300):
            probs = tasks(rng, 200)
            report = PP.compare(result(sample(rng, probs)), result(sample(rng, probs)))
            shift.append(report['rates']['pass']['z'])
            spread.append(report['dispersion']['z'])
            verdicts.append(report['verdict'])
        self.assertLess(abs(statistics.mean(shift)), 0.2)
        self.assertTrue(0.85 < statistics.pstdev(shift) < 1.15, statistics.pstdev(shift))
        self.assertLess(abs(statistics.mean(spread)), 0.25)
        self.assertTrue(0.8 < statistics.pstdev(spread) < 1.2, statistics.pstdev(spread))
        self.assertLessEqual(verdicts.count('different'), 6)          # 0.7% of equal pairs: about 2 in 300

    def test_a_shift_of_all_tasks_is_found(self):
        rng = random.Random(2)
        probs = tasks(rng, 972)
        report = PP.compare(result(sample(rng, probs)), result(sample(rng, [p - 0.05 for p in probs])))
        self.assertEqual(report['verdict'], 'different')
        self.assertLess(report['rates']['pass']['z'], -3)
        self.assertLess(report['mean_delta'], -0.03)
        self.assertIn('pass rate', report['reasons'][0])
        self.assertLess(report['mean_delta_interval'][1], 0)

    def test_tasks_that_move_in_opposite_directions_are_found(self):
        rng = random.Random(3)
        report = PP.compare(result(sample(rng, [0.5] * 600)),
                            result(sample(rng, [0.85 if i % 2 else 0.15 for i in range(600)])))
        self.assertLess(abs(report['rates']['pass']['z']), 3)          # the mean does not move
        self.assertGreater(report['dispersion']['z'], 3)
        self.assertGreater(report['dispersion']['ratio'], 1.5)
        self.assertEqual(report['verdict'], 'different')
        self.assertTrue(any('spread' in reason for reason in report['reasons']))

    def test_more_output_cap_hits_are_found(self):
        rng = random.Random(4)
        probs = tasks(rng, 500)
        report = PP.compare(result(sample(rng, probs, budget=0.02)), result(sample(rng, probs, budget=0.08)))
        self.assertGreater(report['rates']['budget_hit']['z'], 3)
        self.assertEqual(report['verdict'], 'different')
        self.assertTrue(any('budget hit rate' in reason for reason in report['reasons']))


class TestVerdicts(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(7)
        self.probs = tasks(self.rng, 972)
        self.a = result(sample(self.rng, self.probs))
        self.b = result(sample(self.rng, self.probs), wall=400.0)

    def test_equal_layouts_at_gate_size(self):
        report = PP.compare(self.a, self.b)
        self.assertEqual(report['verdict'], 'no_difference_detected')
        self.assertEqual(report['reasons'], [])
        self.assertEqual((report['tasks_compared'], report['tasks_only_a'], report['tasks_only_b']), (972, 0, 0))
        self.assertEqual(len(report['tasks']), 972)
        # One run of 972 tasks x 8 rollouts resolves a shift of 2 to 3 points of pass rate.
        self.assertTrue(0.015 < report['detectable_shift'] < 0.035, report['detectable_shift'])
        self.assertAlmostEqual(report['speedup_b_over_a'], 2.5)
        row = report['tasks'][0]
        self.assertEqual(sorted(row), ['a', 'b', 'delta', 'env_id', 'index', 'p'])
        self.assertAlmostEqual(row['delta'], row['b'][0] / row['b'][1] - row['a'][0] / row['a'][1])
        self.assertEqual([r['p'] for r in report['tasks']], sorted(r['p'] for r in report['tasks']))
        json.dumps(report)

    def test_max_shift_turns_no_difference_into_parity_or_inconclusive(self):
        self.assertEqual(PP.compare(self.a, self.b, max_shift=0.05)['verdict'], 'parity')
        tight = PP.compare(self.a, self.b, max_shift=0.002)
        self.assertEqual(tight['verdict'], 'inconclusive')
        self.assertIn('wider than max_shift', tight['reasons'][0])

    def test_two_checkpoints_are_not_a_head_vs_head_run(self):
        with self.assertRaises(PP.ParityError):
            PP.compare(self.a, result(sample(self.rng, self.probs), cid='d' * 64))
        with self.assertRaises(PP.ParityError):
            PP.compare(self.a, {'no': 'result'})

    def test_sides_of_different_plans_are_not_comparable(self):
        other_seed = PP.compare(self.a, result(sample(self.rng, self.probs), seed='s2'))
        self.assertEqual(other_seed['verdict'], 'not_comparable')
        self.assertIn('the seeds differ', other_seed['reasons'])

        other_k = PP.compare(self.a, result(sample(self.rng, self.probs, k=12)))
        self.assertEqual(other_k['verdict'], 'not_comparable')
        self.assertIn('rollouts per task differ', other_k['reasons'])

        def record(indices, k=8):
            plan = PP.build_plan(CID, indices, 's1', harness=HARNESS, k=k)
            cells = sample(self.rng, [0.5] * len(indices), k=k)
            return dict(version=PS.SIDE_VERSION, checkpoint_id=CID, tag='x', status='ok', plan=PS.plan_identity(plan),
                        serve=dict(layout=PL.LIVE.as_dict(), score_changing=[]), result=result(cells))

        same = PP.compare(record(list(range(50))), record(list(range(50))))
        self.assertEqual(same['verdict'], 'no_difference_detected')
        moved = PP.compare(record(list(range(50))), record(list(range(1, 51))))
        self.assertEqual(moved['verdict'], 'not_comparable')
        self.assertIn('%s: the tasks differ' % 'affine_gatepool', moved['reasons'])

    def test_a_partial_side_is_inconclusive_over_the_shared_tasks(self):
        cells = sample(self.rng, self.probs)
        half = {index: cell for index, cell in cells.items() if int(index) < 400}
        report = PP.compare(self.a, result(half, partial=True))
        self.assertEqual(report['verdict'], 'inconclusive')
        self.assertEqual((report['tasks_compared'], report['tasks_only_a']), (400, 572))
        self.assertIsNone(report['speedup_b_over_a'])

    def test_the_text_report_names_the_verdict_the_rates_and_the_tasks(self):
        text = PP.render(PP.compare(self.a, self.b), top=5)
        self.assertIn('verdict: no_difference_detected', text)
        self.assertIn('tasks compared: 972', text)
        self.assertIn('pass ', text)
        self.assertIn('budget hit', text)
        self.assertIn('B is 2.50 times as fast as A', text)
        self.assertIn('largest differences first (5 of 972)', text)
        self.assertEqual(len(text.split('Fisher p\n')[1].splitlines()), 5)


def update(norm, *, cid=CID, data='d' * 64, step=41, engine='padded', world=None, limit=1.0, **more):
    """The trainer report of one update (report.private.json), with the keys that the trainer writes.

    The reports of two real updates are in test_host_v834f_sharded.TestCompareTrainReports.
    """
    rows = [dict(task_id='t%d' % i, nonce=i, tokens=100, loss_tokens=100, derail_index=None, support_outside_tokens=0,
                 advantage=0.5, stop='stop', old_logprob_mean=-1.0) for i in range(4)]
    report = dict(
        version='affine-mimo-grpo-update-v1', status='updated', input_checkpoint=cid, data_sha256=data,
        optimizer_step_before=step, optimizer_step=step + 1, group_count=1, rollout_count=4,
        input=dict(run='r', epoch=9, checkpoint=cid, contract_sha256='c' * 64, optimizer_step=step),
        config=dict(settings=dict(max_gradient_norm=limit)), preclip_gradient_norm=norm, objective_loss=-0.1,
        trained_rollouts=4, masked_rollouts=0, trained_tokens=400, loss_tokens=400, derailed_rollouts=0,
        trajectories=rows, microbatch=dict(engine=engine, count=1, **({'world': world} if world else {})),
        epoch_metrics=dict(preclip_gradient_norm=norm, max_gradient_norm=limit, gradient_clip_applied=norm > limit))
    report.update(more)
    return report


class TestPreclipIsTracked(unittest.TestCase):
    """The parity report has the gradient norm before clipping of the trainer update of each side."""

    def setUp(self):
        rng = random.Random(21)
        probs = tasks(rng, 300)
        self.a, self.b = result(sample(rng, probs)), result(sample(rng, probs))
        self.b300 = update(0.8)
        self.sharded = update(0.888, engine='packed-zero3-v835', world=8)          # 11% above, as the run of 2026-11-11

    def test_the_report_has_the_norm_of_each_side_and_the_ratio(self):
        report = PP.compare(self.a, self.b, train_a=self.b300, train_b=self.sharded)
        pre = report['preclip']
        self.assertEqual((pre['a']['preclip_gradient_norm'], pre['b']['preclip_gradient_norm']), (0.8, 0.888))
        self.assertAlmostEqual(pre['ratio_b_over_a'], 1.11, places=12)
        self.assertAlmostEqual(pre['relative_difference'], 0.088 / 0.888, places=12)
        # The shadow pass criterion permits 5%: this pair is beyond it, and the report says so.
        self.assertEqual((pre['shadow_tolerance'], pre['within_shadow_tolerance']), (0.05, False))
        self.assertEqual((pre['same_job'], pre['differences'], pre['in_verdict']), (True, [], False))
        self.assertEqual((pre['a']['engine'], pre['a']['ranks'], pre['b']['engine'], pre['b']['ranks']),
                         ('padded', None, 'packed-zero3-v835', 8))
        for side in (pre['a'], pre['b']):
            self.assertEqual((side['input_checkpoint'], side['data_sha256'], side['optimizer_step_before']), (CID, 'd' * 64, 41))
            self.assertEqual((side['from_this_checkpoint'], side['max_gradient_norm'], side['above_limit']), (True, 1.0, False))
            self.assertEqual((side['trained_rollouts'], side['trained_tokens'], side['loss_tokens']), (4, 400, 400))
            self.assertNotIn('trajectories', side)
        json.dumps(report)
        text = PP.render(report, top=3)
        self.assertIn('preclip gradient norm of the trainer update (tracked; not in the verdict): '
                      'A 0.8 [padded], B 0.888 [packed-zero3-v835, 8 ranks], B / A 1.1100\n', text)
        self.assertIn('  relative difference 9.91%, above the 5.00% of the shadow pass criterion\n', text)
        self.assertIn('  the trainer limits the norm to 1: A not above the limit, B not above the limit\n', text)

        close = PP.compare(self.a, self.b, train_a=self.b300, train_b=update(0.83, limit=0.5))['preclip']
        self.assertEqual((close['within_shadow_tolerance'], close['b']['above_limit']), (True, True))
        self.assertEqual(close['differences'], ['config.settings.max_gradient_norm'])     # a setting, not the job
        self.assertEqual(close['same_job'], True)

    def test_the_norm_changes_no_verdict_and_no_exit_code(self):
        plain = PP.compare(self.a, self.b)
        self.assertEqual(plain['verdict'], 'no_difference_detected')
        self.assertEqual(plain['preclip'], dict(
            in_verdict=False, a=None, b=None, same_job=None, differences=[], ratio_b_over_a=None,
            relative_difference=None, shadow_tolerance=0.05, within_shadow_tolerance=None))
        self.assertIn('not in the verdict): no trainer report (give --train-a and --train-b)\n', PP.render(plain))
        for norm in (0.8, 0.888, 80.0):
            tracked = PP.compare(self.a, self.b, train_a=self.b300, train_b=update(norm))
            self.assertEqual({k: v for k, v in tracked.items() if k != 'preclip'},
                             {k: v for k, v in plain.items() if k != 'preclip'})
        # Sides that differ stay different with equal norms.
        rng = random.Random(22)
        probs = tasks(rng, 300)
        worse = PP.compare(result(sample(rng, probs)), result(sample(rng, [p * 0.7 for p in probs])),
                           train_a=self.b300, train_b=self.b300)
        self.assertEqual((worse['verdict'], worse['preclip']['ratio_b_over_a']), ('different', 1.0))
        self.assertFalse(any('preclip' in reason or 'norm' in reason for reason in worse['reasons']))

    def test_updates_of_two_jobs_have_no_ratio(self):
        for other, differs, differences in (
                (update(0.888, data='e' * 64), ['data_sha256'], ['data_sha256']),
                (update(0.888, step=42), ['optimizer_step_before'], ['input.optimizer_step']),
                (update(0.888, cid='f' * 64), ['input_checkpoint'], ['input.checkpoint'])):
            pre = PP.compare(self.a, self.b, train_a=self.b300, train_b=other)['preclip']
            self.assertEqual((pre['same_job'], pre['ratio_b_over_a'], pre['relative_difference'],
                              pre['within_shadow_tolerance']), (False, None, None, None), differs)
            self.assertEqual(pre['differences'], differences)
            self.assertEqual((pre['a']['preclip_gradient_norm'], pre['b']['preclip_gradient_norm']), (0.8, 0.888))
            # Two summaries name the keys of the job that differ.
            cut = lambda report: {k: v for k, v in report.items() if k != 'trajectories'}
            self.assertEqual(PP.track_preclip(cut(self.b300), cut(other))['differences'], differs)
        self.assertEqual(pre['b']['from_this_checkpoint'], False)
        self.assertIn('  the update of B does not start from the checkpoint of the two sides\n', PP.render(
            PP.compare(self.a, self.b, train_a=self.b300, train_b=update(0.888, cid='f' * 64))))
        text = PP.render(PP.compare(self.a, self.b, train_a=self.b300, train_b=update(0.888, data='e' * 64)))
        self.assertIn('A 0.8 [padded], B 0.888 [padded]; no ratio: the two updates are not one job\n'
                      '  inputs of the two updates that differ: data_sha256\n', text)
        self.assertNotIn('does not start from', text)
        # One job that starts from another checkpoint than the two sides serve: a ratio, and the report says so.
        report = PP.compare(self.a, self.b, train_a=update(0.8, cid='f' * 64), train_b=update(0.888, cid='f' * 64))
        self.assertEqual((report['preclip']['same_job'], report['preclip']['a']['from_this_checkpoint']), (True, False))
        self.assertAlmostEqual(report['preclip']['ratio_b_over_a'], 1.11, places=12)
        self.assertIn('  the updates of A and B do not start from the checkpoint of the two sides\n', PP.render(report))

    def test_one_side_without_a_trainer_report_has_no_ratio(self):
        pre = PP.compare(self.a, self.b, train_b=self.sharded)['preclip']
        self.assertEqual((pre['a'], pre['b']['preclip_gradient_norm'], pre['ratio_b_over_a'], pre['same_job']),
                         (None, 0.888, None, None))
        self.assertIn('not in the verdict): A no trainer report, B 0.888 [packed-zero3-v835, 8 ranks]\n',
                      PP.render(PP.compare(self.a, self.b, train_b=self.sharded)))

    def test_the_documents_that_hold_a_trainer_report(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict('os.environ', {'PE_STATE': tmp}):
            stored = copy_training_report_to_host(report_path=None, item_id='i' * 32, epoch=9, candidate_id='e' * 64,
                                                  report_doc=self.sharded)
            on_host = json.loads(Path(stored['host_path']).read_text())
        holders = dict(
            live_training_result=dict(payload=dict(report=dict(training=self.sharded, checkpoint=CID))),
            train_v834e_result=dict(ok=True, report='/host/report.private.json', report_doc=self.sharded),
            host_training_report=on_host)
        for name, doc in holders.items():
            pre = PP.compare(self.a, self.b, train_a=self.b300, train_b=doc)['preclip']
            self.assertEqual(pre, PP.compare(self.a, self.b, train_a=self.b300, train_b=self.sharded)['preclip'], name)
        # markers/train.done and the summary of the host have the norm alone: a ratio, and no word on the job.
        marker = dict(ok=True, report='/workspace/pst/out/report.private.json',
                      training_report_summary=dict(gradient_norm=0.888, engine='packed', update_size=1.0))
        for doc in (marker, dict(summary=on_host['summary'])):
            pre = PP.compare(self.a, self.b, train_a=self.b300, train_b=doc)['preclip']
            self.assertAlmostEqual(pre['ratio_b_over_a'], 1.11, places=12)
            self.assertEqual((pre['same_job'], pre['differences'], pre['b']['input_checkpoint']), (None, [], None))
        self.assertIn('  the reports do not say if the two updates are one job (one checkpoint, one batch, one step)\n',
                      PP.render(PP.compare(self.a, self.b, train_a=self.b300, train_b=marker)))
        # A box summary has 0 where the trainer report had no norm; an update that did not run has none.
        for doc in (dict(training_report_summary=dict(gradient_norm=0.0)), dict(status='no-update'), {}, 'report'):
            with self.assertRaises(PP.ParityError) as caught:
                PP.compare(self.a, self.b, train_a=self.b300, train_b=doc)
            self.assertIn('no gradient norm of a trainer update', str(caught.exception))
        # A document that holds itself has an end.
        loop = dict(status='no-update')
        loop['report'] = loop
        with self.assertRaises(PP.ParityError):
            PP.find_update(loop)

    def test_the_command_line_reads_the_two_trainer_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name, doc in (('a', self.a), ('b', self.b), ('train-a', self.b300), ('train-b', self.sharded),
                              ('no-norm', dict(status='no-update'))):
                paths[name] = str(Path(tmp, name + '.json'))
                Path(paths[name]).write_text(json.dumps(doc))
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = PP.main(['compare', paths['a'], paths['b'], '--train-a', paths['train-a'], '--train-b',
                                paths['train-b'], '--out', str(Path(tmp, 'report.json'))])
                refused = PP.main(['compare', paths['a'], paths['b'], '--train-a', paths['train-a'], '--train-b',
                                   paths['no-norm'], '--out', str(Path(tmp, 'refused.json'))])
            written = json.loads(Path(tmp, 'report.json').read_text())
            self.assertFalse(Path(tmp, 'refused.json').exists())
        self.assertEqual((code, refused), (0, 2))                # 11% apart: the exit code is the one of the verdict
        self.assertIn('B / A 1.1100', out.getvalue())
        self.assertIn('probe parity: the document has no gradient norm of a trainer update', err.getvalue())
        self.assertEqual(written['preclip'], PP.compare(self.a, self.b, train_a=self.b300, train_b=self.sharded)['preclip'])
        self.assertEqual((written['verdict'], written['version']), ('no_difference_detected', PP.REPORT_VERSION))


class TestSidesOnTheBox(BoxCase):
    """Two real sides: the pinned worker probes one checkpoint under two layouts with one plan."""

    def setUp(self):
        super().setUp()
        self.plan = PP.build_plan(self.cid, list(range(24)), 'parity-seed', harness=HARNESS)
        self.plan['grade_workers'] = 2
        self.plan_path = self.tmp / 'plan.json'
        self.plan_path.write_text(json.dumps(self.plan))

    def side(self, layout, **kwargs):
        layout = PL.layout_from(layout)
        out = self.tmp / ('side-%s.json' % (kwargs.get('tag') or layout.name))
        PS.side(str(out), self.cid, str(self.plan_path), layout, self.box, **kwargs)
        return json.loads(out.read_text())

    def test_one_plan_under_two_layouts_gives_the_same_requests(self):
        a = self.side('rtx5090-1gpu')
        b = self.side('rtx5090-tp2')
        self.assertEqual(self.alive(), [])                              # each side stops its servers
        self.assertEqual(a['plan'], b['plan'])
        self.assertEqual((a['serve']['launcher'], b['serve']['launcher']), ('podctl', 'direct'))
        self.assertEqual((a['serve']['runtime']['vllm_version'], b['serve']['runtime']['vllm_version']), ('0.32.0', '0.33.0'))
        self.assertEqual((len(a['serve']['replicas']), len(b['serve']['replicas'])), (4, 2))

        report = PP.compare(a, b)
        self.assertEqual(report['verdict'], 'no_difference_detected')
        self.assertEqual(report['tasks_compared'], 24)
        # The stand-in server answers by (prompt, request seed) alone. Equal counts on every task show that both
        # layouts sent the same prompts with the same seeds.
        self.assertEqual({row['delta'] for row in report['tasks']}, {0.0})
        self.assertEqual(report['same_length_share'], 1.0)
        self.assertTrue(0.1 < report['rates']['pass']['a'] < 0.9)
        self.assertEqual((report['a']['tag'], report['b']['tag']), ('rtx5090-1gpu', 'rtx5090-tp2'))
        self.assertEqual(report['a']['layout']['gpus_per_replica'], 1)
        self.assertEqual(report['b']['layout']['gpus_per_replica'], 2)
        self.assertIn('rtx5090-tp2: 2 GPU per replica, vLLM 0.33.0, max-num-seqs 112', PP.render(report))

    def test_servers_that_give_other_scores_are_found(self):
        self.plan = PP.build_plan(self.cid, list(range(120)), 'parity-seed', harness=HARNESS)
        self.plan['grade_workers'] = 2
        self.plan_path.write_text(json.dumps(self.plan))
        a = self.side('rtx5090-tp2', tag='a')
        self.fb.control(bias=-0.3)                            # the servers of side B pass every task less often
        b = self.side('rtx5090-tp2', tag='b')
        report = PP.compare(a, b)
        self.assertEqual(report['verdict'], 'different')
        self.assertLess(report['rates']['pass']['z'], -3)

    def test_fp8_cannot_be_a_side(self):
        """No FP8 in the probe, and no switch to measure it: the layout is refused before anything reaches the box."""
        commands = []
        pod = self.pod()
        send = pod.run
        pod.run = lambda cmd, *args, **kwargs: commands.append(cmd) or send(cmd, *args, **kwargs)
        for setting in ({'kv_cache_dtype': 'fp8'}, {'quantization': 'fp8'}):
            fp8 = dict({'preset': 'rtx5090-tp2', 'name': 'tp2-fp8'}, **setting)
            # The host side of the harness.
            with self.assertRaises(PL.LayoutError) as caught:
                PP.run_side(pod, checkpoint=self.cid, plan=self.plan, layout=fp8, work_dir=str(self.tmp / 'parity'),
                            root=str(self.fb.root), box_args=self.box.args()[2:], poll_s=0.2)
            self.assertIn('the probe has no FP8', str(caught.exception))
            # The tool on the box, with a layout file that someone wrote by hand.
            layout_file = self.tmp / 'fp8-layout.json'
            layout_file.write_text(json.dumps(fp8))
            with self.assertRaises(PL.LayoutError):
                PS.main([*self.box.args(), 'side', '--checkpoint', self.cid, '--plan', str(self.plan_path),
                         '--layout', str(layout_file), '--out', str(self.tmp / 'side-fp8.json')])
        self.assertEqual((commands, self.fb.started()), ([], []))
        self.assertFalse((self.tmp / 'parity').exists())
        self.assertFalse((self.tmp / 'side-fp8.json').exists())

    def test_a_side_never_resumes_an_earlier_run(self):
        self.side('rtx5090-tp2', tag='t1')
        with self.assertRaises(PS.ServeError) as caught:
            self.side('rtx5090-tp2', tag='t1')
        self.assertIn('already exists', str(caught.exception))
        other = dict(self.plan, checkpoint_id='d' * 64)
        self.plan_path.write_text(json.dumps(other))
        with self.assertRaises(PS.ServeError):
            self.side('rtx5090-tp2', tag='t2')

    def test_the_host_runs_a_side_on_the_box_and_gets_the_record(self):
        record = PP.run_side(
            self.pod(), checkpoint=self.cid, plan=self.plan, layout='rtx5090-tp2', work_dir=str(self.tmp / 'parity'),
            root=str(self.fb.root), box_args=self.box.args()[2:], poll_s=0.2)
        self.assertEqual((record['version'], record['status'], record['tag']), (PS.SIDE_VERSION, 'ok', 'rtx5090-tp2'))
        self.assertEqual(record['plan'], PS.plan_identity(self.plan))
        self.assertEqual(record['result']['samples'], 24 * 8)
        self.assertEqual(PP.compare(record, self.side('rtx5090-1gpu'))['verdict'], 'no_difference_detected')

        self.fb.control(fail_if=['--max-num-seqs'])
        with self.assertRaises(PP.ParityError) as caught:
            PP.run_side(self.pod(), checkpoint=self.cid, plan=self.plan, layout='rtx5090-tp2', tag='again',
                        work_dir=str(self.tmp / 'parity'), root=str(self.fb.root), box_args=self.box.args()[2:], poll_s=0.2)
        self.assertIn('ended without a record', str(caught.exception))
        self.assertIn('exited while starting', str(caught.exception))


class TestCommandLine(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        (self.tmp / 'harness.json').write_text(json.dumps(HARNESS))

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = PP.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_plan_draws_the_same_tasks_for_the_same_seed(self):
        plans = []
        for name in ('p1.json', 'p2.json'):
            code, out, _ = self.run_cli('plan', '--checkpoint', CID, '--seed', 'par-1', '--tasks', '40', '--pool-size', '5000',
                                        '--harness', str(self.tmp / 'harness.json'), '--out', str(self.tmp / name))
            self.assertEqual(code, 0)
            plans.append(json.loads((self.tmp / name).read_text()))
        self.assertEqual(plans[0], plans[1])
        env = plans[0]['envs'][0]
        self.assertEqual((env['env_id'], env['k'], env['max_tokens'], len(set(env['indices']))), ('affine_gatepool', 8, 65536, 40))
        self.assertTrue(all(0 <= i < 5000 for i in env['indices']))
        self.assertEqual((plans[0]['checkpoint_id'], plans[0]['seed'], plans[0]['harness']), (CID, 'par-1', HARNESS['harness']))
        self.assertNotEqual(PP.draw_indices(5000, 40, 'par-2'), env['indices'])

        (self.tmp / 'indices.json').write_text(json.dumps([5, 9, 2]))
        code, _, _ = self.run_cli('plan', '--checkpoint', CID, '--seed', 'par-1', '--indices', str(self.tmp / 'indices.json'),
                                  '--harness', str(self.tmp / 'harness.json'), '--out', str(self.tmp / 'p3.json'))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads((self.tmp / 'p3.json').read_text())['envs'][0]['indices'], [5, 9, 2])

    def test_no_command_has_a_switch_for_a_layout_that_changes_scores(self):
        side = ['--checkpoint', CID, '--plan', 'p.json', '--layout', 'rtx5090-tp2', '--out', 'side.json']
        run_side = ['run-side', '--pod', 'pod.json', '--plan', 'p.json', '--layout', 'rtx5090-tp2', '--out', 'side.json']
        for main, argv in ((PS.main, ['side'] + side), (PP.main, run_side)):
            for switch in ('--allow-score-changing', '--kv-cache-dtype=fp8', '--quantization=fp8'):
                with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()) as err:
                    main(argv + [switch])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn('unrecognized arguments: %s' % switch, err.getvalue())

    def test_plan_needs_tasks_and_the_public_harness(self):
        base = ('plan', '--checkpoint', CID, '--seed', 's', '--out', str(self.tmp / 'p.json'))
        code, _, err = self.run_cli(*base, '--harness', str(self.tmp / 'harness.json'))
        self.assertEqual(code, 2)
        self.assertIn('--indices', err)
        with mock.patch.object(PP, '_load_public_harness', return_value={}):
            code, _, err = self.run_cli(*base, '--tasks', '5', '--pool-size', '10')
        self.assertEqual(code, 2)
        self.assertIn('public harness is required', err)
        self.assertFalse((self.tmp / 'p.json').exists())

    def test_compare_exit_code_follows_the_verdict(self):
        rng = random.Random(11)
        probs = tasks(rng, 400)
        files = {}
        for name, doc in (('a', result(sample(rng, probs))), ('same', result(sample(rng, probs))),
                          ('worse', result(sample(rng, [p * 0.8 for p in probs]))),
                          ('seed2', result(sample(rng, probs), seed='s2'))):
            files[name] = str(self.tmp / (name + '.json'))
            Path(files[name]).write_text(json.dumps(doc))

        code, out, _ = self.run_cli('compare', files['a'], files['same'], '--out', str(self.tmp / 'report.json'))
        self.assertEqual(code, 0)
        self.assertIn('verdict: no_difference_detected', out)
        self.assertEqual(json.loads((self.tmp / 'report.json').read_text())['version'], PP.REPORT_VERSION)
        self.assertEqual(self.run_cli('compare', files['a'], files['worse'])[0], 1)
        self.assertEqual(self.run_cli('compare', files['a'], files['seed2'])[0], 2)
        self.assertEqual(self.run_cli('compare', files['a'], files['same'], '--max-shift', '0.001')[0], 2)
        self.assertEqual(self.run_cli('compare', files['a'], files['same'], '--max-shift', '0.08')[0], 0)


if __name__ == '__main__':
    unittest.main()
