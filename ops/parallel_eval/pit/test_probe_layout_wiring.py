"""Probe layouts in the PIT config and in the eval stage of a box."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from unittest import mock

from ops.parallel_eval.live.pe_box import detached_eval_command
from ops.parallel_eval.live.pe_probe_plan import build_probe_plan
from ops.parallel_eval.pit import DEFAULT_PIT, PitConfig, load_pit_config
from ops.parallel_eval.pit import probe_layout as PL
from ops.parallel_eval.pit import probe_serve as PS
from ops.parallel_eval.pit.box import BOX_ROOT, BoxRunner
from ops.parallel_eval.pit.box_job import BoxJobController, RemoteBoxJobStore
from ops.parallel_eval.pit.probe_fixture import ENV_ID, HARNESS, BoxCase
from ops.parallel_eval.pit.state import ItemState

CID = 'c' * 64
SEED = 'plan-seed-xyz'
INDICES = [3, 7, 11, 19]
JOB = '/workspace/pst/pit-jobs/x'


def shadow(**pit):
    return load_pit_config({'pit': dict({'enabled': True, 'mode': 'shadow'}, **pit)})


def live(**pit):
    return load_pit_config({'pit': dict({'enabled': True, 'mode': 'live', 'cutover': True}, **pit)})


class TestConfig(unittest.TestCase):
    def test_shadow_default_is_tp2_on_rtx5090_and_live_elsewhere(self):
        for cfg in (shadow(), DEFAULT_PIT, PitConfig()):
            self.assertIsNone(cfg.probe_layouts)
            self.assertEqual([l.name for l in cfg.probe_plan('RTX5090').layouts], ['rtx5090-tp2', 'rtx5090-1gpu'])
            for gpu in ('B300', 'B200', 'H200', 'H100', None):
                self.assertTrue(cfg.probe_plan(gpu).is_live(), gpu)

    def test_live_mode_serves_the_live_layout_whatever_the_gpu_or_the_settings(self):
        self.assertTrue(live().probe_plan('RTX5090').is_live())
        self.assertTrue(live(probe_layouts={'RTX5090': 'live'}).probe_plan('RTX5090').is_live())
        for spec in ('rtx5090-tp2', 'rtx5090-1gpu', {'preset': 'live', 'max_num_seqs': 256}, ['live', 'rtx5090-1gpu']):
            with self.assertRaises(ValueError, msg=str(spec)) as caught:
                live(probe_layouts={'RTX5090': spec})
            self.assertIn('only shadow mode', str(caught.exception))
        # A config object that skipped the loader still cannot leave the live layout in live mode.
        by_hand = PitConfig(enabled=True, mode='live', probe_layouts={'RTX5090': 'rtx5090-tp2', 'default': 'rtx5090-tp2'})
        self.assertTrue(by_hand.probe_plan('RTX5090').is_live())
        self.assertTrue(by_hand.probe_plan('B300').is_live())

    def test_every_layout_setting_is_set_from_the_config(self):
        cfg = shadow(probe_layouts={
            'RTX5090': {'preset': 'rtx5090-tp2', 'max_num_seqs': 128, 'gpu_memory_utilization': 0.92,
                        'vllm_version': '0.33.1', 'prefix_caching': True, 'fallback': None},
            'H100': 'rtx5090-1gpu',
            'default': {'preset': 'live', 'name': 'live-256', 'max_num_seqs': 256},
        })
        tp2 = cfg.probe_plan('RTX5090')
        self.assertEqual(len(tp2.layouts), 1)
        self.assertEqual((tp2.primary.max_num_seqs, tp2.primary.gpu_memory_utilization, tp2.primary.vllm_version,
                          tp2.primary.prefix_caching), (128, 0.92, '0.33.1', True))
        self.assertEqual(cfg.probe_plan('H100').primary.name, 'rtx5090-1gpu')
        self.assertEqual(cfg.probe_plan('B300').primary.max_num_seqs, 256)
        self.assertEqual(json.loads(json.dumps(cfg.as_dict()))['probe_layouts']['H100'], 'rtx5090-1gpu')

    def test_settings_that_must_not_serve_are_refused_at_load(self):
        for layouts in (
            {'RTX5090': {'preset': 'rtx5090-tp2', 'kv_cache_dtype': 'fp8'}},          # no FP8 KV cache
            {'RTX5090': ['rtx5090-tp2', {'preset': 'rtx5090-1gpu', 'kv_cache_dtype': 'fp8_e4m3'}]},
            {'RTX5090': {'preset': 'rtx5090-tp2', 'quantization': 'fp8'}},            # no FP8 weights
            {'default': {'preset': 'live', 'quantization': 'fp8'}},
            {'RTX5090': {'preset': 'rtx5090-tp2', 'task_queue': 'per_replica'}},
            {'RTX5090': {'preset': 'rtx5090-tp2', 'dtype': 'float16'}},               # a gate constant, not a setting
            {'RTX5090': {'preset': 'rtx5090-tp2', 'max_model_len': 32768}},
            {'RTX5090': 'no-such-layout'},
            {'A100': 'live'},
            ['rtx5090-tp2'],
        ):
            with self.assertRaises(ValueError, msg=str(layouts)):
                shadow(probe_layouts=layouts)
            with self.assertRaises(ValueError, msg=str(layouts)):
                live(probe_layouts=layouts)


class _Store:
    def __init__(self):
        self.files = {}

    def write_text(self, path, text):
        self.files[path] = text


class _Ctrl:
    job_dir = JOB

    def __init__(self):
        self.store = _Store()


class TestArmDetachedEval(unittest.TestCase):
    def arm(self, cfg, gpu):
        ctrl = _Ctrl()
        with mock.patch('ops.parallel_eval.pit.box._load_public_harness', return_value=HARNESS):
            command = BoxRunner(config=cfg, io=None, lane=None)._arm_detached_eval(
                ctrl, candidate_id=CID, plan_handle=SEED, indices=list(INDICES), gpu=gpu)
        return command, ctrl.store.files

    def live_plan(self):
        return build_probe_plan(CID, INDICES, SEED, k=8, max_tokens=65536, max_num_seqs=512, harness=HARNESS)

    def live_command(self):
        return detached_eval_command(job_dir=JOB, candidate_id=CID, plan_path=JOB + '/eval-plan.json',
                                     podd=BOX_ROOT, max_num_seqs=512)

    def test_the_live_layout_runs_the_live_command_and_the_live_plan_byte_for_byte(self):
        cases = [(live(), gpu) for gpu in ('RTX5090', 'B300', 'H100', None)]
        cases += [(shadow(), gpu) for gpu in ('B300', 'B200', 'H200', 'H100', None)]
        cases += [(shadow(probe_layouts={'RTX5090': 'live'}), 'RTX5090')]
        for cfg, gpu in cases:
            command, files = self.arm(cfg, gpu)
            self.assertEqual(command, self.live_command(), (cfg.mode, gpu))
            self.assertEqual(list(files), [JOB + '/eval-plan.json'])
            self.assertEqual(files[JOB + '/eval-plan.json'], json.dumps(self.live_plan(), sort_keys=True) + '\n')

    def test_a_shadow_rtx5090_box_gets_the_layouts_the_tool_and_the_live_clean_up(self):
        command, files = self.arm(shadow(), 'RTX5090')
        self.assertEqual(sorted(files), [JOB + '/' + name for name in (
            'eval-plan.json', 'probe-layouts.json', 'probe_layout.py', 'probe_serve.py')])
        here = Path(PS.__file__).parent
        for name in PS.TOOL_FILES:
            self.assertEqual(files[JOB + '/' + name], (here / name).read_text())
        self.assertEqual(PL.ProbePlan.from_doc(json.loads(files[JOB + '/probe-layouts.json'])), PL.shadow_plan('RTX5090'))

        # The plan is the live plan. Only the client load per replica follows the layout.
        plan, want = json.loads(files[JOB + '/eval-plan.json']), self.live_plan()
        self.assertEqual(plan.pop('concurrency_per_replica'), 112)
        want.pop('concurrency_per_replica')
        self.assertEqual(plan, want)

        # The clean-up is the live one; the live driver is replaced by one line that starts the tool.
        preamble = self.live_command().split('python3 - ')[0]
        self.assertTrue(command.startswith(preamble))
        tool_line = command[len(preamble):]
        self.assertEqual(tool_line, (
            'python3 {jd}/probe_serve.py --root {root} eval --job-dir {jd} --candidate {cid} '
            '--plan {jd}/eval-plan.json --layouts {jd}/probe-layouts.json\n').format(jd=JOB, root=BOX_ROOT, cid=CID))
        self.assertNotIn('podctl.py serve', command)
        self.assertNotIn("<<'PY'", command)
        # The clean-up ends processes by these patterns; the tool's own command line must match none of them.
        self.assertIsNone(re.search(r'train_pit\.py|v83_dp\.py|vllm serve|podctl\.py serve', tool_line))

    def test_configured_layouts_reach_the_box(self):
        cfg = shadow(probe_layouts={'H100': {'preset': 'rtx5090-tp2', 'max_num_seqs': 96, 'fallback': 'live'}})
        command, files = self.arm(cfg, 'H100')
        self.assertIn('probe_serve.py', command)
        shipped = PL.ProbePlan.from_doc(json.loads(files[JOB + '/probe-layouts.json']))
        self.assertEqual([(l.name, l.max_num_seqs) for l in shipped.layouts], [('rtx5090-tp2-custom', 96), ('live', 512)])
        self.assertEqual(json.loads(files[JOB + '/eval-plan.json'])['concurrency_per_replica'], 96)

    def test_the_host_waits_long_enough_for_the_fallback_layout(self):
        plan = shadow().probe_plan('RTX5090')
        self.assertGreater(plan.eval_timeout_s(), 7200)
        self.assertEqual(shadow().probe_plan('B300').eval_timeout_s(), 7200)     # the live wait is unchanged


class TestHostBoxProtocol(BoxCase):
    """The real host job controller and the real remote store drive the tool on a local box."""

    def setUp(self):
        super().setUp()
        self.store = RemoteBoxJobStore(self.pod())
        self.ctrl = BoxJobController(store=self.store, item_id='e1-item', job_root=str(self.tmp / 'pit-jobs'))

    def arm(self, layouts, probe_plan):
        job_dir = self.ctrl.job_dir
        self.ctrl.ensure_dir()
        plan_path = '%s/eval-plan.json' % job_dir
        self.store.write_text(plan_path, json.dumps(probe_plan, sort_keys=True) + '\n')
        PS.ship(self.store.write_text, job_dir, layouts, read_text=self.store.read_text)
        live_command = detached_eval_command(job_dir=job_dir, candidate_id=self.cid, plan_path=plan_path,
                                             podd=str(self.fb.root), max_num_seqs=512)
        return PS.detached_command(preamble=PS.live_preamble(live_command), job_dir=job_dir, candidate_id=self.cid,
                                   plan_path=plan_path, root=str(self.fb.root), box_args=self.box.args()[2:])

    def run_eval(self, command):
        def forbidden():
            raise AssertionError('the host must not run the eval itself')

        return self.ctrl.run_stage_once('eval', forbidden, detached_command=command, wait_for_marker=True,
                                        poll_s=0.2, timeout_s=300)

    def test_the_remote_store_delivers_the_tool_files_unchanged(self):
        job_dir = self.ctrl.job_dir
        paths = PS.ship(self.store.write_text, job_dir, PL.shadow_plan('RTX5090'))
        here = Path(PS.__file__).parent
        for name in PS.TOOL_FILES:
            self.assertEqual(Path(paths[name]).read_bytes(), (here / name).read_bytes(), name)
        self.assertEqual(Path(paths[PS.LAYOUTS_NAME]).read_text(), PL.dumps(PL.shadow_plan('RTX5090')))

        writes = []
        PS.ship(lambda path, text: writes.append(path), job_dir, PL.shadow_plan('RTX5090'), read_text=self.store.read_text)
        self.assertEqual(writes, [])                         # a host restart does not rewrite the files
        PS.ship(lambda path, text: writes.append(path), job_dir, PL.plan_from('rtx5090-1gpu'), read_text=self.store.read_text)
        self.assertEqual(writes, [paths[PS.LAYOUTS_NAME]])

    def test_detached_eval_under_the_host_controller_returns_the_probe_result(self):
        probe_plan = self.probe_plan()
        marker = self.run_eval(self.arm(PL.shadow_plan('RTX5090'), probe_plan))

        self.assertEqual((marker['ok'], marker['status'], marker['candidate_id']), (True, 'ok', self.cid))
        result = marker['probe_result']
        self.assertIs(result['partial'], False)
        self.assertEqual(result['samples'], 16 * 8)
        self.assertEqual(result['serve_layout']['layout']['name'], 'rtx5090-tp2')
        self.assertEqual(result['serve_layout']['failed_layouts'], [])
        # A host that restarts finds the result and starts nothing.
        again = BoxJobController(store=self.store, item_id='e1-item', job_root=str(self.tmp / 'pit-jobs'))
        started = len(self.fb.started())
        self.assertEqual(again.resume_or_fetch()['probe_result']['samples'], 16 * 8)
        self.assertEqual(again.run_stage_once('eval', lambda: {}, detached_command='false', wait_for_marker=True,
                                              poll_s=0.2, timeout_s=30)['probe_result']['samples'], 16 * 8)
        self.assertEqual(len(self.fb.started()), started)
        log = (Path(self.ctrl.job_dir) / 'job.log').read_text()
        self.assertIn('pit-box-job-eval', log)

    def test_fallback_layout_under_the_host_controller(self):
        self.fb.control(fail_if=['--tensor-parallel-size'])
        marker = self.run_eval(self.arm(PL.shadow_plan('RTX5090'), self.probe_plan()))
        self.assertEqual((marker['ok'], marker['status']), (True, 'ok'))
        served = marker['probe_result']['serve_layout']
        self.assertEqual(served['layout']['name'], 'rtx5090-1gpu')
        self.assertEqual(served['launcher'], 'podctl')
        self.assertEqual([f['layout'] for f in served['failed_layouts']], ['rtx5090-tp2'])
        self.assertEqual(served['queue']['in_flight_limit'], 4 * 64)

    def test_a_box_that_cannot_serve_reports_failed_and_the_host_does_not_wait(self):
        self.fb.control(fail_if=['--max-num-seqs'])
        marker = self.run_eval(self.arm(PL.shadow_plan('RTX5090'), self.probe_plan()))
        self.assertEqual((marker['ok'], marker['status']), (False, 'failed'))
        self.assertIn('no layout of the plan serves', marker['error'])
        self.assertNotIn('probe_result', marker)


class TestEvalStageOnTheBox(BoxCase):
    """``BoxRunner`` runs the eval stage of an item on a local box, and a host that restarts finds its result."""

    def test_a_restarted_host_takes_the_result_from_the_box_and_serves_nothing(self):
        tasks = list(range(16))
        probe, record = self.eval_stage(shadow(), tasks)
        self.assertIs(record.state, ItemState.RESULT)
        self.assertEqual((probe['partial'], probe['samples'], len(probe['envs'][ENV_ID]['per_task'])), (False, 16 * 8, 16))
        self.assertEqual(probe['serve_layout']['layout']['name'], 'rtx5090-tp2')
        started = len(self.fb.started())
        self.assertEqual(started, 2)                          # two TP2 replicas on the four GPUs of this box

        # A host that starts again reads the result before it does anything else.
        record, bundle = self.restarted_host(shadow(), tasks)
        self.assertIs(record.state, ItemState.RESULT)
        self.assertEqual((bundle.candidate_id, bundle.probe_result['envs']), (self.cid, probe['envs']))
        # The box can also finish between that first look and the eval stage. The eval stage then takes the result.
        again, record = self.eval_stage(shadow(), tasks)
        self.assertEqual((record.state, record.meta['last_transition_reason']), (ItemState.RESULT, 'eval-resume-result'))
        self.assertEqual(again['envs'], probe['envs'])
        self.assertEqual(len(self.fb.started()), started)     # nothing served a second time


if __name__ == '__main__':
    unittest.main()
