"""Probe serving layouts: the shadow default for RTX 5090, the fallback, and what a layout may never change."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import unittest
from pathlib import Path

from ops.parallel_eval.pit import GATE_MAX_TOKENS, MAX_SEQS, STALL_FIRST_COMPILE
from ops.parallel_eval.pit import probe_layout as PL
from ops.parallel_eval.pit.config import GATE_DTYPE, PitConfig

PINNED = Path(__file__).resolve().parents[3] / 'miner' / 'validator' / 'probe'

# Prints how the real vLLM argument parser reads a serve command line. Run by a vLLM interpreter, not by this one.
_REAL_PARSER = r'''
import importlib, json, sys
import importlib.metadata as md
for name in ('vllm.entrypoints.common.cli_args', 'vllm.entrypoints.launchers.cli_args', 'vllm.entrypoints.openai.cli_args'):
    try:
        mod = importlib.import_module(name)
    except ImportError:
        continue
    break
from vllm.utils.argparse_utils import FlexibleArgumentParser
args = mod.make_arg_parser(FlexibleArgumentParser()).parse_args(json.loads(sys.argv[1]))
mod.validate_parsed_serve_args(args)
keys = ('host', 'port', 'served_model_name', 'dtype', 'max_model_len', 'gpu_memory_utilization', 'enable_prefix_caching',
        'limit_mm_per_prompt', 'max_num_seqs', 'tensor_parallel_size', 'kv_cache_dtype')
print(json.dumps(dict(vllm=md.version('vllm'), parsed={k: getattr(args, k) for k in keys}), default=str))
'''


def _flags(args):
    """['--a', '1', '--switch', '--b', '2'] -> {'--a': '1', '--switch': None, '--b': '2'}"""
    out, i = {}, 0
    while i < len(args):
        if i + 1 < len(args) and not args[i + 1].startswith('--'):
            out[args[i]] = args[i + 1]
            i += 2
        else:
            out[args[i]] = None
            i += 1
    return out


class TestShadowDefaults(unittest.TestCase):
    def test_rtx5090_serves_tp2_on_vllm_033_with_the_measured_settings(self):
        for gpu in ('RTX5090', 'NVIDIA GeForce RTX 5090', '8xRTX5090', 'rtx 5090'):
            plan = PL.shadow_plan(gpu)
            tp2 = plan.primary
            self.assertEqual(tp2.gpus_per_replica, 2, gpu)
            self.assertEqual(tp2.groups(8), ((0, 1), (2, 3), (4, 5), (6, 7)))   # 4 replicas per box
            self.assertEqual(tp2.vllm_version, '0.33.0')
            self.assertIs(tp2.prefix_caching, False)
            self.assertTrue(96 <= tp2.max_num_seqs <= 128)
            self.assertEqual(tp2.gpu_memory_utilization, 0.94)
            self.assertEqual(tp2.task_queue, 'shared')
            self.assertEqual(tp2.kv_cache_dtype, 'auto')
            self.assertEqual(tp2.score_changing(), [])
            flags = _flags(tp2.vllm_args(18555))
            self.assertEqual(flags['--tensor-parallel-size'], '2')
            self.assertIn('--no-enable-prefix-caching', flags)
            self.assertNotIn('--enable-prefix-caching', flags)
            self.assertEqual(flags['--max-num-seqs'], str(tp2.max_num_seqs))
            self.assertEqual(flags['--gpu-memory-utilization'], '0.94')
            self.assertNotIn('--kv-cache-dtype', flags)

    def test_rtx5090_keeps_a_one_gpu_per_replica_fallback(self):
        plan = PL.shadow_plan('RTX5090')
        self.assertEqual([layout.name for layout in plan.layouts], ['rtx5090-tp2', 'rtx5090-1gpu'])
        one = plan.layouts[1]
        self.assertEqual(one.groups(8), tuple((i,) for i in range(8)))
        self.assertEqual((one.max_num_seqs, one.gpu_memory_utilization), (64, 0.90))
        # It is the measured baseline: the live serve.sh and the live vLLM build, so the box needs no install.
        self.assertEqual(one.launcher(), 'podctl')
        self.assertEqual(one.vllm_version, PL.LIVE_VLLM_VERSION)
        self.assertIs(one.install_vllm, False)
        # 2.65 h measured for 972 tasks x 8: the live 90 min limit would always cut this layout short.
        self.assertGreater(one.probe_wall_s, 2.65 * 3600)

    def test_first_job_limits_match_the_pit_constants_for_a_new_gpu_class(self):
        tp2 = PL.shadow_plan('RTX5090').primary
        self.assertEqual(tp2.serve_timeout_s, STALL_FIRST_COMPILE['serving'])
        self.assertEqual(tp2.probe_wall_s, STALL_FIRST_COMPILE['probing'])

    def test_every_other_gpu_class_keeps_the_live_layout(self):
        for gpu in ('B300', 'B200', 'H200', 'H100', 'NVIDIA H100 80GB', None, ''):
            plan = PL.shadow_plan(gpu)
            self.assertTrue(plan.is_live(), gpu)
            self.assertEqual(plan.primary, PL.LIVE)

    def test_the_host_wait_covers_install_both_serve_attempts_and_the_longest_probe(self):
        plan = PL.shadow_plan('RTX5090')
        tp2, one = plan.layouts
        need = PL.INSTALL_TIMEOUT_S + tp2.serve_timeout_s + one.serve_timeout_s + one.probe_wall_s
        self.assertGreater(plan.eval_timeout_s(), need)
        self.assertLess(plan.eval_timeout_s(), PitConfig().decision_timeout_seconds)


class TestLiveLayout(unittest.TestCase):
    def test_live_layout_is_what_the_pinned_serve_sh_runs(self):
        text = (PINNED / 'serve.sh').read_text()
        exec_line = text[text.index('exec $D/venv/bin/vllm serve "$CK"'):].replace('\\\n', ' ')
        words = shlex.split(
            exec_line.replace('"$PORT"', '18555').replace('"$GMU"', '0.9').replace('${MAX_SEQS:-512}', '512'))
        self.assertEqual(words[:4], ['exec', '$D/venv/bin/vllm', 'serve', '$CK'])
        self.assertEqual(words[-1], '$@')
        self.assertEqual(PL.LIVE.vllm_args(18555), words[4:-1])
        self.assertEqual(PL.LIVE.launcher(), 'podctl')

    def test_live_layout_equals_the_pinned_probe_constants(self):
        pins = json.loads((PINNED / 'runtime' / 'PINS.json').read_text())
        self.assertEqual(PL.LIVE.max_num_seqs, pins['probe']['max_num_seqs'])
        self.assertEqual(PL.LIVE.max_num_seqs, MAX_SEQS)
        self.assertEqual(PL.GATE_MAX_MODEL_LEN, pins['probe']['max_model_len'])
        self.assertEqual(PL.GATE_DTYPE, pins['probe']['dtype'])
        self.assertEqual(PL.GATE_DTYPE, GATE_DTYPE)
        self.assertEqual(PL.LIVE_VLLM_VERSION, pins['runtime']['vllm'])
        self.assertGreater(PL.GATE_MAX_MODEL_LEN, GATE_MAX_TOKENS)
        podctl = (PINNED / 'podctl.py').read_text()
        self.assertIn("D = '%s'; PORT = %d;" % (PL.BOX_ROOT, PL.BASE_PORT), podctl)
        self.assertIn('for i in range(%d)' % PL.MAX_REPLICAS, podctl)

    def test_gate_constants_are_in_every_layout_and_are_no_settings(self):
        for layout in PL.PRESETS.values():
            flags = _flags(layout.vllm_args(18555))
            self.assertEqual(flags['--dtype'], 'bfloat16')
            self.assertEqual(flags['--max-model-len'], '69632')
            self.assertEqual(flags['--host'], '127.0.0.1')
            self.assertEqual(flags['--served-model-name'], 'probe')
            self.assertEqual(json.loads(flags['--limit-mm-per-prompt']), {'image': 0, 'video': 0})
        for key in ('dtype', 'max_model_len', 'host', 'served_model_name'):
            with self.assertRaises(PL.LayoutError):
                PL.layout_from({'preset': 'live', key: 'x'})

    def test_a_layout_with_another_name_but_live_settings_is_live(self):
        self.assertTrue(PL.plan_from({'name': 'mine'}).is_live())
        self.assertFalse(PL.plan_from({'preset': 'live', 'max_num_seqs': 256}).is_live())
        self.assertFalse(PL.plan_from(['live', 'rtx5090-1gpu']).is_live())


class TestSettings(unittest.TestCase):
    def test_every_setting_can_be_set(self):
        layout = PL.layout_from({
            'preset': 'rtx5090-tp2', 'name': 'bench-a', 'gpus_per_replica': 4, 'replicas': 2,
            'vllm_version': '0.34.1', 'vllm_venv': '/opt/vllm-0.34.1', 'vllm_requirements': '/opt/vllm.lock',
            'install_vllm': False, 'runtime_sha256': 'ab' * 32, 'prefix_caching': True, 'max_num_seqs': 128,
            'gpu_memory_utilization': 0.92, 'task_queue': 'shared', 'concurrency_per_replica': 96,
            'kv_cache_dtype': 'auto', 'serve_timeout_s': 600, 'probe_stall_s': 900, 'probe_wall_s': 4000,
        })
        self.assertEqual(layout.groups(8), ((0, 1, 2, 3), (4, 5, 6, 7)))
        flags = _flags(layout.vllm_args(18555))
        self.assertEqual(flags['--tensor-parallel-size'], '4')
        self.assertIn('--enable-prefix-caching', flags)
        self.assertEqual((flags['--max-num-seqs'], flags['--gpu-memory-utilization']), ('128', '0.92'))
        self.assertEqual(layout.concurrency(), 96)
        self.assertEqual((layout.serve_timeout_s, layout.probe_stall_s, layout.probe_wall_s), (600, 900, 4000))
        self.assertEqual(layout.launcher(), 'direct')
        self.assertEqual(PL.ProbeLayout.from_mapping(json.loads(json.dumps(layout.as_dict()))), layout)

    def test_max_num_seqs_covers_the_measured_range(self):
        for seqs in (96, 112, 128):
            layout = PL.layout_from({'preset': 'rtx5090-tp2', 'max_num_seqs': seqs})
            self.assertEqual(_flags(layout.vllm_args(18555))['--max-num-seqs'], str(seqs))
            self.assertEqual(layout.concurrency(), seqs)

    def test_gpu_pairs_can_be_named(self):
        layout = PL.layout_from({'preset': 'rtx5090-tp2', 'gpu_groups': [[0, 4], [1, 5], [2, 6], [3, 7]]})
        self.assertEqual(layout.groups(8), ((0, 4), (1, 5), (2, 6), (3, 7)))
        with self.assertRaises(PL.LayoutError):
            layout.groups(4)
        for groups in ([[0, 1], [1, 2]], [[0], [1]], [[0, -1]], []):
            with self.assertRaises(PL.LayoutError, msg=str(groups)):
                PL.layout_from({'preset': 'rtx5090-tp2', 'gpu_groups': groups})

    def test_gpus_that_do_not_fit_are_refused(self):
        tp2 = PL.PRESETS['rtx5090-tp2']
        self.assertEqual(tp2.groups(4), ((0, 1), (2, 3)))
        for count in (0, 7, True):
            with self.assertRaises(PL.LayoutError, msg=str(count)):
                tp2.groups(count)
        with self.assertRaises(PL.LayoutError):
            PL.layout_from({'preset': 'rtx5090-tp2', 'replicas': 5}).groups(8)

    def test_wrong_values_are_refused(self):
        bad = [
            {'max_num_seqs': 0}, {'max_num_seqs': 513}, {'max_num_seqs': '112'}, {'max_num_seqs': True},
            {'gpu_memory_utilization': 0}, {'gpu_memory_utilization': 1.01}, {'gpu_memory_utilization': '0.94'},
            {'gpus_per_replica': 0}, {'replicas': 17}, {'prefix_caching': 'off'}, {'install_vllm': 1},
            {'vllm_version': 'latest'}, {'vllm_venv': 'venv'}, {'vllm_requirements': '/a b'},
            {'runtime_sha256': 'abc'}, {'concurrency_per_replica': 0}, {'kv_cache_dtype': 'fp8;rm'},
            {'serve_timeout_s': 0}, {'probe_wall_s': 1.5}, {'name': 'Bad Name'}, {'no_such_setting': 1},
        ]
        for change in bad:
            with self.assertRaises(PL.LayoutError, msg=str(change)):
                PL.layout_from(dict({'preset': 'rtx5090-tp2'}, **change))
        for spec in ('no-such-preset', 5, {'preset': 7}, {'preset': 'live', 'fallback': 'live'}):
            with self.assertRaises(PL.LayoutError, msg=str(spec)):
                PL.layout_from(spec)

    def test_the_only_task_queue_is_the_shared_one(self):
        with self.assertRaises(PL.LayoutError) as caught:
            PL.layout_from({'preset': 'rtx5090-tp2', 'task_queue': 'per_replica'})
        self.assertIn('one queue for all replicas', str(caught.exception))

    def test_fallbacks_can_be_replaced_or_removed(self):
        self.assertEqual(len(PL.plan_from({'preset': 'rtx5090-tp2', 'fallback': None}).layouts), 1)
        plan = PL.plan_from({'preset': 'rtx5090-tp2', 'max_num_seqs': 96,
                             'fallback': [{'preset': 'rtx5090-tp2', 'name': 'tp2-032', 'vllm_version': '0.32.0'},
                                          'rtx5090-1gpu']})
        self.assertEqual([l.name for l in plan.layouts], ['rtx5090-tp2-custom', 'tp2-032', 'rtx5090-1gpu'])
        self.assertEqual(PL.ProbePlan.from_doc(json.loads(PL.dumps(plan))), plan)
        with self.assertRaises(PL.LayoutError):
            PL.plan_from(['rtx5090-tp2', 'rtx5090-tp2'])
        with self.assertRaises(PL.LayoutError):
            PL.plan_from([])

    def test_shadow_overrides_are_per_gpu_class(self):
        over = {'RTX5090': {'preset': 'rtx5090-tp2', 'max_num_seqs': 128, 'fallback': None}, 'default': 'live'}
        self.assertEqual(PL.shadow_plan('RTX5090', over).primary.max_num_seqs, 128)
        self.assertEqual(len(PL.shadow_plan('RTX5090', over).layouts), 1)
        self.assertTrue(PL.shadow_plan('H100', over).is_live())
        self.assertEqual(PL.shadow_plan('H100', {'default': 'rtx5090-1gpu'}).primary.name, 'rtx5090-1gpu')
        self.assertEqual(PL.shadow_plan('RTX5090', {'default': 'live'}).primary.name, 'live')


class TestScoreChangingSettings(unittest.TestCase):
    def test_fp8_kv_cache_is_off_in_every_preset(self):
        for layout in PL.PRESETS.values():
            self.assertEqual(layout.kv_cache_dtype, 'auto')
            self.assertNotIn('fp8', ' '.join(layout.vllm_args(18555)))

    def test_fp8_kv_cache_is_marked_and_refused_for_the_gate(self):
        for dtype in ('fp8', 'fp8_e4m3', 'fp8_e5m2'):
            layout = PL.layout_from({'preset': 'rtx5090-tp2', 'kv_cache_dtype': dtype})
            self.assertTrue(layout.score_changing())
            self.assertEqual(_flags(layout.vllm_args(18555))['--kv-cache-dtype'], dtype)
            with self.assertRaises(PL.LayoutError):
                PL.ProbePlan((layout,)).check_gate()
            with self.assertRaises(PL.LayoutError):
                PL.shadow_plan('RTX5090', {'RTX5090': {'preset': 'rtx5090-tp2', 'kv_cache_dtype': dtype}})
            with self.assertRaises(PL.LayoutError):
                PL.shadow_plan('RTX5090', {'RTX5090': ['rtx5090-tp2', layout.as_dict()]})


@unittest.skipUnless(os.environ.get('PIT_VLLM_PYTHONS'), 'set PIT_VLLM_PYTHONS to interpreters that have vLLM')
class TestRealVllmParsers(unittest.TestCase):
    """The real vLLM argument parser reads every preset as the layout means it."""

    def test_presets_parse_as_meant(self):
        seen = set()
        for python in os.environ['PIT_VLLM_PYTHONS'].split(os.pathsep):
            for layout in PL.PRESETS.values():
                argv = ['/ckpt/x'] + layout.vllm_args(18555)
                proc = subprocess.run([python, '-c', _REAL_PARSER, json.dumps(argv)], capture_output=True, text=True,
                                      timeout=600, env=dict(os.environ, VLLM_TARGET_DEVICE='cpu'))
                self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
                doc = json.loads(proc.stdout.strip().splitlines()[-1])
                got = doc['parsed']
                seen.add(doc['vllm'])
                self.assertEqual(got['tensor_parallel_size'], layout.gpus_per_replica)
                self.assertIs(got['enable_prefix_caching'], layout.prefix_caching)
                self.assertEqual(got['max_num_seqs'], layout.max_num_seqs)
                self.assertEqual(got['gpu_memory_utilization'], layout.gpu_memory_utilization)
                self.assertEqual(got['kv_cache_dtype'], 'auto')
                self.assertEqual(got['max_model_len'], 69632)
                self.assertEqual(got['dtype'], 'bfloat16')
                self.assertEqual(got['served_model_name'], ['probe'])
                self.assertEqual(got['limit_mm_per_prompt'], {'image': 0, 'video': 0})
                self.assertEqual((got['host'], got['port']), ('127.0.0.1', 18555))
        self.assertTrue(seen)


if __name__ == '__main__':
    unittest.main()
