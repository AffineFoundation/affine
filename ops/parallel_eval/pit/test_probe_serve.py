"""The box side of a probe layout, on a local box built from the pinned probe code (see probe_fixture)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

from ops.parallel_eval.pit import probe_fixture as PF
from ops.parallel_eval.pit import probe_layout as PL
from ops.parallel_eval.pit import probe_serve as PS
from ops.parallel_eval.pit.probe_fixture import BoxCase

SERVE_SH_ENV = dict(VLLM_LOGGING_LEVEL='WARNING', HF_HUB_OFFLINE='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1')


def flags_of(started):
    """Flags of one started server: {'--port': '18555', '--enable-prefix-caching': None, ...}."""
    argv = started['argv']
    rest = argv[argv.index('serve') + 2:]
    out, i = {}, 0
    while i < len(rest):
        if rest[i] in PS.SWITCHES:
            out[rest[i]] = None
            i += 1
        else:
            out[rest[i]] = rest[i + 1]
            i += 2
    return out


class TestDefault5090Layout(BoxCase):
    GPUS = 8

    def test_shadow_default_serves_four_tp2_replicas_and_probes_through_one_queue(self):
        plan = PL.shadow_plan('RTX5090')
        record = PS.serve(plan, self.cid, self.box)

        self.assertEqual(record['layout']['name'], 'rtx5090-tp2')
        self.assertEqual(record['launcher'], 'direct')
        self.assertEqual(record['gpu_groups'], [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertEqual(record['failed_layouts'], [])
        self.assertEqual(record['runtime']['venv'], str(self.v033))
        self.assertEqual(record['runtime']['runtime_sha256'], hashlib.sha256(b'vllm==0.33.0\n').hexdigest())
        self.assertEqual(record['gpus'], [PF.GPU_NAME] * 8)

        started = sorted(self.fb.started(), key=lambda s: s['port'])
        self.assertEqual(len(started), 4)
        caches = set()
        for i, server in enumerate(started):
            flags = flags_of(server)
            self.assertEqual(server['version'], '0.33.0')
            self.assertEqual(server['env']['CUDA_VISIBLE_DEVICES'], '%d,%d' % (2 * i, 2 * i + 1))
            self.assertEqual(flags['--port'], str(self.box.port + i))
            self.assertEqual(flags['--tensor-parallel-size'], '2')
            self.assertIn('--no-enable-prefix-caching', flags)
            self.assertNotIn('--enable-prefix-caching', flags)
            self.assertEqual(flags['--max-num-seqs'], '112')
            self.assertEqual(flags['--gpu-memory-utilization'], '0.94')
            self.assertEqual((flags['--dtype'], flags['--max-model-len']), ('bfloat16', '69632'))
            self.assertNotIn('--kv-cache-dtype', flags)
            self.assertNotIn('--quantization', flags)
            self.assertNotIn('fp8', json.dumps(server['argv']).lower())
            for key, value in SERVE_SH_ENV.items():       # what the live serve.sh exports
                self.assertEqual(server['env'][key], value)
            caches.add(server['env']['VLLM_CACHE_ROOT'])
            self.assertTrue(os.path.isdir(server['env']['TORCHINDUCTOR_CACHE_DIR']))
        self.assertEqual(len(caches), 4)                  # one compile cache per replica

        # The pinned podctl sees the replicas through the state files this tool wrote.
        status = self.box.status()
        self.assertEqual((status['replicas'], status['replicas_up'], status['serve_ckpt']), (4, 4, self.cid))
        self.assertEqual([r['pid'] for r in record['replicas']], [s['pid'] for s in started])

        probe_plan = self.probe_plan(tasks=24)
        result, how = PS.run_probe(plan.primary, self.cid, probe_plan, self.box)
        self.assertEqual(how, 'ok')
        self.assertIs(result['partial'], False)
        self.assertEqual(result['samples'], 24 * 8)
        self.assertEqual(len(result['envs'][PF.ENV_ID]['per_task']), 24)
        queue = PS.queue_use(result, record)
        self.assertEqual(queue['replicas_fed'], 4)
        self.assertEqual(sum(queue['requests_per_replica']), 24 * 8)
        self.assertEqual(queue['in_flight_limit'], 4 * 112)
        self.assertTrue(queue['as_planned'])

        PS.stop(self.box)
        self.assertEqual(self.alive(), [])
        self.assertEqual(self.box.status()['replicas_up'], 0)


class TestFallback(BoxCase):
    def test_tp2_that_cannot_start_falls_back_to_one_replica_per_gpu_on_the_live_serve_sh(self):
        self.fb.control(fail_if=['--tensor-parallel-size'])
        record = PS.serve(PL.shadow_plan('RTX5090'), self.cid, self.box)

        self.assertEqual(record['layout']['name'], 'rtx5090-1gpu')
        self.assertEqual(record['launcher'], 'podctl')
        self.assertEqual(record['gpu_groups'], [[0], [1], [2], [3]])
        self.assertEqual([f['layout'] for f in record['failed_layouts']], ['rtx5090-tp2'])
        self.assertIn('exited while starting', record['failed_layouts'][0]['error'])
        self.assertIn('fake vllm: cannot start', record['failed_layouts'][0]['error'])   # the replica's log tail

        serving = sorted(self.alive(), key=lambda s: s['port'])
        self.assertEqual(len(serving), 4)
        for i, server in enumerate(serving):
            flags = flags_of(server)
            self.assertEqual(server['version'], '0.32.0')                    # the bundle's vLLM, through serve.sh
            self.assertEqual(server['argv'][0], str(self.fb.root / 'venv' / 'bin' / 'vllm'))
            self.assertEqual(server['env']['CUDA_VISIBLE_DEVICES'], str(i))
            self.assertEqual(server['env']['VLLM_CACHE_ROOT'], str(self.fb.root / 'vcache' / ('r%d' % i) / 'vllm'))
            self.assertEqual(flags['--max-num-seqs'], '64')
            self.assertEqual(flags['--gpu-memory-utilization'], '0.9')
            self.assertIn('--enable-prefix-caching', flags)
            self.assertNotIn('--tensor-parallel-size', flags)

    def test_a_missing_vllm_build_falls_back(self):
        shutil.rmtree(self.v033)
        plan = PL.plan_from({'preset': 'rtx5090-tp2', 'install_vllm': False, 'fallback': 'rtx5090-1gpu'})
        record = PS.serve(plan, self.cid, self.box)
        self.assertEqual(record['layout']['name'], 'rtx5090-1gpu')
        self.assertIn('vLLM 0.33.0 is not on the box', record['failed_layouts'][0]['error'])

    def test_no_layout_serves_is_an_error_and_leaves_nothing_running(self):
        self.fb.control(fail_if=['--max-num-seqs'])
        with self.assertRaises(PS.ServeError) as caught:
            PS.serve(PL.shadow_plan('RTX5090'), self.cid, self.box)
        self.assertIn('rtx5090-tp2', str(caught.exception))
        self.assertIn('rtx5090-1gpu', str(caught.exception))
        self.assertEqual(self.alive(), [])

    def test_a_fallback_that_serves_is_the_layout_that_probes(self):
        self.fb.control(fail_if=['--tensor-parallel-size'])
        record = PS.serve(PL.shadow_plan('RTX5090'), self.cid, self.box)
        layout = PL.ProbeLayout.from_mapping(record['layout'])
        result, _ = PS.run_probe(layout, self.cid, self.probe_plan(), self.box)
        queue = PS.queue_use(result, record)
        self.assertEqual((queue['replicas'], queue['replicas_fed'], queue['in_flight_limit']), (4, 4, 4 * 64))


class TestInstall(BoxCase):
    def test_the_build_is_installed_into_its_own_environment_when_the_box_lacks_it(self):
        shutil.rmtree(self.v033)
        self.fb.add_uv()
        record = PS.serve(PL.shadow_plan('RTX5090'), self.cid, self.box)

        self.assertEqual(record['layout']['name'], 'rtx5090-tp2')
        self.assertIs(record['runtime']['installed'], True)
        self.assertEqual(record['runtime']['venv'], str(self.v033))
        calls = self.fb.uv_calls()
        self.assertEqual([c['args'] for c in calls],
                         [['venv', '-q', '-p', '3.12', str(self.v033)], ['pip', 'install', '-q', 'vllm==0.33.0']])
        self.assertEqual(calls[1]['env'], {'VIRTUAL_ENV': str(self.v033), 'UV_CACHE_DIR': str(self.fb.root / 'uvcache')})
        self.assertEqual({s['version'] for s in self.alive()}, {'0.33.0'})
        # The live bundle is not touched.
        self.assertEqual(PS.venv_version(str(self.fb.root / 'venv')), '0.32.0')

        PS.stop(self.box)
        again = PS.serve(PL.shadow_plan('RTX5090'), self.cid, self.box)
        self.assertIs(again['runtime']['installed'], False)
        self.assertEqual(len(self.fb.uv_calls()), 2)

    def test_a_lock_file_and_its_digest_pin_the_whole_environment(self):
        shutil.rmtree(self.v033)
        self.fb.add_uv()
        lock = self.tmp / 'vllm-0.33.0.lock'
        lock.write_text('vllm==0.33.0\ntorch==2.14.0\ntransformers==5.1.0\n')
        packages = ['torch==2.14.0', 'transformers==5.1.0', 'vllm==0.33.0']
        digest = PS.packages_digest(packages)
        layout = PL.layout_from({'preset': 'rtx5090-tp2', 'vllm_requirements': str(lock), 'runtime_sha256': digest})

        record = PS.serve_one(layout, self.cid, self.box)
        self.assertEqual(self.fb.uv_calls()[1]['args'], ['pip', 'install', '-q', '-r', str(lock)])
        self.assertEqual(record['packages'], packages)
        self.assertEqual(record['runtime']['runtime_sha256'], digest)

        PS.stop(self.box)
        other = PL.layout_from({'preset': 'rtx5090-tp2', 'runtime_sha256': 'ab' * 32})
        with self.assertRaises(PS.ServeError) as caught:
            PS.serve_one(other, self.cid, self.box)
        self.assertIn('the layout pins', str(caught.exception))
        self.assertEqual(self.alive(), [])


class TestVerify(BoxCase):
    def start(self, layout):
        groups = PS.launch(layout, self.cid, self.box, PS.runtime(layout, self.box)['venv'])
        PS.wait_ready(layout, self.cid, self.box, len(groups))
        return groups

    def test_replicas_that_are_not_the_layout_are_refused(self):
        tp2 = PL.PRESETS['rtx5090-tp2']
        groups = self.start(tp2)
        self.assertEqual(len(PS.verify(tp2, self.cid, self.box, groups)), 2)
        for change, why in (
            ({'max_num_seqs': 128}, '--max-num-seqs is 112, the layout needs 128'),
            ({'gpu_memory_utilization': 0.9}, '--gpu-memory-utilization is 0.94'),
            ({'prefix_caching': True}, 'prefix caching is off, the layout needs on'),
            ({'vllm_version': '0.32.0'}, 'runs vLLM 0.33.0, the layout needs 0.32.0'),
            ({'gpu_groups': [[0, 2], [1, 3]]}, 'replica 0 sees GPUs 0,1, the layout gives it 0,2'),
        ):
            other = PL.layout_from(dict({'preset': 'rtx5090-tp2'}, **change))
            with self.assertRaises(PS.ServeError, msg=str(change)) as caught:
                PS.verify(other, self.cid, self.box, other.groups(4))
            self.assertIn(why, str(caught.exception))
        one = PL.layout_from({'preset': 'rtx5090-tp2', 'gpus_per_replica': 1, 'replicas': 2})
        with self.assertRaises(PS.ServeError) as caught:
            PS.verify(one, self.cid, self.box, one.groups(4))
        self.assertIn('--tensor-parallel-size is 2, the layout needs 1', str(caught.exception))

    def test_replicas_that_run_with_an_fp8_argument_never_pass(self):
        """No layout can ask for FP8. Replicas that run with it all the same do not pass as any layout."""
        tp2 = PL.PRESETS['rtx5090-tp2']
        for flag in ('--kv-cache-dtype', '--quantization'):
            # A vLLM build that adds the FP8 argument to every server it starts.
            venv = self.tmp / ('venv-fp8%s' % flag)
            (venv / 'bin').mkdir(parents=True)
            script = venv / 'bin' / 'vllm'
            script.write_text('#!/bin/sh\nexec %s "$@" %s fp8\n' % (self.v033 / 'bin' / 'vllm', flag))
            script.chmod(0o755)
            groups = PS.launch(tp2, self.cid, self.box, str(venv))
            PS.wait_ready(tp2, self.cid, self.box, len(groups))
            self.assertEqual([flags_of(s)[flag] for s in self.alive()], ['fp8', 'fp8'])
            with self.assertRaises(PS.ServeError) as caught:
                PS.verify(tp2, self.cid, self.box, groups)
            self.assertIn('vLLM argument outside the layout: %r' % flag, str(caught.exception))

    def test_a_server_that_reports_another_version_is_refused(self):
        self.fb.control(report_version='0.31.9')
        with self.assertRaises(PS.ServeError) as caught:
            PS.serve_one(PL.PRESETS['rtx5090-tp2'], self.cid, self.box)
        self.assertIn('runs vLLM 0.31.9, the layout needs 0.33.0', str(caught.exception))

    def test_flags_outside_the_layouts_are_refused(self):
        ok = ['/v/bin/python', '/v/bin/vllm', 'serve', '/ckpt/x'] + PL.PRESETS['rtx5090-tp2'].vllm_args(18555)
        self.assertEqual(PS.argv_mismatch(PL.PRESETS['rtx5090-tp2'], ok, ckpt='/ckpt/x', port=18555), [])
        for extra in (['--max-num-seqs', '512'], ['--enable-prefix-caching'], ['--seed', '1'], ['--port']):
            with self.subTest(extra=extra):
                try:
                    bad = PS.argv_mismatch(PL.PRESETS['rtx5090-tp2'], ok + extra, ckpt='/ckpt/x', port=18555)
                except PS.ServeError:
                    continue
                self.assertTrue(bad)
        # FP8 in any form: the KV cache, the weights, the short flag, flag=value, or the value 'auto'.
        for extra in (['--kv-cache-dtype', 'fp8'], ['--kv-cache-dtype', 'fp8_e4m3'], ['--kv-cache-dtype=fp8'],
                      ['--kv-cache-dtype', 'auto'], ['--quantization', 'fp8'], ['--quantization=fp8'], ['-q', 'fp8']):
            for layout in PL.PRESETS.values():
                argv = ['/v/bin/python', '/v/bin/vllm', 'serve', '/ckpt/x'] + layout.vllm_args(18555) + extra
                with self.assertRaises(PS.ServeError, msg='%s %s' % (layout.name, extra)) as caught:
                    PS.argv_mismatch(layout, argv, ckpt='/ckpt/x', port=18555)
                self.assertIn('vLLM argument outside the layout', str(caught.exception))
        self.assertFalse([flag for flag in PS.ALLOWED_FLAGS if 'kv-cache' in flag or 'quant' in flag])
        self.assertTrue(PS.argv_mismatch(PL.PRESETS['rtx5090-tp2'], ok, ckpt='/ckpt/y', port=18555))
        self.assertTrue(PS.argv_mismatch(PL.PRESETS['rtx5090-tp2'], ok, ckpt='/ckpt/x', port=18556))
        with self.assertRaises(PS.ServeError):
            PS.parse_serve_argv(['/usr/bin/python', '-m', 'http.server'])

    def test_a_vllm_server_outside_the_state_files_blocks_the_start(self):
        stray = subprocess.Popen(
            [str(self.v033 / 'bin' / 'vllm'), 'serve', str(self.fb.root / 'ckpt' / self.cid)]
            + PL.PRESETS['rtx5090-tp2'].vllm_args(self.box.port + 9),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.addCleanup(stray.wait)
        self.addCleanup(stray.kill)
        with self.assertRaises(PS.ServeError) as caught:
            PS.serve_one(PL.PRESETS['rtx5090-tp2'], self.cid, self.box)
        self.assertIn('outside the probe state files', str(caught.exception))
        self.assertIn(str(stray.pid), str(caught.exception))

    def test_an_exited_replica_that_nobody_reaped_is_not_running(self):
        child = subprocess.Popen([sys.executable, '-c', 'pass'])
        self.addCleanup(child.wait)
        deadline = time.time() + 30
        while PS._stat(child.pid)[:1] != ['Z'] and time.time() < deadline:
            time.sleep(0.01)
        os.kill(child.pid, 0)                               # the test podctl uses: still "alive"
        pid_file = self.tmp / 'serve.0.pid'
        pid_file.write_text(str(child.pid))
        self.assertIsNone(PS._pid_from(str(pid_file)))
        pid_file.write_text(str(os.getpid()))
        self.assertEqual(PS._pid_from(str(pid_file)), os.getpid())

    def test_busy_gpus_stop_the_serve(self):
        self.fb.set_gpus(4, used_mib=30000)
        with self.assertRaises(PS.ServeError) as caught:
            PS.wait_gpus_free(self.box)
        self.assertIn('GPUs not free', str(caught.exception))


class TestEvalJob(BoxCase):
    def job(self, plan, probe_plan):
        job = self.tmp / 'job'
        job.mkdir(exist_ok=True)
        (job / 'eval-plan.json').write_text(json.dumps(probe_plan))
        (job / PS.LAYOUTS_NAME).write_text(PL.dumps(plan))
        return job

    def run_eval(self, job):
        return PS.eval_job(str(job), self.cid, str(job / 'eval-plan.json'), str(job / PS.LAYOUTS_NAME), self.box)

    def test_eval_writes_the_documents_of_the_live_detached_eval(self):
        probe_plan = self.probe_plan()
        job = self.job(PL.shadow_plan('RTX5090'), probe_plan)
        self.assertEqual(self.run_eval(job), 0)

        marker = json.loads((job / 'markers' / 'eval.done').read_text())
        self.assertEqual(json.loads((job / 'RESULT.json').read_text()), marker)
        self.assertEqual({k: marker[k] for k in ('ok', 'status', 'candidate_id', 'run_name')},
                         dict(ok=True, status='ok', candidate_id=self.cid, run_name=probe_plan['run_name']))
        result = marker['probe_result']
        self.assertIs(result['partial'], False)
        self.assertEqual((result['checkpoint_id'], result['samples'], result['seed']), (self.cid, 16 * 8, 'seed-1'))
        served = result['serve_layout']
        self.assertEqual(served['layout'], PL.PRESETS['rtx5090-tp2'].as_dict())
        self.assertEqual(served['queue']['task_queue'], 'shared')
        self.assertTrue(served['queue']['as_planned'])
        self.assertNotIn('packages', served)
        # The worker ran the plan the host wrote; only the client load follows the layout.
        ran = json.loads((self.fb.root / 'runs' / probe_plan['run_name'] / 'plan.json').read_text())
        for key in ('seed', 'harness', 'manifest_projection', 'checkpoint_id', 'grade_workers'):
            self.assertEqual(ran[key], probe_plan[key], key)
        self.assertEqual([{k: v for k, v in env.items() if k not in ('spec', 'kind')} for env in ran['envs']],
                         probe_plan['envs'])
        self.assertEqual(ran['concurrency_per_replica'], 112)

    def test_a_failure_still_writes_the_marker(self):
        job = self.job(PL.shadow_plan('RTX5090'), self.probe_plan())
        shutil.rmtree(self.fb.root / 'ckpt' / self.cid)
        self.assertEqual(self.run_eval(job), 1)
        marker = json.loads((job / 'markers' / 'eval.done').read_text())
        self.assertEqual((marker['ok'], marker['status']), (False, 'failed'))
        self.assertIn('no layout of the plan serves', marker['error'])
        self.assertNotIn('probe_result', marker)

    def test_a_plan_on_the_box_that_asks_for_fp8_is_refused_before_anything_serves(self):
        for setting in ({'kv_cache_dtype': 'fp8'}, {'quantization': 'fp8'}):
            job = self.job(PL.shadow_plan('RTX5090'), self.probe_plan())
            doc = json.loads((job / PS.LAYOUTS_NAME).read_text())
            doc['layouts'][0].update(setting)
            (job / PS.LAYOUTS_NAME).write_text(json.dumps(doc))
            self.assertEqual(self.run_eval(job), 1)
            marker = json.loads((job / 'markers' / 'eval.done').read_text())
            self.assertEqual((marker['ok'], marker['status']), (False, 'failed'))
            self.assertIn('the probe has no FP8', marker['error'])
            self.assertEqual(self.fb.started(), [])

    def test_a_probe_past_its_wall_limit_is_closed_as_partial(self):
        self.fb.control(delay_ms=4000)
        layout = PL.layout_from({'preset': 'rtx5090-tp2', 'name': 'short', 'probe_wall_s': 1})
        job = self.job(PL.ProbePlan((layout,)), self.probe_plan(layout=layout))
        self.assertEqual(self.run_eval(job), 0)
        marker = json.loads((job / 'markers' / 'eval.done').read_text())
        self.assertEqual((marker['ok'], marker['status']), (False, 'finalized'))
        self.assertIs(marker['probe_result']['partial'], True)

    def test_the_tool_runs_from_the_job_directory_as_the_host_ships_it(self):
        """ship() + detached_command(): the copies on the box run outside the repository, on a plain python3."""
        job = self.tmp / 'job'
        probe_plan = self.probe_plan()

        def write_text(path, text):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)

        paths = PS.ship(write_text, str(job) + '/', PL.shadow_plan('RTX5090'))
        self.assertEqual(sorted(os.path.basename(p) for p in paths.values()),
                         ['probe-layouts.json', 'probe_layout.py', 'probe_serve.py'])
        write_text(str(job / 'eval-plan.json'), json.dumps(probe_plan))
        command = PS.detached_command(
            preamble='set -e\n', job_dir=str(job), candidate_id=self.cid, plan_path=str(job / 'eval-plan.json'),
            root=str(self.fb.root), box_args=self.box.args()[2:])
        proc = subprocess.run(['bash', '-c', command], capture_output=True, text=True, timeout=600, cwd=str(self.tmp),
                              env={'PATH': '%s:/usr/bin:/bin' % self.fb.bin, 'HOME': str(self.tmp)})
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
        marker = json.loads((job / 'markers' / 'eval.done').read_text())
        self.assertEqual((marker['ok'], marker['status']), (True, 'ok'))
        self.assertEqual(marker['probe_result']['serve_layout']['layout']['name'], 'rtx5090-tp2')

    def test_files_that_the_host_cannot_send_are_refused(self):
        with mock.patch.object(PS, 'MAX_SHIP_BYTES', 100):
            with self.assertRaises(PS.ServeError):
                PS.ship(lambda path, text: None, '/job', PL.shadow_plan('RTX5090'))
        for name in PS.TOOL_FILES:
            text = (Path(PS.__file__).parent / name).read_text()
            self.assertNotIn('EOF', text.splitlines())      # write_text sends a file as a heredoc that ends at EOF
            self.assertLess(len(text.encode()), PS.MAX_SHIP_BYTES)

    def test_the_live_preamble_is_the_clean_up_of_the_live_command(self):
        from_live = PS.live_preamble('set -u\npgrep -f "[v]llm serve" | xargs -r kill\npython3 - a b <<\'PY\'\nprint(1)\nPY\n')
        self.assertEqual(from_live, 'set -u\npgrep -f "[v]llm serve" | xargs -r kill\n')
        for text in ('python3 - a\n', 'echo no driver\n', 'pgrep x; python3 - a'):
            with self.assertRaises(PS.ServeError):
                PS.live_preamble(text)


if __name__ == '__main__':
    unittest.main()
