"""The entry points that a controller calls are as docs/PIT_BOX_ENTRY_POINTS.md says.

A controller (the BoxAgent of the batch controller) uses the train command text, the box paths, the command
lines of the box tools and their JSON records. This test runs the real tools and compares them with the
document: a change of an entry point fails here until the document and the callers change too.
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

from ops.parallel_eval import pit
from ops.parallel_eval.durable import canonical
from ops.parallel_eval.live import gate_fetch
from ops.parallel_eval.live import pe_artifact_upload as AU
from ops.parallel_eval.live import pe_box
from ops.parallel_eval.live import pe_probe_plan
from ops.parallel_eval.live import pe_r2_box_upload as BOX
from ops.parallel_eval.pit import box as pit_box
from ops.parallel_eval.pit import (
    box_job, fakepass_sandbox, gpu_policy, pe_score, probe_fingerprint, probe_layout, probe_serve, queue_ingest, shadow,
)

DOC = (Path(__file__).resolve().parents[3] / 'docs' / 'PIT_BOX_ENTRY_POINTS.md').read_text()
DIGEST = 'a' * 64
# The two rules of the controller for a train command (recipe.check_train_cmd of the batch controller).
SETS_EXPECTED_MASTER = re.compile(r'AFFINE_FP32_EXPECTED_MASTER_HASH\s*=\s*["\']?([^\s"\';]*)')
SETS_OVERRIDE = re.compile(r'BOOTSTRAP_OVERRIDE\s*=')


def run_tool(main, argv):
    """A command line of a box tool. -> (exit code, standard output, standard error)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class TestDocument(unittest.TestCase):
    def assert_documented(self, names):
        """Each name is in the document as a word of its own."""
        missing = [name for name in names
                   if re.search(r'(?<![A-Za-z0-9_-])%s(?![A-Za-z0-9_-])' % re.escape(name), DOC) is None]
        self.assertEqual(missing, [])

    def test_a_name_that_is_not_in_the_document_is_found(self):
        with self.assertRaises(AssertionError):
            self.assert_documented(['--settle'])           # only a part of the option --settle-s

    def test_the_box_paths_and_the_job_files_are_the_documented_ones(self):
        self.assertEqual((pe_box.PST, pe_box.TW, pe_box.FP32, pe_box.KDIR, pe_box.BOX_ROOT), (
            '/workspace/pst', '/workspace/affine-mimo-trainer-prod', '/workspace/affine-mimo-trainer-prod/fp32-master',
            '/opt/plain/probe/affine-env-probe/ckpt', '/root/affine-env-probe'))
        self.assertEqual(probe_layout.BOX_ROOT, pe_box.BOX_ROOT)
        job = box_job.job_dir_for('item-1')
        self.assertEqual(job, '/workspace/pst/pit-jobs/item-1')
        self.assertEqual(box_job.marker_path(job, 'train'), job + '/markers/train.done')
        self.assertEqual(box_job.STAGES, ('download', 'train', 'eval'))
        files = (box_job.RESULT_NAME, box_job.PID_NAME, box_job.STAGE_NAME, box_job.LOG_NAME, box_job.CLAIM_NAME,
                 box_job.CLAIM_LOCK_NAME)
        self.assertEqual(files, ('RESULT.json', 'job.pid', 'job.stage', 'job.log', 'start.claim', 'start.flock'))
        self.assertEqual(box_job.START_CLAIM_TTL_S, 300.0)
        self.assert_documented(files + (pe_box.PST, pe_box.FP32, pe_box.KDIR, pe_box.BOX_ROOT, 'STARTED <pid>', 'BUSY <pid>'))

    def test_the_train_text_keeps_the_fp32_rules_of_the_controller(self):
        text = pe_box._train_pit_inner_cmd(expected_master=DIGEST)
        self.assertEqual(SETS_EXPECTED_MASTER.findall(text), [DIGEST])
        self.assertIsNone(SETS_OVERRIDE.search(text))
        self.assertIn('env -u AFFINE_FP32_BOOTSTRAP_OVERRIDE ', text)
        for part in ('rm -rf /workspace/pst/out', '/workspace/pst/train_pit.py > /workspace/pst/train.log 2>&1',
                     'echo rc=$? | tee -a /workspace/pst/train.log'):
            self.assertIn(part, text)
        detached = pe_box.detached_train_command(expected_master=DIGEST, job_dir='/workspace/pst/pit-jobs/item-1/')
        self.assertIn('(%s); ' % text, detached)
        self.assertIn('/workspace/pst/pit-jobs/item-1/markers/train.done', detached)
        # The trainer refuses the override and a missing digest itself; the admission check stays in it.
        trainer = (Path(pe_box.__file__).with_name('train_pit.py')).read_text()
        for part in ("assert len(exp) == 64", "assert not os.environ.get('AFFINE_FP32_BOOTSTRAP_OVERRIDE')",
                     "host.checked(HOST / 'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json')",
                     "assert fp['input_master_digest'] == exp"):
            self.assertIn(part, trainer)
        events = re.findall(r"ev\('([a-z0-9-]+)'", trainer)
        self.assertEqual(events, [
            'root-admission-checked', 'zero3-policy', 'native-installed', 'zero3-installed', 'actor-loaded',
            'vram-preflight-ok', 'vram-preflight-refuse', 'optimizer-built', 'signed-dispatch-prepared',
            'native-grpo-complete', 'zero3-microbatch', 'staged', 'done'])
        self.assert_documented([name for name in events if name != 'vram-preflight-refuse'])

    def test_the_names_that_the_controller_loads_keep_their_arguments(self):
        calls = {
            pe_box.boot_helpers: 'P', pe_box.push_probe_bundle: 'P', pe_box.bundle_trainer: 'P',
            pe_box._ship_train_pit: 'P', pe_box.ship_inputs: 'P, *, epoch_dir, epoch, work_jd',
            pe_box._train_pit_inner_cmd: '*, expected_master',
            pe_box.detached_train_command: '*, expected_master, job_dir',
            pe_box.put: "P, dest, data, mode='644'", pe_box.put_tar: 'P, local_dir, remote_dir',
            pe_box.fetch: 'P, cid, files, urls, dest, threads=32',
            probe_serve.ship: 'write_text, job_dir, plan, *, read_text=None',
            probe_serve.detached_command:
                "*, preamble, job_dir, candidate_id, plan_path, root='/root/affine-env-probe', box_args=()",
            pe_probe_plan.build_probe_plan:
                "cid, indices, plan_seed, k=8, max_tokens=65536, env_id='affine_gatepool', run_name=None, "
                "max_num_seqs=512, harness=None",
            probe_fingerprint.of: 'doc', gpu_policy.effective_gpu_order: 'order=None, *, allow_sharded_gpus=False',
            pit_box._load_public_harness: '',
            pe_score.call_live_host_score: 'host_score, *, epoch, round_id, result, baseline_of=None',
        }
        for call, arguments in calls.items():
            signature = inspect.signature(call)
            plain = [p.replace(annotation=inspect.Parameter.empty) for p in signature.parameters.values()]
            got = str(signature.replace(parameters=plain, return_annotation=inspect.Signature.empty))
            self.assertEqual(got, '(%s)' % arguments, call.__name__)
        self.assert_documented([call.__name__ for call in calls])
        self.assertTrue(issubclass(probe_fingerprint.FingerprintError, Exception))
        # Constants and names of the PIT package that the controller imports.
        self.assertTrue(set(pit.SHARDED_GPUS) <= set(pit.ALLOWED_GPUS))
        self.assertEqual([type(value) for value in (pit.ALLOWED_GPUS, pit.SHARDED_GPUS, pit.REQUIRED_GPU_COUNT,
                                                    pit.PRICE_CEILING_USD_H, pit.ALLOW_SHARDED_GPUS_DEFAULT)],
                         [tuple, tuple, int, float, bool])
        self.assertTrue(issubclass(shadow.ShadowWriteError, Exception))
        self.assertTrue(issubclass(fakepass_sandbox.FakepassError, BaseException))
        more = (fakepass_sandbox.assert_paths_outside_live, fakepass_sandbox.live_roots_from_env, canonical,
                queue_ingest._epoch_index_of, queue_ingest._above_floor_or_pit_open, queue_ingest._epoch_dir_ready)
        for call in more:
            self.assertTrue(callable(call))
        self.assert_documented([call.__name__ for call in more] + [
            'ALLOWED_GPUS', 'SHARDED_GPUS', 'REQUIRED_GPU_COUNT', 'PRICE_CEILING_USD_H', 'ALLOW_SHARDED_GPUS_DEFAULT',
            'ShadowWriteError', 'FakepassError'])

    def test_the_host_only_names_that_the_controller_loads_are_in_the_source(self):
        """pe_common and the live pe_score load on the validator host only: the test reads their source."""
        def arguments(function):
            spec = function.args
            return [a.arg for a in spec.posonlyargs + spec.args] + (['*' + spec.vararg.arg] if spec.vararg else []) + [
                a.arg for a in spec.kwonlyargs]

        found, star = {}, []
        for module in ('pe_common', 'pe_score'):
            tree = ast.parse(Path(pe_box.__file__).with_name(module + '.py').read_text())
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    found['%s.%s' % (module, node.name)] = arguments(node)
                elif isinstance(node, ast.Assign):
                    found.update(('%s.%s' % (module, t.id), None) for t in node.targets if isinstance(t, ast.Name))
                elif isinstance(node, ast.ImportFrom) and [alias.name for alias in node.names] == ['*']:
                    star.append((module, node.module))
        self.assertEqual(found['pe_common.lium'], ['*a', 'timeout'])
        self.assertEqual(found['pe_common.pod_of'], ['d'])
        self.assertEqual(found['pe_score.score'][:4], ['epoch', 'rid', 'result', 'flag'])
        # pe_score has each name of pe_common: HELDOUT of pe_score is the one of pe_common.
        self.assertIn('pe_common.HELDOUT', found)
        self.assertIn(('pe_score', 'pe_common'), star)
        self.assert_documented(['live/pe_common.lium', 'live/pe_common.pod_of', 'live/pe_score.score',
                                'live/pe_score.HELDOUT'])

    def test_the_probe_command_text_and_the_plan_are_what_the_controller_uses(self):
        text = probe_serve.detached_command(preamble='', job_dir='/workspace/batchctl/f1/probe/', candidate_id=DIGEST,
                                            plan_path='/workspace/batchctl/f1/plan.json')
        self.assertEqual(text, 'python3 /workspace/batchctl/f1/probe/probe_serve.py --root /root/affine-env-probe eval '
                               '--job-dir /workspace/batchctl/f1/probe --candidate %s '
                               '--plan /workspace/batchctl/f1/plan.json '
                               '--layouts /workspace/batchctl/f1/probe/probe-layouts.json\n' % DIGEST)
        plan = probe_layout.plan_from(['rtx5090-tp2', 'rtx5090-1gpu'])
        self.assertEqual([layout.name for layout in plan.layouts], ['rtx5090-tp2', 'rtx5090-1gpu'])
        self.assertIs(plan.primary, plan.layouts[0])
        self.assertIsInstance(plan.eval_timeout_s(), int)
        self.assertIsInstance(plan.primary.concurrency(), int)
        written = {}
        paths = probe_serve.ship(written.__setitem__, '/workspace/batchctl/f1/probe/', plan)
        self.assertEqual(paths, {name: '/workspace/batchctl/f1/probe/' + name
                                 for name in ('probe_layout.py', 'probe_serve.py', 'probe-layouts.json')})
        self.assertEqual(sorted(written), sorted(paths.values()))
        layouts = json.loads(written[paths['probe-layouts.json']])
        self.assertEqual((layouts['version'], len(layouts['layouts'])), ('pit-probe-layout-v1', 2))

    def test_the_host_call_takes_the_output_prefix_and_refuses_the_override(self):
        self.assertEqual(list(inspect.signature(pe_box.HostBoxIO).parameters), ['output_prefix'])
        self.assertEqual(list(inspect.signature(pe_box.box_io_factory).parameters), ['output_prefix'])
        self.assertEqual(AU.OUTPUT_PREFIX_ENV, 'PIT_R2_OUTPUT_PREFIX')
        self.assertEqual(AU.output_prefix_env('pit-shadow/batchctl/run1'), {
            'PIT_R2_OUTPUT_PREFIX': 'pit-shadow/batchctl/run1/',
            'PIT_R2_PENDING_PREFIX': 'pit-shadow/batchctl/run1/pending',
            'PIT_R2_FP32_PREFIX': 'pit-shadow/batchctl/run1/fp32-master',
            'PIT_R2_CKPT_PREFIX': 'pit-shadow/batchctl/run1/ckpts'})
        for override in ('fp32_bootstrap_override', 'bootstrap_override'):
            with self.assertRaisesRegex(pe_box.BoxIOError, 'bootstrap-override-refused-by-pit'):
                pe_box.HostBoxIO().train_v834e(**{override: True})

    def test_the_probe_tool_has_the_documented_command_line(self):
        code, out, _ = run_tool(probe_serve.main, ['--help'])
        self.assertEqual(code, 0)
        options = sorted(set(re.findall(r'--[a-z][a-z-]+', out)) - {'--help'})
        self.assertEqual(options, ['--drain-s', '--free-mib', '--gpu-wait-s', '--poll-s', '--port', '--probe-poll-s',
                                   '--python', '--root', '--settle-s'])
        self.assertIn('{start,stop,eval,side}', out)
        needs = {
            'start': '--checkpoint, --layout, --venv',
            'eval': '--job-dir, --candidate, --plan, --layouts',
            'side': '--checkpoint, --plan, --layout, --out',
        }
        for command, arguments in needs.items():
            code, _, err = run_tool(probe_serve.main, [command])
            self.assertEqual(code, 2, command)
            self.assertIn('the following arguments are required: %s\n' % arguments, err, command)
            self.assert_documented(arguments.split(', '))
        self.assertIn('[--tag TAG]', run_tool(probe_serve.main, ['side', '--help'])[1])
        self.assert_documented(options + ['--tag'] + sorted(probe_layout.PRESETS))
        self.assertEqual(probe_serve.TOOL_FILES, ('probe_layout.py', 'probe_serve.py'))
        self.assertEqual((probe_serve.LAYOUTS_NAME, probe_layout.VERSION), ('probe-layouts.json', 'pit-probe-layout-v1'))
        self.assertEqual(probe_fingerprint.KEY, 'probe_fingerprint')
        # No FP8 in the probe: neither FP8 argument is one that a server can run with.
        self.assertFalse({'--kv-cache-dtype', '--quantization'} & set(probe_serve.ALLOWED_FLAGS))

    def test_a_failed_probe_writes_the_documented_marker_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, _ = run_tool(probe_serve.main, [
                '--root', tmp, 'eval', '--job-dir', tmp + '/job', '--candidate', DIGEST, '--plan', tmp + '/no-plan.json',
                '--layouts', tmp + '/no-layouts.json'])
            marker = json.loads(Path(tmp, 'job', 'markers', 'eval.done').read_text())
            result = json.loads(Path(tmp, 'job', 'RESULT.json').read_text())
        self.assertEqual(code, 1)
        self.assertEqual(marker, result)
        self.assertEqual(sorted(result), ['at', 'candidate_id', 'error', 'ok', 'status'])
        self.assertEqual((result['ok'], result['status'], result['candidate_id']), (False, 'failed', DIGEST))

    def test_the_upload_tool_has_the_documented_records(self):
        self.assertEqual(run_tool(BOX.main, ['send', 'x.json'])[0], 2)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'tree').mkdir()
            Path(tmp, 'tree', 'a.bin').write_bytes(b'abc')
            request, job = Path(tmp, 'request.json'), Path(tmp, 'job.json')
            request.write_text(json.dumps({'trees': {'t': {'dir': tmp + '/tree', 'names': None},
                                                     'u': {'dir': tmp + '/tree', 'names': ['a.bin', 'b.bin']}}}))
            code, out, _ = run_tool(BOX.main, ['stat', str(request)])
            stat = json.loads(out.strip().splitlines()[-1])
            job.write_text(json.dumps({'version': BOX.VERSION, 'files': []}))
            done_code, done_out, _ = run_tool(BOX.main, ['upload', str(job)])
            self.assertFalse(job.exists())                 # a job holds signed URLs: the tool removes it
        self.assertEqual(code, 1)                          # a file is missing
        self.assertEqual(stat, {'version': 'pit-r2-upload-v1', 'ok': False, 'trees': {'t': {'a.bin': 3}, 'u': {'a.bin': 3}},
                                'missing': ['u/b.bin']})
        manifest = json.loads(done_out.strip().splitlines()[-1])
        self.assertEqual(done_code, 0)
        self.assertEqual(sorted(manifest), ['bytes_reused', 'bytes_sent', 'errors', 'files', 'ok', 'seconds', 'version'])
        self.assert_documented(sorted(manifest) + sorted(stat) + ['hashed', 'sent', 'etags', 'reused', 'have_etag'])

    def test_the_fetch_tool_has_the_documented_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            urls = Path(tmp, 'urls.private.json')
            urls.write_text('{}')
            code, out, _ = run_tool(gate_fetch.main, [tmp + '/no-descriptor.json', tmp, str(urls)])
            self.assertFalse(urls.exists())                # the URL file is removed in each case
        record = json.loads(out.strip().splitlines()[-1])
        self.assertEqual(code, 1)
        self.assertEqual(sorted(record), ['attempts', 'ckpt', 'errno', 'errno_name', 'error', 'retryable', 'status'])
        self.assertEqual((record['status'], record['errno_name']), ('failed', 'ENOENT'))
        self.assertEqual(pe_box.FETCH_FAILED, 'PIT_FETCH_FAILED')
        self.assert_documented(sorted(record) + ['present', 'materialized', 'PIT_FETCH_FAILED rc=<n>',
                                                 'GATE_FETCH_ATTEMPTS', 'GATE_FETCH_BACKOFF_BASE', 'GATE_FETCH_BACKOFF_MAX'])


if __name__ == '__main__':
    unittest.main()
