"""Detached eval plan must equal live pe_lane.run_probe plan for same inputs."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from ops.parallel_eval.live.pe_probe_plan import build_probe_plan


CID = 'c' * 64
SEED = 'plan-seed-xyz'
INDICES = [3, 7, 11, 19]
HARNESS = {
    'harness': {'version': 'public-v1', 'tasks': ['a']},
    'mimo_answer_projection': {'mode': 'mimo'},
    'sampling_contract': {'temp': 0.0},
}


class TestDetachedEvalPlanParity(unittest.TestCase):
    def test_build_probe_plan_includes_harness_projection(self):
        plan = build_probe_plan(
            CID, INDICES, SEED, k=8, max_tokens=65536, max_num_seqs=512, harness=HARNESS,
        )
        self.assertEqual(plan['checkpoint_id'], CID)
        self.assertEqual(plan['seed'], SEED)
        self.assertEqual(plan['envs'][0]['indices'], INDICES)
        self.assertEqual(plan['harness'], HARNESS['harness'])
        self.assertEqual(
            plan['manifest_projection'],
            {
                'mimo_answer_projection': HARNESS['mimo_answer_projection'],
                'sampling_contract': HARNESS['sampling_contract'],
            },
        )

    def test_detached_plan_equals_live_plan_same_inputs(self):
        """BoxRunner._arm_detached_eval plan keys/values equal live for same inputs."""
        from ops.parallel_eval.pit.box import BoxRunner
        from ops.parallel_eval.pit.config import load_pit_config

        live = build_probe_plan(
            CID, INDICES, SEED, k=8, max_tokens=65536, max_num_seqs=512, harness=HARNESS,
        )
        written = {}

        class _Store:
            def write_text(self, path, text):
                written['path'] = path
                written['plan'] = json.loads(text)

        class _Ctrl:
            job_dir = '/workspace/pst/pit-jobs/x'
            store = _Store()

        cfg = load_pit_config({
            'pit': {
                'enabled': True, 'mode': 'live', 'cutover': True,
                'k_cand': 8, 'max_output_tokens': 65536, 'max_num_seqs': 512,
            },
        })
        runner = BoxRunner(config=cfg, io=None, lane=None)
        with mock.patch('ops.parallel_eval.pit.box._load_public_harness', return_value=HARNESS):
            cmd = runner._arm_detached_eval(
                _Ctrl(), candidate_id=CID, plan_handle=SEED, indices=list(INDICES),
            )
        detached = written['plan']
        self.assertEqual(
            json.dumps(detached, sort_keys=True),
            json.dumps(live, sort_keys=True),
        )
        for key in (
            'run_name', 'checkpoint_id', 'seed', 'concurrency', 'grade_workers',
            'resume', 'bandit_topk', 'concurrency_per_replica', 'envs',
            'harness', 'manifest_projection',
        ):
            self.assertIn(key, detached)
            self.assertEqual(detached[key], live[key])
        self.assertIn('eval-plan.json', written['path'])
        self.assertIn('serve', cmd)
        self.assertIn('finalize-run', cmd)

    def test_pe_lane_run_probe_uses_shared_builder(self):
        """pe_lane.run_probe must call build_probe_plan (not a hand-copied dict)."""
        live_dir = Path(__file__).resolve().parents[1] / 'live'
        # Load pe_probe_plan under bare name as pe_lane does via sys.path.
        if str(live_dir) not in sys.path:
            sys.path.insert(0, str(live_dir))
        # Stub host-only deps so pe_lane imports.
        rp = types.ModuleType('replace_pod')

        class _Pod:
            pass

        rp.Pod = _Pod
        stubs = {
            'replace_pod': rp,
            'pe_rent': types.ModuleType('pe_rent'),
        }
        # Minimal pe_common stub with symbols pe_lane imports at module level.
        pc = types.ModuleType('pe_common')
        pc.PRIV = live_dir
        pc.rj = lambda *a, **k: {}
        pc.GATE_CODE = live_dir
        pc.RA = live_dir
        pc.PODD = '/root/affine-env-probe'
        pc.LOGS = live_dir
        pc.ST = live_dir
        pc.HOST_PY = sys.executable
        pc.cr = lambda: 'cr'
        pc.append = lambda *a, **k: None
        pc.scrub = lambda s: s
        stubs['pe_common'] = pc
        with mock.patch.dict(sys.modules, stubs):
            for name in ('pe_lane', 'pe_probe_plan'):
                sys.modules.pop(name, None)
            # Ensure pe_probe_plan is the real module.
            spec = importlib.util.spec_from_file_location(
                'pe_probe_plan', live_dir / 'pe_probe_plan.py',
            )
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules['pe_probe_plan'] = mod
            spec.loader.exec_module(mod)
            import pe_lane as PL  # type: ignore
            self.assertTrue(callable(getattr(PL, 'build_probe_plan', None)))
            # run_probe source must reference build_probe_plan.
            src = Path(PL.__file__).read_text()
            self.assertIn('build_probe_plan', src)
            self.assertIn('from pe_probe_plan import build_probe_plan', src)
            # Shared builder equality with harness injected as pe_lane would.
            live = mod.build_probe_plan(
                CID, INDICES, SEED, k=8, max_tokens=64, max_num_seqs=512, harness=HARNESS,
            )
            via_lane = PL.build_probe_plan(
                CID, INDICES, SEED, k=8, max_tokens=64, max_num_seqs=512, harness=HARNESS,
            )
            self.assertEqual(live, via_lane)

    def test_detached_eval_command_mentions_serve_wait_and_finalize(self):
        from ops.parallel_eval.live.pe_box import detached_eval_command
        cmd = detached_eval_command(
            job_dir='/workspace/pst/pit-jobs/x',
            candidate_id=CID,
            plan_path='/workspace/pst/pit-jobs/x/eval-plan.json',
        )
        self.assertIn('serve', cmd)
        self.assertIn('replicas_up', cmd)
        self.assertIn('max-model-len 69632', cmd)
        self.assertIn('max-num-seqs 512', cmd)
        self.assertIn('bfloat16', cmd)
        self.assertIn('finalize-run', cmd)
        self.assertIn("status == 'ok'", cmd)
        self.assertIn('eval.done', cmd)
        self.assertIn('failed', cmd)
        # Restart mid-eval: poll nvidia-smi until free (≤5 min), not sleep 2.
        self.assertIn('wait_gpus_free', cmd)
        self.assertIn('nvidia-smi --query-gpu=memory.used', cmd)
        self.assertIn('timeout_s=300', cmd)
        self.assertNotIn('; sleep 2; python3 -', cmd)
        # Must not self-kill via literal pgrep on embedded cmdline patterns.
        self.assertNotIn(
            "pgrep -f 'train_pit.py|v83_dp.py|vllm serve|podctl.py serve'",
            cmd,
        )
        self.assertIn('[v]llm serve', cmd)
        self.assertIn('query-compute-apps=pid', cmd)


if __name__ == '__main__':
    unittest.main()
