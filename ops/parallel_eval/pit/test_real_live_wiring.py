#!/usr/bin/env python3
"""Integration: real live pe_rent / pe_lane / pe_score with fake lium + fake ssh.

NO PIT-level mocks of rent/lane/score. Only the CLI binaries (lium, ssh) are
faked so the live modules can run off-box.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


LIVE = Path(__file__).resolve().parents[1] / 'live'
PODS = Path(__file__).resolve().parents[2] / 'pods'


def _write_exec(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestRealLiveWiring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.state = self.root / 'pe-state'
        self.state.mkdir()
        (self.state / 'pit').mkdir()
        self.boxfs = self.root / 'boxfs'
        self.boxfs.mkdir()
        roles = PODS / 'roles.json'
        self.env = {
            'PATH': '%s:%s' % (self.bin, os.environ.get('PATH', '')),
            'PE_STATE': str(self.state),
            'PE_ST': str(self.state),
            'PIT_STATE': str(self.state / 'pit'),
            'LIUM': str(self.bin / 'lium'),
            'HOME': str(self.root),
            'PODS_ROLES': str(roles),
            'PYTHONPATH': '%s:%s' % (LIVE, PODS),
        }
        pod = {
            'id': '11111111-2222-3333-4444-555555555555',
            'name': 'affine-pe-pit-7-a1',
            'huid': 'exec-b300-test',
            'ip': '127.0.0.1',
            'ssh_port': 2222,
            'port': 2222,
            'status': 'RUNNING',
            'price_per_hour': 40.0,
            'gpu_count': 8,
            'available_gpu_count': 8,
            'config': 'B300',
            'gpu_model': 'B300',
            'ssh_command': 'ssh -p 2222 root@127.0.0.1',
        }
        self.pod = pod
        ls_row = {
            'huid': 'exec-b300-test',
            'name': 'offer-b300',
            'price_per_hour': 40.0,
            'gpu_count': 8,
            'available_gpu_count': 8,
            'country_code': 'US',
            'tier': 't1',
            'gpu_model': 'B300',
        }
        _write_exec(
            self.bin / 'lium',
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json, sys
                argv = sys.argv[1:]
                POD = {pod}
                LS = {ls}
                if argv[:1] == ["ls"]:
                    print(json.dumps([LS])); sys.exit(0)
                if argv[:1] == ["ps"]:
                    print(json.dumps([POD])); sys.exit(0)
                if argv[:1] == ["up"]:
                    print(json.dumps({{"pod": POD}})); sys.exit(0)
                if argv[:1] in (["rm"], ["down"], ["stop"]):
                    print(json.dumps({{"ok": True}})); sys.exit(0)
                print("unknown", argv, file=sys.stderr); sys.exit(2)
                """
            ).format(pod=json.dumps(pod), ls=json.dumps(ls_row)),
        )
        ssh = textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -e
            BOXFS="__BOXFS__"
            while [[ $# -gt 0 ]]; do
              case "$1" in
                -p|-o|-i|-l|-F) shift 2 ;;
                -T|-n|-A|-X|-Y|-tt|-f|-q|-v) shift ;;
                root@*|*[0-9].[0-9]*) shift; break ;;
                *) break ;;
              esac
            done
            CMD="$*"
            mkdir -p "$BOXFS"
            if [[ "$CMD" == *"nvidia-smi"* ]]; then echo "NVIDIA B300"; exit 0; fi
            if [[ "$CMD" == *"podctl.py serve"* ]]; then echo '{"ok": true}'; exit 0; fi
            if [[ "$CMD" == *"podctl.py status"* ]]; then
              echo '{"state":"serving","ready":true,"replicas_up":8,"replicas":8,"served":"'"${SERVE_CID:-cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc}"'"}'
              exit 0
            fi
            if [[ "$CMD" == *"/v1/models"* ]]; then echo '{"data":[{"id":"m"}]}'; exit 0; fi
            if [[ "$CMD" == *"vllm serve"* ]] || [[ "$CMD" == *"ps -eo"* ]]; then
              echo "vllm serve /ckpt/cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc --dtype bfloat16 --max-model-len 69632 --max-num-seqs 512"
              exit 0
            fi
            eval "cd \\"$BOXFS\\" && $CMD" || true
            exit 0
            """
        ).replace('__BOXFS__', str(self.boxfs))
        _write_exec(self.bin / 'ssh', ssh)
        (self.root / '.lium' / 'known_hosts').mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _prep_sys_path(self):
        for p in (str(LIVE), str(PODS)):
            if p not in sys.path:
                sys.path.insert(0, p)

    def _load_live_rent(self):
        self._prep_sys_path()
        import pe_common
        import pe_rent
        pe_common.LIUM = str(self.bin / 'lium')
        pe_common.ST = self.state
        pe_common.HOME = self.root
        pe_rent.ST = self.state
        pe_rent.PJ = self.state / 'pods.json'
        return pe_rent

    def test_rent_name_must_be_affine_pe_prefix(self):
        """Live pe_rent asserts affine-pe-*; PIT PREFIX affine-pe-pit- must pass."""
        with mock.patch.dict(os.environ, self.env, clear=False):
            pe_rent = self._load_live_rent()
            with mock.patch.object(pe_rent.R, 'ssh_ok', return_value=True):
                with self.assertRaises(AssertionError):
                    pe_rent.rent('affine-pit-bad', ttl='4h', gpu='B300', max_eval_nodes=5)
                out = pe_rent.rent('affine-pe-pit-7-a1', ttl='4h', gpu='B300', max_eval_nodes=5)
            self.assertIsNotNone(out)
            self.assertTrue(str(out['name']).startswith('affine-pe-'))
            self.assertEqual(out['id'], self.pod['id'])
            self.assertIn('ip', out)
            self.assertIn('port', out)

    def test_pit_rent_service_uses_live_pe_rent_and_catches_errors(self):
        from ops.parallel_eval.pit import PREFIX
        from ops.parallel_eval.pit.pe_rent import RentService, _bind_offer_precheck

        self.assertTrue(PREFIX.startswith('affine-pe-'))
        with mock.patch.dict(os.environ, self.env, clear=False):
            pe_rent = self._load_live_rent()
            with mock.patch.object(pe_rent.R, 'ssh_ok', return_value=True):
                rent = RentService(
                    require_host=True, max_boxes=2, max_eval_nodes=5,
                    price_ceiling_usd_h=56.0,
                )
                rent.host_rent = pe_rent.rent
                rent.host_release = getattr(pe_rent, 'release', lambda *a, **k: None)
                rent.list_pods = pe_rent.our_pods
                _bind_offer_precheck(rent, pe_rent)
                out = rent.rent(item_id='a' * 32, epoch=7, attempt=1)
            self.assertEqual(out['name'], '%s7-a1' % PREFIX)
            self.assertTrue(out.get('auto_backup_disabled'))
            self.assertEqual(out.get('gpu') or out.get('gpu_model'), 'B300')

    def test_offer_precheck_rejects_over_ceiling(self):
        from ops.parallel_eval.pit.pe_rent import RentError, RentService

        rent = RentService(price_ceiling_usd_h=56.0)

        def _over(*, gpu='B300'):
            return dict(huid='x', price_per_hour=99.0, gpu=gpu, available_gpu_count=8)

        rent.offer_precheck_fn = _over
        with self.assertRaises(RentError):
            rent.precheck_offer()

    def test_pe_lane_serve_requires_pod_handle_not_string(self):
        cid = 'c' * 64
        with mock.patch.dict(os.environ, self.env, clear=False):
            self._prep_sys_path()
            import pe_common
            pe_common.LIUM = str(self.bin / 'lium')
            pe_common.HOME = self.root
            P = pe_common.pod_of(self.pod)
            import pe_lane

            status = json.dumps({
                'state': 'serving', 'ready': True,
                'replicas_up': 8, 'replicas': 8, 'served': cid,
            })
            models = json.dumps({'data': [{'id': 'm'}]})
            vllm = (
                '/usr/bin/vllm serve /ckpt/%s --dtype bfloat16 '
                '--max-model-len 69632 --max-num-seqs 512'
            ) % cid

            def _run(cmd, *a, **kw):
                c = str(cmd)
                if 'podctl.py status' in c:
                    return status
                if '/v1/models' in c:
                    return models
                if 'vllm serve' in c or 'ps -eo' in c:
                    return '\n'.join([vllm] * 8)
                if 'podctl.py serve' in c:
                    return '{"ok": true}'
                return ''

            with mock.patch.object(P, 'run', side_effect=_run):
                with mock.patch.object(pe_lane.time, 'sleep', return_value=None):
                    out = pe_lane.serve(P, cid, max_num_seqs=512)
            self.assertIn('serve_s', out)
            with self.assertRaises(Exception):
                pe_lane.serve('box-id-string', cid, max_num_seqs=512)

    def test_pe_score_imports_real_live_module(self):
        repo = Path(__file__).resolve().parents[3]
        gate = repo / 'miner' / 'validator' / 'gate'
        peg = repo / 'ops' / 'gate_probe_error_guard'
        with mock.patch.dict(os.environ, self.env, clear=False):
            self._prep_sys_path()
            import pe_common
            pe_common.GATE_CODE = gate
            for p in (str(gate), str(peg)):
                if p not in sys.path:
                    sys.path.insert(0, p)
            import pe_score
            self.assertTrue(callable(getattr(pe_score, 'score', None)))
            self.assertEqual(Path(pe_score.__file__).resolve(), (LIVE / 'pe_score.py').resolve())

    def test_pit_state_disagree_fail_closed(self):
        from ops.parallel_eval.pit.durable import pit_state_root
        with mock.patch.dict(os.environ, {
            'PIT_STATE': str(self.root / 'other-pit'),
            'PE_STATE': str(self.state),
        }, clear=False):
            with self.assertRaises(RuntimeError):
                pit_state_root()
        with mock.patch.dict(os.environ, {
            'PIT_STATE': str(self.state / 'pit'),
            'PE_STATE': str(self.state),
        }, clear=False):
            self.assertEqual(pit_state_root().resolve(), (self.state / 'pit').resolve())

    def test_pe_live_dispatch_noop_only_on_cutover(self):
        """Shadow enabled must NOT stop live dispatch; cutover must."""
        src = (LIVE / 'pe_live.py').read_text()
        self.assertIn('if _pit_cutover():', src)
        self.assertNotIn('if _pit_enabled():\n        return {"pit": "dispatch_noop"', src)
        self.assertIn(
            'pit.get("enabled") is True and pit.get("mode") == "live" and pit.get("cutover") is True',
            src,
        )


if __name__ == '__main__':
    unittest.main()
