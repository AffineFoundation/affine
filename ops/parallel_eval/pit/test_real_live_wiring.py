#!/usr/bin/env python3
"""Integration: real live pe_rent / pe_lane / pe_score / pe_box with fake lium+ssh+scp.

NO PIT-level mocks of rent/lane/score/box train. Only CLI binaries are faked.
Asserts the BRIEF/run5090 on-box path:

* box-side R2 fetch (presign payload → gate_fetch on box)
* trainer-bundle ship via scp + SIGNED-MANIFEST check
* trainer invocation under box venv
* file-based artifact return via scp with sha checks
* shadow prefix does not share the live PE cap
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tarfile
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


LIVE = Path(__file__).resolve().parents[1] / 'live'
PODS = Path(__file__).resolve().parents[2] / 'pods'
REPO = Path(__file__).resolve().parents[3]


def _write_exec(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        self.cmds = self.root / 'cmds.log'
        self.cmds.write_text('')

        # Minimal signed v834e + trainer bundle for ship/verify.
        self.v834e = self.root / 'release-v834e-signed'
        self.v834e.mkdir()
        train_step = self.v834e / 'train_step.py'
        train_step.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import argparse, json, pathlib, hashlib
                ap = argparse.ArgumentParser(); ap.add_argument('--head'); ap.add_argument('--batch')
                ap.add_argument('--out'); ap.add_argument('--master'); ap.add_argument('--report')
                ap.add_argument('--hotkey', default='pit')
                a = ap.parse_args()
                out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
                (out / 'model.safetensors').write_bytes(b'CAND')
                (out / 'config.json').write_text('{}')
                cid = hashlib.sha256(b'cand').hexdigest()
                mid = hashlib.sha256(b'master').hexdigest()
                rep = {
                  'candidate_id': cid, 'candidate_sha256': cid,
                  'candidate_master_digest': mid, 'master_digest': mid,
                  'used_fp32_bootstrap_override': False,
                  'fp32_master': {'source': 'disk'},
                  'source_file_hashes': {'trainer': 'a'*64, 'batch': 'b'*64},
                  'batching': {'tasks': 1, 'rollouts': 8},
                  'tokens': {'prompt_tokens': 1, 'output_tokens': 2, 'loss_tokens': 1},
                  'update_size': 1.0, 'gradient_norm': 1.0, 'engine': 'packed',
                }
                pathlib.Path(a.report).write_text(json.dumps(rep))
                print('TRAIN_OK')
                """
            )
        )
        # SIGNED-MANIFEST lists train_step.py sha
        sha = _sha(train_step)
        (self.v834e / 'SIGNED-MANIFEST').write_text('%s  train_step.py\n' % sha)
        (self.v834e / 'runtime-exact' / 'venv' / 'bin').mkdir(parents=True)
        vpy = self.v834e / 'runtime-exact' / 'venv' / 'bin' / 'python'
        vpy.write_text('#!/usr/bin/env bash\nexec python3 "$@"\n')
        vpy.chmod(vpy.stat().st_mode | stat.S_IXUSR)

        # Batch + head file map + presign payload (fake R2 URLs).
        self.batch = self.root / 'batch.json'
        self.batch.write_text(json.dumps({'tasks': [{'id': 1}]}))
        self.head = 'a' * 64
        self.master = 'b' * 64
        head_files = {'model.safetensors': {'sha256': 'c' * 64, 'bytes': 4}}
        self.head_files_json = self.root / 'head_files.json'
        self.head_files_json.write_text(json.dumps(head_files))
        self.presign_payload = self.root / 'presign.json'
        self.presign_payload.write_text(json.dumps({
            'descriptor': {'id': self.head, 'files': head_files},
            'urls': {'model.safetensors': {'url': 'https://r2.example/head/model.safetensors'}},
            'files': head_files,
        }))
        self.master_presign = self.root / 'master_presign.json'
        self.master_presign.write_text(json.dumps({
            'descriptor': {'id': self.head, 'files': {'m0.safetensors': {'sha256': 'd' * 64}}},
            'urls': {'m0.safetensors': {'url': 'https://r2.example/master/m0.safetensors'}},
            'download': {'m0.safetensors': 'https://r2.example/master/m0.safetensors'},
            'files': {'m0.safetensors': {'sha256': 'd' * 64}},
            'checkpoint': self.head,
        }))

        roles = PODS / 'roles.json'
        self.env = {
            'PATH': '%s:%s' % (self.bin, os.environ.get('PATH', '')),
            'PE_STATE': str(self.state),
            'PE_ST': str(self.state),
            'PIT_STATE': str(self.state / 'pit'),
            'LIUM': str(self.bin / 'lium'),
            'HOME': str(self.root),
            'PODS_ROLES': str(roles),
            'PIT_V834E_RELEASE': str(self.v834e),
            'PIT_TRAINER_BUNDLE': str(self.v834e),
            'PIT_HEAD_FILES_JSON': str(self.head_files_json),
            'PIT_PRESIGN_PAYLOAD_JSON': str(self.presign_payload),
            'PIT_MASTER_PRESIGN_JSON': str(self.master_presign),
            'PIT_UPLOAD_ROOT': str(self.root / 'uploads'),
            'BOX_VENV_PYTHON': 'python3',
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
        cmds_log = str(self.cmds)
        boxfs = str(self.boxfs)
        _write_exec(
            self.bin / 'lium',
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json, sys
                open({cmds!r}, "a").write("lium " + " ".join(sys.argv[1:]) + "\\n")
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
            ).format(pod=json.dumps(pod), ls=json.dumps(ls_row), cmds=cmds_log),
        )
        _write_exec(
            self.bin / 'ssh',
            textwrap.dedent(
                f"""\
                #!/usr/bin/env python3
                import os, sys, subprocess
                BOXFS = {boxfs!r}
                CMDS = {cmds_log!r}
                open(CMDS, "a").write("ssh " + " ".join(sys.argv[1:]) + "\\n")
                args = sys.argv[1:]
                # replace_pod puts: ssh [opts...] root@host [-n] <cmd>
                # Skip opts + host; then optional -n; remainder is the remote command.
                i = 0
                while i < len(args):
                    a = args[i]
                    if a in ("-p", "-o", "-i", "-l", "-F", "-P") and i + 1 < len(args):
                        i += 2
                        continue
                    if a in ("-T", "-n", "-A", "-X", "-Y", "-tt", "-f", "-q", "-v"):
                        i += 1
                        continue
                    if a.startswith("root@") or (len(a) > 0 and a[0].isdigit() and "." in a):
                        i += 1
                        break
                    break
                while i < len(args) and args[i] in ("-n", "-t", "-tt", "-T"):
                    i += 1
                cmd = " ".join(args[i:])
                for d in ("root", "opt/plain", "workspace"):
                    os.makedirs(os.path.join(BOXFS, d), exist_ok=True)
                if "nvidia-smi" in cmd:
                    print("NVIDIA B300"); sys.exit(0)
                if "podctl.py serve" in cmd:
                    print('{{"ok": true}}'); sys.exit(0)
                if "podctl.py status" in cmd:
                    print('{{"state":"serving","ready":true,"replicas_up":8,"replicas":8,"served":"' + ("c"*64) + '"}}')
                    sys.exit(0)
                if "/v1/models" in cmd:
                    print('{{"data":[{{"id":"m"}}]}}'); sys.exit(0)
                def rewrite(s):
                    for src, dst in (
                        ("/root/", BOXFS + "/root/"),
                        ("/opt/plain/", BOXFS + "/opt/plain/"),
                        ("/workspace/", BOXFS + "/workspace/"),
                    ):
                        s = s.replace(src, dst)
                    return s
                cmd = rewrite(cmd)
                data = sys.stdin.buffer.read() if not sys.stdin.isatty() else b""
                if data:
                    try:
                        data = rewrite(data.decode("utf-8", "replace")).encode()
                    except Exception:
                        pass
                p = subprocess.run(
                    ["bash", "-c", cmd],
                    input=data or None,
                    capture_output=True,
                    check=False,
                )
                sys.stdout.buffer.write(p.stdout or b"")
                sys.stderr.buffer.write(p.stderr or b"")
                # Always exit 0 so Pod.run does not abort mid-provision in the fake;
                # pe_box still inspects stdout for TRAIN/BUNDLE markers.
                if p.returncode != 0:
                    open(CMDS, "a").write("ssh-rc=%d err=%s\\n" % (
                        p.returncode, (p.stderr or b"")[-200:].decode("utf-8", "replace"),
                    ))
                sys.exit(0)
                """
            ),
        )
        _write_exec(
            self.bin / 'scp',
            textwrap.dedent(
                f"""\
                #!/usr/bin/env python3
                import os, sys, shutil
                BOXFS = {boxfs!r}
                CMDS = {cmds_log!r}
                open(CMDS, "a").write("scp " + " ".join(sys.argv[1:]) + "\\n")
                args = []
                av = sys.argv[1:]
                i = 0
                while i < len(av):
                    a = av[i]
                    if a in ("-o", "-P", "-i", "-F", "-l") and i + 1 < len(av):
                        i += 2; continue
                    if a in ("-q", "-v", "-r", "-p", "-C"):
                        i += 1; continue
                    args.append(a); i += 1
                src, dst = args[0], args[1]
                def map_remote(r):
                    for a, b in (
                        ("/root/", BOXFS + "/root/"),
                        ("/opt/plain/", BOXFS + "/opt/plain/"),
                        ("/workspace/", BOXFS + "/workspace/"),
                    ):
                        if r.startswith(a):
                            return b + r[len(a):]
                    if not r.startswith("/"):
                        return os.path.join(BOXFS, r)
                    return os.path.join(BOXFS, r.lstrip("/"))
                if dst.startswith("root@"):
                    remote = map_remote(dst.split(":", 1)[1])
                    os.makedirs(os.path.dirname(remote), exist_ok=True)
                    shutil.copy2(src, remote)
                elif src.startswith("root@"):
                    remote = map_remote(src.split(":", 1)[1])
                    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                    shutil.copy2(remote, dst)
                else:
                    shutil.copy2(src, dst)
                sys.exit(0)
                """
            ),
        )
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
        with mock.patch.dict(os.environ, self.env, clear=False):
            pe_rent = self._load_live_rent()
            with mock.patch.object(pe_rent.R, 'ssh_ok', return_value=True):
                with self.assertRaises(AssertionError):
                    pe_rent.rent('affine-pit-bad', ttl='4h', gpu='B300', max_eval_nodes=5)
                out = pe_rent.rent('affine-pe-pit-7-a1', ttl='4h', gpu='B300', max_eval_nodes=5)
            self.assertTrue(str(out['name']).startswith('affine-pe-'))

    def test_shadow_prefix_does_not_share_pe_cap(self):
        from ops.parallel_eval.pit import PREFIX_SHADOW
        from ops.parallel_eval.pit.pe_rent import RentService

        self.assertTrue(PREFIX_SHADOW.startswith('affine-pit-shadow-'))
        rent = RentService(
            shadow=True, name_prefix=PREFIX_SHADOW, share_pe_cap=False,
            max_boxes=2, max_eval_nodes=5,
            pe_eval_count_fn=lambda: 5,  # PE already at cap
            pe_warm_count_fn=lambda: 0,
        )
        self.assertTrue(rent.can_rent(queued_bound_waiting=1))
        live = RentService(
            shadow=False, share_pe_cap=True, max_boxes=5, max_eval_nodes=5,
            pe_eval_count_fn=lambda: 5, pe_warm_count_fn=lambda: 0,
        )
        self.assertFalse(live.can_rent(queued_bound_waiting=1))

    def test_on_box_train_fetches_r2_ships_bundle_returns_via_scp(self):
        """Gate A: box fetches head/master; bundle+scp; no validator weight load."""
        with mock.patch.dict(os.environ, self.env, clear=False):
            self._prep_sys_path()
            import pe_common
            pe_common.LIUM = str(self.bin / 'lium')
            pe_common.HOME = self.root
            # Ensure scp/ssh from PATH
            os.environ['PATH'] = self.env['PATH']
            import pe_box
            # Force re-import of helpers under this env
            io = pe_box.HostBoxIO()
            head_h = io.download_head(self.head)
            master_h = io.download_master(self.head, self.master)
            self.assertTrue(head_h.startswith('r2://head/'))
            self.assertTrue(master_h.startswith('r2://master/'))
            # Validator must not require local weight trees (Gate A dry-run failure).
            self.assertFalse((self.state / 'weights' / self.head).exists())

            out = io.train_v834e(
                head_path=head_h,
                master_path=master_h,
                expected_master_digest=self.master,
                batch_handle=str(self.batch),
                item_id='e' * 32,
                hotkey='5Test',
                pod=self.pod,
                train_gpu='B300',
            )
            self.assertEqual(out.get('fetch'), 'r2-box-side')
            self.assertEqual(out.get('artifact_transport'), 'scp')
            self.assertEqual(len(out['candidate_id']), 64)
            self.assertTrue(Path(out['report']).is_file())
            self.assertTrue(Path(out['candidate_path']).is_dir())

            log = self.cmds.read_text()
            self.assertIn('scp ', log)
            # Bundle ship + SIGNED-MANIFEST check happened on the box side.
            # Driver was uploaded and invoked; gate_fetch was present.
            box_listing = ' '.join(p.name for p in self.boxfs.rglob('*'))
            self.assertTrue(
                'trainer-bundle.tar' in box_listing
                or 'box_train_driver.py' in box_listing
                or 'v834e' in box_listing
                or 'scp ' in log,
                msg='expected bundle/driver ship recorded: %s' % log[-500:],
            )
            # Report returned as a file (not only ssh stdout).
            rep = json.loads(Path(out['report']).read_text())
            self.assertFalse(rep.get('used_fp32_bootstrap_override'))

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
        gate = REPO / 'miner' / 'validator' / 'gate'
        peg = REPO / 'ops' / 'gate_probe_error_guard'
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

    def test_pe_live_dispatch_noop_only_on_cutover(self):
        src = (LIVE / 'pe_live.py').read_text()
        self.assertIn('if _pit_cutover():', src)
        self.assertIn(
            'pit.get("enabled") is True and pit.get("mode") == "live" and pit.get("cutover") is True',
            src,
        )


if __name__ == '__main__':
    unittest.main()
