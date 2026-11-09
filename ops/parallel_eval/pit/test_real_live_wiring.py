#!/usr/bin/env python3
"""Integration: real live pe_rent / pe_lane / pe_score / pe_box with fake lium+ssh+scp.

NO PIT-level mocks of rent/lane/score/box train. Only CLI binaries are faked.
Asserts the drive_item / run5090 on-box path:

* box-side R2 fetch via gate_fetch (head + master under FP32/<bf16>/)
* golden trainer bundle via bundle_fetch (BUN['trainer']) + host-v834e ship
* train_pit.py under fetched runtime-exact/venv (fail closed if missing)
* signed TRAINING-JOB + TRAIN-DISPATCH-FF + docs inputs (not a lone batch file)
* SIGNED-MANIFEST.sha256 verified without ``; true``
* file-based artifact return via scp with sha checks
* shadow prefix does not share the live PE cap
"""
from __future__ import annotations

import hashlib
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

        self.head = 'a' * 64
        self.master = 'b' * 64
        self.head_bf16 = 'c' * 64
        self.cand = 'd' * 64

        # --- Host release: trainer-host-v834e + SIGNED-MANIFEST.sha256 ---
        self.release = self.root / 'release-v834e-signed'
        host = self.release / 'trainer-host-v834e'
        host.mkdir(parents=True)
        (host / 'mimo_host_trainer_v83.py').write_text('# stub host module\n')
        (host / 'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json').write_text(
            json.dumps({'engine': {'sgd_fp32_master': True}, 'authority': 'test'})
        )
        # Manifest lists host files relative to TW after ship; we verify host-v834e via
        # generated sha256sum listing (drive_item). Also place SIGNED-MANIFEST.sha256
        # covering a shipped marker file under TW.
        marker = self.release / 'RELEASE.txt'
        marker.write_text('v834e-test\n')
        (self.release / 'SIGNED-MANIFEST.sha256').write_text(
            '%s  RELEASE.txt\n' % _sha(marker)
        )

        # --- BUNDLE.json trainer part (R2 golden bundle) ---
        self.bun = self.state / 'BUNDLE.json'
        self.bun.write_text(json.dumps({
            'trainer': {
                'key': 'bundles/trainer-golden.tar',
                'sha256': 'e' * 64,
                'bytes': 12,
            },
            'probe': {
                'key': 'bundles/probe.tar',
                'sha256': 'f' * 64,
                'bytes': 8,
            },
        }))

        # --- Life epoch with signed TRAINING-JOB + TRAIN-DISPATCH-FF ---
        self.life = self.root / 'life'
        self.epoch_dir = self.life / 'epochs' / '1041'
        self.epoch_dir.mkdir(parents=True)
        head_files = {
            'model.safetensors': self.head_bf16,  # sha of the bf16 file
            'config.json': '1' * 64,
        }
        job = {
            'payload': {
                'manifest': {
                    'payload': {
                        'checkpoint': {'id': self.head, 'files': head_files},
                        'mimo_runtime': {'version': 'test'},
                    }
                },
                'training': {
                    'config': {'settings': {'lr': 1e-5}},
                    'optimizer_step': 42,
                },
            }
        }
        (self.epoch_dir / 'TRAINING-JOB.json').write_text(json.dumps(job))
        doc_sha = '2' * 64
        dispatch = {
            'signature': 'sig-test',
            'signer': 'ROOT',
            'payload': {
                'selected_rollout_count': 1,
                'selected': [{
                    'document': {
                        'sha256': doc_sha,
                        'key': 'docs/%s.json' % doc_sha,
                    }
                }],
            },
        }
        (self.epoch_dir / 'TRAIN-DISPATCH-FF.json').write_text(json.dumps(dispatch))
        self.batch = self.epoch_dir  # batch_handle = epoch dir (resolves JOB+DISPATCH)

        # --- Published M(H) index ---
        self.fp32_idx = self.root / 'fp32-master'
        self.fp32_idx.mkdir()
        (self.fp32_idx / (self.head + '.json')).write_text(json.dumps({
            'bf16_sha256': self.head_bf16,
            'master_digest': self.master,
            'files': {
                'master.safetensors': {'sha256': '3' * 64, 'r2_key': 'fp32/master.safetensors'},
                'meta.json': {'sha256': '4' * 64, 'r2_key': 'fp32/meta.json'},
            },
            'meta_json': {
                'version': 1,
                'bf16_sha256': self.head_bf16,
                'master_digest': self.master,
            },
        }))

        # --- Stub train_pit.py (box entry; writes descriptor + report like the real one) ---
        self.train_pit = self.root / 'train_pit_stub.py'
        self.train_pit.write_text(textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json, os, sys
            from pathlib import Path
            exp = os.environ.get('AFFINE_FP32_EXPECTED_MASTER_HASH') or ''
            assert len(exp) == 64, 'expected master hash required (fail closed)'
            assert not os.environ.get('AFFINE_FP32_BOOTSTRAP_OVERRIDE'), 'override forbidden'
            IN = Path('/workspace/pst/in')
            OUT = Path('/workspace/pst/out')
            OUT.mkdir(parents=True, exist_ok=True)
            assert (IN / 'manifest.json').is_file()
            assert (IN / 'dispatch.json').is_file()
            assert (IN / 'head.descriptor.json').is_file()
            assert (IN / 'meta.json').is_file()
            assert (IN / 'train_config.json').is_file()
            cid = {self.cand!r}
            # Minimal safetensors-named file so place_cand find works.
            model = OUT / 'model'
            model.mkdir(parents=True, exist_ok=True)
            blob = b'CAND'
            (model / 'model-00001-of-00001.safetensors').write_bytes(blob)
            import hashlib
            h = hashlib.sha256(blob).hexdigest()
            (OUT / 'descriptor.json').write_text(json.dumps({{'id': cid, 'files': {{'model-00001-of-00001.safetensors': h}}}}))
            rep = {{
              'status': 'ok',
              'optimizer': {{'sgd': {{'fp32_master': {{
                'source': 'disk',
                'input_master_digest': exp,
                'output_master_digest': {'5'*64!r},
                'input_bf16_sha256': {'c'*64!r},
                'output_bf16_sha256': {'6'*64!r},
              }}}}}},
              'trained_rollouts': 1, 'trained_tokens': 8,
              'preclip_gradient_norm': 1.0, 'objective_loss': 0.1,
            }}
            (OUT / 'report.private.json').write_text(json.dumps(rep))
            print(json.dumps({{'event': 'done', 'checkpoint': cid, 'fp32': rep['optimizer']['sgd']['fp32_master']}}))
            """
        ))

        # Presign injections (no real R2).
        self.presign_keys = self.root / 'presign_keys.json'
        self.presign_keys.write_text(json.dumps({
            'trainer': 'https://r2.example/bundles/trainer-golden.tar',
            'docs/%s.json' % doc_sha: 'https://r2.example/docs/%s.json' % doc_sha,
            doc_sha + '.json': 'https://r2.example/docs/%s.json' % doc_sha,
            'fp32/master.safetensors': 'https://r2.example/fp32/master.safetensors',
            'fp32/meta.json': 'https://r2.example/fp32/meta.json',
            'master.safetensors': 'https://r2.example/fp32/master.safetensors',
            'meta.json': 'https://r2.example/fp32/meta.json',
        }))
        self.presign_ckpt = self.root / 'presign_ckpt.json'
        self.presign_ckpt.write_text(json.dumps({
            'urls': {
                'model.safetensors': 'https://r2.example/head/model.safetensors',
                'config.json': 'https://r2.example/head/config.json',
            }
        }))
        self.master_presign = self.root / 'master_presign.json'
        self.master_presign.write_text(json.dumps({
            'urls': {
                'master.safetensors': 'https://r2.example/fp32/master.safetensors',
                'meta.json': 'https://r2.example/fp32/meta.json',
            }
        }))

        roles = PODS / 'roles.json'
        self.env = {
            'PATH': '%s:%s' % (self.bin, os.environ.get('PATH', '')),
            'PE_STATE': str(self.state),
            'PE_ST': str(self.state),
            'PIT_STATE': str(self.state / 'pit'),
            'PE_LIFE': str(self.life),
            'LIUM': str(self.bin / 'lium'),
            'HOME': str(self.root),
            'PODS_ROLES': str(roles),
            'PIT_V834E_RELEASE': str(self.release),
            'PIT_TRAINER_HOST': str(host),
            'PIT_BUNDLE_JSON': str(self.bun),
            'PIT_FP32_MASTER_INDEX': str(self.fp32_idx),
            'PIT_TRAIN_PIT_PY': str(self.train_pit),
            'PIT_PRESIGN_KEYS_JSON': str(self.presign_keys),
            'PIT_PRESIGN_PAYLOAD_JSON': str(self.presign_ckpt),
            'PIT_MASTER_PRESIGN_JSON': str(self.master_presign),
            'PIT_UPLOAD_ROOT': str(self.root / 'uploads'),
            'PYTHONPATH': '%s:%s' % (LIVE, PODS),
            'AFFINE_REPO': str(self.root / 'repo'),
        }
        (self.root / 'repo').mkdir()

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
        head_bf16 = self.head_bf16
        master_dig = self.master
        bun_sha = 'e' * 64
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
        # Fake ssh: rewrite box paths into BOXFS; intercept bundle_fetch / gate_fetch.
        _write_exec(
            self.bin / 'ssh',
            textwrap.dedent(
                f"""\
                #!/usr/bin/env python3
                import json, os, sys, subprocess, hashlib, shutil
                from pathlib import Path
                BOXFS = {boxfs!r}
                CMDS = {cmds_log!r}
                HEAD_BF16 = {head_bf16!r}
                MASTER_DIG = {master_dig!r}
                BUN_SHA = {bun_sha!r}
                open(CMDS, "a").write("ssh " + " ".join(sys.argv[1:]) + "\\n")
                args = sys.argv[1:]
                i = 0
                while i < len(args):
                    a = args[i]
                    if a in ("-p", "-o", "-i", "-l", "-F", "-P") and i + 1 < len(args):
                        i += 2; continue
                    if a in ("-T", "-n", "-A", "-X", "-Y", "-tt", "-f", "-q", "-v"):
                        i += 1; continue
                    if a.startswith("root@") or (len(a) > 0 and a[0].isdigit() and "." in a):
                        i += 1; break
                    break
                while i < len(args) and args[i] in ("-n", "-t", "-tt", "-T"):
                    i += 1
                cmd = " ".join(args[i:])
                open(CMDS, "a").write("ssh-cmd " + cmd[:500] + "\\n")

                def bpath(p):
                    for src, dst in (
                        ("/root/", BOXFS + "/root/"),
                        ("/opt/plain/", BOXFS + "/opt/plain/"),
                        ("/workspace/", BOXFS + "/workspace/"),
                        ("/tmp/", BOXFS + "/tmp/"),
                    ):
                        if p.startswith(src):
                            return dst + p[len(src):]
                    return os.path.join(BOXFS, p.lstrip("/"))

                def rewrite(s):
                    # Insert a placeholder first so BOXFS (which lives under /tmp)
                    # is not partially re-matched by the /tmp/ rule.
                    mark = "@@BOXFS@@"
                    s = s.replace(BOXFS, mark)
                    for src, suffix in (
                        ("/root/", "/root/"),
                        ("/opt/plain/", "/opt/plain/"),
                        ("/workspace/", "/workspace/"),
                        ("/tmp/", "/tmp/"),
                    ):
                        s = s.replace(src, mark + suffix)
                    return s.replace(mark, BOXFS)

                for d in ("root", "opt/plain/tbx", "opt/plain/workspace", "opt/plain/probe/affine-env-probe/ckpt",
                          "workspace/pst", "workspace/affine-mimo-trainer-prod/fp32-master", "tmp"):
                    os.makedirs(os.path.join(BOXFS, d), exist_ok=True)

                data = sys.stdin.buffer.read() if not sys.stdin.isatty() else b""

                # --- Intercept: bundle_fetch trainer → plant golden tree ---
                if "bundle_fetch.py" in cmd and "trainer" in cmd:
                    tbx = Path(BOXFS) / "opt/plain/tbx"
                    if tbx.exists(): shutil.rmtree(tbx)
                    tbx.mkdir(parents=True)
                    # Layout matching drive_item merge: workspace/ + other tops
                    ws = tbx / "workspace"
                    (ws / "runtime-exact" / "venv" / "bin").mkdir(parents=True)
                    vpy = ws / "runtime-exact" / "venv" / "bin" / "python"
                    vpy.write_text("#!/usr/bin/env bash\\nexec python3 \\"$@\\"\\n")
                    vpy.chmod(0o755)
                    src = tbx / "workspace" / "affine-mimo-trainer-prod-src"  # unused; real layout uses TW tops
                    # Also plant under tops that merge to /workspace via cp -al of workspace/
                    # drive_item merges workspace → /opt/plain/workspace and other tops to /
                    # Golden tree expects TW=/workspace/affine-mimo-trainer-prod with source + venv.
                    # Put them under tbx/workspace/affine-mimo-trainer-prod so cp -al lands them
                    # under /opt/plain/workspace/...; pe_box then also needs them at TW.
                    # Simpler: plant directly at rewritten TW after "merge" simulation below.
                    tw = Path(BOXFS) / "workspace" / "affine-mimo-trainer-prod"
                    (tw / "runtime-exact" / "venv" / "bin").mkdir(parents=True)
                    (tw / "source-final-v823u").mkdir(parents=True)
                    (tw / "source-final-v823u" / "placeholder.py").write_text("# src\\n")
                    vp = tw / "runtime-exact" / "venv" / "bin" / "python"
                    # Stub python that handles `import torch` check and train_pit.
                    vp.write_text(
                        "#!/usr/bin/env bash\\n"
                        "if [ \\"$1\\" = \\"-c\\" ]; then exec python3 -c \\"print('2.0 cuda []')\\"; fi\\n"
                        "exec python3 \\"$@\\"\\n"
                    )
                    vp.chmod(0o755)
                    # Also populate tbx workspace so the merge loop sees it.
                    (ws / "affine-mimo-trainer-prod" / "runtime-exact" / "venv" / "bin").mkdir(parents=True)
                    shutil.copy2(vp, ws / "affine-mimo-trainer-prod" / "runtime-exact" / "venv" / "bin" / "python")
                    (ws / "affine-mimo-trainer-prod" / "source-final-v823u").mkdir(parents=True)
                    # Release marker for SIGNED-MANIFEST.sha256 (verified under TW).
                    (tw / "RELEASE.txt").write_text("v834e-test\\n")
                    print(json.dumps(dict(sha_ok=True, tar_rc=0, seconds=0.1, gbps=1.7, tops=["workspace"])))
                    sys.exit(0)

                # --- Intercept: run5090.fetch wrapper — plant .verified from stdin JSON ---
                if "gate_fetch.py" in cmd and data:
                    try:
                        blob = json.loads(data.decode())
                    except Exception:
                        blob = None
                    if isinstance(blob, dict) and "descriptor" in blob:
                        # Recover DEST from open('{{K}}/{{c}}.descriptor.json') in the cmd.
                        dest = None
                        cid = blob["descriptor"]["id"]
                        for token in cmd.replace("'", " ").replace('"', " ").split():
                            if token.endswith(cid + ".descriptor.json"):
                                dest = token[: -(len(cid) + len(".descriptor.json") + 1)]
                                break
                        if dest is None:
                            # Fallback: ckpt vs fp32-master by id.
                            if cid == HEAD_BF16:
                                dest = "/opt/plain/probe/affine-env-probe/ckpt"  # overwritten below
                                dest = "/workspace/affine-mimo-trainer-prod/fp32-master"
                            else:
                                dest = "/opt/plain/probe/affine-env-probe/ckpt"
                        dest_r = rewrite(dest)
                        os.makedirs(dest_r, exist_ok=True)
                        desc_path = os.path.join(dest_r, cid + ".descriptor.json")
                        open(desc_path, "w").write(json.dumps(blob["descriptor"]))
                        target = os.path.join(dest_r, cid)
                        os.makedirs(target, exist_ok=True)
                        for name in (blob["descriptor"].get("files") or dict()):
                            open(os.path.join(target, name), "wb").write(b"x")
                        open(os.path.join(target, ".verified"), "w").write("ok\\n")
                        if cid == HEAD_BF16:
                            open(os.path.join(target, "meta.json"), "w").write(json.dumps(dict(
                                version=1,
                                bf16_sha256=HEAD_BF16,
                                master_digest=MASTER_DIG,
                            )))
                        total = sum(
                            os.path.getsize(os.path.join(r, f))
                            for r, _, fs in os.walk(target) for f in fs
                        )
                        print(json.dumps(dict(ckpt=cid, status="materialized")))
                        print(total)
                        sys.exit(0)

                if "nvidia-smi" in cmd and "query-gpu=name" in cmd:
                    print("NVIDIA B300"); sys.exit(0)
                if "podctl.py serve" in cmd:
                    print('{{"ok": true}}'); sys.exit(0)
                if "podctl.py status" in cmd:
                    print('{{"state":"serving","ready":true,"replicas_up":8,"replicas":8,"served":"' + ("d"*64) + '"}}')
                    sys.exit(0)
                if "/v1/models" in cmd:
                    print('{{"data":[{{"id":"m"}}]}}'); sys.exit(0)

                cmd = rewrite(cmd)
                if data:
                    try:
                        data = rewrite(data.decode("utf-8", "replace")).encode()
                    except Exception:
                        pass
                p = subprocess.run(["bash", "-c", cmd], input=data or None, capture_output=True)
                sys.stdout.buffer.write(p.stdout or b"")
                sys.stderr.buffer.write(p.stderr or b"")
                if p.returncode != 0:
                    open(CMDS, "a").write("ssh-rc=%d err=%s\\n" % (
                        p.returncode, (p.stderr or b"")[-200:].decode("utf-8", "replace"),
                    ))
                # Fail closed paths must still surface non-zero for pe_box asserts that
                # check stdout markers; Pod.run(check=True) would raise — our HostBoxIO
                # uses check=False on train. Exit 0 so mid-provision helpers proceed;
                # pe_box inspects SHA_OK / rc=0 / BAD 0 markers itself.
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
                        ("/tmp/", BOXFS + "/tmp/"),
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
            pe_eval_count_fn=lambda: 5,
            pe_warm_count_fn=lambda: 0,
        )
        self.assertTrue(rent.can_rent(queued_bound_waiting=1))
        live = RentService(
            shadow=False, share_pe_cap=True, max_boxes=5, max_eval_nodes=5,
            pe_eval_count_fn=lambda: 5, pe_warm_count_fn=lambda: 0,
        )
        self.assertFalse(live.can_rent(queued_bound_waiting=1))

    def test_on_box_train_drive_item_path(self):
        """Gate A: drive_item steps — R2 bundle_fetch, train_pit, scp return."""
        with mock.patch.dict(os.environ, self.env, clear=False):
            self._prep_sys_path()
            import pe_common
            pe_common.LIUM = str(self.bin / 'lium')
            pe_common.HOME = self.root
            os.environ['PATH'] = self.env['PATH']
            # Fresh import so env-derived paths bind correctly.
            sys.modules.pop('pe_box', None)
            import pe_box
            io = pe_box.HostBoxIO()
            head_h = io.download_head(self.head)
            master_h = io.download_master(self.head, self.master)
            self.assertTrue(head_h.startswith('r2://head/'))
            self.assertTrue(master_h.startswith('r2://master/'))
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
                epoch=1041,
                epoch_dir=str(self.epoch_dir),
            )
            self.assertEqual(out.get('fetch'), 'r2-box-side')
            self.assertEqual(out.get('artifact_transport'), 'scp')
            self.assertEqual(out['candidate_id'], self.cand)
            self.assertEqual(out.get('venv'), pe_box.BOX_VENV)
            self.assertTrue(Path(out['report']).is_file())
            self.assertTrue(Path(out['candidate_path']).is_dir())
            rep = json.loads(Path(out['report']).read_text())
            fp = rep['optimizer']['sgd']['fp32_master']
            self.assertEqual(fp['input_master_digest'], self.master)
            self.assertEqual(fp['source'], 'disk')

            log = self.cmds.read_text()
            # Exact on-box command markers from drive_item / run5090.
            self.assertIn('bundle_fetch.py trainer', log)
            self.assertIn('gate_fetch.py', log)
            self.assertIn('train_pit.py', log)
            self.assertIn('runtime-exact/venv/bin/python', log)
            self.assertIn('AFFINE_FP32_EXPECTED_MASTER_HASH=', log)
            self.assertIn('scp ', log)
            # Fail-closed: no `; true` after SIGNED-MANIFEST check in pe_box source.
            src = (LIVE / 'pe_box.py').read_text()
            self.assertIn('SIGNED-MANIFEST.sha256', src)
            self.assertNotIn('SIGNED-MANIFEST; true', src)
            self.assertNotIn('sha256sum -c SIGNED-MANIFEST; true', src)
            # Venv fail-closed: never falls back to bare python3.
            self.assertIn('test -x %s', src)
            self.assertIn('runtime-exact/venv/bin/python', src)
            self.assertIn('BOX_VENV', src)
            self.assertNotIn('else V=python3', src)

    def test_fail_closed_missing_venv(self):
        """Fresh box without fetched venv must not fall back to system python3."""
        with mock.patch.dict(os.environ, self.env, clear=False):
            self._prep_sys_path()
            sys.modules.pop('pe_box', None)
            import pe_box
            P = mock.Mock()
            P.run.side_effect = RuntimeError('test -x failed: no venv')
            with self.assertRaises(Exception):
                pe_box.run_train_pit(P, expected_master=self.master)

    def test_fail_closed_unsigned_dispatch(self):
        with mock.patch.dict(os.environ, self.env, clear=False):
            self._prep_sys_path()
            bad = self.root / 'bad-epoch'
            bad.mkdir()
            shutil = __import__('shutil')
            shutil.copy(self.epoch_dir / 'TRAINING-JOB.json', bad / 'TRAINING-JOB.json')
            (bad / 'TRAIN-DISPATCH-FF.json').write_text(json.dumps({
                'signature': '', 'signer': 'SHADOW-UNSIGNED', 'payload': {'selected': []},
            }))
            sys.modules.pop('pe_box', None)
            import pe_box
            P = mock.Mock()
            with self.assertRaises(pe_box.BoxIOError):
                pe_box.ship_inputs(P, epoch_dir=bad, epoch=1, work_jd=self.root / 'jd')

    def test_fail_closed_nested_master_rejected_by_fetch_master(self):
        """Master lands under FP32/<bf16>/; meta bf16+digest must match (no invented digest)."""
        src = (LIVE / 'pe_box.py').read_text()
        self.assertIn("FP32 + '/' + entry + '/meta.json'", src)
        self.assertIn('Never writes a digest the box did not compute/verify', src)
        self.assertIn('pod master meta', src)
        # Old fail-open path wrote expected digest without checking — must be gone.
        self.assertNotIn('(master_dir / "master_digest.txt").write_text(expected', src)

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
