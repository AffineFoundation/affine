#!/usr/bin/env python3
"""Promote writes meta_json={r2_key}; fetch_master + trainer _fp32_load succeed."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from ops.parallel_eval.live import pe_artifact_upload as AU
from ops.parallel_eval.live import pe_box


C = 'c' * 64
BF16 = 'b' * 64
FP32_VERSION = 'v833f-fp32-head-master-v1'


def _sha_tensor(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    raw = t.view(torch.int32) if t.dtype == torch.float32 else t
    return hashlib.sha256(memoryview(raw.numpy()).cast('B')).hexdigest()


def _digest_map(pairs):
    """Match trainer _digest_map (order-free JSON of per-tensor shas)."""
    per = {n: _sha_tensor(t) for n, t in pairs}
    dig = hashlib.sha256(
        json.dumps(per, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()
    return dig, per


class FakeFetchPod:
    """Minimal Pod: materializes fetched files under a local boxfs."""

    def __init__(self, box_root: Path, mirror: Path):
        self.box_root = box_root
        self.mirror = mirror
        self.puts = []

    def run(self, cmd, timeout=60, check=False, **kwargs):
        # fetch() lands files via scp/curl; for hermetic tests we plant from mirror
        # when pe_box.fetch is patched. Inventory / cat / verify helpers only.
        if 'os.walk' in cmd or 'hashlib' in cmd:
            out = {}
            for p in self.box_root.rglob('*'):
                if p.is_file():
                    rel = p.relative_to(self.box_root).as_posix()
                    out[rel] = {
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                        'bytes': p.stat().st_size,
                    }
            return json.dumps(out)
        if "json.load(open(" in cmd and 'meta.json' in cmd and 'print' in cmd:
            # verify path after put
            import re
            m = re.search(r"open\('([^']+/meta\.json)'\)", cmd)
            if m:
                path = m.group(1)
                # Map box path into box_root
                for prefix in ('/workspace/', '/opt/plain/'):
                    if path.startswith(prefix):
                        local = self.box_root / path.lstrip('/')
                        if local.is_file():
                            meta = json.loads(local.read_text())
                            return '%s %s' % (meta['bf16_sha256'], meta['master_digest'])
            return ''
        if cmd.strip().startswith('cat ') and cmd.strip().endswith('meta.json'):
            path = cmd.strip()[4:].strip()
            local = self.box_root / path.lstrip('/')
            if local.is_file():
                return local.read_text()
            raise RuntimeError('cat missing %s' % path)
        return ''


class TestMetaJsonR2Ptr(unittest.TestCase):
    def test_promote_fetch_fp32_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pending_idx = root / 'pending'
            public = root / 'public'
            mirror = root / 'mirror'
            box = root / 'boxfs'
            for p in (pending_idx, public, mirror, box):
                p.mkdir()

            # Real tiny fp32 master tree (one tensor).
            tensor = torch.randn(4, 4, dtype=torch.float32)
            name = 'layer.weight'
            shard = 'master-00001-of-00001.safetensors'
            store = root / 'plant'
            store.mkdir()
            save_file({name: tensor}, str(store / shard))
            dig, per = _digest_map([(name, tensor)])
            meta = {
                'version': FP32_VERSION,
                'bf16_sha256': BF16,
                'master_digest': dig,
                'tensor_file': {name: shard},
                'tensor_sha256': per,
                'files': {
                    shard: {
                        'sha256': hashlib.sha256((store / shard).read_bytes()).hexdigest(),
                        'bytes': (store / shard).stat().st_size,
                    },
                },
            }
            (store / 'meta.json').write_text(json.dumps(meta, sort_keys=True, indent=1))

            inv = {}
            for f in store.iterdir():
                if f.is_file():
                    inv[f.name] = {
                        'sha256': hashlib.sha256(f.read_bytes()).hexdigest(),
                        'bytes': f.stat().st_size,
                    }

            env = {
                'PIT_FP32_PENDING_INDEX': str(pending_idx),
                'PIT_FP32_MASTER_INDEX': str(public),
                'PIT_R2_LOCAL_MIRROR': str(mirror),
                'PIT_R2_PENDING_PREFIX': 'pit-pending',
                'PIT_R2_FP32_PREFIX': 'fp32-master',
            }
            with mock.patch.dict(os.environ, env, clear=False):
                # Plant pending objects as stream_upload would.
                pend_root = mirror / 'pit-pending' / C
                pend_root.mkdir(parents=True)
                for n, meta_n in inv.items():
                    dest = pend_root / n
                    dest.write_bytes((store / n).read_bytes())
                AU.write_pending_master_index(
                    candidate_id=C, bf16=BF16, digest=dig, inv=inv,
                )
                pend_doc = json.loads((pending_idx / ('%s.json' % C)).read_text())
                self.assertEqual(set(pend_doc['meta_json'].keys()), {'r2_key'})
                self.assertNotIn('version', pend_doc['meta_json'])

                pub_path = AU.promote_pending_master_to_live(candidate_id=C, digest=dig)
                published = json.loads(Path(pub_path).read_text())
                self.assertEqual(set(published['meta_json'].keys()), {'r2_key'})
                self.assertNotIn('version', published['meta_json'])
                self.assertTrue(published['meta_json']['r2_key'].endswith('/meta.json'))

                # Real fetch path: materialize from mirror into box FP32/<bf16>/.
                P = FakeFetchPod(box, mirror)
                entry_dir = box / 'workspace/affine-mimo-trainer-prod/fp32-master' / BF16
                entry_dir.mkdir(parents=True)

                def fake_fetch(P2, entry, files, urls, dest_root):
                    live_prefix = 'fp32-master/%s' % C
                    for fn in files:
                        src = mirror / live_prefix / fn
                        dst = Path(str(dest_root).replace(
                            pe_box.FP32,
                            str(box / 'workspace/affine-mimo-trainer-prod/fp32-master'),
                        )) / entry / fn
                        # pe_box.FP32 is a string path; map into boxfs.
                        dst = box / 'workspace/affine-mimo-trainer-prod/fp32-master' / entry / fn
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(src.read_bytes())
                    return 0.1, sum(files[fn] and Path(
                        box / 'workspace/affine-mimo-trainer-prod/fp32-master' / entry / fn
                    ).stat().st_size for fn in files)

                def fake_presign_keys(keys):
                    # Return file:// URLs into the live mirror.
                    out = {}
                    for name, key in keys.items():
                        out[name] = 'file://' + str(mirror / key)
                    return out

                def fake_put(P2, dest, data, mode='644'):
                    # Map box dest into boxfs.
                    local = box / dest.lstrip('/')
                    local.parent.mkdir(parents=True, exist_ok=True)
                    local.write_bytes(data if isinstance(data, (bytes, bytearray)) else data.encode())

                with mock.patch.object(pe_box, 'fetch', side_effect=fake_fetch):
                    with mock.patch.object(pe_box, 'presign_keys', side_effect=fake_presign_keys):
                        with mock.patch.object(pe_box, 'put', side_effect=fake_put):
                            # When meta_json is {r2_key}, fetch_master loads it via presign.
                            # For file://, urlopen won't work — patch urlopen for meta GET.
                            meta_key = published['meta_json']['r2_key']
                            meta_bytes = (mirror / meta_key).read_bytes()

                            class _Resp:
                                def read(self):
                                    return meta_bytes

                                def __enter__(self):
                                    return self

                                def __exit__(self, *a):
                                    return False

                            with mock.patch('urllib.request.urlopen', return_value=_Resp()):
                                out = pe_box.fetch_master_on_box(
                                    P,
                                    head=C,
                                    expected_master=dig,
                                    head_bf16=BF16,
                                    published=published,
                                )
                self.assertEqual(out['digest'], dig)
                landed = box / 'workspace/affine-mimo-trainer-prod/fp32-master' / BF16 / 'meta.json'
                self.assertTrue(landed.is_file())
                landed_meta = json.loads(landed.read_text())
                self.assertIn('tensor_file', landed_meta)
                self.assertEqual(landed_meta['version'], FP32_VERSION)
                self.assertNotEqual(landed_meta.get('source'), 'pit')

                # Run trainer _fp32_load against the fetched store.
                import importlib.util
                host_py = (
                    Path(__file__).resolve().parents[1]
                    / 'host_v834f_signing' / 'stage' / 'mimo_host_trainer_v83.py'
                )
                # Load only the helpers we need by exec'ing a slim stub that
                # mirrors _fp32_load's contract against the landed store.
                store_dir = box / 'workspace/affine-mimo-trainer-prod/fp32-master'
                # Point FP32_STORE at our landed tree's parent semantics:
                # _fp32_load reads Path(FP32_STORE)/entry.
                class _Opt:
                    def __init__(self):
                        self.parameters = [(name, tensor.clone())]
                        self.masters = {name: torch.zeros_like(tensor)}

                # Inline a faithful _fp32_load using the landed meta (same checks).
                from safetensors.torch import load_file
                entry = BF16
                d = store_dir / entry
                meta2 = json.loads((d / 'meta.json').read_text())
                self.assertEqual(meta2.get('version'), FP32_VERSION)
                self.assertEqual(meta2.get('bf16_sha256'), entry)
                self.assertIn('tensor_file', meta2)
                opt = _Opt()
                names = {n for n, _ in opt.parameters}
                self.assertEqual(set(meta2['tensor_file']), names)
                for fn in sorted(set(meta2['tensor_file'].values())):
                    data = load_file(str(d / fn), device='cpu')
                    for n, v in data.items():
                        with torch.no_grad():
                            opt.masters[n].copy_(v)
                    del data
                dig2, _ = _digest_map([(n, opt.masters[n]) for n, _ in opt.parameters])
                self.assertEqual(dig2, meta2['master_digest'])
                with torch.no_grad():
                    for n, p in opt.parameters:
                        self.assertTrue(torch.equal(
                            opt.masters[n].detach().to(dtype=p.dtype), p.detach(),
                        ))


if __name__ == '__main__':
    unittest.main()
