#!/usr/bin/env python3
"""Host box I/O for PIT (download head/master, train-v834e ON the rented box).

Training MUST run on the rented GPU box (B300), never as a local process on the
validator (no GPU). Flow (shadow driver / E1041–E1046 path):

1. Resolve pod handle from rent dict (``pod_of``).
2. Upload signed v834e host files + head + fp32 master + K1 batch over ssh.
3. Run trainer-v834e on the box; pull report.json + candidate paths back.
4. Caller then serves + probes on the same pod via pe_lane.

Fail-closed: missing host paths / trainer entry / expected-master mismatch
raises. Never passes ``--fp32-bootstrap-override``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional

sys.dont_write_bytecode = True

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# On-box work root (same tree pe_lane uses for serve/probe).
BOX_ROOT = '/root/affine-env-probe'
BOX_PIT = '/root/pit-train'


class BoxIOError(RuntimeError):
    """Fail-closed host box I/O error."""


def _as_pod(pod: Any) -> Any:
    """Accept a Pod object or a live pe_rent dict → Pod with .run()."""
    if pod is None:
        raise BoxIOError('pod handle required for on-box train')
    if hasattr(pod, 'run') and callable(getattr(pod, 'run')):
        return pod
    if isinstance(pod, Mapping):
        try:
            import pe_common as PC  # live code dir
            return PC.pod_of(pod)
        except Exception:
            pass
        try:
            from ops.parallel_eval.live import pe_common as PC
            return PC.pod_of(pod)
        except Exception as exc:
            raise BoxIOError('cannot build Pod from rent dict: %s' % exc) from exc
    raise BoxIOError('pod must be a Pod or pe_rent dict, got %r' % type(pod))


def _push_bytes(P: Any, dest: str, data: bytes, *, mode: str = '644') -> None:
    """Upload bytes to the box via ssh stdin (lium/ssh; no host-local GPU)."""
    parent = str(Path(dest).parent)
    P.run(
        "mkdir -p %s && cat > %s.tmp && chmod %s %s.tmp && mv %s.tmp %s" % (
            parent, dest, mode, dest, dest, dest,
        ),
        inp=data,
        timeout=max(120, min(3600, 30 + len(data) // (256 * 1024))),
    )


def _push_tree(P: Any, src: Path, dest: str) -> None:
    """Tar a local directory and extract it on the box."""
    src = Path(src)
    if not src.exists():
        raise BoxIOError('push_tree source missing: %s' % src)
    with tempfile.NamedTemporaryFile(suffix='.tar') as tmp:
        with tarfile.open(tmp.name, 'w') as tar:
            tar.add(str(src), arcname=src.name if src.is_dir() else src.name)
        data = Path(tmp.name).read_bytes()
    parent = str(Path(dest).parent)
    P.run("mkdir -p %s" % parent, timeout=60)
    P.run(
        "mkdir -p %s && tar -C %s -xf - && echo TAR_OK" % (dest if src.is_dir() else parent, parent),
        inp=data,
        timeout=max(300, min(7200, 60 + len(data) // (128 * 1024))),
    )


def _pull_bytes(P: Any, remote: str) -> bytes:
    out = P.run("cat %s" % remote, timeout=600)
    if isinstance(out, bytes):
        return out
    return out.encode('utf-8', errors='replace') if isinstance(out, str) else bytes(out or b'')


def _detect_box_gpu(P: Any) -> Optional[str]:
    try:
        out = P.run(
            "nvidia-smi --query-gpu=name --format=csv,noheader | head -1",
            timeout=30, check=False,
        )
    except Exception:
        return None
    text = (out or '').strip().splitlines()
    if not text:
        return None
    name = text[0].strip()
    upper = name.upper()
    for tag in ('B300', 'B200', 'H200', 'H100', 'A100'):
        if tag in upper:
            return tag
    return name


def _rj(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _pe_state() -> Path:
    env = os.environ.get('PE_STATE') or os.environ.get('PE_ST')
    if env:
        return Path(env)
    return Path(
        '/home/const/subnet120-rewrite/state/root-audits/'
        '20261106-parallel-cutover/state'
    )


def _repo() -> Path:
    env = os.environ.get('AFFINE_REPO') or os.environ.get('REPO')
    if env:
        return Path(env)
    return Path.home() / 'subnet120-rewrite'


def _weights_root() -> Path:
    env = os.environ.get('PE_WEIGHTS') or os.environ.get('AFFINE_WEIGHTS')
    if env:
        return Path(env)
    return _pe_state() / 'weights'


def _master_root() -> Path:
    env = os.environ.get('PE_MASTERS') or os.environ.get('AFFINE_FP32_MASTERS')
    if env:
        return Path(env)
    return _pe_state() / 'masters'


def _upload_root() -> Path:
    env = os.environ.get('PIT_UPLOAD_ROOT')
    if env:
        return Path(env)
    return _pe_state() / 'pit' / 'uploads'


def _trainer_entry() -> Path:
    env = os.environ.get('PIT_TRAIN_ENTRY') or os.environ.get('AFFINE_PIT_TRAIN_ENTRY')
    if env:
        return Path(env)
    # Prefer kit train_step (v834e) when present on the host/box image.
    candidates = [
        _repo() / 'miner' / 'kit' / 'train_step.py',
        Path('/workspace/affine-mimo-trainer-prod/train_step.py'),
        Path('/root/affine-env-probe/train_step.py'),
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise BoxIOError('trainer-v834e entry not found (set PIT_TRAIN_ENTRY)')


def _sha_dir_id(path: Path) -> str:
    files = {}
    for p in sorted(path.iterdir()):
        if p.is_file() and not p.name.startswith('.') and p.name not in (
            'CHECKPOINT.json',
        ) and p.suffix in ('.json', '.safetensors', '.jinja'):
            files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def _read_master_digest(master_dir: Path) -> Optional[str]:
    for name in ('FP32-MASTER.json', 'MASTER.json'):
        doc = _rj(master_dir / name)
        if isinstance(doc, Mapping):
            dig = doc.get('master_digest') or doc.get('digest') or doc.get('sha256')
            if isinstance(dig, str) and len(dig) == 64:
                return dig
    tag = master_dir / 'master_digest.txt'
    if tag.is_file():
        dig = tag.read_text().strip()
        if len(dig) == 64:
            return dig
    return None


class HostBoxIO:
    """Host-side BoxIO used by PIT ``BoxRunner`` in live mode."""

    def download_head(self, head: str) -> str:
        if not isinstance(head, str) or len(head) != 64:
            raise BoxIOError('head must be hex64')
        root = _weights_root() / head
        if not root.is_dir():
            # Common alternate layout: weights/<head>/bf16
            alt = _weights_root() / head / 'bf16'
            if alt.is_dir():
                root = alt
            else:
                raise BoxIOError('head weights missing for %s' % head[:12])
        return str(root)

    def download_master(self, head: str, expected_digest: str) -> str:
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise BoxIOError('expected_digest must be hex64')
        # Prefer digest-addressed store, then head-named sibling.
        candidates = [
            _master_root() / expected_digest,
            _master_root() / head,
            Path(self.download_head(head) + '-fp32-master'),
            _weights_root() / head / 'fp32-master',
        ]
        for root in candidates:
            if not root.is_dir():
                continue
            got = _read_master_digest(root)
            if got is None:
                # Directory of tensors: trust expected_digest only if INDEX pins it.
                idx = _rj(_pe_state() / 'MASTER-INDEX.json') or {}
                if isinstance(idx, Mapping) and idx.get(head) == expected_digest:
                    return str(root)
                continue
            if got != expected_digest:
                raise BoxIOError(
                    'expected-master-mismatch: got %s want %s' % (got[:12], expected_digest[:12])
                )
            return str(root)
        raise BoxIOError('expected-master-mismatch: fp32 master missing for head %s' % head[:12])

    def train_v834e(self, **kwargs: Any) -> Mapping[str, Any]:
        """Train ON the rented box (never a local validator process)."""
        if kwargs.get('fp32_bootstrap_override') or kwargs.get('bootstrap_override'):
            raise BoxIOError('bootstrap-override-refused-by-pit')
        head_path = kwargs.get('head_path')
        master_path = kwargs.get('master_path')
        expected = kwargs.get('expected_master_digest')
        batch_handle = kwargs.get('batch_handle')
        item_id = kwargs.get('item_id') or 'pit-item'
        pod = kwargs.get('pod')
        if not head_path or not master_path or not expected:
            raise BoxIOError('train_v834e requires head_path, master_path, expected_master_digest')
        if not batch_handle or str(batch_handle).startswith('secret:'):
            raise BoxIOError('train_v834e requires opaque non-secret batch_handle')
        if str(batch_handle).startswith('batch-handle:'):
            raise BoxIOError(
                'refuse synthetic batch-handle; pass real K1 batch path '
                '(E1041-class divergence risk vs live trainer)'
            )
        batch_path = Path(str(batch_handle))
        if not batch_path.is_file():
            raise BoxIOError('batch file missing: %s' % batch_path)
        got = _read_master_digest(Path(master_path))
        if got is not None and got != expected:
            raise BoxIOError('expected-master-mismatch before train')

        P = _as_pod(pod)
        entry = _trainer_entry()
        work = '%s/%s-%d' % (BOX_PIT, item_id[:16], int(time.time()))
        remote_head = work + '/head'
        remote_master = work + '/master'
        remote_batch = work + '/batch.json'
        remote_out = work + '/candidate'
        remote_report = work + '/report.json'
        remote_train = work + '/train_step.py'

        P.run('mkdir -p %s %s/ckpt %s/runs' % (BOX_ROOT, BOX_ROOT, BOX_ROOT), timeout=120)
        _push_bytes(P, remote_train, entry.read_bytes(), mode='755')
        _push_tree(P, Path(head_path), remote_head)
        _push_tree(P, Path(master_path), remote_master)
        _push_bytes(P, remote_batch, batch_path.read_bytes())

        py = os.environ.get('BOX_PY') or 'python3'
        train_cmd = (
            "set -e; "
            "export AFFINE_FP32_EXPECTED_MASTER_HASH=%s; "
            "unset AFFINE_FP32_BOOTSTRAP_OVERRIDE FP32_BOOTSTRAP_OVERRIDE; "
            "mkdir -p %s; "
            "%s %s --head %s --batch %s --out %s --master %s --report %s --hotkey %s; "
            "test -f %s; echo TRAIN_OK"
        ) % (
            expected,
            remote_out,
            py, remote_train,
            remote_head, remote_batch, remote_out, remote_master, remote_report,
            str(kwargs.get('hotkey') or 'pit'),
            remote_report,
        )
        try:
            P.run(train_cmd, timeout=int(os.environ.get('PIT_TRAIN_TIMEOUT_S') or 7200))
        except Exception as exc:
            raise BoxIOError('on-box trainer failed: %s' % str(exc)[:400]) from exc

        try:
            report_bytes = _pull_bytes(P, remote_report)
        except Exception as exc:
            raise BoxIOError(
                'training report missing after train (required before release): %s' % exc
            ) from exc
        rep = json.loads(report_bytes.decode('utf-8'))
        if not isinstance(rep, dict) or not rep:
            raise BoxIOError('training report missing after train (required before release)')
        if rep.get('used_fp32_bootstrap_override') or (
            (rep.get('fp32_master') or {}).get('source') == 'bootstrap-upcast'
        ):
            raise BoxIOError('bootstrap-override-refused-by-pit')

        host_out = _upload_root() / ('train-%s-%d' % (item_id[:16], int(time.time())))
        host_out.mkdir(parents=True, exist_ok=True)
        report = host_out / 'report.json'
        report.write_bytes(report_bytes)
        os.chmod(report, 0o600)

        cand_dir = host_out / 'candidate'
        cand_dir.mkdir(parents=True, exist_ok=True)
        try:
            tar_out = P.run('tar -C %s -cf - .' % remote_out, timeout=3600)
            tar_bytes = tar_out.encode('latin-1') if isinstance(tar_out, str) else bytes(tar_out)
            with tempfile.NamedTemporaryFile(suffix='.tar') as tmp:
                tmp.write(tar_bytes)
                tmp.flush()
                with tarfile.open(tmp.name, 'r') as tar:
                    tar.extractall(cand_dir)
        except Exception as exc:
            raise BoxIOError('failed to pull candidate from box: %s' % str(exc)[:300]) from exc

        candidate_id = _sha_dir_id(cand_dir)
        if len(candidate_id) != 64:
            candidate_id = str(rep.get('candidate_id') or rep.get('candidate_sha256') or '')
        if len(candidate_id) != 64:
            raise BoxIOError('candidate id digest failed')

        master_out = host_out / 'fp32-master'
        remote_master_out = remote_out + '/fp32-master'
        try:
            marker = P.run(
                'test -d %s && echo YES || echo NO' % remote_master_out,
                timeout=60, check=False,
            )
            if 'YES' in str(marker):
                master_out.mkdir(parents=True, exist_ok=True)
                tar_out = P.run('tar -C %s -cf - .' % remote_master_out, timeout=3600)
                tar_bytes = tar_out.encode('latin-1') if isinstance(tar_out, str) else bytes(tar_out)
                with tempfile.NamedTemporaryFile(suffix='.tar') as tmp:
                    tmp.write(tar_bytes)
                    tmp.flush()
                    with tarfile.open(tmp.name, 'r') as tar:
                        tar.extractall(master_out)
        except Exception:
            master_out = Path(master_path)
        master_dig = _read_master_digest(master_out) or ''
        if len(master_dig) != 64:
            master_dig = str(rep.get('candidate_master_digest') or rep.get('master_digest') or '')
        if len(master_dig) != 64:
            raise BoxIOError('candidate master digest missing')

        train_gpu = kwargs.get('train_gpu') or _detect_box_gpu(P)

        return dict(
            candidate_id=candidate_id,
            candidate_path=str(cand_dir),
            candidate_master_path=str(master_out),
            candidate_master_digest=master_dig,
            candidate_sha256=candidate_id,
            used_fp32_bootstrap_override=False,
            fp32_master_source=(rep.get('fp32_master') or {}).get('source') or 'disk',
            report=str(report),
            report_doc=rep,
            training_report=rep,
            train_gpu=train_gpu,
            pod_id=getattr(P, 'id', None),
            box_work=work,
        )

    def upload_candidate(self, *, candidate_id: str, candidate_path: str) -> Mapping[str, Any]:
        src = Path(candidate_path)
        if not src.exists():
            raise BoxIOError('candidate_path missing for upload')
        dst = _upload_root() / candidate_id
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / src.name)
        man = {
            'id': candidate_id,
            'sha256': candidate_id,
            'uploaded_at': time.time(),
            'source': 'pit',
        }
        (dst / 'UPLOAD.json').write_text(json.dumps(man, indent=1, sort_keys=True) + '\n')
        os.chmod(dst / 'UPLOAD.json', 0o600)
        return dict(ok=True, path=str(dst))

    def upload_master_during_probe(self, *, candidate_id: str, master_path: str) -> Mapping[str, Any]:
        return upload_master(candidate_id=candidate_id, path=master_path)

    def probe_result_fixture(self) -> Mapping[str, Any]:
        # Live BoxRunner must use lane.run_probe; fixture only for synthetic tests.
        raise BoxIOError('live HostBoxIO refuses probe_result_fixture (use pe_lane.run_probe)')


def box_io_factory() -> HostBoxIO:
    return HostBoxIO()


# Alias expected by host_wiring.
BoxIO = HostBoxIO


def upload_master(*, candidate_id: str, path: Optional[str] = None, **_kw: Any) -> Mapping[str, Any]:
    if not path:
        raise BoxIOError('upload_master requires path')
    src = Path(path)
    if not src.exists():
        raise BoxIOError('master path missing')
    dst = _upload_root() / candidate_id / 'fp32-master'
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / src.name)
    dig = _read_master_digest(dst)
    if dig:
        idx_path = _pe_state() / 'MASTER-INDEX.json'
        idx = _rj(idx_path) or {}
        if not isinstance(idx, dict):
            idx = {}
        idx[candidate_id] = dig
        tmp = idx_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(idx, indent=1, sort_keys=True) + '\n')
        os.chmod(tmp, 0o600)
        os.replace(tmp, idx_path)
    return dict(ok=True, path=str(dst), master_digest=dig)


def upload_candidate(*, candidate_id: str, path: Optional[str] = None, **_kw: Any) -> Mapping[str, Any]:
    io = HostBoxIO()
    return io.upload_candidate(candidate_id=candidate_id, candidate_path=str(path or ''))


def destroy_box_state(
    *,
    box_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    reason: str = 'destroy',
    **_kw: Any,
) -> Mapping[str, Any]:
    removed = []
    if candidate_id:
        for root in (
            _upload_root() / candidate_id,
            _pe_state() / 'candidates' / candidate_id,
        ):
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
                removed.append(str(root))
    return dict(ok=True, box_id=box_id, candidate_id=candidate_id, reason=reason, removed=removed)
