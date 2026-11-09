#!/usr/bin/env python3
"""Host box I/O for PIT (download head/master, train-v834e, uploads, destroy).

Fail-closed: missing host paths / trainer entry / expected-master mismatch
raises. Never passes ``--fp32-bootstrap-override``. Live boxes must train from
the published **fp32 master** of the bound head (not a silent bf16 upcast) with
trainer-v834e, using the real K1 batch file.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

sys.dont_write_bytecode = True

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


class BoxIOError(RuntimeError):
    """Fail-closed host box I/O error."""


def _detect_local_gpu() -> Optional[str]:
    """Best-effort GPU class for train/eval stamps on the box result."""
    try:
        smi = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if smi.returncode != 0 or not smi.stdout.strip():
        return None
    name = smi.stdout.strip().splitlines()[0]
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
        if kwargs.get('fp32_bootstrap_override') or kwargs.get('bootstrap_override'):
            raise BoxIOError('bootstrap-override-refused-by-pit')
        head_path = kwargs.get('head_path')
        master_path = kwargs.get('master_path')
        expected = kwargs.get('expected_master_digest')
        batch_handle = kwargs.get('batch_handle')
        item_id = kwargs.get('item_id') or 'pit-item'
        if not head_path or not master_path or not expected:
            raise BoxIOError('train_v834e requires head_path, master_path, expected_master_digest')
        if not batch_handle or str(batch_handle).startswith('secret:'):
            raise BoxIOError('train_v834e requires opaque non-secret batch_handle')
        # Refuse the controller placeholder — live must pass the real K1 batch path.
        if str(batch_handle).startswith('batch-handle:'):
            raise BoxIOError(
                'refuse synthetic batch-handle; pass real K1 batch path '
                '(E1041-class divergence risk vs live trainer)'
            )
        batch_path = Path(str(batch_handle))
        if not batch_path.is_file():
            raise BoxIOError('batch file missing: %s' % batch_path)

        # Re-check master digest before train (fail closed).
        got = _read_master_digest(Path(master_path))
        if got is not None and got != expected:
            raise BoxIOError('expected-master-mismatch before train')

        entry = _trainer_entry()
        out_dir = _upload_root() / ('train-%s-%d' % (item_id[:16], int(time.time())))
        out_dir.mkdir(parents=True, exist_ok=True)
        report = out_dir / 'report.json'
        env = os.environ.copy()
        env['AFFINE_FP32_EXPECTED_MASTER_HASH'] = str(expected)
        env.pop('AFFINE_FP32_BOOTSTRAP_OVERRIDE', None)
        env.pop('FP32_BOOTSTRAP_OVERRIDE', None)

        cmd = [
            env.get('HOST_PY') or sys.executable,
            str(entry),
            '--head', str(head_path),
            '--batch', str(batch_path),
            '--out', str(out_dir / 'candidate'),
            '--master', str(master_path),
            '--report', str(report),
            '--hotkey', str(kwargs.get('hotkey') or 'pit'),
        ]
        # Host/box images may wrap train_step differently; honor PIT_TRAIN_ARGV_PREFIX.
        prefix = os.environ.get('PIT_TRAIN_ARGV_PREFIX')
        if prefix:
            cmd = prefix.split() + cmd

        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=int(
                    os.environ.get('PIT_TRAIN_TIMEOUT_S') or 7200
                ),
            )
        except FileNotFoundError as exc:
            raise BoxIOError('trainer binary missing: %s' % exc) from exc
        except subprocess.TimeoutExpired as exc:
            raise BoxIOError('trainer timeout') from exc
        if proc.returncode != 0:
            raise BoxIOError(
                'trainer-v834e failed rc=%s: %s' % (
                    proc.returncode, (proc.stderr or proc.stdout or '')[-400:],
                )
            )

        cand_dir = out_dir / 'candidate'
        if not cand_dir.is_dir():
            raise BoxIOError('trainer produced no candidate dir')
        candidate_id = _sha_dir_id(cand_dir)
        master_out = cand_dir / 'fp32-master'
        if not master_out.is_dir():
            # Some trainers write sibling fp32-master.
            alt = out_dir / 'fp32-master'
            if alt.is_dir():
                master_out = alt
            else:
                raise BoxIOError('trainer produced no candidate fp32 master')
        master_dig = _read_master_digest(master_out) or ''
        if len(master_dig) != 64:
            # Fall back to hashing MASTER.json if present.
            man = master_out / 'FP32-MASTER.json'
            if man.is_file():
                master_dig = hashlib.sha256(man.read_bytes()).hexdigest()
            else:
                raise BoxIOError('candidate master digest missing')

        rep = _rj(report) or {}
        if not isinstance(rep, dict) or not rep:
            raise BoxIOError('training report missing after train (required before release)')
        if rep.get('used_fp32_bootstrap_override') or (
            (rep.get('fp32_master') or {}).get('source') == 'bootstrap-upcast'
        ):
            raise BoxIOError('bootstrap-override-refused-by-pit')

        train_gpu = kwargs.get('train_gpu') or _detect_local_gpu()

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
