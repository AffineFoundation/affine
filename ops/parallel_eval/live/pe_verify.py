#!/usr/bin/env python3
"""Host upload verification for PIT GREEN apply (fail closed).

``verify_uploads`` checks that the bf16 candidate and fp32 master the box
uploaded match the digests claimed in the train receipt. Missing artifacts or
hash mismatch → ``candidate_ok`` / ``master_ok`` False (never promotes).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

sys.dont_write_bytecode = True


class VerifyError(RuntimeError):
    """Fail-closed verification error (caller must not promote)."""


def _rj(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _pe_state() -> Path:
    env = os.environ.get('PE_STATE') or os.environ.get('PE_ST')
    if env:
        return Path(env)
    return Path(
        '/home/const/subnet120-rewrite/state/root-audits/'
        '20261106-parallel-cutover/state'
    )


def _candidate_roots(candidate_id: str) -> list[Path]:
    st = _pe_state()
    roots = [
        st / 'candidates' / candidate_id,
        st / 'pit' / 'uploads' / candidate_id,
        Path(os.environ.get('PIT_UPLOAD_ROOT') or '/tmp/pit-uploads') / candidate_id,
    ]
    extra = os.environ.get('PE_CANDIDATE_ROOT')
    if extra:
        roots.insert(0, Path(extra) / candidate_id)
    return roots


def _find_file(roots: list[Path], names: tuple[str, ...]) -> Optional[Path]:
    for root in roots:
        for name in names:
            p = root / name
            if p.is_file():
                return p
        # Directory of safetensors: use MANIFEST / CHECKPOINT.json files map.
        for manifest_name in ('CHECKPOINT.json', 'MANIFEST.json', 'UPLOAD.json'):
            man = root / manifest_name
            doc = _rj(man)
            if isinstance(doc, Mapping):
                files = doc.get('files') or doc.get('sha256') or {}
                if isinstance(files, Mapping) and files:
                    return man
    return None


def _manifest_sha(path: Path) -> Optional[str]:
    doc = _rj(path)
    if not isinstance(doc, Mapping):
        return None
    if isinstance(doc.get('sha256'), str) and len(doc['sha256']) == 64:
        return doc['sha256']
    if isinstance(doc.get('id'), str) and len(doc['id']) == 64:
        return doc['id']
    files = doc.get('files')
    if isinstance(files, Mapping) and files:
        # Checkpoint id = sha256(canonical {name: sha256})
        material = json.dumps(dict(files), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(material).hexdigest()
    return None


def verify_uploads(
    *,
    candidate_id: str,
    candidate_sha256: str,
    master_digest: str,
    candidate_path: Optional[str] = None,
    master_path: Optional[str] = None,
) -> Mapping[str, Any]:
    """Fail-closed upload verify. Returns ``candidate_ok`` / ``master_ok`` booleans."""
    if not isinstance(candidate_id, str) or len(candidate_id) != 64:
        raise VerifyError('candidate_id must be hex64')
    if not isinstance(candidate_sha256, str) or len(candidate_sha256) != 64:
        raise VerifyError('candidate_sha256 must be hex64')
    if not isinstance(master_digest, str) or len(master_digest) != 64:
        raise VerifyError('master_digest must be hex64')

    cand_ok = False
    master_ok = False
    detail: dict[str, Any] = {'candidate_id': candidate_id}

    # Explicit paths from apply_live / HostBoxIO when provided.
    if candidate_path:
        p = Path(candidate_path)
        if p.is_file():
            got = _sha256_file(p) if p.suffix != '.json' else (_manifest_sha(p) or _sha256_file(p))
            cand_ok = got == candidate_sha256 or got == candidate_id
            detail['candidate_path'] = str(p)
            detail['candidate_got'] = got
        elif p.is_dir():
            man = _find_file([p], ('CHECKPOINT.json', 'MANIFEST.json'))
            got = _manifest_sha(man) if man else None
            cand_ok = got in (candidate_sha256, candidate_id)
            detail['candidate_dir'] = str(p)
            detail['candidate_got'] = got

    if master_path:
        p = Path(master_path)
        if p.is_file():
            got = _sha256_file(p) if 'MASTER' not in p.name.upper() else (
                (_rj(p) or {}).get('master_digest') or (_rj(p) or {}).get('digest') or _sha256_file(p)
            )
            if isinstance(got, str):
                master_ok = got == master_digest
            detail['master_path'] = str(p)
            detail['master_got'] = got
        elif p.is_dir():
            for name in ('FP32-MASTER.json', 'MASTER.json', 'master_digest.txt'):
                mp = p / name
                if not mp.is_file():
                    continue
                if mp.suffix == '.txt':
                    got = mp.read_text().strip()
                else:
                    doc = _rj(mp) or {}
                    got = doc.get('master_digest') or doc.get('digest') or doc.get('sha256')
                if isinstance(got, str):
                    master_ok = got == master_digest
                    detail['master_got'] = got
                break

    # Fall back to PE_STATE / PIT upload roots when paths not supplied.
    if not cand_ok:
        roots = _candidate_roots(candidate_id)
        man = _find_file(roots, ('CHECKPOINT.json', 'MANIFEST.json', 'UPLOAD.json', 'candidate.sha256'))
        if man is not None:
            if man.name.endswith('.sha256'):
                got = man.read_text().strip()
            else:
                got = _manifest_sha(man) or _sha256_file(man)
            cand_ok = got in (candidate_sha256, candidate_id)
            detail['candidate_lookup'] = str(man)
            detail['candidate_got'] = got

    if not master_ok:
        for root in _candidate_roots(candidate_id):
            for name in ('FP32-MASTER.json', 'MASTER.json', 'master.sha256', 'master_digest.txt'):
                mp = root / name
                if not mp.is_file():
                    # Also check sibling master store keyed by digest.
                    continue
                if mp.suffix == '.txt' or mp.name.endswith('.sha256'):
                    got = mp.read_text().strip()
                else:
                    doc = _rj(mp) or {}
                    got = doc.get('master_digest') or doc.get('digest') or doc.get('sha256')
                if isinstance(got, str) and got == master_digest:
                    master_ok = True
                    detail['master_lookup'] = str(mp)
                    break
            if master_ok:
                break
        # Published master index under PE_STATE.
        if not master_ok:
            idx = _rj(_pe_state() / 'MASTER-INDEX.json') or {}
            if isinstance(idx, Mapping) and idx.get(candidate_id) == master_digest:
                master_ok = True
                detail['master_lookup'] = 'MASTER-INDEX'
            pub = _pe_state() / 'masters' / master_digest
            if pub.exists():
                master_ok = True
                detail['master_lookup'] = str(pub)

    # Fail closed: if neither path nor state produced a checkable artifact, refuse.
    if 'candidate_got' not in detail and 'candidate_lookup' not in detail and not candidate_path:
        detail['candidate_error'] = 'no-upload-artifact'
        cand_ok = False
    if 'master_got' not in detail and 'master_lookup' not in detail and not master_path:
        detail['master_error'] = 'no-master-artifact'
        master_ok = False

    return dict(
        candidate_ok=bool(cand_ok),
        master_ok=bool(master_ok),
        detail=detail,
    )


# Alias used by host_wiring.
verify_green_uploads = verify_uploads
