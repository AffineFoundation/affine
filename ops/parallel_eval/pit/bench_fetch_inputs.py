"""Host-side descriptor + short-lived presigned URLs for the bench fetch stage.

Resolves the new head's checkpoint files the same way PIT item download does
(TRAINING-JOB / lifecycle publication receipt), presigns read-only R2 URLs on
the host (``pe_box.ckpt_urls``), and pushes ``descriptor.json`` +
``urls.private.json`` into the on-box job dir **before** the fetch stage
launches. The box deletes the URLs file after ``gate_fetch`` (and we never log
or commit URLs).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional


def flatten_files(files: Mapping[str, Any]) -> Dict[str, str]:
    """Normalize ``{name: sha}`` or ``{name: {sha256:...}}`` to ``{name: sha}``."""
    out: Dict[str, str] = {}
    for name, val in files.items():
        if isinstance(val, Mapping):
            sha = val.get('sha256') or val.get('digest')
        else:
            sha = val
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError('checkpoint file %r missing hex64 sha256' % name)
        out[str(name)] = sha
    if not out:
        raise ValueError('checkpoint files map is empty')
    return out


def _default_presign(head: str, files: Mapping[str, Any]) -> Mapping[str, Any]:
    # Host-only seam: pe_box pulls live R2 helpers. Keep deferred so hermetic
    # tests inject ``presign_fn`` without importing the live host stack.
    from ops.parallel_eval.live.pe_box import ckpt_urls  # noqa: PLC0415

    return ckpt_urls(head, files)


def resolve_head_files(
    head: str,
    *,
    files: Optional[Mapping[str, Any]] = None,
    life_root: Optional[Path] = None,
) -> Dict[str, str]:
    """Return flat ``{name: sha256}`` for ``head``.

    Prefer explicit ``files`` (from the GREEN item's ``candidate_files``).
    Otherwise scan ``$LIFE_ROOT/epochs/*/TRAINING-RESULT.json`` publication
    receipts (same source as the standing bench fetch).
    """
    if files:
        return flatten_files(files)
    root = life_root
    if root is None:
        for key in ('LIFE_ROOT', 'PE_LIFE'):
            val = os.environ.get(key)
            if val:
                root = Path(val)
                break
    if root is None:
        raise RuntimeError(
            'head files unavailable: pass candidate_files or set LIFE_ROOT/PE_LIFE'
        )
    epochs = root / 'epochs'
    if not epochs.is_dir():
        raise RuntimeError('LIFE epochs dir missing: %s' % epochs)
    for d in sorted(epochs.iterdir(), key=lambda p: -int(p.name) if p.name.isdigit() else 1):
        if not d.name.isdigit():
            continue
        path = d / 'TRAINING-RESULT.json'
        if not path.is_file():
            # Also accept TRAINING-JOB checkpoint files for the same id.
            job_path = d / 'TRAINING-JOB.json'
            if job_path.is_file():
                try:
                    job = json.loads(job_path.read_text())
                    payload = job.get('payload') or job
                    ck = ((payload.get('manifest') or {}).get('payload') or {}).get('checkpoint') or {}
                    if ck.get('id') == head and isinstance(ck.get('files'), Mapping):
                        return flatten_files(ck['files'])
                except Exception:
                    pass
            continue
        try:
            raw = json.loads(path.read_text())
        except Exception:
            continue
        payload = raw.get('payload') or raw
        report = payload.get('report') or {}
        receipt = report.get('publication_receipt') or {}
        desc = (receipt.get('payload') or receipt).get('descriptor') or {}
        if desc.get('id') == head and isinstance(desc.get('files'), Mapping):
            return flatten_files(desc['files'])
    raise RuntimeError('no checkpoint files for head %s' % head[:12])


def presign_head_urls(
    head: str,
    files: Mapping[str, Any],
    *,
    presign_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
) -> Dict[str, str]:
    """Short-lived read URLs for ``files`` (never logged by callers)."""
    if presign_fn is not None:
        urls = presign_fn(head, files)
    else:
        urls = _default_presign(head, files)
    if not isinstance(urls, Mapping) or not urls:
        raise RuntimeError('presign returned no urls')
    # Ensure every file has a URL.
    flat = flatten_files(files)
    missing = [n for n in flat if n not in urls]
    if missing:
        raise RuntimeError('presign missing urls for %s' % missing[:5])
    return {str(k): str(v) for k, v in urls.items() if k in flat}


def push_fetch_inputs(
    store: Any,
    job_dir: str,
    *,
    head: str,
    files: Optional[Mapping[str, Any]] = None,
    resolve_fn: Optional[Callable[..., Mapping[str, str]]] = None,
    presign_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Write ``descriptor.json`` + ``urls.private.json`` into the box job dir.

    Returns a redacted receipt (file names + counts only — **no URLs**).
    """
    store.ensure_dir(job_dir)
    if resolve_fn is not None:
        flat = dict(resolve_fn(head, files=files))
    else:
        flat = resolve_head_files(head, files=files)
    urls = presign_head_urls(head, flat, presign_fn=presign_fn)
    descriptor = {'id': head, 'files': flat}
    desc_path = '%s/descriptor.json' % job_dir.rstrip('/')
    urls_path = '%s/urls.private.json' % job_dir.rstrip('/')
    store.write_text(desc_path, json.dumps(descriptor, indent=1, sort_keys=True) + '\n')
    # URLs via stdin (item-path ``inp=``), never a heredoc/argv — so they never
    # appear in process lists. Mode 0600 on the box.
    urls_blob = json.dumps(urls, indent=1, sort_keys=True) + '\n'
    put = getattr(store, 'write_text_stdin', None)
    if not callable(put):
        raise RuntimeError(
            'store missing write_text_stdin; refuse putting urls.private.json on argv'
        )
    put(urls_path, urls_blob, mode='600')
    return {
        'ok': True,
        'head': head,
        'n_files': len(flat),
        'names': sorted(flat.keys()),
        'descriptor_path': desc_path,
        'urls_path': urls_path,
        'urls_via': 'stdin',
        # Explicitly absent: urls
    }
