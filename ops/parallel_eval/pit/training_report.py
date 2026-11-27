"""Training-report copy-back + GPU metadata for PIT boxes.

Before release the box must return its trainer report (source file hashes,
batching, tokens, update size, gradient norm). The host stores it on the item.
Also records train/eval GPU types for shadow SCORE-vs-live.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class TrainingReportError(RuntimeError):
    pass


# Fields required in the host-stored summary (Jacob 2026-11-09).
REPORT_SUMMARY_KEYS = (
    'source_file_hashes',
    'batching',
    'tokens',
    'update_size',
    'gradient_norm',
)


def _rj(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _pe_state() -> Path:
    env = os.environ.get('PE_STATE') or os.environ.get('PE_ST')
    if env:
        return Path(env)
    pit = os.environ.get('PIT_STATE')
    if pit:
        # PIT_STATE is …/pit; host training-reports live under PE_STATE/pit/.
        p = Path(pit)
        return p.parent if p.name == 'pit' else p
    raise RuntimeError(
        'PE_STATE unset (fail closed; no host-path defaults for training-reports)'
    )


def extract_report_summary(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a trainer-v834e / kit report into the host summary schema."""
    if not isinstance(raw, Mapping):
        raise TrainingReportError('training report must be a mapping')
    inner = raw.get('report') if isinstance(raw.get('report'), Mapping) else raw
    file_summary = raw.get('file_summary') or inner.get('file_summary') or {}
    sgd = ((inner.get('optimizer') or {}).get('sgd') or {})
    fp32 = sgd.get('fp32_master') if isinstance(sgd.get('fp32_master'), Mapping) else {}

    source_hashes = (
        raw.get('source_file_hashes')
        or file_summary.get('source_file_hashes')
        or file_summary.get('files')
        or file_summary.get('sha256')
        or inner.get('source_file_hashes')
    )
    if source_hashes is None and isinstance(file_summary, Mapping):
        # kit validate summary often carries document digests under groups/receipts.
        source_hashes = {
            k: file_summary[k]
            for k in ('batch_sha256', 'head_id', 'manifest_sha256', 'documents')
            if k in file_summary
        } or dict(file_summary)

    batching = (
        raw.get('batching')
        or file_summary.get('batching')
        or {
            k: file_summary.get(k)
            for k in ('tasks', 'rollouts', 'groups', 'max_tasks', 'max_rollouts', 'train')
            if k in file_summary
        }
        or inner.get('batching')
    )

    tokens = (
        raw.get('tokens')
        or file_summary.get('tokens')
        or {
            k: (file_summary.get(k) if isinstance(file_summary, Mapping) else None)
            for k in ('prompt_tokens', 'output_tokens', 'loss_tokens', 'total_tokens')
            if isinstance(file_summary, Mapping) and k in file_summary
        }
        or inner.get('tokens')
        or inner.get('token_stats')
    )

    update_size = (
        raw.get('update_size')
        or sgd.get('raw_update_norm')
        or fp32.get('update_norm_master')
        or inner.get('raw_update_norm')
        or inner.get('update_size')
    )
    gradient_norm = (
        raw.get('gradient_norm')
        or raw.get('preclip_gradient_norm')
        or inner.get('preclip_gradient_norm')
        or inner.get('gradient_norm')
    )

    out = dict(
        source_file_hashes=source_hashes if source_hashes is not None else {},
        batching=batching if batching is not None else {},
        tokens=tokens if tokens is not None else {},
        update_size=update_size,
        gradient_norm=gradient_norm,
        engine=raw.get('engine') or inner.get('engine'),
        head=raw.get('head') or inner.get('head') or inner.get('input_checkpoint'),
        status=inner.get('status') or raw.get('status'),
        objective_loss=inner.get('objective_loss'),
        wall_seconds=raw.get('wall_seconds'),
    )
    # Optional richer fields kept for shadow compare.
    if isinstance(file_summary, Mapping) and file_summary:
        out['file_summary'] = dict(file_summary)
    if sgd:
        out['sgd'] = {
            k: sgd.get(k)
            for k in ('raw_update_norm', 'tensors_rel_capped', 'mode')
            if k in sgd
        }
        if fp32:
            out['sgd']['update_norm_master'] = fp32.get('update_norm_master')
    return out


def copy_training_report_to_host(
    *,
    report_path: Optional[str],
    item_id: str,
    epoch: int,
    candidate_id: Optional[str] = None,
    report_doc: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Copy box report.json onto the host PIT state and return the stored summary.

    Fail-closed if neither a readable report file nor an in-memory report_doc
    is available — the host must not release without a report.
    """
    raw: Optional[Mapping[str, Any]] = None
    src: Optional[Path] = None
    if report_path:
        src = Path(report_path)
        if src.is_file():
            raw = _rj(src)
    if raw is None and report_doc is not None:
        raw = dict(report_doc)
    if not isinstance(raw, Mapping):
        raise TrainingReportError(
            'training report missing (box must copy report before release)'
        )

    summary = extract_report_summary(raw)
    summary = dict(summary)
    try:
        try:
            root = _pe_state() / 'pit' / 'training-reports'
        except RuntimeError as exc:
            # No PE_STATE: never fall back to a host path. Unit tests land in
            # a fresh tempfile; production must export PE_STATE.
            root = Path(tempfile.mkdtemp(prefix='pit-training-reports-'))
            summary['host_store_error'] = str(exc)[:200]
        root.mkdir(parents=True, exist_ok=True)
        dest = root / ('e%d-%s.json' % (int(epoch), (candidate_id or item_id)[:16]))
        payload = dict(
            item_id=item_id,
            epoch=int(epoch),
            candidate_id=candidate_id,
            summary=summary,
            report=raw,
        )
        tmp = dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + '\n')
        os.chmod(tmp, 0o600)
        os.replace(tmp, dest)
        if src is not None and src.is_file() and src.resolve() != dest.resolve():
            # Keep a verbatim copy next to the summary for audit.
            verbatim = dest.with_suffix('.raw.json')
            try:
                shutil.copy2(src, verbatim)
                os.chmod(verbatim, 0o600)
            except OSError:
                pass
        summary['host_path'] = str(dest)
    except OSError as exc:
        # Content is still returned for item.meta; durable path is best-effort
        # when the configured PE_STATE tree is not writable.
        summary['host_path'] = None
        summary['host_store_error'] = str(exc)[:200]
    return summary


def gpu_type_of(pod_or_meta: Any) -> Optional[str]:
    """Best-effort GPU class string (B300 / H200 / …)."""
    if pod_or_meta is None:
        return None
    if isinstance(pod_or_meta, str):
        text = pod_or_meta.upper()
        for name in ('RTX5090', 'B300', 'B200', 'H200', 'H100', 'A100', '5090'):
            if name in text:
                return 'RTX5090' if name == '5090' else name
        return pod_or_meta
    if isinstance(pod_or_meta, Mapping):
        for key in ('gpu', 'gpu_model', 'gpu_name', 'train_gpu', 'eval_gpu'):
            val = pod_or_meta.get(key)
            if val:
                return gpu_type_of(val)
    return None


def detect_local_gpu() -> Optional[str]:
    """Query nvidia-smi for the local GPU class (train or eval on the box)."""
    try:
        smi = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if smi.returncode != 0 or not smi.stdout.strip():
        return None
    return gpu_type_of(smi.stdout.strip().splitlines()[0])
