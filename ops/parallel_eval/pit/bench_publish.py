"""Land PIT on-demand bench results where ``affine-bench-publish`` already reads.

``publish_fleet`` / ``affine-bench-publish`` rebuilds ``website/benchmarks.json``
every ~2 minutes from ``$BENCH_RESULTS_DIR/results/<model_id>/<benchmark>.json``.
This module writes those result files (same schema as the standing daemon's
``collect()``). It does **not** merge into ``benchmarks.json`` and does **not**
import ``bench.affine_bench``.

Results root is config/env only (``bench_results_dir`` / ``BENCH_RESULTS_DIR`` /
``AFFINE_BENCH_RESULTS``) — fail closed if unset. No host-path defaults.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

_NX_GPU = re.compile(r'^\d+[xX]')


def results_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    for key in ('BENCH_RESULTS_DIR', 'AFFINE_BENCH_RESULTS'):
        val = os.environ.get(key)
        if val:
            return Path(val)
    raise RuntimeError(
        'BENCH_RESULTS_DIR/AFFINE_BENCH_RESULTS unset (fail closed; no host-path defaults)'
    )


def results_dir(root: Optional[Path] = None) -> Path:
    return (root or results_root()) / 'results'


def format_node_gpu(
    gpu: Optional[str],
    *,
    gpu_count: Optional[int] = None,
    rent_meta: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render publish_fleet GPU column (e.g. ``8xB300``), never ``?`` when known."""
    meta = dict(rent_meta or {})
    raw = (
        gpu
        or meta.get('gpu')
        or meta.get('gpu_model')
        or meta.get('config')
        or ''
    )
    raw = str(raw).strip()
    if not raw:
        raise ValueError('gpu type missing for node record')
    if _NX_GPU.match(raw):
        return raw
    if ':' in raw:
        kind, _, rest = raw.partition(':')
        if rest.isdigit():
            return '%dx%s' % (int(rest), kind)
        raw = kind
    count = gpu_count
    if count is None:
        for key in ('gpu_count', 'available_gpu_count', 'n_gpu'):
            if meta.get(key) is not None:
                count = int(meta[key])
                break
    if count is None:
        count = 8  # PIT/bench pin is full 8-GPU B300 nodes
    return '%dx%s' % (int(count), raw)


def node_record(
    *,
    box_id: Optional[str] = None,
    box_name: Optional[str] = None,
    gpu: Optional[str] = None,
    gpu_count: Optional[int] = None,
    rent_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Standing-daemon ``collect()`` node shape for publish_fleet ``_gpu(r)``."""
    meta = dict(rent_meta or {})
    return {
        'pod': box_name or meta.get('name') or meta.get('pod'),
        'lium_id': box_id or meta.get('box_id') or meta.get('id') or meta.get('lium_id'),
        'gpu': format_node_gpu(gpu, gpu_count=gpu_count, rent_meta=meta),
    }


def _normalize_summary(
    raw: Mapping[str, Any],
    *,
    head: str,
    model_id: str,
    node: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Match the standing daemon collect() / node_run summary.json shape."""
    bid = raw.get('benchmark') or raw.get('benchmark_id')
    if not bid:
        raise ValueError('result row missing benchmark id')
    sampling = dict(raw.get('sampling') or {})
    if 'presence_penalty' not in sampling:
        sampling['presence_penalty'] = float(
            raw.get('presence_penalty') if raw.get('presence_penalty') is not None else 0.0
        )
    if float(sampling.get('presence_penalty', 1.0)) != 0.0:
        raise RuntimeError('refuse result write: Affine king presence_penalty must be 0')
    out = dict(raw)
    node_doc = dict(node) if node else dict(raw.get('node') or {})
    out.update(
        model_id=model_id,
        model_kind=raw.get('model_kind') or 'king',
        benchmark=str(bid),
        king=raw.get('king') or head,
        profile=raw.get('profile') or sampling.get('profile') or 'qwen3.5-family-pp0',
        presence_penalty=0.0,
        sampling=sampling,
        finished_at=raw.get('finished_at') or time.time(),
    )
    if node_doc:
        out['node'] = node_doc
    # Required numeric fields publish.py/_row_from_disk reads.
    for key, default in (
        ('score', None),
        ('se', None),
        ('tasks', raw.get('n') or 0),
        ('samples', 0),
        ('rollouts', 1),
        ('cap_rate', 0.0),
        ('mean_out_tokens', None),
        ('wall_seconds', raw.get('wall_s') or 0),
    ):
        out.setdefault(key, default)
    if 'bench_version' not in out and raw.get('env_version'):
        out['bench_version'] = raw.get('env_version')
    return out


def write_results(
    *,
    head: str,
    result: Mapping[str, Any],
    results_dir_path: Optional[str] = None,
    now: Optional[float] = None,
    node: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write ``results/<model_id>/<benchmark>.json`` for publish_fleet consumption."""
    if float(result.get('presence_penalty', 0.0) or 0.0) != 0.0:
        raise RuntimeError('refuse write: Affine king presence_penalty must be 0')
    root = results_root(results_dir_path)
    out_dir = results_dir(root)
    mid = str(result.get('model_id') or ('king-%s' % head[:16]))
    model_dir = out_dir / mid
    model_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    rows = list(result.get('results') or [])
    if not rows:
        raise RuntimeError('bench RESULT has no results rows to land')
    at = float(now if now is not None else time.time())
    node_doc = dict(node) if node else dict(result.get('node') or {})
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        doc = _normalize_summary(raw, head=head, model_id=mid, node=node_doc or None)
        doc['collected_at'] = at
        bid = str(doc['benchmark'])
        path = model_dir / ('%s.json' % bid)
        tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
        written.append(str(path))
    return {
        'ok': True,
        'merge_only': True,  # publish_fleet owns website merge; we never rewrite benchmarks.json
        'landed': True,
        'results_root': str(root),
        'model_id': mid,
        'files': written,
        'n': len(written),
        'presence_penalty': 0.0,
        'node_gpu': (node_doc or {}).get('gpu'),
    }


def merge_publish(
    *,
    head: str,
    result: Mapping[str, Any],
    site: Optional[str] = None,
    results_dir_path: Optional[str] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Deprecated name kept for callers — lands results for publish_fleet; no website merge.

    ``site`` is ignored (website/benchmarks.json is owned by affine-bench-publish).
    """
    del site  # never write website/benchmarks.json from PIT
    return write_results(
        head=head, result=result, results_dir_path=results_dir_path, now=now,
        node=result.get('node') if isinstance(result.get('node'), Mapping) else None,
    )
