"""Merge-only publish of PIT on-demand bench results into website/benchmarks.json.

Host-side only — does **not** import ``bench.affine_bench``. SITE path comes from
``AFFINE_SITE`` or ``SITE`` (fail closed if unset). Never touches
``bench-results/daemon.json`` or standing-daemon globals.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def site_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    for key in ('AFFINE_SITE', 'SITE'):
        val = os.environ.get(key)
        if val:
            return Path(val)
    raise RuntimeError('AFFINE_SITE/SITE unset (fail closed; no host-path defaults)')


def benchmarks_path(site: Optional[Path] = None) -> Path:
    return (site or site_root()) / 'benchmarks.json'


def _row_key(r: Mapping[str, Any]) -> tuple:
    return (
        str(r.get('model_id') or ''),
        str(r.get('benchmark_id') or r.get('benchmark') or ''),
        str(r.get('env_version') or ''),
    )


def _normalize_result_row(raw: Mapping[str, Any], *, head: str) -> Dict[str, Any]:
    bid = raw.get('benchmark_id') or raw.get('benchmark')
    mid = raw.get('model_id') or ('king-%s' % head[:16])
    return {
        'model_id': mid,
        'model_kind': raw.get('model_kind') or 'king',
        'benchmark_id': bid,
        'suite': raw.get('suite'),
        'env_version': raw.get('env_version') or raw.get('bench_version') or '',
        'score': raw.get('score'),
        'se': raw.get('se'),
        'n': raw.get('n') or raw.get('tasks'),
        'samples': raw.get('samples'),
        'samples_expected': raw.get('samples_expected'),
        'errors': raw.get('errors'),
        'complete': raw.get('complete'),
        'rollouts': raw.get('rollouts'),
        'cap_rate': raw.get('cap_rate'),
        'budget': raw.get('budget') or (raw.get('sampling') or {}).get('max_tokens'),
        'profile': raw.get('profile') or 'qwen3.5-family-pp0',
        'presence_penalty': float(raw.get('presence_penalty') if raw.get('presence_penalty') is not None else 0.0),
        'mean_out_tokens': raw.get('mean_out_tokens'),
        'run_at': raw.get('run_at') or raw.get('finished_at'),
        'wall_s': raw.get('wall_s'),
        'gpu': raw.get('gpu'),
        'king': raw.get('king') or head,
    }


def merge_publish(
    *,
    head: str,
    result: Mapping[str, Any],
    site: Optional[str] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Merge RESULT rows into ``$SITE/benchmarks.json`` (never drop prior rows)."""
    if float(result.get('presence_penalty', 0.0) or 0.0) != 0.0:
        raise RuntimeError('refuse publish: Affine king presence_penalty must be 0')
    root = site_root(site)
    out_path = benchmarks_path(root)
    root.mkdir(parents=True, exist_ok=True)
    prev: Dict[str, Any] = {}
    if out_path.is_file():
        try:
            prev = json.loads(out_path.read_text())
        except Exception:
            prev = {}
    prior = list(prev.get('results') or [])
    models = list(prev.get('models') or [])
    new_rows = [
        _normalize_result_row(r, head=head)
        for r in list(result.get('results') or [])
        if isinstance(r, Mapping)
    ]
    merged: Dict[tuple, Dict[str, Any]] = {}
    for r in prior + new_rows:
        if not r.get('model_id') or not r.get('benchmark_id'):
            continue
        merged[_row_key(r)] = dict(r)
    mid = str(result.get('model_id') or ('king-%s' % head[:16]))
    model_doc = {
        'model_id': mid,
        'label': 'king-%s' % head[:8],
        'kind': 'king',
        'king': head,
        'status': 'scored',
        'profile': result.get('profile') or 'qwen3.5-family-pp0',
        'budget': (result.get('sampling') or {}).get('max_tokens'),
        'green': True,
    }
    models = [m for m in models if m.get('model_id') != mid]
    models.append(model_doc)
    at = float(now if now is not None else time.time())
    doc = {
        'version': prev.get('version') or 'affine-bench-v2',
        'updated_at': at,
        'serving': prev.get('serving') or {},
        'about': prev.get('about') or (
            'MERGE-ONLY publish from PIT on-demand GREEN bench. '
            'Affine kings use presence_penalty 0 (qwen3.5-family-pp0).'
        ),
        'benchmarks': prev.get('benchmarks') or [],
        'suites': prev.get('suites') or [],
        'models': models,
        'results': list(merged.values()),
        'results_prior': prev.get('results_prior') or [],
        'node': prev.get('node') or {},
        'progress': prev.get('progress') or {},
        'pit_bench': {
            'head': head,
            'profile': result.get('profile'),
            'presence_penalty': 0.0,
            'published_at': at,
            'merge_only': True,
        },
    }
    tmp = out_path.with_name('.%s.%d.tmp' % (out_path.name, os.getpid()))
    tmp.write_text(json.dumps(doc, separators=(',', ':')))
    os.chmod(tmp, 0o644)
    os.replace(tmp, out_path)
    return {
        'ok': True,
        'merge_only': True,
        'path': str(out_path),
        'rows_added': len(new_rows),
        'rows_total': len(doc['results']),
        'presence_penalty': 0.0,
    }
