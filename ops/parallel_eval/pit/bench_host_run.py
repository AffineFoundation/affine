"""Host helpers for on-demand GREEN bench (no ``bench.affine_bench`` import).

Box setup/run lives in ``bench_box`` (detached commands). Publish lives in
``bench_publish``. This module only wires a Remote store from a rented pod.
"""
from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional

from ops.jobs.runner import JobRecord

from .bench_box import RemoteBenchBoxStore


def make_bench_box_store_factory(
    *,
    list_pods_fn: Optional[Callable[[], List[Mapping[str, Any]]]] = None,
    rent_meta_fn: Optional[Callable[[str], Optional[Mapping[str, Any]]]] = None,
) -> Callable[[JobRecord], Any]:
    """Build a ``box_store_factory(job) → RemoteBenchBoxStore`` for live pods."""

    def factory(job: JobRecord) -> RemoteBenchBoxStore:
        # Lazy: pe_common pulls live host seams; hermetic unit tests never call this.
        from ops.parallel_eval.live import pe_common as PC  # noqa: PLC0415 — host-only seam

        box_id = job.box_id
        row: Optional[Mapping[str, Any]] = None
        if list_pods_fn is not None and box_id:
            for r in list(list_pods_fn() or []):
                if str(r.get('id') or r.get('box_id') or '') == str(box_id):
                    row = r
                    break
        if row is None and rent_meta_fn is not None and box_id:
            row = rent_meta_fn(str(box_id))
        if row is None:
            raise RuntimeError('bench box %s not listed for store' % (box_id,))
        if 'ip' not in row or 'port' not in row:
            raise RuntimeError('bench box row missing ip/port for SSH store')
        pod = PC.pod_of(dict(row))
        return RemoteBenchBoxStore(pod)

    return factory
