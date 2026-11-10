"""Host-side runner for the on-demand PIT GREEN bench job.

Rents come from PIT (``affine-pe-bench-*``, B300, $56/h). This module drives the
existing ``bench.affine_bench`` sampling/publish path on the rented box and never
decides promotion.

No host paths, IPs, ports, or systemd unit names are hardcoded — SITE / STATE /
SSH come from ``affine_bench`` env config and the rented pod handle.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import BENCH_PROFILE
from .bench_job import king_sampling_for_bench


def make_host_bench_run(
    *,
    box_exec_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    allow_sampling_only: bool = False,
) -> Callable[..., Mapping[str, Any]]:
    """Build a ``run_fn`` for ``BenchJobService``.

    When ``box_exec_fn`` is provided it receives
    ``(head=, profile=, box_id=, sampling=, request_name=)`` and must run the
    existing affine_bench node path (fetch king → node-run → collect) on that
    box, returning a result dict. Comparison-model settings are never applied
    to Affine kings (registry ``king_profile`` / presence_penalty 0).

    Without ``box_exec_fn``, refuse success unless ``allow_sampling_only``
    (tests only) — never fake a published bench on a live GREEN.
    """

    def run_fn(
        *,
        head: str,
        profile: str = BENCH_PROFILE,
        box_id: Optional[str] = None,
        request_name: Optional[str] = None,
        sampling: Optional[Mapping[str, Any]] = None,
        **_kw: Any,
    ) -> Dict[str, Any]:
        st = dict(sampling or king_sampling_for_bench(head, profile=profile))
        if float(st.get('presence_penalty', 1.0)) != 0.0:
            raise RuntimeError('Affine king bench requires presence_penalty 0')
        if box_exec_fn is not None:
            out = dict(box_exec_fn(
                head=head,
                profile=profile,
                box_id=box_id,
                sampling=st,
                request_name=request_name,
            ))
            out.setdefault('profile', profile)
            out.setdefault('presence_penalty', 0.0)
            out.setdefault('sampling', st)
            return out
        if allow_sampling_only:
            return {
                'ok': True,
                'profile': profile,
                'presence_penalty': 0.0,
                'sampling': st,
                'box_id': box_id,
                'request_name': request_name,
                'benchmarks': 0,
                'note': 'sampling-only (allow_sampling_only)',
            }
        raise RuntimeError(
            'PIT bench box executor not wired; refusing sampling-only success'
        )

    return run_fn


def make_pe_box_exec(
    *,
    list_pods_fn: Optional[Callable[[], List[Mapping[str, Any]]]] = None,
    rent_meta_fn: Optional[Callable[[str], Optional[Mapping[str, Any]]]] = None,
    poll_s: float = 30.0,
    max_wall_s: float = 7200.0,
) -> Callable[..., Mapping[str, Any]]:
    """Run the existing affine_bench king path on a PIT-rented pe box.

    Uses ``pe_common.pod_of`` for SSH and ``affine_bench`` fetch / node-run /
    collect / registry (king_profile → presence_penalty 0). Does not rent or
    release the box (BenchJobService owns lifecycle).
    """

    def box_exec_fn(
        *,
        head: str,
        profile: str = BENCH_PROFILE,
        box_id: Optional[str] = None,
        sampling: Optional[Mapping[str, Any]] = None,
        request_name: Optional[str] = None,
        **_kw: Any,
    ) -> Dict[str, Any]:
        from ops.parallel_eval.live import pe_common as PC
        from bench.affine_bench import daemon as D
        from bench.affine_bench import registry as R
        from bench.affine_bench import config as BC
        from bench.affine_bench.fetch import descriptor, fetch as fetch_king

        st = dict(sampling or king_sampling_for_bench(head, profile=profile))
        row: Optional[Mapping[str, Any]] = None
        if list_pods_fn is not None and box_id:
            for r in list(list_pods_fn() or []):
                if str(r.get('id') or r.get('box_id') or '') == str(box_id):
                    row = r
                    break
        if row is None and rent_meta_fn is not None and box_id:
            row = rent_meta_fn(str(box_id))
        if row is None:
            raise RuntimeError('bench box %s not listed for exec' % (box_id,))
        pod = PC.pod_of(dict(row))

        class _Node:
            def run(self, cmd, inp=None, timeout=120):
                return pod.run(cmd, inp=inp, timeout=timeout, check=False)

            def out(self, cmd, timeout=120):
                try:
                    p = self.run(cmd, timeout=timeout)
                except Exception:
                    return ''
                return (getattr(p, 'stdout', None) or '') if p is not None else ''

        node = _Node()
        # Point daemon helpers at this rented box for the duration of the oneshot.
        prev_node, prev_pod = D.NODE, dict(D.C_POD)
        D.NODE = node
        D.C_POD.clear()
        D.C_POD.update({
            'id': row.get('id') or box_id,
            'name': row.get('name'),
            'huid': row.get('huid'),
            'gpu': row.get('gpu') or row.get('gpu_model') or 'B300',
            'cost': row.get('price_usd_h') or row.get('cost'),
            'bootstrapped': True,
        })
        try:
            if descriptor(head) is None:
                raise RuntimeError('no king descriptor for %s' % head[:12])
            fr = fetch_king(head, node)
            if not (fr.get('status') or '').startswith('rc0') and 'verified' not in str(fr.get('out') or '').lower():
                # fetch_king returns rcN; accept only clean fetches.
                if fr.get('status') != 'rc0':
                    raise RuntimeError('king fetch failed: %s' % fr)
            mid = 'king-%s' % head[:16]
            model = dict(
                model_id=mid,
                king=head,
                label='king-%s' % head[:8],
                kind='king',
                green=True,
                reasoning_parser='qwen3',
                why='pit-green',
            )
            bs = R.benchmarks(enabled_only=True, runnable_only=True)
            bids = [b['id'] for b in bs]
            if not bids:
                raise RuntimeError('no runnable benchmarks')
            # Ensure sampling profile is the king pp0 profile (comparison models untouched).
            resolved = R.resolve_settings(mid, kind='king')
            if resolved.get('profile') != profile or float(resolved.get('presence_penalty', 1)) != 0.0:
                raise RuntimeError('king sampling mismatch: %r' % resolved)
            D.launch(model, bids)
            started = time.time()
            while time.time() - started < float(max_wall_s):
                run = D.S.get('running')
                if not run:
                    raise RuntimeError('bench launch lost running state')
                st_node = D.node_status(run['run_key'], run.get('pid'))
                if st_node is None:
                    time.sleep(float(poll_s))
                    continue
                if st_node.get('done') == '1':
                    scores = D.collect(run)
                    D.S['running'] = None
                    D.save()
                    return {
                        'ok': True,
                        'profile': profile,
                        'presence_penalty': 0.0,
                        'sampling': st,
                        'benchmarks': len(scores),
                        'scores': scores,
                        'box_id': box_id,
                        'request_name': request_name,
                        'node_root': getattr(BC, 'NODE_ROOT', None),
                    }
                if st_node.get('alive') != '1' and st_node.get('done') != '1':
                    raise RuntimeError('bench runner died: %s' % st_node)
                time.sleep(float(poll_s))
            raise RuntimeError('bench wall timeout (%ss); never promotes' % int(max_wall_s))
        finally:
            D.NODE = prev_node
            D.C_POD.clear()
            D.C_POD.update(prev_pod)

    return box_exec_fn


def make_job_log_tail(
    *,
    read_fn: Optional[Callable[..., str]] = None,
) -> Callable[..., str]:
    """Return last lines of the box ``job.log`` (or host substitute)."""

    def log_tail_fn(*, box_id: Optional[str] = None, job: Any = None, **_kw: Any) -> str:
        if read_fn is not None:
            return str(read_fn(box_id=box_id, job=job) or '')
        meta = getattr(job, 'meta', None) or {}
        if isinstance(meta, Mapping) and meta.get('job_log_tail'):
            return str(meta.get('job_log_tail'))
        return ''

    return log_tail_fn


def summarize_bench_event(event: Mapping[str, Any]) -> str:
    """Compact one-line summary for host logs (no secrets)."""
    return json.dumps({
        'event': event.get('event'),
        'head': (event.get('head') or '')[:12],
        'job_id': event.get('job_id'),
        'ok': event.get('ok'),
        'error': (event.get('error') or '')[:160],
    }, sort_keys=True)
