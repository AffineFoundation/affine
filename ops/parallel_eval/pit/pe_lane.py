"""PIT eval-lane seam (BRIEF §1.3).

Host RA ``pe_lane.py`` owns boot/serve/run_probe/release. This git port exposes
the call shapes PIT needs and loads the host module when present. Shadow mode
never promotes on a partial probe.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from . import GATE_K_CAND, GATE_MAX_TOKENS, MAX_SEQS


class LaneError(RuntimeError):
    pass


REQUIRED_SERVE_FLAGS = {
    'dtype': 'bfloat16',
    'max_model_len': 69632,
    'max_num_seqs': MAX_SEQS,
    'fp8': False,
}


def verify_serve_flags(flags: Mapping[str, Any]) -> Dict[str, Any]:
    """Parity guard (DESIGN §7 / pe_lane.verify_serve_flags)."""
    bad = []
    if flags.get('dtype') != REQUIRED_SERVE_FLAGS['dtype']:
        bad.append('dtype')
    if int(flags.get('max_model_len') or 0) != REQUIRED_SERVE_FLAGS['max_model_len']:
        bad.append('max_model_len')
    if int(flags.get('max_num_seqs') or 0) != REQUIRED_SERVE_FLAGS['max_num_seqs']:
        bad.append('max_num_seqs')
    if flags.get('fp8') not in (False, None, 0, '0', 'false', 'False'):
        bad.append('fp8')
    if bad:
        raise LaneError('serve flag parity failed: %s' % ','.join(bad))
    return dict(ok=True, flags=dict(REQUIRED_SERVE_FLAGS))


def assert_complete_probe(result: Mapping[str, Any], *, planned_samples: int) -> None:
    """Never promote on partial / incomplete results."""
    if result.get('partial') is not False:
        raise LaneError('incomplete result: partial')
    samples = result.get('samples')
    if type(samples) is not int or samples != int(planned_samples):
        raise LaneError('incomplete result: samples %r != planned %d' % (samples, planned_samples))


PodHandle = Union[str, Mapping[str, Any], Any]


@dataclass
class LaneService:
    require_host: bool = False
    host_boot: Optional[Callable[..., Mapping[str, Any]]] = None
    host_serve: Optional[Callable[..., Mapping[str, Any]]] = None
    host_run_probe: Optional[Callable[..., Mapping[str, Any]]] = None
    host_release: Optional[Callable[..., Mapping[str, Any]]] = None

    def boot(self, pod: PodHandle, **kwargs: Any) -> Dict[str, Any]:
        if self.require_host and self.host_boot is None:
            raise LaneError('live lane boot requires host pe_lane module')
        if self.host_boot is None:
            box_id = pod if isinstance(pod, str) else str((pod or {}).get('id') or pod)
            return dict(ok=True, box_id=box_id, synthetic=True)
        return dict(self.host_boot(pod, **kwargs))

    def serve(
        self,
        pod: PodHandle,
        candidate_id: str,
        *,
        max_num_seqs: int = MAX_SEQS,
        gmu: str = '0.9',
        flags: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        flags = dict(flags or {
            'dtype': 'bfloat16',
            'max_model_len': 69632,
            'max_num_seqs': max_num_seqs,
            'fp8': False,
        })
        verify_serve_flags(flags)
        if self.require_host and self.host_serve is None:
            raise LaneError('live lane serve requires host pe_lane module')
        if self.host_serve is None:
            box_id = pod if isinstance(pod, str) else str((pod or {}).get('id') or pod)
            return dict(
                ok=True, box_id=box_id, candidate_id=candidate_id,
                flags=flags, synthetic=True,
            )
        return dict(self.host_serve(pod, candidate_id, gmu=gmu, max_num_seqs=max_num_seqs))

    def run_probe(
        self,
        pod: PodHandle,
        candidate_id: str,
        *,
        indices: Optional[Sequence[int]] = None,
        plan_seed: Optional[str] = None,
        indices_count: Optional[int] = None,
        plan_seed_handle: Optional[str] = None,
        k: int = GATE_K_CAND,
        max_tokens: int = GATE_MAX_TOKENS,
        env_id: str = 'affine_gatepool',
        result: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if k != GATE_K_CAND or max_tokens != GATE_MAX_TOKENS:
            raise LaneError('refuse non-live gate probe settings')
        seed = plan_seed or plan_seed_handle
        if not seed:
            raise LaneError('plan_seed handle required')
        idx_list: Optional[List[int]] = None
        if indices is not None:
            idx_list = list(indices)
        if self.host_run_probe is not None or self.require_host:
            if self.require_host and self.host_run_probe is None:
                raise LaneError('live lane run_probe requires host pe_lane module')
            if idx_list is None:
                raise LaneError('indices list required for host run_probe')
            planned = len(idx_list) * int(k)
            out = dict(self.host_run_probe(
                pod, candidate_id, idx_list, seed,
                k=k, max_tokens=max_tokens, env_id=env_id,
            ))
        else:
            if idx_list is None:
                if indices_count is None:
                    raise LaneError('indices list or indices_count required')
                planned = int(indices_count) * int(k)
            else:
                planned = len(idx_list) * int(k)
            if result is None:
                raise LaneError('synthetic lane requires injected result')
            out = dict(result)
        assert_complete_probe(out, planned_samples=planned)
        out.setdefault('k', k)
        out.setdefault('dtype', 'bfloat16')
        out.setdefault('planned_samples', planned)
        out['candidate_id'] = candidate_id
        return out


def load_host_pe_lane(module_path: Optional[str] = None):
    for name in ([module_path] if module_path else []) + ['pe_lane']:
        if not name:
            continue
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def bind_host_lane_service(service: LaneService, *, module_path: Optional[str] = None) -> LaneService:
    mod = load_host_pe_lane(module_path)
    if mod is None:
        return service
    service.host_boot = getattr(mod, 'boot', None)
    service.host_serve = getattr(mod, 'serve', None)
    service.host_run_probe = getattr(mod, 'run_probe', None)
    service.host_release = getattr(mod, 'release', None)
    return service
