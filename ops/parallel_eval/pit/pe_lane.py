"""PIT eval-lane seam (BRIEF §1.3).

Host RA ``pe_lane.py`` owns boot/serve/run_probe/release. This git port exposes
the call shapes PIT needs and loads the host module when present. Shadow mode
never promotes on a partial probe.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

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


@dataclass
class LaneService:
    host_boot: Optional[Callable[..., Mapping[str, Any]]] = None
    host_serve: Optional[Callable[..., Mapping[str, Any]]] = None
    host_run_probe: Optional[Callable[..., Mapping[str, Any]]] = None
    host_release: Optional[Callable[..., Mapping[str, Any]]] = None

    def boot(self, box_id: str, **kwargs: Any) -> Dict[str, Any]:
        if self.host_boot is None:
            return dict(ok=True, box_id=box_id, synthetic=True)
        return dict(self.host_boot(box_id=box_id, **kwargs))

    def serve(self, box_id: str, candidate_id: str, *, max_num_seqs: int = MAX_SEQS, flags: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        flags = dict(flags or {
            'dtype': 'bfloat16',
            'max_model_len': 69632,
            'max_num_seqs': max_num_seqs,
            'fp8': False,
        })
        verify_serve_flags(flags)
        if self.host_serve is None:
            return dict(ok=True, box_id=box_id, candidate_id=candidate_id, flags=flags, synthetic=True)
        return dict(self.host_serve(box_id=box_id, candidate_id=candidate_id, max_num_seqs=max_num_seqs, flags=flags))

    def run_probe(
        self,
        box_id: str,
        candidate_id: str,
        *,
        indices_count: int,
        plan_seed_handle: str,
        k: int = GATE_K_CAND,
        max_tokens: int = GATE_MAX_TOKENS,
        env_id: str = 'affine_gatepool',
        result: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if k != GATE_K_CAND or max_tokens != GATE_MAX_TOKENS:
            raise LaneError('refuse non-live gate probe settings')
        if not plan_seed_handle:
            raise LaneError('plan_seed handle required')
        planned = int(indices_count) * int(k)
        if self.host_run_probe is not None:
            out = dict(self.host_run_probe(
                box_id=box_id, candidate_id=candidate_id,
                plan_seed=plan_seed_handle, k=k, max_tokens=max_tokens, env_id=env_id,
            ))
        else:
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
