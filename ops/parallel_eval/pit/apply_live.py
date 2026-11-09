"""Live GREEN_FRESH apply path (SPEC §3.1, BRIEF §1.9).

GREEN_FRESH: verify bf16 C + M(C) uploads, then hand to pe_apply / HEAD-CHAIN.
GREEN_STALE: paid, merge_blocked, destroy box state — never applied.
RED / NO_DECISION: destroy box state; RED earns nothing (last-10-GREEN rule).
Never promote on timeout or fail-closed error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from .config import PitConfig
from .state import ItemRecord


class ApplyLiveError(RuntimeError):
    pass


class PromoteRefused(ApplyLiveError):
    """Timeout / fail-closed / incomplete probe must never become GREEN."""


@dataclass
class LiveApplyResult:
    applied: bool
    decision: str
    merge_blocked: bool
    paid: bool
    destroyed_box_state: bool
    head_advanced: bool
    pe_apply_accepted: bool = False
    master_verified: bool = False
    candidate_verified: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            'applied': self.applied,
            'decision': self.decision,
            'merge_blocked': self.merge_blocked,
            'paid': self.paid,
            'destroyed_box_state': self.destroyed_box_state,
            'head_advanced': self.head_advanced,
            'pe_apply_accepted': self.pe_apply_accepted,
            'master_verified': self.master_verified,
            'candidate_verified': self.candidate_verified,
            'detail': dict(self.detail),
        }


def refuse_if_unpromotable(scored: Mapping[str, Any]) -> None:
    """Never promote on timeout, partial probe, or fail-closed."""
    decision = str(scored.get('decision') or '').lower()
    if scored.get('partial') or scored.get('timeout') or scored.get('incomplete'):
        raise PromoteRefused('never promote on partial/timeout/incomplete')
    if scored.get('fail_closed') or scored.get('fp32_fail_closed'):
        raise PromoteRefused('never promote on fail-closed')
    if decision in ('pending', 'no_decision', 'timeout', 'error'):
        raise PromoteRefused('decision %r is not promotable' % decision)


def destroy_box_state(
    destroy_fn: Callable[..., Mapping[str, Any]],
    *,
    box_id: Optional[str],
    candidate_id: Optional[str],
    reason: str,
) -> Dict[str, Any]:
    return dict(destroy_fn(box_id=box_id, candidate_id=candidate_id, reason=reason))


def verify_green_uploads(
    verify_fn: Callable[..., Mapping[str, Any]],
    *,
    candidate_id: str,
    candidate_sha256: str,
    master_digest: str,
) -> Dict[str, Any]:
    out = dict(verify_fn(
        candidate_id=candidate_id,
        candidate_sha256=candidate_sha256,
        master_digest=master_digest,
    ))
    if not out.get('candidate_ok') or not out.get('master_ok'):
        raise ApplyLiveError('GREEN upload sha verify failed')
    return out


@dataclass
class LiveApplicator:
    config: PitConfig
    accept_pit_decision: Optional[Callable[[Mapping[str, Any]], Any]] = None
    verify_uploads: Optional[Callable[..., Mapping[str, Any]]] = None
    destroy_state: Optional[Callable[..., Mapping[str, Any]]] = None
    upload_master: Optional[Callable[..., Mapping[str, Any]]] = None
    upload_candidate: Optional[Callable[..., Mapping[str, Any]]] = None

    def apply(self, item: ItemRecord, *, scored: Mapping[str, Any]) -> LiveApplyResult:
        if not self.config.may_write_live_artifacts():
            raise ApplyLiveError('live apply refused: mode is not live or revert is set')
        decision = (item.decision or str(scored.get('decision') or '')).lower()
        if decision == 'green':
            decision = 'green_fresh' if item.fresh else 'green_stale'

        if decision == 'green_fresh':
            return self._apply_fresh(item, scored=scored)
        if decision == 'green_stale':
            return self._stale_or_red(item, decision='green_stale', paid=True, merge_blocked=True)
        if decision == 'red':
            # Last-10-GREEN rule: REDs earn nothing.
            if item.paid:
                raise ApplyLiveError('RED must not be paid')
            return self._stale_or_red(item, decision='red', paid=False, merge_blocked=False)
        if decision in ('no_decision',):
            return self._stale_or_red(item, decision='no_decision', paid=False, merge_blocked=False)
        raise ApplyLiveError('unsupported decision %r' % decision)

    def _apply_fresh(self, item: ItemRecord, *, scored: Mapping[str, Any]) -> LiveApplyResult:
        refuse_if_unpromotable({**dict(scored), 'decision': 'green'})
        if item.merge_blocked or item.fresh is False:
            raise ApplyLiveError('fresh GREEN cannot be merge_blocked/stale')
        cand = item.candidate_id
        if not cand or len(cand) != 64:
            raise ApplyLiveError('missing candidate_id')
        # Upload masters only on GREEN (SPEC §3.1).
        if self.upload_candidate is not None:
            self.upload_candidate(candidate_id=cand, path=(item.meta or {}).get('candidate_path'))
        if self.upload_master is not None:
            self.upload_master(candidate_id=cand, path=(item.meta or {}).get('candidate_master_path'))
        master_digest = (item.meta or {}).get('candidate_master_digest') or (item.meta or {}).get('master_digest')
        cand_sha = (item.meta or {}).get('candidate_sha256') or cand
        verified = {'candidate_ok': True, 'master_ok': True}
        if self.verify_uploads is not None:
            if not master_digest:
                raise ApplyLiveError('missing master digest for GREEN verify')
            verified = verify_green_uploads(
                self.verify_uploads,
                candidate_id=cand,
                candidate_sha256=str(cand_sha),
                master_digest=str(master_digest),
            )
        pe_ok = False
        if self.accept_pit_decision is not None:
            payload = {
                'source': 'pit',
                'epoch': item.epoch,
                'candidate': cand,
                'training_head': item.bound_head,
                'green_at': (item.meta or {}).get('decided_at_mono'),
                'decision': 'green_fresh',
                'merge_blocked': False,
                'files': (item.meta or {}).get('files'),
            }
            self.accept_pit_decision(payload)
            pe_ok = True
        return LiveApplyResult(
            applied=True,
            decision='green_fresh',
            merge_blocked=False,
            paid=True,
            destroyed_box_state=False,
            head_advanced=True,
            pe_apply_accepted=pe_ok,
            master_verified=bool(verified.get('master_ok')),
            candidate_verified=bool(verified.get('candidate_ok')),
            detail={'candidate': cand, 'verify': verified},
        )

    def _stale_or_red(
        self,
        item: ItemRecord,
        *,
        decision: str,
        paid: bool,
        merge_blocked: bool,
    ) -> LiveApplyResult:
        destroyed = False
        if self.destroy_state is not None:
            destroy_box_state(
                self.destroy_state,
                box_id=item.box_id,
                candidate_id=item.candidate_id,
                reason=decision,
            )
            destroyed = True
        return LiveApplyResult(
            applied=False,
            decision=decision,
            merge_blocked=merge_blocked,
            paid=paid,
            destroyed_box_state=destroyed,
            head_advanced=False,
            detail={'candidate': item.candidate_id},
        )
