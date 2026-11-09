"""Live GREEN_FRESH apply path (SPEC §3.1, BRIEF §1.9, CUTOVER-PLAN blocker 5).

GREEN_FRESH: verify bf16 C + M(C) uploads (**fail closed**), then hand to pe_apply
(first GREEN from a given head becomes HEAD).
GREEN_STALE / stranded: paid + merge_blocked, never becomes HEAD; destroy box state.
RED / NO_DECISION: destroy box state; RED earns nothing (last-10-GREEN rule).
Never promote on timeout or fail-closed error.

``verify_uploads`` is **mandatory** in live. Missing verifier or a failed
fp32-master / bf16 hash check raises ApplyLiveError (never treats as verified).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .checkpoint_auth import build_checkpoint_auth_payload, sign_checkpoint_authorization
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
    stranded: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            'applied': self.applied,
            'decision': self.decision,
            'merge_blocked': self.merge_blocked,
            'stranded': self.stranded or self.merge_blocked,
            'paid': self.paid,
            'destroyed_box_state': self.destroyed_box_state,
            'head_advanced': self.head_advanced,
            'pe_apply_accepted': self.pe_apply_accepted,
            'master_verified': self.master_verified,
            'candidate_verified': self.candidate_verified,
            'detail': dict(self.detail),
        }


def refuse_if_unpromotable(scored: Mapping[str, Any]) -> None:
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
    if not out.get('candidate_ok'):
        raise ApplyLiveError('bf16 candidate hash verify failed (fail closed)')
    if not out.get('master_ok'):
        raise ApplyLiveError('fp32 master hash verify failed (fail closed)')
    return out


def _write_pending_json(path: Path, doc: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + '\n')
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return str(path)


def _pending_life_dir() -> Path:
    life = os.environ.get('PE_LIFE_STATE')
    if life:
        return Path(life) / 'pending'
    pe = os.environ.get('PE_STATE')
    if pe:
        return Path(pe) / 'pending'
    return Path('/tmp/pit-pending')


@dataclass
class LiveApplicator:
    config: PitConfig
    accept_pit_decision: Optional[Callable[[Mapping[str, Any]], Any]] = None
    verify_uploads: Optional[Callable[..., Mapping[str, Any]]] = None
    destroy_state: Optional[Callable[..., Mapping[str, Any]]] = None
    upload_master: Optional[Callable[..., Mapping[str, Any]]] = None
    upload_candidate: Optional[Callable[..., Mapping[str, Any]]] = None
    checkpoint_signer: Optional[Any] = None

    def apply(self, item: ItemRecord, *, scored: Mapping[str, Any]) -> LiveApplyResult:
        if not self.config.may_write_live_artifacts():
            raise ApplyLiveError('live apply refused: mode is not live or revert is set')
        decision = (item.decision or str(scored.get('decision') or '')).lower()
        if decision == 'green':
            decision = 'green_fresh' if item.fresh else 'green_stale'

        if decision == 'green_fresh':
            return self._apply_fresh(item, scored=scored)
        if decision == 'green_stale':
            return self._stale_or_red(
                item, decision='green_stale', paid=True, merge_blocked=True, stranded=True,
            )
        if decision == 'red':
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
        # FAIL CLOSED: verify_uploads is mandatory in live (CUTOVER-PLAN blocker 5).
        if self.verify_uploads is None:
            raise ApplyLiveError('live apply refused: verify_uploads is required (fail closed)')
        if self.upload_candidate is not None:
            self.upload_candidate(candidate_id=cand, path=(item.meta or {}).get('candidate_path'))
        if self.upload_master is not None:
            self.upload_master(candidate_id=cand, path=(item.meta or {}).get('candidate_master_path'))
        master_digest = (item.meta or {}).get('candidate_master_digest') or (item.meta or {}).get('master_digest')
        cand_sha = (item.meta or {}).get('candidate_sha256')
        if not master_digest or len(str(master_digest)) != 64:
            raise ApplyLiveError('missing/invalid master digest for GREEN verify (fail closed)')
        if not cand_sha or len(str(cand_sha)) != 64:
            raise ApplyLiveError('missing/invalid bf16 candidate sha for GREEN verify (fail closed)')
        verified = verify_green_uploads(
            self.verify_uploads,
            candidate_id=cand,
            candidate_sha256=str(cand_sha),
            master_digest=str(master_digest),
        )
        meta = item.meta or {}
        files = meta.get('files') or (meta.get('checkpoint') or {}).get('files')
        ckpt = dict(id=cand)
        if files:
            ckpt['files'] = dict(files)
        optimizer_step = int(meta.get('optimizer_step') or scored.get('optimizer_step') or 0)
        auth_payload = build_checkpoint_auth_payload(
            checkpoint_id=cand,
            input_head=item.bound_head or '',
            optimizer_step=optimizer_step,
            files=ckpt.get('files'),
            publication_receipt_sha256=meta.get('publication_receipt_sha256'),
        )
        authorization = sign_checkpoint_authorization(auth_payload, signer=self.checkpoint_signer)
        pending_dir = _pending_life_dir()
        pending_path = pending_dir / ('e%d-%s.json' % (item.epoch, cand[:16]))
        pend_doc = dict(
            epoch=item.epoch,
            candidate=cand,
            training_head=item.bound_head,
            checkpoint=ckpt,
            optimizer_step=optimizer_step,
            checkpoint_authorization=authorization,
            pe_keep=True,
            source='pit',
            at=time.time(),
        )
        pending_abs = _write_pending_json(pending_path, pend_doc)
        pe_ok = False
        if self.accept_pit_decision is not None:
            payload = {
                'source': 'pit',
                'epoch': item.epoch,
                'candidate': cand,
                'training_head': item.bound_head,
                'green_at': meta.get('decided_at_mono'),
                'decision': 'green_fresh',
                'merge_blocked': False,
                'files': files,
                'pending': pending_abs,
            }
            self.accept_pit_decision(payload)
            pe_ok = True
        return LiveApplyResult(
            applied=True,
            decision='green_fresh',
            merge_blocked=False,
            stranded=False,
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
        stranded: bool = False,
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
            stranded=bool(stranded or merge_blocked),
            paid=paid,
            destroyed_box_state=destroyed,
            head_advanced=False,
            detail={'candidate': item.candidate_id},
        )
