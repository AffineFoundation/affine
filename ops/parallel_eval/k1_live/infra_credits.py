"""K1 infra credits — pay-only GREEN for infra-lost epochs (Jacob 2026-11-10 11:00 CR).

Reads ``staged/K1-INFRA-CREDITS.json``. When UID, hotkey, reg_block, candidate sha
and evidence hash all match and z>1, the K1 verdict becomes green tagged
``credited_infra_loss``. Once accepted the credit is final for pay; a later
rerun only affects HEAD. Mismatches are ignored and logged.

Never touches HEAD-CHAIN, APPLY, GREENS, DECISIONS, items, or the queue.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

LOG = logging.getLogger('k1.infra_credits')

SCHEMA_VERSION = 1


def example_doc() -> Dict[str, Any]:
    """Format example only — real rows are filled on the host by the operator."""
    return {
        'version': SCHEMA_VERSION,
        'enabled': True,
        'note': (
            'Jacob 2026-11-10 11:00 CR: GREENs lost to infra are credited for pay. '
            'Host fills real rows for e1060/665, e1062/847, e1065/681, e1073/898.'
        ),
        'credits': [
            {
                'epoch': 0,
                'uid': 0,
                'hotkey': '5REPLACE_WITH_SS58',
                'reg_block': 0,
                'candidate': '0' * 64,
                'evidence_hash': '0' * 64,
                'z': 1.01,
                'note': 'example only — not a live credit',
            },
        ],
    }


def load_infra_credits(path: Path) -> Dict[str, Any]:
    """Load credits file; empty/disabled on missing or corrupt (fail soft).

    Always logs on exception — never silently disable.
    """
    try:
        doc = json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        LOG.info('infra_credits_missing path=%s err=%s', path, str(exc)[:200])
        return {'enabled': False, 'credits': [], 'error': str(exc)[:200]}
    except Exception as exc:
        LOG.exception('infra_credits_unreadable path=%s', path)
        return {'enabled': False, 'credits': [], 'error': str(exc)[:200]}
    if not isinstance(doc, dict):
        return {'enabled': False, 'credits': [], 'error': 'not-a-dict'}
    credits = doc.get('credits') if isinstance(doc.get('credits'), list) else []
    return {
        'enabled': bool(doc.get('enabled') is True),
        'credits': [c for c in credits if isinstance(c, dict)],
        'version': doc.get('version'),
        'note': doc.get('note'),
    }


def _hex64(val: Any) -> Optional[str]:
    if isinstance(val, str) and len(val) == 64:
        return val
    return None


def match_credit(
    credit: Mapping[str, Any],
    *,
    uid: Any,
    hotkey: Any,
    reg_block: Any,
    candidate: Any,
    evidence_hash: Any,
    z: Any,
) -> Tuple[bool, str]:
    """Return (ok, reason). All identity fields must match; z must be > 1."""
    try:
        if int(credit.get('uid')) != int(uid):
            return False, 'uid-mismatch'
    except (TypeError, ValueError):
        return False, 'uid-invalid'
    if str(credit.get('hotkey') or '') != str(hotkey or ''):
        return False, 'hotkey-mismatch'
    try:
        if int(credit.get('reg_block')) != int(reg_block):
            return False, 'reg_block-mismatch'
    except (TypeError, ValueError):
        return False, 'reg_block-invalid'
    c_cred = _hex64(credit.get('candidate'))
    c_got = _hex64(candidate)
    if not c_cred or not c_got or c_cred != c_got:
        return False, 'candidate-mismatch'
    e_cred = _hex64(credit.get('evidence_hash'))
    e_got = _hex64(evidence_hash)
    if not e_cred or not e_got or e_cred != e_got:
        return False, 'evidence_hash-mismatch'
    try:
        z_cred = float(credit.get('z'))
        z_got = float(z) if z is not None else float('nan')
    except (TypeError, ValueError):
        return False, 'z-invalid'
    if not (z_cred > 1.0 and z_got > 1.0):
        return False, 'z-not-gt-1'
    return True, 'ok'


def credit_for_epoch(doc: Mapping[str, Any], epoch: int) -> Optional[Dict[str, Any]]:
    if not doc.get('enabled'):
        return None
    for row in doc.get('credits') or []:
        try:
            if int(row.get('epoch')) == int(epoch):
                return dict(row)
        except (TypeError, ValueError):
            continue
    return None


def apply_infra_credit(
    rec: Dict[str, Any],
    *,
    turn: Mapping[str, Any],
    decision: Mapping[str, Any],
    credit: Mapping[str, Any],
    log_fn=None,
) -> Dict[str, Any]:
    """Mutate a verdict record when credit matches; else log and leave unchanged.

    On match: ``verdict='green'``, ``credited_infra_loss=True``, ``credit_final=True``.
    """
    log = log_fn or (lambda **kw: LOG.info('%s', kw))
    cand = decision.get('candidate') or rec.get('candidate') or credit.get('candidate')
    evidence = (
        turn.get('file_sha256')
        or turn.get('evidence_hash')
        or decision.get('evidence_hash')
        or decision.get('commitment')
    )
    z = (decision.get('evaluated') or {}).get('z')
    if z is None:
        z = rec.get('z')
    if z is None:
        z = credit.get('z')
    ok, reason = match_credit(
        credit,
        uid=turn.get('uid', rec.get('uid')),
        hotkey=turn.get('hotkey', rec.get('hotkey')),
        reg_block=turn.get('reg_block', rec.get('reg_block')),
        candidate=cand,
        evidence_hash=evidence,
        z=z,
    )
    if not ok:
        log(
            event='infra_credit_mismatch',
            epoch=rec.get('epoch_index') or credit.get('epoch'),
            reason=reason,
            uid=turn.get('uid'),
            candidate=(str(cand)[:12] if cand else None),
        )
        return rec
    out = dict(rec)
    out['signed_verdict'] = out.get('signed_verdict') or out.get('verdict')
    out['verdict'] = 'green'
    out['credited_infra_loss'] = True
    out['credit_final'] = True
    out['infra_credit_note'] = str(credit.get('note') or 'credited_infra_loss')[:200]
    if out.get('z') is None:
        try:
            out['z'] = float(credit.get('z'))
        except (TypeError, ValueError):
            pass
    if not out.get('candidate'):
        out['candidate'] = _hex64(cand)
    log(
        event='infra_credit_accepted',
        epoch=out.get('epoch_index'),
        uid=out.get('uid'),
        candidate=(out.get('candidate') or '')[:12],
        z=out.get('z'),
    )
    return out


def apply_credits_to_verdicts(
    recs: Sequence[Mapping[str, Any]],
    *,
    credits_doc: Mapping[str, Any],
    turns_by_epoch: Mapping[int, Mapping[str, Any]],
    decisions_by_epoch: Mapping[int, Mapping[str, Any]],
    log_fn=None,
) -> List[Dict[str, Any]]:
    """Apply matching infra credits onto a verdicts list (pure; for tests)."""
    out: List[Dict[str, Any]] = []
    for rec in recs:
        row = dict(rec)
        try:
            epoch = int(row.get('epoch_index'))
        except (TypeError, ValueError):
            out.append(row)
            continue
        # Already accepted credit is final for pay — do not strip.
        if row.get('credit_final') and row.get('credited_infra_loss'):
            out.append(row)
            continue
        credit = credit_for_epoch(credits_doc, epoch)
        if credit is None:
            out.append(row)
            continue
        turn = turns_by_epoch.get(epoch) or {
            'uid': row.get('uid'),
            'hotkey': row.get('hotkey'),
            'reg_block': row.get('reg_block'),
            'file_sha256': row.get('evidence_hash') or row.get('file_sha256'),
        }
        decision = decisions_by_epoch.get(epoch) or {
            'candidate': row.get('candidate'),
            'evaluated': {'z': row.get('z')},
            'evidence_hash': row.get('evidence_hash'),
        }
        out.append(apply_infra_credit(
            row, turn=turn, decision=decision, credit=credit, log_fn=log_fn,
        ))
    return out
