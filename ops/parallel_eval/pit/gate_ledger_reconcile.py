"""Reconcile PE promotion-gate ledger rows against HEAD-CHAIN + archives.

Owned by PIT (PR #96). Live ``affine-gate-ledger`` runs a different file
(``20261026-promotion-gate/code/gate_ledger.py``); see
``artifacts/gate_ledger_live_patch.py`` for the minimal operator patch.

Rules:
  * Apply HEAD-CHAIN promotion checks ONLY to PIT epochs (``PIT-DISPATCHED.json``).
  * Pre-PE / non-PIT ledger rows are left untouched.
  * A PIT GREEN is ``promoted`` only when a HEAD-CHAIN row exists for its candidate.
  * Archived attempt-1 decisions are ``superseded`` (never listed as promoted).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set


def _rj(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def _wj(path: Path, obj: Any, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def head_chain_candidates(chain: Mapping[str, Any]) -> set:
    found = set()
    for row in chain.get('applied') or []:
        if not isinstance(row, Mapping):
            continue
        for key in ('candidate', 'new_head'):
            val = row.get(key)
            if isinstance(val, str) and len(val) == 64:
                found.add(val)
    head = chain.get('head') or chain.get('current_head')
    if isinstance(head, str) and len(head) == 64:
        found.add(head)
    return found


def epoch_decision_archived(gate_epochs: Path, epoch: int) -> bool:
    return (Path(gate_epochs) / str(int(epoch)) / 'attempt-1' / 'DECISION.json').is_file()


def epoch_is_pit(life_epochs: Path, epoch: int) -> bool:
    """True when lifecycle marked the epoch as PIT-owned (PIT-DISPATCHED)."""
    return (Path(life_epochs) / str(int(epoch)) / 'PIT-DISPATCHED.json').is_file()


def reconcile_entry(
    ent: Mapping[str, Any],
    *,
    chain_candidates: set,
    archived: bool,
    is_pit: bool = False,
) -> Dict[str, Any]:
    """Return a corrected ledger entry (does not mutate input).

    Non-PIT rows are returned unchanged so pre-PE history keeps its verdicts.
    """
    out = dict(ent)
    if not is_pit:
        return out
    cand = out.get('candidate')
    decision = out.get('decision')
    if archived:
        out['verdict'] = 'superseded'
        out['superseded'] = True
        out['reconcile'] = 'archived-attempt'
        return out
    if decision == 'green':
        if cand in chain_candidates:
            out['verdict'] = 'promoted'
            out['reconcile'] = 'head-chain'
        else:
            # PIT GREEN without HEAD-CHAIN — not actually promoted (e.g. e1062).
            out['verdict'] = 'rejected'
            out['reconcile'] = 'green-without-head-chain'
            out['promoted_refused'] = True
    elif decision == 'red':
        out['verdict'] = 'rejected'
        out.setdefault('reconcile', 'red')
    return out


def reconcile_ledger(
    *,
    ledger_path: Path,
    head_chain_path: Path,
    gate_epochs: Path,
    life_epochs: Path,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Rewrite ledger rows with PIT-only HEAD-CHAIN / archive reconcile.

    Writes ``out_path`` (default: sibling ``LEDGER.reconciled.jsonl``) and a
    summary JSON. Never deletes the original ledger. Pre-PE rows unchanged.
    """
    rows = load_jsonl(Path(ledger_path))
    chain = _rj(Path(head_chain_path)) or {}
    cands = head_chain_candidates(chain if isinstance(chain, Mapping) else {})
    fixed: List[dict] = []
    n_promoted = n_rejected = n_superseded = n_untouched = 0
    for ent in rows:
        ep = int(ent.get('epoch') or -1)
        is_pit = epoch_is_pit(Path(life_epochs), ep)
        archived = epoch_decision_archived(Path(gate_epochs), ep) if is_pit else False
        before = dict(ent)
        row = reconcile_entry(
            ent, chain_candidates=cands, archived=archived, is_pit=is_pit,
        )
        fixed.append(row)
        if not is_pit:
            n_untouched += 1
            continue
        v = row.get('verdict')
        if v == 'promoted':
            n_promoted += 1
        elif v == 'superseded':
            n_superseded += 1
        else:
            n_rejected += 1
        # Keep silence on unused before.
        del before
    dest = Path(out_path) if out_path else Path(ledger_path).with_name('LEDGER.reconciled.jsonl')
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name('.%s.tmp' % dest.name)
    with open(tmp, 'w') as fh:
        for row in fixed:
            fh.write(json.dumps(row, sort_keys=True) + '\n')
    os.chmod(tmp, 0o644)
    os.replace(tmp, dest)
    summary = {
        'ok': True,
        'source': str(ledger_path),
        'out': str(dest),
        'counts': {
            'total': len(fixed),
            'promoted': n_promoted,
            'rejected': n_rejected,
            'superseded': n_superseded,
            'untouched_non_pit': n_untouched,
        },
        'reconciled_at': time.time(),
    }
    _wj(dest.with_suffix('.summary.json'), summary, 0o644)
    return summary


def reconcile_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    chain_candidates: Optional[set] = None,
    archived_epochs: Optional[set] = None,
    pit_epochs: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    """Pure helper for tests."""
    cands = set(chain_candidates or ())
    archived = set(archived_epochs or ())
    pit = set(pit_epochs or ())
    out = []
    for ent in rows:
        ep = int(ent.get('epoch') or -1)
        out.append(reconcile_entry(
            ent,
            chain_candidates=cands,
            archived=ep in archived,
            is_pit=ep in pit,
        ))
    return out
