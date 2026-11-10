#!/usr/bin/env python3
"""Minimal patch for the LIVE affine-gate-ledger unit file.

Target: the file the live ``affine-gate-ledger`` unit already runs
(``20261026-promotion-gate/code/gate_ledger.py``). Do NOT apply this to the
PR rule-text copies under ``ops/zbar_flat`` / ``ops/green_only_reveal`` in a
way that diverges them.

What it changes (PIT epochs only):
  * Wrap ``entry`` — never replace it wholesale.
  * GREEN ⇒ ``promoted`` only when HEAD-CHAIN (via the live file's existing
    ``PE_ST``) lists the candidate.
  * Archived attempt-1 with a still-returned entry ⇒ ``superseded``.
  * Non-PIT rows are returned byte-identical to the original ``entry``.

Result for the lost GREENs: e1062 becomes ``rejected`` now; once archived by
a sanctioned rerun (and ``entry`` still returns a row) it becomes
``superseded``.

Print the snippet with ``python3 -m ops.parallel_eval.pit.artifacts.gate_ledger_live_patch``.
This file is documentation + snippet, not an auto-editor of the live host.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Snippet to splice into live gate_ledger.py AFTER the existing entry().
# Assumes PE_ST, EP, LIFE, rj already exist — do not invent host-path fallbacks.
# ---------------------------------------------------------------------------

LIVE_SNIPPET = r'''
# --- PIT sanctioned-rerun reconcile (2026-11-10) ---------------------------
# Wrap entry(); never rebuild the public row. HEAD-CHAIN under existing PE_ST.
HEAD_CHAIN = PE_ST / 'HEAD-CHAIN.json'
_entry_orig = entry


def _head_chain_candidates():
    doc = rj(HEAD_CHAIN) or {}
    out = set()
    for row in (doc.get('applied') or []):
        if isinstance(row, dict):
            for k in ('candidate', 'new_head'):
                v = row.get(k)
                if isinstance(v, str) and len(v) == 64:
                    out.add(v)
    h = doc.get('head') or doc.get('current_head')
    if isinstance(h, str) and len(h) == 64:
        out.add(h)
    return out


def _decision_archived(e):
    return (EP / str(e) / 'attempt-1' / 'DECISION.json').exists()


def _is_pit_epoch(e):
    return (LIFE / 'epochs' / str(e) / 'PIT-DISPATCHED.json').exists()


def entry(e):
    r = _entry_orig(e)
    if r is None or not _is_pit_epoch(e):
        return r
    # PIT only: adjust verdict / reconcile / superseded. All other fields stay.
    out = dict(r)
    if _decision_archived(e) and not (EP / str(e) / 'DECISION.json').exists():
        out['verdict'] = 'superseded'
        out['reconcile'] = 'archived-attempt'
        out['superseded'] = True
        return out
    if out.get('decision') == 'green':
        if out.get('candidate') in _head_chain_candidates():
            out['verdict'] = 'promoted'
            out['reconcile'] = 'head-chain'
            out.pop('superseded', None)
        else:
            out['verdict'] = 'rejected'
            out['reconcile'] = 'green-without-head-chain'
            out.pop('superseded', None)
    return out
# --- end PIT reconcile snippet --------------------------------------------
'''


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--print', action='store_true', help='print the live snippet')
    args = p.parse_args()
    print(LIVE_SNIPPET)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
