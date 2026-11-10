#!/usr/bin/env python3
"""Minimal patch for the LIVE affine-gate-ledger unit file.

Target: the file the live ``affine-gate-ledger`` unit already runs
(``20261026-promotion-gate/code/gate_ledger.py``). Do NOT apply this to the
PR rule-text copies under ``ops/zbar_flat`` / ``ops/green_only_reveal`` in a
way that diverges them.

What it changes (PIT epochs only):
  * GREEN ⇒ ``promoted`` only when HEAD-CHAIN (via the live file's existing
    ``PE_ST``) lists the candidate.
  * Archived ``attempt-1/DECISION.json`` with no live DECISION ⇒ ``superseded``.
  * Pre-PE / non-PIT rows keep their existing verdicts untouched.

Result for the lost GREENs: e1062 becomes ``rejected`` now; once archived by
a sanctioned rerun it becomes ``superseded``.

Print the snippet with ``python3 -m ops.parallel_eval.pit.artifacts.gate_ledger_live_patch``.
This file is documentation + snippet, not an auto-editor of the live host.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Snippet to splice into live gate_ledger.py (replace ``entry``).
# Assumes the live module already defines PE_ST, EP, LIFE, rj, mtime, cr,
# manifest_window, pre_probe — do not invent host-path fallbacks here.
# ---------------------------------------------------------------------------

LIVE_SNIPPET = r'''
# --- PIT sanctioned-rerun reconcile (2026-11-10) ---------------------------
# HEAD-CHAIN lives under the live unit's existing PE_ST (never invent a path).
HEAD_CHAIN = PE_ST / 'HEAD-CHAIN.json'


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
    d = EP / str(e); dec = rj(d / 'DECISION.json')
    archived = _decision_archived(e)
    if (not dec or dec.get('decision') not in ('green', 'red')) and archived:
        dec = rj(d / 'attempt-1' / 'DECISION.json')
    if not dec or dec.get('decision') not in ('green', 'red'):
        return None
    life = LIFE / 'epochs' / str(e)
    applied = rj(life / 'PROMOTION-GATE.json') or {}
    tl = rj(d / 'TIMELINE.json') or {}
    ev = dec.get('evaluated') or {}
    start, deadline = manifest_window(e)
    is_pit = _is_pit_epoch(e)
    if is_pit and archived and not (d / 'DECISION.json').exists():
        verdict = 'superseded'
        reconcile = 'archived-attempt'
    elif is_pit and dec['decision'] == 'green':
        if dec.get('candidate') in _head_chain_candidates():
            verdict = 'promoted'; reconcile = 'head-chain'
        else:
            verdict = 'rejected'; reconcile = 'green-without-head-chain'
    else:
        # Non-PIT (pre-PE) history: keep the classic green→promoted mapping.
        verdict = 'promoted' if dec['decision'] == 'green' else 'rejected'
        reconcile = 'pre-pit' if not is_pit else ('red' if dec['decision'] == 'red' else 'head-chain')
    return dict(
        epoch=e, head=dec['head'], candidate=dec['candidate'], decision=dec['decision'], verdict=verdict,
        applied=applied.get('decision'), mode=dec.get('mode'),
        z=ev.get('z'), candidate_cap_rate=ev.get('c_cap_rate'), n=ev.get('n'),
        head_pass=ev.get('h_pass'), candidate_pass=ev.get('c_pass'),
        batch_tasks_dropped=dec.get('batch_overlap'), reasons=dec.get('reasons'),
        reason=('; '.join(dec.get('reasons') or []) or (('z %s %.1f and cap-hit < %.2f' % ('>=' if dec.get('z_inclusive') else '>', dec.get('z_min', -1.0), dec.get('cap_max', 0.20)))
                if verdict == 'promoted' else None)),
        thresholds=dict(z_min=dec.get('z_min', -1.0), z_inclusive=bool(dec.get('z_inclusive')), cap_max=dec.get('cap_max', 0.20),
                        min_tasks=dec.get('min_tasks', 200)), red_credit=dec.get('red_credit'),
        commitment=dec.get('commitment'),
        timestamps=dict(gather_start=start, upload_deadline=deadline,
                        training_result=tl.get('result_mtime') or mtime(life / 'TRAINING-RESULT.json'),
                        pre_probe=pre_probe(dec['head'], start),
                        post_probe=dict(start=tl.get('fetch_start') or tl.get('h_ready'), fetch_start=tl.get('fetch_start'),
                                        fetched=tl.get('fetched'), serve_start=tl.get('serve_start'), served=tl.get('served'),
                                        run_start=tl.get('run_start'), end=tl.get('run_done')),
                        decided_at=dec.get('decided_at'), finalized=mtime(life / 'FINALIZED.json')),
        decided_at_cr=dec.get('decided_at_cr'), timing_s=dec.get('timing'),
        reconcile=reconcile,
        superseded=(verdict == 'superseded'),
        pit=is_pit)
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
