#!/usr/bin/env python3
"""Apply the SN120 grade-stall fix (2026-11-01) gate hunks to a gate_service.py copy (anchored; refuses on a missing anchor).
usage: patch_gate_stallfix.py FILE..."""
import sys
PAIRS = [
("""                delta_after_result=d.get('delta_after_result') is True)
""", """                delta_after_result=d.get('delta_after_result') is True,
                # stall fix 20261101 (RA/20261101-grade-stall-fix): no-progress salvage after this many seconds (0 = off)
                stall_salvage_s=float(d.get('stall_salvage_s', 300)))
"""),
("""def run_candidate(e, C, tasks, seed, f, tl, kp=None, side='C', pseed=None):""", '''def stall_salvage(slot, name, ru, planned, event, **info):
    """No-progress salvage (stall fix 20261101, Jacob's condition for the next gate switch): a run that is still 'running' on the
    pod with every sample generated and its graded count unchanged for opts()['stall_salvage_s'] (default 300 s) is finalized with
    the existing `podctl finalize-run` (partial result from the graded rows). The caller's normal result path then applies the usual
    coverage check (baseline min_cover / candidate 0.9 partial-retry) to that partial. `slot` (persisted by the caller) keeps
    the last graded count and since when it is unchanged."""
    lim = opts()['stall_salvage_s']; pg = ru.get('progress') or {}
    gen, graded = pg.get('generated'), pg.get('samples')
    if not lim or gen is None or graded is None: return None
    now = time.time(); key = '%s:%s' % (name, graded)
    if slot.get('stall_key') != key:
        slot['stall_key'] = key; slot['stall_since'] = now; return None
    if int(gen) < int(planned) or int(graded) >= int(planned) or now - float(slot['stall_since']) < lim: return None
    if slot.get('stall_salvaged') == key and now - float(slot.get('stall_salvaged_at', 0)) < 120: return None
    slot['stall_salvaged'] = key; slot['stall_salvaged_at'] = now
    try: fin = podctl('finalize-run', name, '30', timeout=90)
    except Exception as x: fin = dict(finalized=False, failed=str(x)[-200:])
    log(event=event, run=name, generated=gen, graded=graded, planned=planned, stalled_s=round(now - float(slot['stall_since'])), info=fin, **info)
    return fin


def run_candidate(e, C, tasks, seed, f, tl, kp=None, side='C', pseed=None):'''),
("""        fin = podctl('finalize-run', name, '30', timeout=90)
        log(event='gate_run_salvage', epoch=e, info=fin)
    return None
""", """        fin = podctl('finalize-run', name, '30', timeout=90)
        log(event='gate_run_salvage', epoch=e, info=fin)
    elif r.get('running'):
        stall_salvage(tl, name, r, len(tasks) * (1 if kp is None else int(kp['k'])), 'gate_run_stall_salvage', epoch=e)
    return None
"""),
("""        elif ru.get('running'):
            running += 1
""", """        elif ru.get('running'):
            running += 1
            stall_salvage(bs, name, ru, len(tasks) * int(r['k_block']), 'heldout_baseline_stall_salvage', round=rid, block=bi)
"""),
]
rc = 0
for p in sys.argv[1:]:
    s = open(p).read()
    for a, b in PAIRS:
        if b in s: continue
        if s.count(a) != 1: print('ANCHOR missing in %s: %r' % (p, a[:60])); rc = 2; break
        s = s.replace(a, b)
    else:
        open(p, 'w').write(s); print('patched', p)
sys.exit(rc)
