"""affine.io/baseline IRT curves (LIVE from 2026-11-02, Jacob pt1083u): binned shapes of the irt-info draw the gate uses.

Run by affine-site-baseline.timer (every 15 min) after build_baseline_curves.py. Reads, read-only: the live prior table (task order
only, same order as /baseline/curves.json), the private IRT params (RA/20261102-irt-baseline/private/live) and the head ability
theta of the current irt-info round (else the MAP of the head's own rollouts, exactly as the gate computes it at round open), and
writes website/baseline/irt-curves.json with only, per equal-count bin (2 decimals):
  * select_pct_irt: the per-round selection probability (%) of the live irt-info draw (gate code: baseline_v1.irt_score + inclusion_from);
  * info_irt: the draw term a'^2 P(1-P) at the head / its mean;  info_current: the previous prior-v1 term E[p(1-p)] / its mean;
  * a_rel_median: the median (shrunk, clipped) discrimination a' / pool median.
No task ids, no per-task values, no theta, no head id, no fit parameters (guarded). Pure Python (system python3 -I)."""
from __future__ import annotations
import json, os, statistics, sys, tempfile, time, types
from pathlib import Path

RA = Path('/home/const/subnet120-rewrite/state/root-audits')
LIVE_CODE = RA / '20261026-promotion-gate/code'
HELD = RA / '20261026-promotion-gate/state/heldout'
OUT = Path('/home/const/subnet120/affine/website/baseline/irt-curves.json')
LIVE_SINCE = RA / '20261102-irt-live/state/SWITCH_CR'
BINS = 160
sys.path.insert(0, str(LIVE_CODE))
import baseline_v1 as BV   # noqa: E402
import prior_gate as PG    # noqa: E402


def rj(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None


def main() -> int:
    f = rj(PG.FLAGP) or {}; pv = PG.pv_of(f)
    if not hasattr(BV, 'irt_score'): print('irt code not installed'); return 0
    P = rj(pv['irt_params']); ab = {t: (float(v[0]), float(v[1])) for t, v in P['tasks'].items()}
    pool = PG.load_pool(pv); rows = BV.load_rows(pv['table'], pool['ids'])
    cur = rj(HELD / 'CURRENT.private.json') or {}; H = cur.get('head'); th = None; src = None
    for rd in sorted(HELD.glob('r2*'), reverse=True)[:30]:
        r = rj(rd / 'ROUND.private.json') or {}
        if r.get('weighting') == 'irt-info' and (r.get('irt') or {}).get('theta') is not None and (H is None or r.get('head') == H):
            th = float(r['irt']['theta']); src = 'current round'; H = r['head']; break
    if th is None and H:
        S = types.SimpleNamespace(rounds=lambda: types.SimpleNamespace(root=HELD))
        cells, _ = PG.head_cells(S, pv, H, pv['table'])
        if cells: th, _, nr = BV.theta_map(cells, ab, float(pv.get('theta_prior_sd', 2.0))); src = 'head rollouts' if nr >= 200 else None
    if th is None: th = float(P['theta_fallback']); src = 'fit fallback'
    k = {x: float(pv.get(x, BV.IRT_DEFAULTS[x])) for x in BV.IRT_DEFAULTS}; med = float(P['a_median'])
    score = BV.irt_score(rows, ab, th, med, k['a_shrink'], k['a_lo'], k['a_hi'])
    pi = BV.inclusion_from(rows, score, int(pv['N']), float(pv['floor']), float(pv['cap']), pv['shares'])
    pts = []
    for r, s in zip(rows, score):
        if r['n'] <= 0: continue
        au = BV.irt_a_used(ab[r['task_id']][0], med, k['a_shrink'], k['a_lo'], k['a_hi'])
        pts.append(((r['s'] / r['n'], (r['s'] + 1) / (r['n'] + 2)), pi[r['task_id']], s * s, BV.post_var(r['s'], r['n']), au / med))
    pts.sort(key=lambda t: t[0]); n = len(pts); kb = min(BINS, n)
    mi = sum(p[2] for p in pts) / n; mv = sum(p[3] for p in pts) / n
    def bins(j, f=lambda v: sum(v) / len(v), scale=1.0):
        return [round(float(f([p[j] for p in pts[(b * n) // kb:((b + 1) * n) // kb]])) * scale, 2) for b in range(kb)]
    doc = dict(schema='affine-baseline-irt-curves-v2', status='live', bins=kb, live_since=(LIVE_SINCE.read_text().strip() if LIVE_SINCE.exists() else None),
               model='2PL item response, P = sigmoid(a (theta - b)), MAP fit on every recorded checkpoint x gate-pool task pass/fail (refit daily)',
               theta_from=src, fitted=P.get('fitted_at_cr'),
               select_pct_irt=[round(x, 2) for x in bins(1, scale=100.0)], info_irt=bins(2, scale=1 / mi), info_current=bins(3, scale=1 / mv),
               a_rel_median=bins(4, statistics.median), expected_n=round(sum(pi.values()), 1),
               at_cap=sum(1 for v in pi.values() if v >= float(pv['cap']) - 1e-9),
               generated=time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime()),
               note='Shapes only, binned and rounded; same x order as /baseline/curves.json. The live gate draw (irt-info weighting).')
    body = json.dumps(doc, separators=(',', ':'))
    for bad in ('math:', 'dapo:', 'task_id', 'sqlite', 'digest', 'sha256', '"pi"', '"prior"', '"s"', '"n"', '"theta"', '"head"'):
        if bad in body: raise SystemExit('irt curves guard: %r in output' % bad)
    for key in ('select_pct_irt', 'info_irt', 'info_current', 'a_rel_median'):
        assert len(doc[key]) <= 200 and all(isinstance(x, float) for x in doc[key])
    old = OUT.read_text() if OUT.exists() else ''
    if old:
        try:
            o = json.loads(old); o.pop('generated', None); d2 = dict(doc); d2.pop('generated', None)
            if o == d2: return 0
        except Exception: pass
    fd, tmp = tempfile.mkstemp(dir=str(OUT.parent), prefix='.irt-curves.')
    with os.fdopen(fd, 'w') as fh: fh.write(body + '\n')
    os.chmod(tmp, 0o644); os.replace(tmp, OUT)
    print('irt curves: %d tasks, %d bins, theta from %s -> %s' % (n, kb, src, OUT)); return 0


if __name__ == '__main__':
    sys.exit(main())
