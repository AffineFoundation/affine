"""affine.io/baseline IRT overlay (Jacob pt1080u, 2026-11-02): static, binned shapes of a 2PL item-response fit, PROPOSED, NOT LIVE.

One-off offline build (the fit is not re-run by the 15-min timer). Reads the private fit outputs of
RA/20261102-irt-baseline/private (per-task arrays, never published) and writes website/baseline/irt-curves.json with only:
  * tasks in the baseline page's order (head pass rate s/n ascending, ties by (s+1)/(n+2); prior-table snapshot of the fit),
    cut into BINS equal-count bins, exactly like build_baseline_curves.py;
  * per bin, rounded to 2 decimals: the per-round selection probability (%) if the draw weight were sqrt(a^2 P (1-P)) at the
    head (same N / floor / cap as live), the normalised information a^2 P(1-P) and the normalised current E[p(1-p)]
    (each divided by its mean over all tasks), and the median discrimination a divided by its pool median (model-dependent).
No task ids, no per-task values, no head id, no fit parameters. Same guard as the live generator.
Run: ~/subnet120-rewrite/.venv/bin/python site-tools/build_baseline_irt_curves.py"""
from __future__ import annotations
import json, os, sys, tempfile, time
import numpy as np

PRIV = '/home/const/subnet120-rewrite/state/root-audits/20261102-irt-baseline/private'
OUT = '/home/const/subnet120/affine/website/baseline/irt-curves.json'
TAG = 'ab_1_4_chain'
BINS = 160


def main() -> int:
    F = np.load('%s/fit_%s.npz' % (PRIV, TAG)); D = np.load('%s/derived_%s.npz' % (PRIV, TAG))
    xpos = D['xpos']; ok = np.isfinite(xpos); idx = np.where(ok)[0][np.argsort(xpos[ok])]
    P = len(idx); k = min(BINS, P)
    a, I, v = F['a'][idx], D['I'][idx], D['v'][idx]
    In, vn, ar = I / I.mean(), v / v.mean(), a / np.median(a)
    def bins(x, f=np.mean):
        return [round(float(f(x[(j * P) // k:((j + 1) * P) // k])), 2) for j in range(k)]
    doc = dict(schema='affine-baseline-irt-curves-v1', status='proposed, not live', bins=k,
               model='2PL item response, P = sigmoid(a (theta - b)), MAP fit on past gate/probe pass-fail counts of one training chain',
               select_pct_irt=bins(100.0 * D['pi_irt'][idx]), info_irt=bins(In), info_current=bins(vn),
               a_rel_median=bins(ar, np.median), generated=time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime()),
               note='Shapes only, binned and rounded; same x order as /baseline/curves.json at the time of the fit. Not used by the gate.')
    body = json.dumps(doc, separators=(',', ':'))
    for bad in ('math:', 'dapo:', 'task_id', 'sqlite', 'digest', 'sha256', '"pi"', '"prior"', '"s"', '"n"', '"theta"', '"head"'):
        if bad in body: raise SystemExit('irt curves guard: %r in output' % bad)
    for key in ('select_pct_irt', 'info_irt', 'info_current', 'a_rel_median'):
        assert len(doc[key]) <= 200 and all(isinstance(x, float) for x in doc[key])
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(OUT), prefix='.irt-curves.')
    with os.fdopen(fd, 'w') as f: f.write(body + '\n')
    os.chmod(tmp, 0o644); os.replace(tmp, OUT)
    print('irt curves: %d tasks, %d bins -> %s' % (P, k, OUT)); return 0


if __name__ == '__main__':
    sys.exit(main())
