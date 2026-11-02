"""affine.io/baseline data: the SHAPE of the live prior-v1 baseline table, nothing more (Jacob, 2026-11-02).

Reads the live prior table read-only (sqlite mode=ro) through the gate's own code (baseline_v1.load_rows / inclusion, with the
N / floor / cap / shares / n_init of the current round), and writes website/baseline/curves.json with:
  * tasks sorted by pass rate s/n (ascending), cut into BINS equal-count bins (~120 tasks each);
  * per bin: the mean pass rate (2 decimals) and the mean per-round inclusion probability pi of the Neyman/Hajek draw
    (in %, 2 decimals), in the same sorted order, so the two curves line up;
  * aggregates only: N tasks, mean pass rate (2 decimals), expected tasks per round, uniform pi.
No task ids, no per-task values, no table digest. A guard refuses to write anything that looks like a task id or a long array.
Run by affine-site-baseline.timer (every 15 min); rewrites the file only when its content changes."""
from __future__ import annotations
import json, math, os, sys, tempfile, time
from pathlib import Path

LIVE_CODE = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code')
CURRENT = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/heldout/CURRENT.private.json')
OUT = Path('/home/const/subnet120/affine/website/baseline/curves.json')
BINS = 160
SINCE_EPOCH = 497          # prior-v1 baseline live from the e497 round (RA/20261101-baseline-switch)

sys.path.insert(0, str(LIVE_CODE))
import baseline_v1 as BV   # noqa: E402
import prior_gate as PG    # noqa: E402


def live_cfg() -> dict:
    pv = PG._merge({})
    try:
        cur = json.loads(CURRENT.read_text())
        r = json.loads((CURRENT.parent / cur['id'] / 'ROUND.private.json').read_text())
        if r.get('baseline') == 'prior-v1':
            for k in ('N', 'floor', 'cap', 'shares', 'n_init', 'pool_file'):
                if r.get('cfg', {}).get(k) is not None: pv[k] = r['cfg'][k]
    except Exception:
        pass
    return pv


def bins(vals, k):
    n = len(vals); out = []
    for b in range(k):
        lo, hi = (b * n) // k, ((b + 1) * n) // k
        seg = vals[lo:hi]; out.append(sum(seg) / len(seg))
    return out


def main() -> int:
    pv = live_cfg()
    pool = PG.load_pool(pv)
    rows = BV.load_rows(pv['table'], pool['ids'])          # read-only
    pi = BV.inclusion(rows, int(pv['N']), float(pv['floor']), float(pv['cap']), pv['shares'], float(pv['n_init']))
    # sort key: pass rate s/n; ties (e.g. the many 0/n and n/n tasks) by the posterior mean (s+1)/(n+2), so among tasks that
    # always pass, those with more evidence sit further right. pi itself is never a sort key.
    pts = [((r['s'] / r['n'], (r['s'] + 1) / (r['n'] + 2)), pi[r['task_id']]) for r in rows if r['n'] > 0]
    pts.sort(key=lambda t: t[0])
    pts = [(key[0], q) for key, q in pts]
    P = len(pts); k = min(BINS, P)
    doc = dict(
        schema='affine-baseline-curves-v1', baseline='prior-v1', since_epoch=SINCE_EPOCH,
        n_tasks=P, per_round=int(pv['N']), mean_pass=round(sum(p for p, _ in pts) / P, 2),
        uniform_pct=round(100.0 * int(pv['N']) / P, 2), bins=k,
        cfg=dict(N=int(pv['N']), floor=float(pv['floor']), cap=float(pv['cap']), n_init=float(pv['n_init']), shares=str(pv['shares'])),
        pass_rate=[round(v, 2) for v in bins([p for p, _ in pts], k)],
        select_pct=[round(100.0 * v, 2) for v in bins([q for _, q in pts], k)],
        note='Shapes only: tasks sorted by pass rate, averaged over equal-count bins; no task ids or per-task values.')
    body = json.dumps(doc, separators=(',', ':'))
    for bad in ('math:', 'dapo:', 'task_id', 'sqlite', 'digest', 'sha256', '"pi"', '"prior"', '"s"', '"n"'):
        if bad in body: raise SystemExit('baseline curves guard: %r in output' % bad)
    if not all(isinstance(v, (int, float)) for v in doc['pass_rate'] + doc['select_pct']):
        raise SystemExit('baseline curves guard: non-numeric curve value')
    assert len(doc['pass_rate']) <= 200 and len(doc['select_pct']) <= 200
    old = OUT.read_text() if OUT.exists() else None
    stamp = dict(json.loads(old)) if old else {}
    if old and {k2: v for k2, v in stamp.items() if k2 != 'updated'} == doc:
        return 0
    doc['updated'] = time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(OUT.parent), prefix='.curves.')
    with os.fdopen(fd, 'w') as f: f.write(json.dumps(doc, separators=(',', ':')) + '\n')
    os.chmod(tmp, 0o644); os.replace(tmp, OUT)
    print('baseline curves: %d tasks, %d bins, mean %.2f' % (P, k, doc['mean_pass']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
