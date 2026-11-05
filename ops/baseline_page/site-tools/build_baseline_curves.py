"""affine.io/baseline data: the SHAPE of the live gate-pool baseline table, nothing more.

Reads the live prior table read-only (sqlite mode=ro) through the gate's own code (baseline_v1.load_rows /
inclusion, with the N / floor / cap / shares / n_init of the current round), over the live flag pool
(gate-pool-v2). For hard-v1 tasks the prior sampler has not yet covered, uses the 32K filter pass rate as an
interim value (labelled in the JSON). Writes website/baseline/curves.json with:
  * tasks sorted by pass rate s/n (ascending), cut into BINS equal-count bins;
  * per bin: the mean pass rate (2 decimals) and the mean per-round inclusion probability pi of the
    Neyman/Hajek draw (in %, 2 decimals);
  * aggregates only: N tasks, mean pass rate (2 decimals), expected tasks per round, uniform pi,
    pass-rate source counts.
No task ids, no per-task values, no table digest. A guard refuses to write anything that looks like a task id
or a long array. Run by affine-site-baseline.timer (every 15 min); rewrites the file only when content changes.
"""
from __future__ import annotations
import json, os, sys, tempfile, time
from pathlib import Path

LIVE_CODE = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code')
CURRENT = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/heldout/CURRENT.private.json')
HARD_SLICE = Path('/home/const/subnet120-rewrite/state/task-slices/hard-v1')
SLICE = HARD_SLICE / 'private/slice.jsonl'
PASSRATES = HARD_SLICE / 'private/passrates.private.jsonl'
OUT = Path('/home/const/subnet120/affine/website/baseline/curves.json')
BINS = 160
SINCE_EPOCH = 497  # prior-v1 baseline live from the e497 round (RA/20261101-baseline-switch)

sys.path.insert(0, str(LIVE_CODE))
import baseline_v1 as BV   # noqa: E402
import prior_gate as PG    # noqa: E402


def live_cfg() -> dict:
    """Flag pool (gate-pool-v2) + round knobs for N/floor/cap/shares/n_init. Do NOT freeze pool_file to the
    open round: rounds opened before the v2 switch still name gate-pool-v1, but the live baseline is v2."""
    f = {}
    try:
        f = json.loads(Path(PG.FLAGP).read_text())
    except Exception:
        pass
    pv = PG.pv_of(f) if f else PG._merge({})
    try:
        cur = json.loads(CURRENT.read_text())
        r = json.loads((CURRENT.parent / cur['id'] / 'ROUND.private.json').read_text())
        if r.get('baseline') == 'prior-v1' or r.get('weighting') == 'irt-info':
            for k in ('N', 'floor', 'cap', 'shares', 'n_init'):
                if r.get('cfg', {}).get(k) is not None:
                    pv[k] = r['cfg'][k]
    except Exception:
        pass
    return pv


def hard_interim_pass() -> dict:
    """pool task_id -> (p, n) from the hard-v1 32K filter. Private read; never written out per-task."""
    if not SLICE.exists() or not PASSRATES.exists():
        return {}
    uid_to_cand = {}
    with SLICE.open() as fh:
        for line in fh:
            o = json.loads(line)
            uid_to_cand[o['uid']] = o['cand_id']
    cand_pr = {}
    with PASSRATES.open() as fh:
        for line in fh:
            o = json.loads(line)
            cand_pr[o['id']] = (float(o['p']), float(o['n']))
    out = {}
    src = HARD_SLICE / 'public/gate-pool-v2.jsonl'
    if not src.exists():
        return {}
    with src.open() as fh:
        for line in fh:
            o = json.loads(line)
            if o.get('slice') != 'hard-v1':
                continue
            cand = uid_to_cand.get(o.get('uid'))
            if cand in cand_pr:
                out[o['id']] = cand_pr[cand]
    return out


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
    interim = hard_interim_pass()
    # Prefer live prior-table samples; else hard-v1 32K filter interim.
    pts = []
    n_prior = 0
    n_interim = 0
    for r in rows:
        tid = r['task_id']
        if r['n'] > 0:
            p = r['s'] / r['n']
            tie = (r['s'] + 1) / (r['n'] + 2)
            n_prior += 1
        elif tid in interim:
            p, n_f = interim[tid]
            s_f = p * n_f
            tie = (s_f + 1) / (n_f + 2)
            n_interim += 1
        else:
            continue
        pts.append(((p, tie), pi[tid]))
    pts.sort(key=lambda t: t[0])
    pts = [(key[0], q) for key, q in pts]
    P = len(pts); k = min(BINS, P)
    pool_name = 'gate-pool-v2' if P >= 20000 else 'gate-pool-v1'
    note = (
        'Shapes only: tasks sorted by pass rate, averaged over equal-count bins; no task ids or per-task values. '
        + pool_name
        + ('.' if n_interim == 0 else
           '. Pass rates: prior-table where sampled; hard-v1 32K-filter interim for tasks the prior sampler has not covered yet '
           '(switches to prior-table once n>0).')
    )
    doc = dict(
        schema='affine-baseline-curves-v1', baseline='prior-v1', pool=pool_name, since_epoch=SINCE_EPOCH,
        n_tasks=P, per_round=int(pv['N']), mean_pass=round(sum(p for p, _ in pts) / P, 2),
        uniform_pct=round(100.0 * int(pv['N']) / P, 2), bins=k,
        cfg=dict(N=int(pv['N']), floor=float(pv['floor']), cap=float(pv['cap']), n_init=float(pv['n_init']), shares=str(pv['shares'])),
        pass_sources=dict(prior_table=n_prior, hard_v1_32k_filter_interim=n_interim),
        pass_rate=[round(v, 2) for v in bins([p for p, _ in pts], k)],
        select_pct=[round(100.0 * v, 2) for v in bins([q for _, q in pts], k)],
        note=note)
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
    print('baseline curves: %d tasks (%d prior, %d hard-v1 interim), %d bins, mean %.2f' % (P, n_prior, n_interim, k, doc['mean_pass']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
