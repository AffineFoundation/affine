"""real-data checks of the group-sequential rule (no floor, OBF bound c/sqrt(t), futility z<0 at interim looks):
(1) aa-noise A/A 8v8 splits (899 tasks): looks 300/500/700 -> false-GREEN; (2) 59 live turns: looks 300/full (~500) -> agreement with live."""
import sys, json, random, statistics as st, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import replay as RP, seqgate as SG
c3 = float(sys.argv[1]); c2 = float(sys.argv[2]); fut = 0.0
def gs(zg, looks, c, last):
    for k, (z, g) in enumerate(zg):
        b = c / (looks[k] / last) ** 0.5
        if z > b and g > 0: return 'green', looks[k]
        if k < len(zg) - 1 and z < fut: return 'red', looks[k]
    return 'red', looks[-1]
# (1) A/A
import itertools
A = RP.RA / '20261101-aa-noise/private'; b = RP.rj(A / 'baseline.private.json'); cb = b['cells']; sizes = dict(affine_math=7496, dapo=11813)
X = {}
for l in open(A / 'run2/rollouts.jsonl'):
    r = json.loads(l)
    if r['arm'] not in 'SAB': continue
    c = X.setdefault((r['arm'], r['round'], r['task_id']), [0, 0, 0]); c[0] += 1; c[1] += r['kind'] == 'positive'; c[2] += bool(r['capped'])
rounds = sorted({k[1] for k in X}); ids = sorted(cb); drawn = {t: dict(env=cb[t]['env'], pi=cb[t]['pi']) for t in ids}
aa = []
for arm in 'SAB':
    for R1 in itertools.combinations(rounds, 4):
        R2 = tuple(x for x in rounds if x not in R1)
        if R1 > R2: continue
        H = {}; C = {}
        for t in ids:
            h = [X.get((arm, q, t)) for q in R1]; c = [X.get((arm, q, t)) for q in R2]
            if None in h or None in c: continue
            H[t] = dict(n=sum(x[0] for x in h), c=sum(x[1] for x in h), b=sum(x[2] for x in h)); C[t] = dict(n=sum(x[0] for x in c), c=sum(x[1] for x in c), b=sum(x[2] for x in c))
        for swap in (False, True):
            h, c_ = (C, H) if swap else (H, C)
            od = SG.visit_order(list(h), SG.order_seed('aa-%s-%s' % (arm, R1), int(swap)))
            zg = []
            for L in (300, 500, 700):
                ev = RP.live_rule({t: drawn[t] for t in od[:L]}, h, c_, sizes, 0.0)['evaluated']; zg.append((ev['z'], ev['gain']))
            ev5 = RP.live_rule({t: drawn[t] for t in od[:500]}, h, c_, sizes, 0.003)['evaluated']
            d, n = gs(zg, (300, 500, 700), c3, 700); aa.append((d, n, ev5['z'] > 1 and ev5['gain'] > 0))
print('A/A splits', len(aa), 'GS false-GREEN %.3f' % (sum(x[0] == 'green' for x in aa) / len(aa)), 'mean tasks %.0f' % st.mean(x[1] for x in aa),
      '| today z>1@500(floor) %.3f' % (sum(x[2] for x in aa) / len(aa)))
# (2) live turns
rows = []
for e in sorted(int(x) for x in os.listdir(RP.ST / 'epochs') if x.isdigit()):
    d = RP.ST / 'epochs' / str(e)
    try: D = RP.rj(d / 'DECISION.json')
    except Exception: continue
    if (D.get('heldout') or {}).get('baseline') != 'prior-v1' or not (d / 'C-PROBE.private.json').exists(): continue
    r, pv, PI, w, N = RP.ctx_prior(D['heldout']['round_id'])
    head = RP.cells(RP.rj(d / 'H-PROBE.private.json')['per_task'], pv); cand = RP.cells(RP.rj(d / 'C-PROBE.private.json')['per_task'], pv)
    pool = r['pool']['per_env']; dec = []
    for k in range(20):
        od = [t for t in SG.visit_order(list(r['ids']), SG.order_seed(r['seed'], e * 1000 + k)) if t in head and t in cand]
        zg = []
        for L in (300, len(od)):
            ev = RP.live_rule({t: dict(env=PI['env'][t], pi=PI['pi'][t]) for t in od[:L]}, head, cand, pool, 0.0)['evaluated']; zg.append((ev['z'], ev['gain']))
        dec.append(gs(zg, (300, len(od)), c2, len(od)))
    rows.append((e, D['decision'], sum(x[0] == 'green' for x in dec) / 20, st.mean(x[1] for x in dec), len(od)))
g = [x for x in rows if x[1] == 'green']; rd = [x for x in rows if x[1] != 'green']
print('live turns', len(rows), '| live GREEN n=%d: mean P(GS green) %.2f' % (len(g), st.mean(x[2] for x in g)), [(x[0], x[2]) for x in g])
print('live RED n=%d: mean P(GS green) %.3f' % (len(rd), st.mean(x[2] for x in rd)), 'flips>0.5:', [(x[0], x[2]) for x in rd if x[2] > 0.5])
print('mean tasks GS %.0f vs live %.0f' % (st.mean(x[3] for x in rows), st.mean(x[4] for x in rows)))
