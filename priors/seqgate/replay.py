"""Offline replay + A/A of the sequential gate on existing per-task grades (SHADOW; read-only on live state).
Writes private/REPLAY.private.json (per-turn detail) and private/REPLAY.summary.json (aggregates only, no task ids / values)."""
import json, math, os, random, sys, statistics as st, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import seqgate as SG
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); G = RA / '20261026-promotion-gate'; ST = G / 'state'
OUT = RA / '20261102-seqgate-shadow/private'
spec = importlib.util.spec_from_file_location('bv_ro', G / 'code/baseline_v1.py'); BV = importlib.util.module_from_spec(spec); spec.loader.exec_module(BV)
rj = lambda p: json.load(open(p))
CFG = dict(SG.DEFAULTS); ORDERS = int(os.environ.get('ORDERS', 20))


def ctx_prior(rid):
    d = ST / 'heldout' / rid; r = rj(d / 'ROUND.private.json'); PI = rj(d / 'PI.private.json'); pv = r['cfg']
    P = sum(r['pool']['per_env'].values()); sh = {e: n / P for e, n in r['pool']['per_env'].items()}
    w = {t: sh[e] / r['pool']['per_env'][e] for t, e in PI['env'].items()}; N = sum(PI['pi'].values())
    return r, pv, PI, w, N


def cells(pt, pv):
    return {BV.task_of(int(k), pv): v for k, v in pt.items() if isinstance(v, dict) and v.get('n')}


def seq(order, head, cand, pi, w, N, cfg=CFG, budget=None):
    terms = SG.ht_terms(order, head, cand, pi, w, N)
    caps = [((cand[t].get('b') or 0), cand[t]['n']) for t, *_ in terms]
    c = dict(cfg); c['budget'] = budget or c['budget']
    o = SG.run_sequential(terms, caps, c); o['avail'] = len(terms); return o


def boot_ext(order, head, cand, pi, w, N, rng, n=2000):
    """bootstrap-extend a turn to n tasks: resample drawn tasks (prob ~ 1, since drawn set ~ q already) with replacement."""
    ts = [t for t in order if t in head and t in cand]
    return [rng.choice(ts) for _ in range(n)]


def live_rule(drawn, head, cand, pool, floor=0.003):
    return BV.estimate(drawn, head, cand, pool, sigma_run_floor=floor)


def hyper8(cell, rng):   # subsample 8 of the head's 12 rollouts (hypergeometric on c of n)
    n, c = cell['n'], cell['c']; k = min(8, n); bag = [1] * c + [0] * (n - c); rng.shuffle(bag)
    return dict(n=k, c=sum(bag[:k]), b=round((cell.get('b') or 0) * k / n))


def replay():
    rows = []
    for e in sorted(int(x) for x in os.listdir(ST / 'epochs') if x.isdigit()):
        d = ST / 'epochs' / str(e)
        try: D = rj(d / 'DECISION.json')
        except Exception: continue
        if (D.get('heldout') or {}).get('baseline') != 'prior-v1': continue
        if not (d / 'C-PROBE.private.json').exists(): continue
        rid = D['heldout']['round_id']; r, pv, PI, w, N = ctx_prior(rid)
        head = cells(rj(d / 'H-PROBE.private.json')['per_task'], pv); cand = cells(rj(d / 'C-PROBE.private.json')['per_task'], pv)
        rng = random.Random(e); outs = []; outs8 = []; ext = []
        for k in range(ORDERS):
            od = SG.visit_order(list(r['ids']), SG.order_seed(r['seed'], e * 1000 + k))
            outs.append(seq(od, head, cand, PI['pi'], w, N, cfg=(CFG if k == 0 else dict(CFG, bounds=False))))
            h8 = {t: hyper8(v, rng) for t, v in head.items()}
            outs8.append(seq(od, h8, cand, PI['pi'], w, N, cfg=dict(CFG, bounds=False)))
            eo = boot_ext(od, head, cand, PI['pi'], w, N, rng)
            # bootstrap duplicates: keep as separate visits (list of task ids may repeat)
            terms = SG.ht_terms(eo, head, cand, PI['pi'], w, N); caps = [((cand[t].get('b') or 0), cand[t]['n']) for t, *_ in terms]
            ext.append(SG.run_sequential(terms, caps, dict(CFG, bounds=False)))
        ev = D['evaluated']
        rows.append(dict(epoch=e, round=rid, live=D['decision'], z=ev['z'], gain=ev['gain'], n=ev['n'], cap=ev.get('c_cap_rate'),
                         z_bar=(D.get('z_bar') or {}).get('value', D.get('z_min')),
                         run_s=(D.get('timing') or {}).get('run_s'),
                         seq0=dict(decision=outs[0]['decision'], reason=outs[0]['reason'], tasks=outs[0]['tasks_used'], lcb=outs[0]['trace'][-1]['lcb'],
                                   ucb=outs[0]['trace'][-1]['ucb']),
                         p_green=sum(o['decision'] == 'green' for o in outs) / ORDERS,
                         p_early_red=sum(bool(o.get('early')) for o in outs) / ORDERS,
                         tasks_mean=st.mean(o['tasks_used'] for o in outs),
                         p_green_h8=sum(o['decision'] == 'green' for o in outs8) / ORDERS,
                         tasks_mean_h8=st.mean(o['tasks_used'] for o in outs8),
                         ext2000=dict(p_green=sum(o['decision'] == 'green' for o in ext) / ORDERS,
                                      p_early_red=sum(bool(o.get('early')) for o in ext) / ORDERS,
                                      tasks=[o['tasks_used'] for o in ext])))
        print(e, D['decision'], round(ev['z'], 2), rows[-1]['seq0']['decision'], rows[-1]['p_green'], rows[-1]['tasks_mean'], rows[-1]['ext2000']['p_green'],
              st.mean(rows[-1]['ext2000']['tasks']), flush=True)
    return rows


# ------------------------------------------------------------------ A/A
def aa_rule_pair(head, cand, pi, w, N, pool, rid_seed, tag, budget):
    drawn = {t: dict(env=None, pi=pi[t]) for t in head if t in cand}
    return head, cand, drawn


def aa_ivw():
    """ivw-v2 rounds (2000 uniform affine_math tasks, 4 head blocks x 4 rollouts): blocks {a,b} vs {c,d} = 8 v 8, separate runs."""
    out = []
    for rid in sorted(os.listdir(ST / 'heldout')):
        d = ST / 'heldout' / rid
        try: r = rj(d / 'ROUND.private.json')
        except Exception: continue
        if r.get('schema') != 'sn120-heldout-gate-v2': continue
        bl = []
        for b in range(4):
            p = d / ('BLOCK-%d.private.json' % b)
            if not p.exists(): break
            bl.append({str(k): v for k, v in rj(p)['per_task'].items() if isinstance(v, dict) and v.get('n')})
        if len(bl) < 4: continue
        ids = sorted(set.intersection(*[set(x) for x in bl]))
        if len(ids) < 900: continue
        pi = {t: 1.0 for t in ids}; w = {t: 1.0 / len(ids) for t in ids}; N = float(len(ids))   # uniform draw -> r_i = 1
        for (a, b), (c, dd) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
            H = {t: BV_merge([bl[a][t], bl[b][t]]) for t in ids}; C = {t: BV_merge([bl[c][t], bl[dd][t]]) for t in ids}
            for swap in (False, True):
                h, c_ = (C, H) if swap else (H, C)
                od = SG.visit_order(ids, SG.order_seed(rid, a * 10 + c + 100 * swap))
                o = seq(od, h, c_, pi, w, N, budget=2000)
                drawn = {t: dict(env='affine_math', pi=len(ids) / 7496.0) for t in ids}
                lr = live_rule(drawn, h, c_, {'affine_math': 7496}, 0.003); lr0 = live_rule(drawn, h, c_, {'affine_math': 7496}, 0.0)
                lr500 = live_rule({t: drawn[t] for t in od[:500]}, h, c_, {'affine_math': 7496}, 0.003)
                lr500_0 = live_rule({t: drawn[t] for t in od[:500]}, h, c_, {'affine_math': 7496}, 0.0)
                out.append(dict(src='ivw-v2-blocks', round=rid, tasks=len(ids), split='%d%d-%d%d%s' % (a, b, c, dd, '-swap' if swap else ''),
                                seq=o['decision'], seq_tasks=o['tasks_used'], seq_early=bool(o.get('early')),
                                z2000=lr['evaluated']['z'], z2000_nofloor=lr0['evaluated']['z'], z500=lr500['evaluated']['z'],
                                z500_nofloor=lr500_0['evaluated']['z'], gain=lr0['evaluated']['gain']))
    return out


def BV_merge(cs):
    return dict(n=sum(c['n'] for c in cs), c=sum(c['c'] for c in cs), b=sum((c.get('b') or 0) for c in cs))


def aa_prior_blocks():
    """prior-v1 rounds (~500 pi-weighted tasks, 3 head blocks x 4 rollouts): block i vs block j = 4 v 4 (truncated budget)."""
    out = []
    for rid in sorted(os.listdir(ST / 'heldout')):
        d = ST / 'heldout' / rid
        try: r = rj(d / 'ROUND.private.json')
        except Exception: continue
        if r.get('schema') != 'sn120-heldout-gate-prior-v1' or not (d / 'BLOCK-2.private.json').exists(): continue
        r, pv, PI, w, N = ctx_prior(rid)
        bl = [cells(rj(d / ('BLOCK-%d.private.json' % b))['per_task'], pv) for b in range(3)]
        ids = sorted(set.intersection(*[set(x) for x in bl]))
        for a, b in ((0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)):
            od = SG.visit_order(ids, SG.order_seed(r['seed'], 7000 + a * 10 + b))
            o = seq(od, bl[a], bl[b], PI['pi'], w, N, cfg=dict(CFG, bounds=False))
            drawn = {t: dict(env=PI['env'][t], pi=PI['pi'][t]) for t in ids}
            lr = live_rule(drawn, bl[a], bl[b], r['pool']['per_env'], 0.003); lr0 = live_rule(drawn, bl[a], bl[b], r['pool']['per_env'], 0.0)
            out.append(dict(src='prior-v1-blocks', round=rid, tasks=len(ids), split='%d-%d' % (a, b), seq=o['decision'], seq_tasks=o['tasks_used'],
                            seq_early=bool(o.get('early')), z=lr['evaluated']['z'], z_nofloor=lr0['evaluated']['z'], gain=lr0['evaluated']['gain']))
    return out


def aa_synth(reps=1000, seed=7, shifts=(0.0,)):
    """parametric A/A (shift 0) and power (shift > 0) at the full design: 2000 tasks, 8 v 8, pi/w from the current prior-v1 round's PI,
    per-task true p from the round's head baseline (12 rollouts, Beta(1,1)-smoothed on mixed tasks), tasks resampled from the drawn set
    (already ~q-distributed). Candidate true p = clip(p + shift / f_mid) on 0<p<1 tasks; true Delta reported."""
    rid = sorted(x for x in os.listdir(ST / 'heldout') if x.startswith('r2026'))[-1]
    r, pv, PI, w, N = ctx_prior(rid); B = rj(ST / 'heldout' / rid / 'BASELINE.private.json')['per_task']
    ids = [t for t in r['ids'] if t in B]; rng = random.Random(seed); res = []
    P0 = {t: ((B[t]['c'] + 1) / (B[t]['n'] + 2) if 0 < B[t]['c'] < B[t]['n'] else B[t]['c'] / B[t]['n']) for t in ids}
    fmid = sum(1 for t in ids if 0 < P0[t] < 1) / len(ids)
    for shift in shifts:
        P1 = {t: (min(1.0, max(0.0, P0[t] + shift / fmid)) if 0 < P0[t] < 1 else P0[t]) for t in ids}
        rr = {t: w[t] * N / PI['pi'][t] for t in ids}
        true = sum(rr[t] * (P1[t] - P0[t]) for t in ids) / len(ids)
        for k in range(reps):
            tk = [rng.choice(ids) for _ in range(2000)]
            H = {}; C = {}; drawn = {}
            for j, t in enumerate(tk):
                hc = sum(rng.random() < P0[t] for _ in range(8)); cc = sum(rng.random() < P1[t] for _ in range(8))
                key = '%s#%d' % (t, j); H[key] = dict(n=8, c=hc, b=0); C[key] = dict(n=8, c=cc, b=0)
                drawn[key] = dict(env=PI['env'][t], pi=PI['pi'][t])
            pi = {kk: PI['pi'][kk.split('#')[0]] for kk in H}; ww = {kk: w[kk.split('#')[0]] for kk in H}
            order = list(H); nb = dict(CFG, bounds=False)
            o = seq(order, H, C, pi, ww, N, cfg=nb, budget=2000)
            o5 = seq(order[:500], H, C, pi, ww, N, cfg=nb, budget=500)
            op = seq(order, H, C, pi, ww, N, cfg=dict(nb, lam_mode='prpl', c=0.5, s2_0=0.25 * 4 * 1.35 ** 2), budget=2000)
            lr = live_rule({kk: drawn[kk] for kk in order[:500]}, H, C, r['pool']['per_env'], 0.003)
            lr0 = live_rule({kk: drawn[kk] for kk in order[:500]}, H, C, r['pool']['per_env'], 0.0)
            lr2 = live_rule(drawn, H, C, r['pool']['per_env'], 0.0)
            res.append(dict(shift=shift, true_delta=true, seq2000=o['decision'], seq2000_tasks=o['tasks_used'], seq2000_early=bool(o.get('early')),
                            seq500=o5['decision'], prpl2000=op['decision'], prpl2000_tasks=op['tasks_used'],
                            z500=lr['evaluated']['z'], z500_nofloor=lr0['evaluated']['z'], z2000_nofloor=lr2['evaluated']['z'],
                            gain500=lr0['evaluated']['gain']))
        print('shift', shift, 'true', round(true, 5), flush=True)
    return res


def aa_noise():
    """A/A run-noise experiment (RA/20261101-aa-noise run2): 899 pi-weighted tasks, head cb92bdc5, 8 rounds x arms S/A/B x k=2.
    Head = 4 rounds of one arm (8 rollouts/task), candidate = the other 4 rounds of the same arm (8): separate runs, 8 v 8."""
    import itertools
    A = RA / '20261101-aa-noise/private'; b = rj(A / 'baseline.private.json'); cellsb = b['cells']
    sizes = dict(affine_math=7496, dapo=11813); P = sum(sizes.values()); sh = {e: n / P for e, n in sizes.items()}
    pi = {t: v['pi'] for t, v in cellsb.items()}; w = {t: sh[v['env']] / sizes[v['env']] for t, v in cellsb.items()}; N = float(b['design']['N'])
    X = {}
    for l in open(A / 'run2/rollouts.jsonl'):
        r = json.loads(l)
        if r['arm'] not in 'SAB': continue
        c = X.setdefault((r['arm'], r['round'], r['task_id']), [0, 0, 0]); c[0] += 1; c[1] += r['kind'] == 'positive'; c[2] += bool(r['capped'])
    rounds = sorted({k[1] for k in X}); ids = sorted(cellsb); out = []
    drawn = {t: dict(env=cellsb[t]['env'], pi=pi[t]) for t in ids}
    for arm in 'SAB':
        for R1 in itertools.combinations(rounds, 4):
            R2 = tuple(x for x in rounds if x not in R1)
            if R1 > R2: continue
            H = {}; C = {}
            for t in ids:
                h = [X.get((arm, q, t)) for q in R1]; c = [X.get((arm, q, t)) for q in R2]
                if None in h or None in c: continue
                H[t] = dict(n=sum(x[0] for x in h), c=sum(x[1] for x in h), b=sum(x[2] for x in h))
                C[t] = dict(n=sum(x[0] for x in c), c=sum(x[1] for x in c), b=sum(x[2] for x in c))
            for swap in (False, True):
                h, c_ = (C, H) if swap else (H, C)
                od = SG.visit_order(list(h), SG.order_seed('aa-%s-%s' % (arm, R1), int(swap)))
                o = seq(od, h, c_, pi, w, N, cfg=dict(CFG, bounds=False))
                dd = {t: drawn[t] for t in h}
                lr = live_rule(dd, h, c_, sizes, 0.003); lr0 = live_rule(dd, h, c_, sizes, 0.0)
                lr5 = live_rule({t: dd[t] for t in od[:500]}, h, c_, sizes, 0.003)
                out.append(dict(src='aa-noise-run2', arm=arm, split='%s|%s%s' % (R1, R2, '-swap' if swap else ''), tasks=len(h), seq=o['decision'],
                                seq_tasks=o['tasks_used'], seq_early=bool(o.get('early')), z=lr['evaluated']['z'], z_nofloor=lr0['evaluated']['z'],
                                z500=lr5['evaluated']['z'], gain=lr0['evaluated']['gain']))
    return out


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    out = {}
    if what in ('all', 'replay'): out['replay'] = replay()
    if what in ('all', 'aa'):
        out['aa_noise'] = aa_noise(); print('aa_noise', len(out['aa_noise']), flush=True)
        out['aa_prior'] = aa_prior_blocks(); print('aa_prior', len(out['aa_prior']), flush=True)
    if what in ('all', 'synth'): out['aa_synth'] = aa_synth(int(os.environ.get('REPS', 400)), shifts=tuple(float(x) for x in os.environ.get('SHIFTS', '0').split(','))); print('synth', len(out['aa_synth']), flush=True)
    p = OUT / ('REPLAY-%s.private.json' % what); json.dump(dict(cfg=CFG, orders=ORDERS, **out), open(p, 'w')); os.chmod(p, 0o600)
    print('wrote', p)
