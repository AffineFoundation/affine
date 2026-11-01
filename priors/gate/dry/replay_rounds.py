#!/usr/bin/env python3
"""OFFLINE replay of the prior-v1 estimator on the last finished held-out rounds (RA/20261101-baseline-switch). Read-only: reads the
gate's private round files and a COPY of the prior table; writes one private JSON. Nothing live is touched.
Real data per turn: head = round baseline (blocks 1-3 = 12 rollouts on kept tasks; selection block 0 = 4 rollouts elsewhere),
candidate = the turn's C-PROBE (k_cand rollouts on kept tasks); on non-kept tasks (the head solved 0/4 or 4/4) there is no candidate
data, so the candidate is imputed = head there (the old gate's implicit assumption). Pool = the round's own task set S (math only):
pi from the table copy with the prior-v1 formula (Neyman sqrt(E[p(1-p)]) under Beta(s+1,n-s+1), 10% floor), N = 25% of |S|
(the production cap 0.10 is infeasible on a 1-2K pool; cap 0.5 here). R seeded draws per turn -> P(GREEN), mean z, vs the live verdict.
usage: replay_rounds.py TABLE_COPY OUT.private.json ROUND_ID... [--draws R] [--tau T]"""
import json, os, sys, statistics as stt
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_v1 as BV
GS = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state')
def rj(p):
    try: return json.load(open(p))
    except Exception: return None
args = [a for a in sys.argv[1:] if not a.startswith('--')]
R_ = int(sys.argv[sys.argv.index('--draws') + 1]) if '--draws' in sys.argv else 200
TAU = float(sys.argv[sys.argv.index('--tau') + 1]) if '--tau' in sys.argv else 0.0457
table, outp, rids = args[0], args[1], [a for a in args[2:] if a.startswith('r2')]
out = dict(table=table, draws=R_, tau=TAU, rounds=[])
for rid in rids:
    rd = GS / 'heldout' / rid; r = rj(rd / 'ROUND.private.json'); b = rj(rd / 'BASELINE.private.json')
    S = [int(i) for i in r['indices']]; kept = set(int(k) for k in b['kept']); sel = {int(k): v for k, v in b['selection'].items()}
    merged = {int(k): v for k, v in b['per_task'].items()}
    head = {}
    for i in S:
        c = merged.get(i) if i in kept else sel.get(i)
        if isinstance(c, dict) and c.get('n'): head['math:%d' % i] = c
    rows = BV.load_rows(table, ['math:%d' % i for i in S])
    N = int(0.25 * len(S)); pi = BV.inclusion(rows, N=N, floor=0.10, cap=0.5)
    rec = dict(round=rid, set_size=len(S), kept=len(kept), N=N, turns=[])
    for t in [json.loads(l) for l in open(rd / 'TURNS.private.jsonl')]:
        e = t['epoch']; cp = rj(GS / 'epochs' / str(e) / 'C-PROBE.private.json')
        if not cp or t.get('fail_closed'): continue
        cpt = {int(k): v for k, v in cp['per_task'].items()}
        cand = {}
        for i in S:
            tid = 'math:%d' % i
            if i in kept and isinstance(cpt.get(i), dict) and cpt[i].get('n'): cand[tid] = cpt[i]
            elif i not in kept and tid in head: cand[tid] = head[tid]          # imputed: no candidate data off the kept set
        zs, gs, gr = [], [], 0
        for k in range(R_):
            seed = BV.seed_from('replay-%s-%d' % (rid, e), k, '0x00')
            ids = BV.draw(pi, seed); drawn = {x: dict(env='affine_math', pi=pi[x]) for x in ids}
            d = BV.estimate(drawn, head, cand, {'affine_math': len(S)}, tau=TAU, min_scored=50)
            zs.append(d['evaluated']['z']); gs.append(d['evaluated']['gain']); gr += d['decision'] == 'green'
        full = BV.estimate({x: dict(env='affine_math', pi=1.0) for x in head}, head, cand, {'affine_math': len(S)}, tau=TAU, min_scored=50)
        rec['turns'].append(dict(epoch=e, live=t['decision'], live_z=t.get('z'), live_gain=(t.get('evaluated') or {}).get('gain'),
                                 new_p_green=gr / R_, new_z_mean=stt.mean(zs), new_z_sd=stt.pstdev(zs), new_gain_mean=stt.mean(gs),
                                 census_gain=full['evaluated']['gain'], census_z=full['evaluated']['z'], census_decision=full['decision'],
                                 first_draw_decision=None))
        # one concrete decision (draw 0) for the record
        seed = BV.seed_from('replay-%s-%d' % (rid, e), 0, '0x00'); ids = BV.draw(pi, seed)
        d0 = BV.estimate({x: dict(env='affine_math', pi=pi[x]) for x in ids}, head, cand, {'affine_math': len(S)}, tau=TAU, min_scored=50)
        rec['turns'][-1].update(first_draw_decision=d0['decision'], first_draw_z=d0['evaluated']['z'], first_draw_n=d0['evaluated']['n'],
                                first_draw_reasons=d0['reasons'])
    out['rounds'].append(rec)
fd = os.open(outp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, 'w') as f: json.dump(out, f, indent=1)
for rec in out['rounds']:
    print('%s |S|=%d kept=%d N=%d' % (rec['round'], rec['set_size'], rec['kept'], rec['N']))
    for t in rec['turns']:
        print('  e%d live %-5s z %+.2f | new P(GREEN) %.2f z %+.2f+-%.2f gain %+.4f | census gain %+.4f z %+.2f %s | draw0 %s z %+.2f n %d' % (
            t['epoch'], t['live'], t['live_z'] or 0, t['new_p_green'], t['new_z_mean'], t['new_z_sd'], t['new_gain_mean'],
            t['census_gain'], t['census_z'], t['census_decision'], t['first_draw_decision'], t['first_draw_z'], t['first_draw_n']))
