import json, math, glob, os, sys
from pathlib import Path
N=Path('/home/const/subnet120-rewrite/state/root-audits/20261020-measured-impact-v9')
Q=Path('/home/const/subnet120-rewrite/state/live-math-launch-preparation-v1/reward-state-v1/current-assessment-v9-live-1')
PO=dict(temperature=1.0, top_k=10, min_units=3, fallback_top_n=3)
def ap(g):
    el=[(float(r['stouffer_z']),m) for m,r in g.items() if int(r.get('B') or 0)>=3 and r.get('stouffer_z') is not None and math.isfinite(float(r['stouffer_z']))]
    el.sort(key=lambda t:(-t[0],t[1])); pos=[t for t in el if t[0]>0]; pool=pos[:10] if pos else el[:3]
    zm=pool[0][0]; e={m:math.exp(z-zm) for z,m in pool}; t=sum(e.values()); return {m:v/t for m,v in e.items()}
S={}; U={}; Z={}
for f in sorted(N.glob('scores/score-e*.json')):
    p=json.load(open(f))['payload']; E=p['epoch']; S[E]=ap(p['window_gate']); Z[E]={m:r for m,r in p['window_gate'].items()}
    for m,r in p['window_gate'].items(): U[m]=r['uid']
    S[E]['_n']=None; del S[E]['_n']
EP=sorted(S)
def churn(V, label):
    rows=[]
    for a,b in zip(EP,EP[1:]):
        if a not in V or b not in V: continue
        x,y=V[a],V[b]; ks=set(x)|set(y); d={k:abs(y.get(k,0)-x.get(k,0)) for k in ks}
        t10=lambda v:set(sorted(v,key=lambda k:-v[k])[:10])
        rows.append(dict(t=f'e{a}->e{b}', L1=round(sum(d.values()),3), maxdUID=round(max(d.values()),3), top10_out=len(t10(x)-t10(y)), paid=len([k for k in y if y[k]>1e-4]), maxw=round(max(y.values()),3)))
    return rows
def ema(h, lookback=12):
    a=1-2**(-1/h); out={}
    for L in EP:
        acc={}; wt=0
        for k in range(lookback):
            E=L-k
            if E not in S: continue
            c=a*(1-a)**k; wt+=c
            for m,v in S[E].items(): acc[m]=acc.get(m,0)+c*v
        out[L]={m:v/wt for m,v in acc.items()}
    return out
res={'current':churn(S,'cur')}
for h in (2,4,6): res[f'h{h}']=churn(ema(h),f'h{h}')
def summ(rows, sel):
    r=[x for x in rows if sel(x)]
    if not r: return None
    f=lambda k: round(sum(x[k] for x in r)/len(r),3)
    return dict(n=len(r), L1_mean=f('L1'), maxdUID_mean=f('maxdUID'), top10_out_mean=f('top10_out'), paid_mean=f('paid'), maxw_mean=f('maxw'), L1_max=max(x['L1'] for x in r))
normal=lambda x:int(x['t'].split('->e')[1])<=77
out={k:dict(all=summ(v,lambda x:True), normal_e69_77=summ(v,normal), abnormal_into_e78=summ(v,lambda x:not normal(x)), rows=v) for k,v in res.items()}
# per-epoch miners scored / positive / eligible
out['epoch_stats']={E:dict(scored=len(Z[E]), elig=sum(1 for r in Z[E].values() if int(r.get('B') or 0)>=3), pos=sum(1 for r in Z[E].values() if int(r.get('B') or 0)>=3 and (r.get('stouffer_z') or 0)>0), zmax=round(max(float(r['stouffer_z']) for r in Z[E].values() if r.get('stouffer_z') is not None),2)) for E in EP}
# empirical ramp/decay: UIDs entering/leaving current top-10 pool, their EMA share trajectory
tr={}
for k in ('current','h2','h4','h6'):
    V=S if k=='current' else ema(int(k[1:]))
    tr[k]=V
ev=[]
for a,b in zip(EP,EP[1:]):
    for m in set(S[b])-set(S[a]): ev.append(('enter',b,m))
    for m in set(S[a])-set(S[b]): ev.append(('leave',b,m))
out['events']=[]
for typ,E,m in ev:
    row=dict(type=typ, epoch=E, uid=U[m])
    for k,V in tr.items(): row[k]=[round(V[x].get(m,0),3) for x in EP if x>=E-1][:5]
    out['events'].append(row)
# actual on-chain assessment churn per hourly file grouped by score_epoch
acts={}
for f in sorted(Q.glob('assessment-*.json')):
    p=json.load(open(f))['payload']; se=(p.get('v9') or {}).get('score_epoch'); 
    if se is None: continue
    acts.setdefault(se,{m:v for m,v in p['points'].items()})
out['assessment_epochs']=sorted(acts)
out['assessment_matches_replay']={E: (max(abs(acts[E].get(m,0)-S[E].get(m,0)) for m in set(acts[E])|set(S[E])) if E in S else None) for E in acts}
out['assessment_burn']={E: acts[E].get('__burn_uid0_owner__',0) for E in acts}
print(json.dumps(out))
