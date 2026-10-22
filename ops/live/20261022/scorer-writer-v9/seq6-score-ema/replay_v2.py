import json, math, sys
sys.argv=['x']; exec(open('/home/const/subnet120-rewrite/state/root-audits/20261021-v9-score-ema/replay_v1.py').read().split("res={'current'")[0])
def emav(h, lookback, prune=0.0, topn=None):
    V=ema(h,lookback); out={}
    for L,v in V.items():
        if topn: v=dict(sorted(v.items(),key=lambda t:(-t[1],t[0]))[:topn])
        if prune: v={m:x for m,x in v.items() if x>=prune}
        t=sum(v.values()); out[L]={m:x/t for m,x in v.items()}
    return out
def gated(V, need_pos):
    out={}
    for L,v in V.items():
        ok={m:x for m,x in v.items() if m in Z[L] and int(Z[L][m].get('B') or 0)>=3 and Z[L][m].get('stouffer_z') is not None and (not need_pos or float(Z[L][m]['stouffer_z'])>0)}
        t=sum(ok.values()); out[L]={m:x/t for m,x in ok.items()}
    return out
def inelig(V):
    r=[max([x for m,x in V[L].items() if not (m in Z[L] and int(Z[L][m].get('B') or 0)>=3)] or [0]) for L in EP]; return round(max(r),3)
def stale(V):
    r=[]
    for L in EP:
        st={m:x for m,x in V[L].items() if m not in S[L]}
        r.append((sum(st.values()), max(st.values()) if st else 0))
    return round(sum(a for a,_ in r)/len(r),3), round(max(b for _,b in r),3)
rows=[]
variants=[('current',S)]
for h in (2,4,6):
    variants.append((f'h{h} lb{3*h} prune1%', emav(h,3*h,0.01)))
variants.append(('h4 lb12 no-prune', emav(4,12)))
variants.append(('h4 lb12 prune2%', emav(4,12,0.02)))
variants.append(('h4 lb12 top20', emav(4,12,0,20)))
for h in (2,4,6):
    variants.append((f'h{h} lb{3*h} p1% +elig', gated(emav(h,3*h,0.01),False)))
    variants.append((f'h{h} lb{3*h} p1% +elig z>0', gated(emav(h,3*h,0.01),True)))
for name,V in variants:
    c=churn(V,name); n=[x for x in c if int(x['t'].split('->e')[1])<=77]; ab=[x for x in c if int(x['t'].split('->e')[1])>77]
    f=lambda r,k: round(sum(x[k] for x in r)/len(r),3)
    sm,sx=stale(V)
    rows.append(dict(v=name, L1_norm=f(n,'L1'), L1_max_norm=max(x['L1'] for x in n), maxdUID_norm=f(n,'maxdUID'), top10out_norm=f(n,'top10_out'),
      paid_mean=f(n,'paid'), paid_last=c[-1]['paid'], maxw_mean=f(n,'maxw'), L1_e77_78=ab[0]['L1'] if ab else None, stale_mass_mean=sm, stale_uid_max=sx, inelig_uid_max=inelig(V)))
for r in rows: print(json.dumps(r))
# empirical ramp: UIDs entering current pool at E and staying >=2 epochs; ratio EMA share / current share at E, E+1, E+2
V4=gated(emav(4,12,0.01),True)
ev=[]
for a,b in zip(EP,EP[1:]):
    for m in set(S[b])-set(S[a]): ev.append(('enter',b,m))
    for m in set(S[a])-set(S[b]): ev.append(('leave',b,m))
for typ,E,m in ev:
    if typ=='enter':
        seq=[x for x in EP if x>=E][:3]
        if all(m in S[x] for x in seq[:2]) and len(seq)>=2:
            print('ramp uid',U[m],'E',E,[ (round(S[x].get(m,0),3), round(V4[x].get(m,0),3)) for x in seq])
    else:
        seq=[x for x in EP if x>=E][:4]
        if all(m not in S[x] for x in seq):
            print('decay uid',U[m],'E',E,'prev',round(S[E-1][m],3),[round(V4[x].get(m,0),3) for x in seq])
