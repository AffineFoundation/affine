import importlib.util, json, math
N='/home/const/subnet120-rewrite/state/root-audits/20261020-measured-impact-v9'
spec=importlib.util.spec_from_file_location('w9',N+'/weights_v9.py'); w=importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
AU='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
c=w.load_config(N+'/LIVE-CONFIG-v9.ACTIVE.signed.json',AU); c0={k:v for k,v in c.items() if k!='smoothing'}
sh,_,i=w.v9_vector(c,80); raw,_,_=w.v9_vector(c0,80)
g=w._score(c,79)[1]['window_gate']
rows=sorted(((g[m]['uid'],round(v,3),round(raw.get(m,0),3),g[m]['B'],round(g[m]['stouffer_z'],2)) for m,v in sh.items()),key=lambda r:-r[1])
for r in rows: print('uid %d smoothed %.3f raw %.3f B %d z %.2f'%r)
print('in_raw_top10',sum(1 for m in sh if m in raw),'of',len(sh),'mass_outside_raw_top10',round(sum(v for m,v in sh.items() if m not in raw),3),'max_outside',round(max([v for m,v in sh.items() if m not in raw] or [0]),3), 'min_z_paid',round(min(g[m]['stouffer_z'] for m in sh),2))
sh78,_,_=w.v9_vector(c,79) if False else (None,None,None)
# churn e78->e79 raw vs smoothed (abnormal epochs)
def vec(cfg,E):
    import types
    p=w._score(cfg,E)[1]; s,_=w.always_pay(p['window_gate'],cfg['payout'],1.0)
    if 'smoothing' in cfg: s,_=w.smooth(cfg,E,p['window_gate'],s)
    return s
for name,cfg in (('raw',c0),('smoothed',c)):
    a,b=vec(cfg,78),vec(cfg,79); ks=set(a)|set(b)
    t=lambda v:set(sorted(v,key=lambda k:-v[k])[:10])
    print(name,'e78->e79 L1',round(sum(abs(a.get(k,0)-b.get(k,0)) for k in ks),3),'maxd',round(max(abs(a.get(k,0)-b.get(k,0)) for k in ks),3),'top10_out',len(t(a)-t(b)))
st={E:dict(scored=len(w._score(c,E)[1]['window_gate']), elig=sum(1 for r in w._score(c,E)[1]['window_gate'].values() if int(r.get('B') or 0)>=3), miners=w._score(c,E)[1]['miners']) for E in (76,77,78,79)}
print(st)
