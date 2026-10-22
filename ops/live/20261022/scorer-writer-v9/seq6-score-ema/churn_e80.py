import importlib.util
N='/home/const/subnet120-rewrite/state/root-audits/20261020-measured-impact-v9'
spec=importlib.util.spec_from_file_location('w9',N+'/weights_v9.py'); w=importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
c=w.load_config(N+'/LIVE-CONFIG-v9.ACTIVE.signed.json','3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'); c0={k:v for k,v in c.items() if k!='smoothing'}
def vec(cfg,E):
    p=w._score(cfg,E)[1]; s,_=w.always_pay(p['window_gate'],cfg['payout'],1.0)
    if 'smoothing' in cfg: s,_=w.smooth(cfg,E,p['window_gate'],s)
    return s
t=lambda v:set(sorted(v,key=lambda k:-v[k])[:10])
for a,b in ((78,79),(79,80)):
    for n,cfg in (('raw',c0),('smoothed',c)):
        x,y=vec(cfg,a),vec(cfg,b); ks=set(x)|set(y)
        print(f'e{a}->e{b}',n,'L1',round(sum(abs(x.get(k,0)-y.get(k,0)) for k in ks),3),'maxd',round(max(abs(x.get(k,0)-y.get(k,0)) for k in ks),3),'top10_out',len(t(x)-t(y)),'paid',len(y),'maxw',round(max(y.values()),3))
s=vec(c,80); r=vec(c0,80); g=w._score(c,80)[1]['window_gate']
print('e80 smoothed: paid',len(s),'raw_top10_included',sum(1 for m in s if m in r),'mass_outside_raw',round(sum(v for m,v in s.items() if m not in r),3),'max_outside',round(max([v for m,v in s.items() if m not in r] or [0]),3),'min_B',min(g[m]['B'] for m in s),'min_z',round(min(g[m]['stouffer_z'] for m in s),2),'miners_e80',w._score(c,80)[1]['miners'])
