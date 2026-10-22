import importlib.util, json, math, sys
sys.argv=['x']; exec(open('/home/const/subnet120-rewrite/state/root-audits/20261021-v9-score-ema/replay_v2.py').read().split("rows=[]")[0])
spec=importlib.util.spec_from_file_location('w9n','/home/const/subnet120-rewrite/state/root-audits/20261021-v9-score-ema/staged/weights_v9.py'); w=importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
c=w.load_config('/home/const/subnet120-rewrite/state/root-audits/20261020-measured-impact-v9/LIVE-CONFIG-v9.ACTIVE.signed.json','3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd')
cs=dict(c, smoothing=dict(rule='share-ema-v1', half_life_epochs=4, lookback_epochs=12, min_share=0.01))
ref=gated(emav(4,12,0.01),False)
for L in range(70,80):
    sh0,b0,i0=w.v9_vector(c,L); sh,b,i=w.v9_vector(cs,L); E=i['score_epoch']
    old=S[E]; dif0=max(abs(sh0.get(m,0)-old.get(m,0)) for m in set(sh0)|set(old))
    r=ref[E]; dif=max(abs(sh.get(m,0)-r.get(m,0)) for m in set(sh)|set(r))
    L1=sum(abs(sh.get(m,0)-sh0.get(m,0)) for m in set(sh)|set(sh0))
    top=sorted(sh.values())[-1]; topuid=U[max(sh,key=sh.get)]
    print(dict(L=L,E=E,legacy_cfg_vs_seq5=round(dif0,12),smoothed_vs_replay=round(dif,12),paid=len(sh),burn=round(b,9),maxw=round(top,3),maxw_uid=topuid,maxw_uid_raw=round(sh0.get(max(sh,key=sh.get),0),3),L1_vs_unsmoothed=round(L1,3),sm={k:i['smoothing'][k] for k in ('epochs_used','epochs_skipped','candidates','dropped_ineligible','dropped_below_min','paid')}))
