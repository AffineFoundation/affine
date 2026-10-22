import sys, json, glob, math, importlib.util, collections, statistics
sys.path.insert(0,'/workspace/sn120-v9fix'); from lib import S, U, cr
from ss58 import ss58
spec=importlib.util.spec_from_file_location('w9','/workspace/sn120-v9fix/new/weights_v9.py'); w=importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
meta={x['hotkey']:x for x in json.load(open('/workspace/sn120-v9fix/meta.json'))} if len(sys.argv)<3 else {}
SCOR='749593a955893cb69a8025b4cfef47060ca0fa9f87eb96a7d89dfabf01a65f6d'
def cold(m):
    x=meta.get(ss58(m)); return x['coldkey'] if x else '?'
def uid(m): return U.get(m)
# actual paid vector per score epoch (last hourly assessment that used that score epoch)
act={}
for f in sorted(glob.glob('/workspace/sn120-v9fix/asmt/assessment-*.json')):
    a=json.load(open(f))['payload']; v=a.get('v9') or {}
    if v.get('score_epoch') is not None: act[v['score_epoch']]={m:s for m,s in a['points'].items() if m!=w.BURN}
def grp(E):
    p=S[E]; return {m:sum(u['groups'] for u in r['units']) for m,r in p['epoch_units'].items()}
def metrics(E, sh, label):
    g=grp(E); n=len(sh)
    if not n: return dict(label=label,E=E,seats=0)
    full9=[m for m,x in g.items() if x>=9]; 
    ck=collections.Counter(); 
    for m,s in sh.items(): ck[cold(m)]+=s
    top2=sum(v for _,v in ck.most_common(2))
    zero=[m for m in sh if g.get(m,0)==0]
    # overlap: share of weight on miners in the newest epoch, weighted by groups
    tot_g=sum(g.values()); grp_share={m:x/tot_g for m,x in g.items()}
    overlap=sum(min(sh.get(m,0),grp_share.get(m,0)) for m in set(sh)|set(grp_share))
    return dict(label=label,E=E,seats=n,max_share=round(max(sh.values()),4),top2_coldkeys=round(top2,3),
        zero_group_paid=len(zero), zero_group_weight=round(sum(sh[m] for m in zero),3),
        full9=len(full9), full9_unpaid=sum(1 for m in full9 if m not in sh), overlap_with_groups=round(overlap,3),
        uid310=round(sh.get(next((m for m in sh if uid(m)==310),None),0),4), uid270=round(sh.get(next((m for m in sh if uid(m)==270),None),0),4))
def newvec(E, W, G=1, N=0):
    c=dict(scores_dir='/workspace/sn120-v9fix/scores', scorer_pubkey=SCOR, miner_share=1.0, window_payout=dict(rule=w.WINDOW_RULE,window_epochs=W,min_window_groups=G,min_newest_groups=N))
    sh,rows,d=w.window_credit(c,E,S[E]); return sh
def churn(a,b):  # L1/2 distance between consecutive vectors
    return round(0.5*sum(abs(a.get(m,0)-b.get(m,0)) for m in set(a)|set(b)),3)
if __name__=='__main__':
    Es=[E for E in sorted(S) if 80<=E<=90]
    rows=[]
    for lab,f in [('live seq6',lambda E: act.get(E,{})),('W=2',lambda E:newvec(E,2)),('W=3',lambda E:newvec(E,3)),('W=4',lambda E:newvec(E,4)),('W=3,N=1',lambda E:newvec(E,3,1,1)),('W=4,N=1',lambda E:newvec(E,4,1,1))]:
        prev=None
        for E in Es:
            sh=f(E); m=metrics(E,sh,lab); m['churn_vs_prev']=churn(prev,sh) if prev is not None else None; prev=sh; rows.append(m)
    json.dump(rows,open('/workspace/sn120-v9fix/backtest.json','w'),indent=1)
    keys=['label','E','seats','max_share','top2_coldkeys','zero_group_paid','zero_group_weight','full9','full9_unpaid','overlap_with_groups','churn_vs_prev','uid310','uid270']
    print('\t'.join(keys))
    for r in rows: print('\t'.join(str(r.get(k)) for k in keys))
