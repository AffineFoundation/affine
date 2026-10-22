import json,glob,datetime as dt
S={}
for f in sorted(glob.glob('/workspace/sn120-v9fix/scores/score-e*.json')):
    p=json.load(open(f))['payload']; S[p['epoch']]=p
U={}
for E in sorted(S):
    p=S[E]
    for m,r in p['window_gate'].items():
        if r['uid'] is not None: U[m]=r['uid']
    for m,r in p['epoch_units'].items():
        if r['uid'] is not None: U[m]=r['uid']
def groups(E):
    p=S.get(E); 
    if not p: return None
    return {U.get(m,m): sum(x['groups'] for x in r['units']) for m,r in p['epoch_units'].items()}
def cr(ts): return (dt.datetime.utcfromtimestamp(ts)-dt.timedelta(hours=6)).strftime('%m-%d %H:%M')
