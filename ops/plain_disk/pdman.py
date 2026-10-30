# sha256 identity of a gocryptfs original vs its plain-disk copy, over files present in both (skips in-progress *.partial dirs / tmp files).
import os,sys,hashlib,json
from concurrent.futures import ThreadPoolExecutor
o,n=sys.argv[1],sys.argv[2]; L=[]
for r,ds,fs in os.walk(o):
    ds[:]=[d for d in ds if not d.endswith('.partial')]
    for f in fs:
        p=os.path.relpath(os.path.join(r,f),o)
        if not os.path.islink(os.path.join(o,p)) and not f.endswith(('.tmp','.partial','.lock')) and '.tmp' not in f: L.append(p)
def h(base,p):
    s=hashlib.sha256()
    try:
        with open(os.path.join(base,p),'rb') as f:
            while b:=f.read(1<<22): s.update(b)
    except FileNotFoundError: return None
    return s.hexdigest()
both=[p for p in L if os.path.isfile(os.path.join(n,p))]
with ThreadPoolExecutor(16) as ex: A=list(ex.map(lambda p:h(o,p),both)); B=list(ex.map(lambda p:h(n,p),both))
k=[(p,a,b) for p,a,b in zip(both,A,B) if a is not None and b is not None]
bad=[p for p,a,b in k if a!=b]
dg=lambda i:hashlib.sha256(json.dumps(sorted((x[0],x[i]) for x in k)).encode()).hexdigest()[:16]
print(json.dumps({"files":len(k),"pruned_in_new":len(L)-len(both),"bad":len(bad),"bad_first":bad[:5],"orig":dg(1),"new":dg(2)}))
