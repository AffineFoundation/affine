import json, os, glob, hashlib, sqlite3, collections, sys
RA="/home/const/subnet120-rewrite/state/root-audits"
G=RA+"/20261026-promotion-gate/state"
pool=[json.loads(l)["id"] for l in open(RA+"/20261101-baseline-switch/public/gate-pool-v1.jsonl")]
POOL=set(pool)
DO=7496
def tid(idx, pv1):
    i=int(idx)
    if pv1 and i>=DO: return "dapo:%d"%(i-DO)
    return "math:%d"%i
cells=collections.defaultdict(lambda:[0,0])   # (ckpt8, task) -> [n,s]
src=collections.Counter(); srcck=collections.defaultdict(set)
seen=set()
def add(ck, t, n, s, tag):
    ck = ck[:8] if len(ck)==64 else ck
    if t not in POOL or not n: return
    c=cells[(ck,t)]; c[0]+=int(n); c[1]+=int(s); src[tag]+=int(n); srcck[tag].add(ck)
def add_pt(ck, pt, pv1, tag):
    h=hashlib.sha256(json.dumps([ck,sorted((k,v.get("n"),v.get("c")) for k,v in pt.items() if isinstance(v,dict))]).encode()).hexdigest()
    if h in seen: return False
    seen.add(h)
    for k,v in pt.items():
        if isinstance(v,dict) and v.get("n"): add(ck, tid(k,pv1), v["n"], v["c"], tag)
    return True
# A prior table
con=sqlite3.connect("file:%s/20261101-prior-table/private/table/prior_table.private.sqlite?mode=ro"%RA, uri=True)
for h,t,n,c in con.execute("select head,task_id,sum(n),sum(c) from cells group by head,task_id"): add(h,t,n,c,"prior_sampler")
con.close()
# B heldout rounds
rounds=set()
for d in sorted(glob.glob(G+"/heldout/r*")):
    try: R=json.load(open(d+"/ROUND.private.json"))
    except Exception: continue
    pv1 = R.get("baseline")=="prior-v1" or "cfg" in R and R.get("env")!="affine_math"
    H=R.get("head")
    blocks=sorted(glob.glob(d+"/BLOCK*.private.json"))
    used=False
    for b in blocks:
        B=json.load(open(b)); used|=add_pt(B.get("head") or H, B.get("per_task",{}), pv1, "heldout_head_blocks")
    if not blocks:
        for b in glob.glob(d+"/BASELINE*.private.json"):
            B=json.load(open(b)); used|=add_pt(B.get("head") or H, B.get("per_task",{}), pv1, "heldout_head_baseline")
    rounds.add(os.path.basename(d))
# C/D epochs
for e in sorted(os.listdir(G+"/epochs"), key=int):
    for f,tag in (("H-PROBE.private.json","epoch_head_probe"),("C-PROBE.private.json","epoch_candidate_probe")):
        p=G+"/epochs/%s/%s"%(e,f)
        if not os.path.exists(p): continue
        try: D=json.load(open(p))
        except Exception: continue
        if f.startswith("H") and D.get("heldout_round") in rounds: continue
        ck=D.get("head") or D.get("candidate")
        if not ck: continue
        add_pt(ck, D.get("per_task") or {}, bool(D.get("prior_v1")), tag)
# E aa-noise
for run in ("run1","run2"):
    p=RA+"/20261101-aa-noise/private/%s/rollouts.jsonl"%run
    if not os.path.exists(p): continue
    for l in open(p):
        r=json.loads(l)
        if r.get("head","").startswith("cb92bdc5"): add("cb92bdc5", r["task_id"], 1, r.get("kind")=="positive", "aa_noise")
# F allan 2x2 offline arms
for d in glob.glob(RA+"/20261030-allan-2x2/results/out/*/"):
    arm=os.path.basename(d.rstrip("/"))
    for f in glob.glob(d+"gate_s*.jsonl"):
        step=f.split("gate_s")[1].split(".")[0]
        if step=="00": ck="7fa0305a" if arm.startswith("A7fa") else "6b16e298"
        else: ck="x%s_%s"%(arm,step)
        agg=collections.defaultdict(lambda:[0,0])
        for l in open(f):
            r=json.loads(l); a=agg[r["index"]]; a[0]+=1; a[1]+=bool(r["correct"])
        pt={str(k):dict(n=v[0],c=v[1]) for k,v in agg.items()}
        add_pt(ck, pt, False, "allan2x2_offline")
out=open(os.environ.get("IRT_WORK", "/tmp/irt") + "/cells.tsv","w")
for (ck,t),(n,s) in cells.items(): out.write("%s\t%s\t%d\t%d\n"%(ck,t,n,s))
out.close()
cks=set(k for k,_ in cells); ts=set(t for _,t in cells)
print("cells",len(cells),"ckpts",len(cks),"tasks",len(ts),"rollouts",sum(v[0] for v in cells.values()))
for k,v in src.items(): print(" ",k,"rollouts",v,"ckpts",len(srcck[k]))
