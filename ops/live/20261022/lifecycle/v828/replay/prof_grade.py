"""v8.2.8 profile (READ-ONLY R2): time serial R2 document fetch and in-process native grade_batch for a sample
of e<N> documents, with cProfile on grading. Aggregates only (no miner data printed)."""
import cProfile, io, json, pstats, sys, time
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
import rohost
from pathlib import Path
EPOCH=int(sys.argv[1]); N=int(sys.argv[2])
config, actor, policy, tok, real = rohost.build()
stats = rohost.new_stats(); bucket = rohost.ROBucket(real, stats)
from subnet.backend_jobs import signed
from subnet.storage import sha, canonical
from subnet.complete_group_capture import _get
from subnet.complete_group_codec import limits, loads
from subnet.complete_group_native import grade_batch
live = rohost.STATE/'epochs'/str(EPOCH)/'coordinator'
manifest = signed(json.loads((live/'OPENING.json').read_bytes()), actor.id)
t=time.perf_counter(); pop = signed(json.loads((live/'POPULATION.json').read_bytes()), actor.id); t_pop=time.perf_counter()-t
items=[]
for env in pop['captures']:
    c=signed(env, actor.id)
    for it in c['documents']: items.append((c['miner'], it))
step=max(1,len(items)//N); sample=items[::step][:N]
fetch=[]; grade=[]; nroll=0; docs=[]
for miner,it in sample:
    t=time.perf_counter(); data,_=_get(bucket,it['key'],limits(manifest).metadata_bytes); t1=time.perf_counter()
    assert sha(data)==it['sha256']; doc=loads(data,limits(manifest).metadata_bytes); t2=time.perf_counter()
    fetch.append((t1-t, t2-t1, len(data))); docs.append((miner,doc['batch']))
pr=cProfile.Profile()
for miner,batch in docs:
    t=time.perf_counter(); pr.enable(); g=grade_batch(batch, manifest, tok, miner=miner); pr.disable(); grade.append(time.perf_counter()-t); nroll+=len(batch['rollouts'])
s=io.StringIO(); ps=pstats.Stats(pr,stream=s).sort_stats('cumulative'); ps.print_stats(45)
out=dict(epoch=EPOCH, documents_total=len(items), sample=len(sample), rollouts_sampled=nroll, population_load_verify_s=round(t_pop,2),
  fetch_get_s_mean=round(sum(f[0] for f in fetch)/len(fetch),3), decode_s_mean=round(sum(f[1] for f in fetch)/len(fetch),3),
  doc_bytes_mean=int(sum(f[2] for f in fetch)/len(fetch)), grade_s_mean=round(sum(grade)/len(grade),3), grade_s_max=round(max(grade),3),
  grade_s_min=round(min(grade),3))
print(json.dumps(out,indent=1)); print(s.getvalue()[:12000])
