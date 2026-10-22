"""Profile original Coordinator.collect on N e76 miners (read-only R2, temp root). Aggregates only."""
import json, sys, time
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
import rohost
from pathlib import Path
N = int(sys.argv[1])
config, actor, policy, tok, real = rohost.build()
stats = rohost.new_stats(); bucket = rohost.ROBucket(real, stats)
from subnet.complete_group_native import resolve_task
import subnet.complete_group_capture as C
T = rohost.new_stats()
def wrap(mod, name):
    f = getattr(mod, name)
    def g(*a, **k):
        t = time.perf_counter()
        try: return f(*a, **k)
        finally: T[name] += time.perf_counter() - t; T['n_'+name] += 1
    setattr(mod, name, g)
for n in ('validate_commitment', 'validate_document', '_submission_get', '_submission_response', '_immutable'): wrap(C, n)
def resolve(m, e, i):
    t = time.perf_counter()
    try: return resolve_task(m, e, i, tok)
    finally: T['resolve_task'] += time.perf_counter() - t; T['n_resolve_task'] += 1
root = Path('/tmp/v826-profile-root')
coord, live = rohost.coordinator_copy(76, root, bucket, actor, resolve, policy)
miners = sorted(p.stem for p in (live/'captured').glob('*.json'))[:N]
t0 = time.perf_counter(); same = 0; per = []
for m in miners:
    t = time.perf_counter()
    rec = coord.collect(m, now=int(time.time()))
    per.append(time.perf_counter() - t)
    same += (root/'captured'/(m+'.json')).read_bytes() == (live/'captured'/(m+'.json')).read_bytes()
out = dict(miners=len(miners), identical_records=same, wall=round(time.perf_counter()-t0, 2), per_miner=[round(x, 2) for x in per],
           phases={k: round(v, 3) for k, v in T.items() if k != 'lock'}, r2={k: round(v, 3) for k, v in stats.items() if k != 'lock'})
print(json.dumps(out, indent=1))
