"""v8.2.7 offline replay (READ-ONLY R2, temp root): pass1 cold, pass2 forced full re-collect of every miner
(exercises the incremental memo path for every old group), pass3 warm probes. Aggregates only."""
import importlib.util, json, sys, time
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
import rohost
from pathlib import Path
from types import SimpleNamespace
EPOCH = int(sys.argv[1]); WORKERS = int(sys.argv[2]); STAGE = Path(sys.argv[3]); SHARD, NSHARD = map(int, sys.argv[4].split('/'))
config, actor, policy, tok, real = rohost.build()
stats = rohost.new_stats(); bucket = rohost.ROBucket(real, stats)
from subnet.complete_group_native import resolve_task
from subnet.backend_jobs import signed
spec = importlib.util.spec_from_file_location('affine_host_capture_cache', STAGE/'mimo_host_capture_cache.py')
M = importlib.util.module_from_spec(spec); sys.modules[spec.name] = M; spec.loader.exec_module(M)
root = Path('/tmp/v827-replay-e%d-s%d' % (EPOCH, SHARD))
coord, live = rohost.coordinator_copy(EPOCH, root, bucket, actor, lambda m, e, i: resolve_task(m, e, i, tok), policy)
manifest = signed(json.loads((root/'OPENING.json').read_bytes()), actor.id)
out = dict(epoch=EPOCH, workers=WORKERS, shard=SHARD, nshard=NSHARD, version=M.VERSION)
def committed(m):
    found = bucket.complete_commitment_listing(m['epoch'], m['capabilities'], cutoff=time.time() + 30)
    return found[SHARD::NSHARD]
life = SimpleNamespace(services=SimpleNamespace(committed_miners=committed))
cp = M.CapturePass(workers=WORKERS)
for n in (1, 2, 3):
    if n == 2: cp.accepted.clear()   # force original collect of every miner; memo tables stay warm
    t = time.time(); rej = cp(life, coord, manifest, int(time.time()))
    out['pass%d' % n] = dict(cp.last_metrics, wall=round(time.time() - t, 1), rejected_reasons=sorted(set(v[:60] for v in rej.values())))
out['r2_ops'] = {k: round(v, 1) for k, v in stats.items() if k != 'lock'}
print(json.dumps(out, indent=1))
json.dump(out, open('/home/const/subnet120-rewrite/state/root-audits/20261021-v827-incremental-capture/replay/REPLAY-e%d-s%d.json' % (EPOCH, SHARD), 'w'), indent=1)
