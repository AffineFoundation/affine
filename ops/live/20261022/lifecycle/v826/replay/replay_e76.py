"""Offline replay of e76 capture with the v8.2.6 CapturePass: READ-ONLY R2, temp coordinator root.
Compares every capture record and the seeded 256 selection with the live e76 result. Aggregates/UIDs only."""
import importlib.util, json, sys, time, faulthandler, signal
faulthandler.register(signal.SIGUSR1, all_threads=True)
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
import rohost
from pathlib import Path
from types import SimpleNamespace
EPOCH = int(sys.argv[1]); WORKERS = int(sys.argv[2]); STAGE = Path(sys.argv[3]); SHARD, NSHARD = map(int, sys.argv[4].split('/'))
config, actor, policy, tok, real = rohost.build()
stats = rohost.new_stats(); bucket = rohost.ROBucket(real, stats)
from subnet.complete_group_native import resolve_task
from subnet.storage import canonical, sha
from subnet.backend_jobs import signed
spec = importlib.util.spec_from_file_location('affine_host_capture_cache', STAGE/'mimo_host_capture_cache.py')
M = importlib.util.module_from_spec(spec); sys.modules[spec.name] = M; spec.loader.exec_module(M)
root = Path('/tmp/v826-replay-e%d-%s-s%d' % (EPOCH, STAGE.name, SHARD))
coord, live = rohost.coordinator_copy(EPOCH, root, bucket, actor, lambda m, e, i: resolve_task(m, e, i, tok), policy)
manifest = signed(json.loads((root/'OPENING.json').read_bytes()), actor.id)
def committed(m):
    found = bucket.complete_commitment_listing(m['epoch'], m['capabilities'], cutoff=time.time() + 30)
    out['listed_total'] = len(found)
    return found[SHARD::NSHARD]
life = SimpleNamespace(services=SimpleNamespace(committed_miners=committed))
cp = M.CapturePass(workers=WORKERS)
out = dict(epoch=EPOCH, workers=WORKERS, shard=SHARD, nshard=NSHARD, now_after_deadline=time.time() >= manifest['deadline'])
for n in (1, 2):
    t = time.time(); rej = cp(life, coord, manifest, int(time.time()))
    out['pass%d' % n] = dict(cp.last_metrics, wall=round(time.time() - t, 1), rejected_reasons=sorted(set(v[:60] for v in rej.values())))
out['r2_ops'] = {k: round(v, 1) for k, v in stats.items() if k != 'lock'}
print(json.dumps(out, indent=1))
json.dump(out, open('/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay/REPLAY-e%d-%s-w%d-s%d.json' % (EPOCH, STAGE.name, WORKERS, SHARD), 'w'), indent=1)
