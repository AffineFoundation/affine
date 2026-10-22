import cProfile, pstats, io, json, sys, time, importlib.util
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
import rohost
from pathlib import Path
config, actor, policy, tok, real = rohost.build()
stats = rohost.new_stats(); bucket = rohost.ROBucket(real, stats)
from subnet.complete_group_native import resolve_task
spec = importlib.util.spec_from_file_location('affine_host_capture_cache', '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/stage/mimo_host_capture_cache.py')
M = importlib.util.module_from_spec(spec); sys.modules[spec.name] = M; spec.loader.exec_module(M)
M._install_pure_memo()
live = rohost.STATE/'epochs/76/coordinator'
coord, _ = rohost.coordinator_copy(76, Path('/tmp/v826-profmemo-root'), bucket, actor, lambda a, e, i: resolve_task(a, e, i, tok), policy)
miners = sorted(p.stem for p in (live/'captured').glob('*.json'))[100:103]
pm = {}
pr = cProfile.Profile(); t = time.time(); pr.enable()
for mm in miners:
    M._active.memo = {'collect': {}, 'pass': pm}
    try: coord.collect(mm, now=int(time.time()))
    finally: M._active.memo = None
pr.disable(); print('per miner', (time.time() - t) / 3)
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(45); print(s.getvalue()[-9000:])
