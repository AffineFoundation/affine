import json, sys, time, glob
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
import rohost
config, actor, policy, tok, real = rohost.build()
from subnet.backend_jobs import signed
from subnet.storage import canonical, sha
from subnet.mimo_release_succession import ReleaseChain
chain = ReleaseChain(config['release_succession'], actor.id, allow_public=True)
from subnet.mimo_evaluation_scheduler import validate_sets
paths = sorted(glob.glob(str(rohost.STATE/'epochs/*/coordinator/OPENING.json')))
t = time.process_time(); w = time.time(); raws = [open(p, 'rb').read() for p in paths]; r = time.time() - w
t = time.process_time(); ops = [json.loads(x) for x in raws]; parse = time.process_time() - t
t = time.process_time(); ms = [signed(o, actor.id) for o in ops]; sig = time.process_time() - t
t = time.process_time(); bad = 0
for m in ms:
    try: chain.validate_manifest(m)
    except Exception: bad += 1
val = time.process_time() - t
t = time.process_time(); [validate_sets(m) for m in ms]; vs = time.process_time() - t
t = time.process_time(); [sha(canonical(o)) for o in ops]; dg = time.process_time() - t
print(json.dumps(dict(openings=len(paths), total_mb=round(sum(map(len, raws))/1e6), read_s=round(r, 2), parse_cpu=round(parse, 2), signed_cpu=round(sig, 2),
    validate_manifest_cpu=round(val, 2), validate_failed=bad, validate_sets_cpu=round(vs, 2), digest_cpu=round(dg, 2))))
