import cProfile, pstats, json, sys, time, io
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
import rohost
from pathlib import Path
config, actor, policy, tok, real = rohost.build()
from subnet.complete_group_native import resolve_task
from subnet.backend_jobs import signed
live = rohost.STATE/'epochs/76/coordinator'
m = signed(json.loads((live/'OPENING.json').read_bytes()), actor.id)
docs = []
for p in sorted((live/'captured').glob('*.json'))[:3]:
    for d in signed(json.loads(p.read_bytes()), actor.id)['documents']: docs.append(d)
dispatch = signed(json.loads((live/'DISPATCH.json').read_bytes()), actor.id)
tasks = [(r['env_id'], r['index']) for r in dispatch['selected'][:12]]
pr = cProfile.Profile(); t = time.time(); pr.enable()
for e, i in tasks: resolve_task(m, e, i, tok)
pr.disable(); print('per task', (time.time() - t) / len(tasks))
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(30); print(s.getvalue()[-6000:])
