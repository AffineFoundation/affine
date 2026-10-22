"""v8.2.8 profile (READ-ONLY R2, temp coordinator copy; no queue DB access): time every step one lifecycle tick
repeats after seal, plus the one-time audit-population build and training-job validation. Aggregates only."""
import json, shutil, sys, time
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline/replay')
import rohost
from pathlib import Path
from types import SimpleNamespace
EPOCH=int(sys.argv[1])
config, actor, policy, tok, real = rohost.build()
stats = rohost.new_stats(); bucket = rohost.ROBucket(real, stats)
from subnet.backend_jobs import signed
from subnet.mimo_coordinator import Coordinator
from subnet.mimo_lifecycle import Lifecycle
from subnet.mimo_release_policy import digest
from subnet.complete_group_capture import _get
from subnet.storage import sha, canonical
E = rohost.STATE/'epochs'/str(EPOCH); live = E/'coordinator'
root = Path('/tmp/v828-tick-e%d' % EPOCH)
if root.exists(): shutil.rmtree(root)
(root/'coordinator').mkdir(parents=True, mode=0o700)
for n in ('OPENING.json','OPENING-REQUEST.json','POPULATION.json','SELECTION.json','DISPATCH.json'): shutil.copy2(live/n, root/'coordinator'/n)
T = {}
def tm(name, f):
    t = time.perf_counter(); r = f(); T[name] = round(time.perf_counter()-t, 2); return r
plan = tm('load_plan', lambda: json.loads((E/'PLAN.json').read_bytes()))
import importlib.util
_h=importlib.util.spec_from_file_location('uidcap','/home/const/subnet120-rewrite/state/root-audits/20261021-v827-incremental-capture/release-v827-signed/mimo_uid_capacity.py');H=importlib.util.module_from_spec(_h);_h.loader.exec_module(H)
Coordinator.open=H.expanded_method(Coordinator.open,'Coordinator.open','5a66eb7afd4307583a9cdbf2879b21a0d21fd882beaa625d25e6402c95d05503')
coord = Coordinator(root/'coordinator', bucket, actor, resolve_task=None, grade_batch=None, release_policy=policy)
opening = tm('coordinator_open_existing(incl R2 compare)', lambda: coord.open(plan['manifest'], plan['miners']))
manifest = tm('signed(opening)', lambda: signed(opening, actor.id))
dispatch = tm('seal_existing_dispatch', lambda: coord.seal(max_tasks=256, now=int(time.time())))
self = SimpleNamespace(_load=lambda p: json.loads(p.read_bytes()), bucket=bucket, identity=actor, policy=policy)
current = plan['input']
# one-time build (fresh dir, original method) and the per-tick re-authentication (file present)
fresh = root/'fresh'; fresh.mkdir()
doc = tm('audit_population_BUILD_original(once)', lambda: Lifecycle._audit_population(self, fresh, coord, opening, dispatch, plan, current))
T['audit_population_identical_to_live'] = (fresh/'AUDIT-POPULATION.json').read_bytes() == (E/'AUDIT-POPULATION.json').read_bytes()
shutil.copy2(E/'AUDIT-POPULATION.json', root/'AUDIT-POPULATION.json')
tm('audit_population_REAUTH_original(every tick)', lambda: Lifecycle._audit_population(self, root, coord, opening, dispatch, plan, current))
from subnet.mimo_audit_bridge import build_population, authenticate_population
d2 = tm('authenticate_population_only', lambda: authenticate_population(json.loads((E/'AUDIT-POPULATION.json').read_bytes()), actor.id, manifest_guard=policy.validate_manifest))
job_env = tm('load_training_job', lambda: json.loads((E/'TRAINING-JOB.json').read_bytes()))
job = tm('signed(job)', lambda: signed(job_env, actor.id))
from subnet.mimo_training_worker import validate_job
tm('validate_job', lambda: validate_job(job, manifest, actor.id, allow_public=True))
tm('signed(dispatch)', lambda: signed(dispatch, actor.id))
tm('digest(dispatch)', lambda: digest(dispatch))
tm('head_256_selected_docs', lambda: [bucket.client.head_object(Bucket=bucket.name, Key=r['document']['key']) for r in signed(dispatch, actor.id)['selected']])
T['training_job_audit_keys'] = sorted(k for k in json.dumps(job_env['payload']['training'].keys() and list(job_env['payload']['training'].keys())).split('"') if 'audit' in k.lower())
T['training_job_training_keys'] = sorted(job_env['payload']['training'].keys())
T['r2_ops'] = {k: round(v, 1) for k, v in stats.items() if k != 'lock'}
print(json.dumps(T, indent=1))
json.dump(T, open('/home/const/subnet120-rewrite/state/root-audits/20261021-v828-audit-off-path/replay/PROFILE-TICK-e%d.json' % EPOCH, 'w'), indent=1)
shutil.rmtree(root)
