"""Build the probe env registry on the pod: the live signed affine_math spec + every surveyed single-turn text env
that resets/grades with the pinned evaluator source (snapshots pinned, source hashes recomputed). Writes envs.json."""
import json, os, sys, glob, shutil
SRC = '/root/affine-mimo-public-evaluator-20261013/source-final-v822b'; D = '/root/affine-env-probe-20261022'
sys.path.insert(0, SRC); sys.path.append(D + '/extra-site')
from subnet import environments as E
live = json.load(open(D + '/live_math_manifest_subset.json'))
out = {'affine_math': dict(kind='live-math', pool=live['spec']['num_samples'], spec=live['spec'], note='exact live signed spec (7496-task snapshot)')}
os.makedirs(D + '/envs', exist_ok=True)
MAXCH = 24000  # ~6k tokens; leaves >= 6k of the 12288 context for output
for f in sorted(glob.glob(D + '/survey/affine_*.json')):
    r = json.load(open(f)); env = r['env']
    if env == 'affine_math': continue
    if not r.get('ok'): out_skip = r.get('error', '')[:120]; print('skip', env, out_skip); continue
    if r.get('tools'): print('skip', env, 'tool env (multi-turn)'); continue
    if r.get('prompt_chars', 0) > MAXCH: print('skip', env, 'prompt exceeds context'); continue
    snap = D + f'/envs/snap-{env}.json'; shutil.copyfile(D + f'/survey/snap-{env}.json', snap)
    spec = E.build_spec(env, num_samples=r['pool'], max_turns=1, max_output_tokens=8192)
    spec = E.EnvironmentSpec.from_dict(dict(spec.to_dict(), config=dict(spec.config, task_snapshot=snap)))
    spec = E.EnvironmentSpec.from_dict(dict(spec.to_dict(), source_hash=E._source_hash(spec)))
    out[env] = dict(kind='text', pool=r['pool'], spec=spec.to_dict(), note='first %d tasks of the registry taskset' % r['pool'])
json.dump(out, open(D + '/envs.json', 'w'), indent=1)
print(json.dumps({k: v['pool'] for k, v in out.items()}))
