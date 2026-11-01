#!/usr/bin/env python3
"""usage: patch_probe_worker.py apply|revert|check PROBE_WORKER
The lenient affine_math fix (_allan/_lenient_positive) reads gold answers from the ENV's own task snapshot instead of always
original-math7496: needed for the affine_gatepool env (math 0..7495 + dapo 7496..). For affine_math the snapshot is the same file
(assets/original-math7496.tasks.json resolved against source_root), so its grades do not change. Exact-anchor, idempotent, reversible."""
import sys
PAIRS = [
("""def _allan():
    if 'ns' not in _AV:
        ns = {'__name__': 'allan_lenient_v1'}
        exec(compile('import re\\nfrom math_verify import parse, verify\\n' + _ALLAN_SRC, 'allan_lenient_v1', 'exec'), ns)
        _AV['ns'] = ns
        _AV['tasks'] = json.load(open(os.path.join(PLAN['source_root'], 'assets', 'original-math7496.tasks.json')))
    return _AV['ns'], _AV['tasks']
def _lenient_positive(index, tok, out, stop):
    if stop != 'eos' or not out or out[-1] not in EOS: return None
    ns, tasks = _allan()
""",
"""def _allan(env_id=None):
    if 'ns' not in _AV:
        ns = {'__name__': 'allan_lenient_v1'}
        exec(compile('import re\\nfrom math_verify import parse, verify\\n' + _ALLAN_SRC, 'allan_lenient_v1', 'exec'), ns)
        _AV['ns'] = ns; _AV['by_env'] = {}
    # gatepool-v1: gold answers come from the env's own task snapshot (affine_math -> the same original-math7496 file as before)
    snap = 'assets/original-math7496.tasks.json'
    try: snap = (_G['envs'][env_id]['spec'].get('config') or {}).get('task_snapshot') or snap
    except Exception: pass
    path = snap if os.path.isabs(snap) else os.path.join(PLAN['source_root'], snap)
    if path not in _AV['by_env']: _AV['by_env'][path] = json.load(open(path))
    return _AV['ns'], _AV['by_env'][path]
def _lenient_positive(index, tok, out, stop, env_id=None):
    if stop != 'eos' or not out or out[-1] not in EOS: return None
    ns, tasks = _allan(env_id)
"""),
("                    if _lenient_positive(index, tok, out, stop): kind = 'positive'\n",
 "                    if _lenient_positive(index, tok, out, stop, env_id): kind = 'positive'\n"),
]
mode, p = sys.argv[1], sys.argv[2]; s = open(p).read()
state = ['new' if n in s else 'old' if o in s else 'missing' for o, n in PAIRS]
if mode == 'check': print(p, state); sys.exit(0 if 'missing' not in state else 1)
if 'missing' in state: print('anchor missing', state); sys.exit(1)
for o, n in PAIRS: s = s.replace(o, n) if mode == 'apply' else s.replace(n, o)
open(p, 'w').write(s); print(mode, p)
