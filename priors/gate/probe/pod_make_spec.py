#!/usr/bin/env python3
"""Run on the probe pod with the evaluator python: add/remove envs.json entry 'affine_gatepool' (kind live-math) = the live signed
affine_math spec with task_snapshot -> the combined gate-pool snapshot (absolute path), num_samples 19309, source_hash recomputed
by the evaluator's own _source_hash. usage: pod_make_spec.py add|remove|smoke SNAPSHOT"""
import json, os, sys
SRC = '/root/affine-mimo-public-evaluator-20261013/source-final-v822b'; D = '/root/affine-env-probe'
sys.path.insert(0, SRC); sys.path.append(D + '/extra-site')
mode, snap = sys.argv[1], sys.argv[2]
envs = json.load(open(D + '/envs.json'))
def save(d):
    t = D + '/envs.json.tmp-gatepool'; open(t, 'w').write(json.dumps(d, indent=1)); os.replace(t, D + '/envs.json')
if mode == 'remove':
    envs.pop('affine_gatepool', None); save(envs); print('removed'); raise SystemExit
from subnet import environments as E
if mode == 'add':
    n = len(json.load(open(snap)))
    base = envs['affine_math']['spec']
    spec = E.EnvironmentSpec.from_dict(dict(base, num_samples=n, config=dict(base['config'], task_snapshot=snap)))
    spec = E.EnvironmentSpec.from_dict(dict(spec.to_dict(), source_hash=E._source_hash(spec)))
    envs['affine_gatepool'] = dict(kind='live-math', pool=n, spec=spec.to_dict(),
                                   note='SN120 gate pool v1 (prior-v1 baseline): affine_math 0..7495 + dapo 7496..; same adapter/grader as affine_math')
    save(envs); print(json.dumps(dict(added='affine_gatepool', pool=n, source_hash=spec.source_hash)))
elif mode == 'smoke':
    tasks = json.load(open(snap)); sp = E.EnvironmentSpec.from_dict(envs['affine_gatepool']['spec'])
    assert E._source_hash(sp) == sp.source_hash, 'source hash mismatch'
    s = E.create_session(sp); bad = []
    for i in (0, 1, 7495, 7496, 7497, 12000, len(tasks) - 1):
        r = s.reset(i, 0); txt = json.dumps(r.get('messages'), ensure_ascii=False)
        if tasks[i]['data']['problem'][:60] not in json.loads(json.dumps(txt)): bad.append(i)
    try: s.close_runtime()
    except Exception: pass
    from math_verify import parse, verify
    gv = [i for i in range(7496, len(tasks), 997) if not verify(parse('\\boxed{%s}' % tasks[i]['data']['answer']), parse('\\boxed{%s}' % tasks[i]['data']['answer']))]
    print(json.dumps(dict(smoke='ok' if not bad and not gv else 'FAIL', prompt_mismatch=bad, gold_not_self_equal=gv)))
    sys.exit(1 if bad or gv else 0)
