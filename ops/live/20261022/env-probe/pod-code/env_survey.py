"""Survey which registered envs are runnable single-turn text envs on this host (CPU only)."""
import sys, json, time, itertools, signal, os
SRC = sys.argv[1]; ENV = sys.argv[2]; CAP = int(sys.argv[3]); OUT = sys.argv[4]
sys.path.insert(0, SRC); sys.path.append('/root/affine-env-probe-20261022/extra-site')
from subnet import environments as E
t0 = time.time(); res = dict(env=ENV)
try:
    spec = E.build_spec(ENV, num_samples=1, max_turns=1, max_output_tokens=8192)
    ts = E._taskset(spec)
    rows = []
    for t in itertools.islice(iter(ts), CAP):
        rows.append(dict(task_class=type(t).__name__, data=t.data.model_dump(mode='json'), task_config=t.config.model_dump(mode='json')))
    res['pool'] = len(rows); res['capped'] = len(rows) >= CAP; res['snapshot_s'] = round(time.time() - t0, 1)
    snap = os.path.join(os.path.dirname(OUT), f'snap-{ENV}.json')
    open(snap, 'wb').write(E._canonical(rows))
    spec2 = E.EnvironmentSpec.from_dict(dict(spec.to_dict(), num_samples=len(rows), config=dict(spec.config, task_snapshot=os.path.abspath(snap))))
    spec2 = E.EnvironmentSpec.from_dict(dict(spec2.to_dict(), source_hash=E._source_hash(spec2)))
    s = E.create_session(spec2); r = s.reset(0, 0)
    res['tools'] = len(r.get('tools') or []); res['messages'] = len(r['messages'])
    res['prompt_chars'] = sum(len(m.get('content') or '') if isinstance(m.get('content'), str) else 0 for m in r['messages'])
    st = s.step({'text': 'I do not know.'}); res['step'] = dict(done=st['done'], reward=st['reward'], cls=st['classification'])
    s.close(); res['ok'] = True
except BaseException as e:
    res['ok'] = False; res['error'] = f'{type(e).__name__}: {str(e)[:200]}'
res['seconds'] = round(time.time() - t0, 1)
json.dump(res, open(OUT, 'w')); print(json.dumps(res))
