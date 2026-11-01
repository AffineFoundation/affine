#!/usr/bin/env python3
"""OFFLINE fake probe pod for the prior-v1 end-to-end dry run (HG_PROBE_SSH='["python3","fakepod.py"]'). Emulates the podctl commands
gate_service/prior_gate issue (status, serve, start-run, finalize-run, cat result.json, fingerprint, round settings). Results are
synthetic: p_i from FAKEPOD_P (json {probe index: p}), plus a per-checkpoint logit shift (FAKEPOD_SHIFT json {ckpt prefix: shift})
and a per-run logit offset N(0, tau). Never contacts a real pod."""
import hashlib, json, math, os, random, sys
D = os.environ['FAKEPOD_DIR']; os.makedirs(D + '/runs', exist_ok=True)
st_p = D + '/state.json'
st = json.load(open(st_p)) if os.path.exists(st_p) else dict(served=None, ckpts=[])
args = [a for a in sys.argv[1:] if a != '-n']; cmd = args[-1] if args else ''
def save(): json.dump(st, open(st_p, 'w'))
def status():
    runs = {}
    for n in sorted(os.listdir(D + '/runs')): runs[n] = dict(result=os.path.exists('%s/runs/%s/result.json' % (D, n)), running=False)
    return dict(at=0, paused=False, hold=False, replicas=8, serve_procs=8 if st['served'] else 0, serve_ckpt=st['served'],
                replicas_up=8, served=st['served'], ckpts=json.loads(os.environ.get('FAKEPOD_CKPTS', '[]')) + st['ckpts'], fetching=False, gpu=[], runs=runs)
class All(list):
    pass
if 'podctl.py status' in cmd:
    s = status(); print(json.dumps(s))
elif 'podctl.py serve ' in cmd:
    ck = cmd.split('podctl.py serve ')[1].split()[0]; st['served'] = ck; st['ckpts'] = sorted(set(st['ckpts'] + [ck])); save()
    print(json.dumps(dict(serving=ck)))
elif 'podctl.py start-run' in cmd:
    plan = json.loads(sys.stdin.read()); P = json.load(open(os.environ['FAKEPOD_P'])); B = json.load(open(os.environ['FAKEPOD_B']))
    shift = json.loads(os.environ.get('FAKEPOD_SHIFT', '{}')); tau = float(os.environ.get('FAKEPOD_TAU', '0.0457'))
    ck = plan['checkpoint_id']; sh = sum(v for k, v in shift.items() if ck.startswith(k))
    rng = random.Random(int(hashlib.sha256((plan['seed'] + ck).encode()).hexdigest(), 16)); off = rng.gauss(0, tau)
    envs = {}
    for e in plan['envs']:
        pt = {}
        for i in e['indices']:
            p = min(max(P[str(i)], 1e-6), 1 - 1e-6); lp = math.log(p / (1 - p)) + sh + off; q = 1 / (1 + math.exp(-lp))
            c = sum(rng.random() < q for _ in range(e['k'])); bfrac = B.get(str(i), 0.5)
            b = sum(rng.random() < bfrac for _ in range(e['k'] - c))
            pt[str(i)] = dict(n=e['k'], c=c, w=e['k'] - c - b, u=b, b=b, x=0, tok=0, L=[])
        envs[e['env_id']] = dict(per_task=pt, pass_rate=None)
    os.makedirs('%s/runs/%s' % (D, plan['run_name']), exist_ok=True)
    json.dump(dict(envs=envs, wall_seconds=1.0, partial=False), open('%s/runs/%s/result.json' % (D, plan['run_name']), 'w'))
    print(json.dumps(dict(started=plan['run_name'])))
elif 'gate_fetch.py' in cmd:
    sys.stdin.read(); print('ok')   # never reached when FAKEPOD_CKPTS lists the checkpoints
elif 'podctl.py finalize-run' in cmd:
    print('{}')
elif cmd.startswith('cat ') and 'result.json' in cmd:
    name = cmd.split('/runs/')[1].split('/result.json')[0]; print(open('%s/runs/%s/result.json' % (D, name)).read())
elif "live_math_manifest_subset.json'));print" in cmd:
    print(json.dumps(dict(harness=dict(temperature=1.0, top_p=0.95, top_k=20, enable_thinking=True), sampling_contract='fake')))
    print('fake-runtime-line')
elif 'sha256sum probe_worker.py serve.sh podctl.py envs.json live_math' in cmd:
    print('fake-fingerprint')
else:
    sys.stderr.write('fakepod: unknown command %r\n' % cmd[:120]); sys.exit(2)
