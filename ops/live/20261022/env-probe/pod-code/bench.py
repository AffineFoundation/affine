"""Scaling bench: run the probe worker against the live vLLM server with explicit (envs, N, k, max_tokens, concurrency).
usage: bench.py LABEL N K MAXTOK CONC env1,env2,...|all [SEED]"""
import json, os, random, subprocess, sys, hashlib
D = '/root/affine-env-probe-20261022'
label, N, K, MT, C, envs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
seed = sys.argv[7] if len(sys.argv) > 7 else 'bench-' + label
reg = json.load(open(D + '/envs.json')) if os.path.exists(D + '/envs.json') else {}
live = json.load(open(D + '/live_math_manifest_subset.json'))
reg.setdefault('affine_math', dict(kind='live-math', pool=live['spec']['num_samples'], spec=live['spec']))
names = sorted(reg) if envs == 'all' else envs.split(',')
plan = dict(checkpoint_id=open(D + '/serve.ckpt').read().strip(), seed=seed, concurrency=C, grade_workers=32,
            harness=live['harness'], manifest_projection=dict(mimo_answer_projection=live['mimo_answer_projection'], sampling_contract=live['sampling_contract']),
            source_root='/root/affine-mimo-public-evaluator-20261013/source-final-v822b', vllm_url='http://127.0.0.1:18555', served_model='probe',
            pause_file=D + '/PAUSE', envs=[])
plan['ckpt_dir'] = f"{D}/ckpt/{plan['checkpoint_id']}"
for e in names:
    rng = random.Random(int(hashlib.sha256(f'{seed}:{e}'.encode()).hexdigest(), 16))
    plan['envs'].append(dict(env_id=e, kind=reg[e]['kind'], spec=reg[e]['spec'], k=K, max_tokens=MT, indices=sorted(rng.sample(range(reg[e]['pool']), min(N, reg[e]['pool'])))))
os.makedirs(D + '/bench', exist_ok=True); pf = f'{D}/bench/{label}.plan.json'; json.dump(plan, open(pf, 'w'))
r = subprocess.run(['nice', '-n', '15', '/root/affine-mimo-20261013/venv/bin/python', '-B', D + '/probe_worker.py', pf, f'{D}/bench/{label}.json',
                    '--progress', f'{D}/bench/{label}.progress.json'], capture_output=True, text=True)
print(r.stdout[-2000:]); print(r.stderr[-3000:], file=sys.stderr)
