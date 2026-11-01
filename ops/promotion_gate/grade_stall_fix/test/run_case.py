"""Run one probe_worker.py on a stub plan; usage: run_case.py WORKER RUNDIR PORT TIMEOUT_S [n_tasks]"""
import json, os, subprocess, sys, time
W, RD, PORT, TO = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]); N = int(sys.argv[5]) if len(sys.argv) > 5 else 12
os.makedirs(RD, exist_ok=True)
for f in ('result.json', 'grades.jsonl', 'progress.json', 'FINALIZE'):
    try: os.remove(os.path.join(RD, f))
    except FileNotFoundError: pass
here = os.path.dirname(os.path.abspath(__file__))
plan = dict(checkpoint_id='stub', seed='s', ckpt_dir=RD, vllm_url='http://127.0.0.1:%d' % PORT, served_model='probe', source_root=here + '/src',
            concurrency=64, grade_workers=int(os.environ.get('GW', 4)), grade_timeout_s=6, harness=dict(enable_thinking=True, max_context_tokens=100000, temperature=1.0, top_p=1.0, top_k=-1),
            manifest_projection={}, envs=[dict(env_id='stubenv', spec=dict(id='stubenv', source_hash='abc'), kind='text', indices=list(range(N)), k=4, max_tokens=64)])
json.dump(plan, open(RD + '/plan.json', 'w'))
t0 = time.time()
p = subprocess.Popen([sys.executable, '-B', W, RD + '/plan.json', RD + '/result.json', '--progress', RD + '/progress.json'],
                     stdout=open(RD + '/worker.out', 'w'), stderr=open(RD + '/worker.err', 'w'), start_new_session=True, env=dict(os.environ, PYTHONPATH=here + '/src'))
try: rc = p.wait(TO)
except subprocess.TimeoutExpired:
    rc = 'HUNG'
    prog = {}
    try: prog = json.load(open(RD + '/progress.json'))
    except Exception: pass
    print(json.dumps(dict(worker=os.path.basename(W), rc=rc, seconds=round(time.time() - t0, 1), progress=dict(generated=prog.get('generated'), graded=prog.get('samples')))))
    os.killpg(p.pid, 9); sys.exit(0)
res = json.load(open(RD + '/result.json')); e = res['envs']['stubenv']
print(json.dumps(dict(worker=os.path.basename(W), rc=rc, seconds=round(time.time() - t0, 1), tasks=e['tasks'], samples=e['samples'], partial=res['partial'],
                      per_task={k: dict(n=v['n'], c=v['c'], w=v['w'], x=v['x']) for k, v in sorted(e['per_task'].items(), key=lambda kv: int(kv[0]))},
                      reasons=e['unresolved_reasons'], supervisor={k: v for k, v in (res.get('grade_supervisor') or {}).items() if k != 'events'})))
