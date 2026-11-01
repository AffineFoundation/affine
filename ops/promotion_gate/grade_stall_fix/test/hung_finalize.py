import json, os, subprocess, sys, time
W = sys.argv[1]; RD = '/tmp/podtest/runs/' + sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run(['rm', '-rf', RD]); os.makedirs(RD)
plan = dict(checkpoint_id='stub', seed='s', ckpt_dir=RD, vllm_url='http://127.0.0.1:18999', served_model='probe', source_root=here + '/src',
            concurrency=64, grade_workers=4, grade_timeout_s=6, harness=dict(enable_thinking=True, max_context_tokens=100000, temperature=1.0, top_p=1.0, top_k=-1),
            manifest_projection={}, envs=[dict(env_id='stubenv', spec=dict(id='stubenv', source_hash='abc'), kind='text', indices=list(range(12)), k=4, max_tokens=64)])
json.dump(plan, open(RD + '/plan.json', 'w'))
p = subprocess.Popen([sys.executable, '-B', W, RD + '/plan.json', RD + '/result.json', '--progress', RD + '/progress.json'], cwd=RD,
                     stdout=open(RD + '/worker.out', 'w'), stderr=open(RD + '/worker.err', 'w'), start_new_session=True, env=dict(os.environ, PYTHONPATH=here + '/src'))
open(RD + '/worker.pid', 'w').write(str(p.pid)); time.sleep(15)
print('status before:', json.dumps({k: v for k, v in json.loads(subprocess.run([sys.executable, '/tmp/podtest/podctl.py', 'status'], capture_output=True, text=True).stdout)['runs'][sys.argv[2]].items() if k != 'err_tail'}))
t0 = time.time(); out = subprocess.run([sys.executable, '/tmp/podtest/podctl.py', 'finalize-run', sys.argv[2], '30'], capture_output=True, text=True)
time.sleep(2)
print('finalize-run (%.1fs): %s' % (time.time() - t0, out.stdout.strip()), '| worker alive after:', p.poll() is None)
r = json.load(open(RD + '/result.json')); print('result tasks', r['envs']['stubenv']['tasks'], 'partial', r['partial'], r['stop_reason'])
if p.poll() is None: os.killpg(p.pid, 9)
