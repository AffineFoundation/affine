"""Offline unit test of the gate no-progress salvage (stall_salvage + run_candidate branch + baseline branch wiring)."""
import importlib.util, json, os, sys, tempfile, types
D = sys.argv[1]; os.environ['HG_GATE_STATE'] = tempfile.mkdtemp(); os.environ['HG_SITE'] = tempfile.mkdtemp()
sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location('gs', os.path.join(D, 'gate_service.py')); gs = importlib.util.module_from_spec(spec); spec.loader.exec_module(gs)
clock = [1000.0]; gs.time = types.SimpleNamespace(time=lambda: clock[0], strftime=__import__('time').strftime, gmtime=__import__('time').gmtime, sleep=lambda s: None)
calls = []; logs = []
gs.log = lambda **kw: logs.append(kw)
gs.opts = lambda: dict(stall_salvage_s=300.0)
state = dict(run=dict(running=True, progress=dict(generated=8000, samples=7996)))
def podctl(*a, **k):
    calls.append(a)
    if a[0] == 'status': return dict(runs={NAME: state['run']})
    if a[0] == 'finalize-run':
        state['run'] = dict(running=False, result=True); return dict(finalized=True, partial=True, samples=7996)
    raise AssertionError(a)
gs.podctl = podctl
gs.pod = lambda cmd, **k: json.dumps(dict(partial=True, wall_seconds=1, envs=dict(affine_math=dict(per_task={str(i): dict(n=4, c=1) for i in range(1999)}))))
ok = True
def check(name, cond):
    global ok; print(('PASS ' if cond else 'FAIL ') + name); ok &= bool(cond)
# 1. stall_salvage: graded count moving -> never fires; gen < planned -> never fires; unchanged >= 300 s -> fires once
slot = {}
for t, g in ((0, 7000), (200, 7500), (400, 7990), (600, 7996)):
    clock[0] = 1000 + t; gs.stall_salvage(slot, 'x', dict(progress=dict(generated=8000, samples=g)), 8000, 'ev')
check('no salvage while graded count moves', not [c for c in calls if c[0] == 'finalize-run'])
clock[0] = 1000 + 600 + 299; gs.stall_salvage(slot, 'x', dict(progress=dict(generated=8000, samples=7996)), 8000, 'ev')
check('no salvage before 300 s unchanged', not [c for c in calls if c[0] == 'finalize-run'])
slot2 = {}
for t in (0, 400, 800): clock[0] = 1000 + t; gs.stall_salvage(slot2, 'y', dict(progress=dict(generated=7000, samples=7000)), 8000, 'ev')
check('no salvage while samples still generating', not [c for c in calls if c[0] == 'finalize-run'])
clock[0] = 1000 + 600 + 301; state['run'] = dict(running=True, progress=dict(generated=8000, samples=7996))
fin = gs.stall_salvage(slot, 'x', dict(progress=dict(generated=8000, samples=7996)), 8000, 'heldout_baseline_stall_salvage', round='r', block=0)
check('salvage fires after 300 s unchanged with all generated', fin and fin.get('finalized') and calls[-1] == ('finalize-run', 'x', '30'))
check('salvage logged (event has no "error")', logs[-1]['event'] == 'heldout_baseline_stall_salvage' and logs[-1]['graded'] == 7996 and 'error' not in logs[-1]['event'])
n = len(calls); gs.stall_salvage(slot, 'x', dict(progress=dict(generated=8000, samples=7996)), 8000, 'ev')
check('no immediate repeat finalize', len(calls) == n)
# 2. run_candidate: hung candidate run -> finalize-run after 300 s -> partial result returned through the normal 0.9 coverage path
NAME = None; calls.clear(); state['run'] = dict(running=True, progress=dict(generated=2000 * 4, samples=1999 * 4))
tl = dict(run_start=0.0); clock[0] = 5000.0
kp = dict(k=4, seeded=True); tasks = list(range(2000)); f = dict(env='affine_math', gmu='0.9')
import hashlib
pseed = 'ps'; NAME = 'gate-e%06d-%s' % (7, 'c' * 12) + '-%s-k%d%s' % ('C', 4, 's') + '-ho' + hashlib.sha256(('%s:%s' % (pseed, ','.join(map(str, sorted(tasks))))).encode()).hexdigest()[:8]
r1 = gs.run_candidate(7, 'c' * 64, tasks, {}, f, tl, kp=kp, pseed=pseed)
clock[0] += 301; r2 = gs.run_candidate(7, 'c' * 64, tasks, {}, f, tl, kp=kp, pseed=pseed)
check('candidate: finalize-run called on the stalled run', ('finalize-run', NAME, '30') in calls and r1 is None and r2 is None)
clock[0] += 15; r3 = gs.run_candidate(7, 'c' * 64, tasks, {}, f, tl, kp=kp, pseed=pseed)
check('candidate: partial result (1999/2000 >= 0.9) returned', isinstance(r3, dict) and len(r3['envs']['affine_math']['per_task']) == 1999)
# 3. salvage switched off (stall_salvage_s 0)
gs.opts = lambda: dict(stall_salvage_s=0.0); calls.clear(); s3 = {}
for t in (0, 1000): clock[0] = t; gs.stall_salvage(s3, 'z', dict(progress=dict(generated=8, samples=4)), 8, 'ev')
check('stall_salvage_s 0 disables it', not calls)
# 4. baseline wiring present in ensure_round_baseline
import inspect
check('baseline running branch calls stall_salvage', "stall_salvage(bs, name, ru, len(tasks) * int(r['k_block']), 'heldout_baseline_stall_salvage'" in inspect.getsource(gs.ensure_round_baseline))
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
