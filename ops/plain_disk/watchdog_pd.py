"""plain-disk auto-revert watchdog (e276..e278). Reverts (MODE=revert activate_pd.sh <pod> <item>):
- probe grader/worker paths pd4, pd3 on partial-run retry / salvage / epoch_error / fail-closed / decision error, or > 5 indeterminate (x)
  in any finished baseline block or candidate run;
- probe serving paths pd2v, pd2b, pd1 on gate_reserve, serve_s > 320, decision error, or EngineDead/FileNotFoundError in a serve log after arming
  (pd1 ckpt revert waits until no gate_delta/gate_fetch is running on the pod);
- trainer pt1 on trainer_restore_done ok=false or epoch_error; eval pe1+pe2 if the evaluator distributed_worker disappears.
Logs to PROGRESS.md."""
import json, time, subprocess, os
D = '/home/const/subnet120-rewrite/state/root-audits/20261030-plain-disk'
G = '/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/gate.log'
FROM, TO = int(os.environ.get('FROM', 276)), int(os.environ.get('TO', 278))
def pod(p, cmd, t=60):
    try: return subprocess.run([D + '/pod.sh', p, cmd], capture_output=True, text=True, timeout=t).stdout.strip()
    except Exception: return ''
def prog(m): open(D + '/PROGRESS.md', 'a').write('- %s: watchdog_pd: %s\n' % (time.strftime('%H:%M:%S CR', time.gmtime(time.time() - 6 * 3600)), m))
rev = set()
def revert(pdn, item, why):
    if (pdn, item) in rev: return
    rev.add((pdn, item))
    if item == 'pd1':
        for _ in range(180):
            if not pod('probe', "pgrep -f 'gate_(delta|fetch)\\.py' >/dev/null && echo busy"): break
            time.sleep(5)
    r = subprocess.run(['bash', D + '/activate_pd.sh', pdn, item], env=dict(os.environ, MODE='revert'), capture_output=True, text=True)
    prog('AUTO-REVERT %s %s: %s (%s)' % (pdn, item, why, r.stdout.strip().replace('\n', ' ')[:300]))
GRADE = [('probe', 'pd4'), ('probe', 'pd3')]; SERVE = [('probe', 'pd2v'), ('probe', 'pd2b'), ('probe', 'pd1')]
def revert_all(lst, why):
    for p, i in lst: revert(p, i, why)
def xcount(run):
    out = pod('probe', "python3 -c \"import json;print(sum(json.loads(l)['x'] for l in open('/root/affine-env-probe/runs/%s/grades.jsonl')))\"" % run)
    try: return int(out)
    except Exception: return None
LOGMARK = pod('probe', 'for i in 0 1 2 3 4 5 6 7; do stat -c %s /root/affine-env-probe/serve.$i.log; done').split()
def serve_errors():
    if len(LOGMARK) != 8: return ''
    return pod('probe', ' ; '.join("tail -c +%d /root/affine-env-probe/serve.%d.log | grep -m1 -E 'EngineDeadError|FileNotFoundError'" % (int(LOGMARK[i]) + 1, i) for i in range(8)))
EVPID = pod('eval', 'pgrep -f subnet.distributed_worker | head -1')
prog('armed e%d..e%d (probe pd1 pd2v pd2b pd3 pd4, trainer pt1, eval pe1 pe2; eval worker pid %s)' % (FROM, TO, EVPID))
pos = os.path.getsize(G); done = set(); last_slow = 0
while len(done) < TO - FROM + 1:
    with open(G) as f:
        f.seek(pos); lines = f.readlines(); pos = f.tell()
    for l in lines:
        try: ev = json.loads(l)
        except Exception: continue
        e = ev.get('epoch'); k = ev.get('event')
        if k == 'gate_reserve': revert_all(SERVE, k)
        if k in ('gate_run_partial_retry', 'epoch_error', 'gate_run_salvage', 'heldout_baseline_salvage', 'heldout_baseline_partial_retry', 'heldout_fail_closed'):
            revert_all(GRADE, k)
        if k == 'epoch_error' or (k == 'trainer_restore_done' and not ev.get('ok')): revert('trainer', 'pt1', k)
        if k in ('heldout_baseline_block_done', 'gate_run_done'):
            st = pod('probe', 'ls -t /root/affine-env-probe/runs | head -1')
            x = xcount(st) if st else None; prog('%s run %s x=%s' % (k, st, x))
            if x is not None and x > 5: revert_all(GRADE, 'run %s x=%d' % (st, x))
        if k == 'decision' and e and FROM <= e <= TO:
            t = ev.get('timing') or {}; done.add(e)
            if (t.get('serve_s') or 0) > 320: revert_all(SERVE, 'serve_s > 320')
            if ev.get('decision') == 'error': revert_all(GRADE + SERVE, 'decision error')
            prog(('OK ' if not rev else '') + 'e%d %s TR->DEC %s fetch %s serve %s run %s' % (e, ev.get('decision'), t.get('result_to_decision_s'), t.get('fetch_s'), t.get('serve_s'), t.get('run_s')))
    if time.time() - last_slow > 60:
        last_slow = time.time()
        se = serve_errors()
        if se: revert_all(SERVE, 'serve log: ' + se[:200])
        if EVPID and not pod('eval', 'kill -0 %s 2>/dev/null && echo up' % EVPID):
            revert('eval', 'pe1', 'evaluator worker %s gone' % EVPID); revert('eval', 'pe2', 'evaluator worker gone'); EVPID = ''
    time.sleep(5)
prog('disarmed (reverted: %s)' % (sorted(rev) or 'none'))
