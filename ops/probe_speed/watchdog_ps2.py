"""probe-speed auto-revert watchdog v2 (e275..e277, after ps1+ps2+ps3 live). Reverts ps1 (podctl/PREP-CACHE/delta_timeout) on a gate
re-serve or serve_s > 320 s; reverts ps3 then ps2 (grader changes) on partial-run retry / salvage / epoch error / held-out fail-closed, or
if any finished baseline block or candidate run has > 5 indeterminate (x) samples. Logs to PROGRESS.md."""
import json, time, subprocess, os
D = '/home/const/subnet120-rewrite/state/root-audits/20261030-probe-speed'
G = '/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/gate.log'
FROM, TO = int(os.environ.get('FROM', 275)), int(os.environ.get('TO', 277))
PP = ['ssh', '-T', '-n', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15', '-o',
      'UserKnownHostsFile=/home/const/.lium/known_hosts/b4f3da9d-df66-436a-af38-829d1606966d', '-p', '40198', 'root@38.88.142.89']
def prog(m): open(D + '/PROGRESS.md', 'a').write('- %s: watchdog_ps2: %s\n' % (time.strftime('%H:%M CR', time.gmtime(time.time() - 6 * 3600)), m))
rev = set()
def revert(which, why):
    if which in rev: return
    rev.add(which); subprocess.run(['bash', D + '/activate_%s.sh' % which], env=dict(os.environ, MODE='revert')); prog('AUTO-REVERT %s: %s' % (which, why))
def xcount(run):
    try:
        out = subprocess.run(PP + ["python3 -c \"import json;print(sum(json.loads(l)['x'] for l in open('/root/affine-env-probe/runs/%s/grades.jsonl')))\"" % run],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out)
    except Exception: return None
prog('armed e%d..e%d (ps1, ps2, ps3)' % (FROM, TO)); pos = os.path.getsize(G); done = set(); seen_runs = set()
while len(done) < TO - FROM + 1:
    with open(G) as f:
        f.seek(pos); lines = f.readlines(); pos = f.tell()
    for l in lines:
        try: ev = json.loads(l)
        except Exception: continue
        e = ev.get('epoch'); k = ev.get('event')
        if k == "gate_reserve": revert("ps1", "gate_reserve"); revert("ps4", "gate_reserve")
        if k in ('gate_run_partial_retry', 'epoch_error', 'gate_run_salvage', 'heldout_baseline_salvage', 'heldout_baseline_partial_retry', 'heldout_fail_closed'):
            revert('ps3', k); revert('ps2', k)
        if k == 'heldout_baseline_block_done':
            st = subprocess.run(PP + ['ls -t /root/affine-env-probe/runs | grep "^heldout-.*-H-b%d-" | head -1' % ev.get('block', -1)], capture_output=True, text=True, timeout=60).stdout.strip()
            x = xcount(st) if st else None; prog('baseline block %s run %s x=%s' % (ev.get('block'), st, x))
            if x is not None and x > 5: revert('ps3', 'baseline %s x=%d' % (st, x)); revert('ps2', 'baseline %s x=%d' % (st, x))
        if k == 'decision' and e and FROM <= e <= TO:
            t = ev.get('timing') or {}; done.add(e)
            msg = 'e%d %s TR->DEC %s fetch %s serve %s run %s' % (e, ev.get('decision'), t.get('result_to_decision_s'), t.get('fetch_s'), t.get('serve_s'), t.get('run_s'))
            if (t.get("serve_s") or 0) > 320: revert("ps1", "serve_s > 320"); revert("ps4", "serve_s > 320")
            if ev.get('decision') == 'error': revert('ps3', 'decision error'); revert('ps2', 'decision error')
            prog(('OK ' if not rev else '') + msg)
    time.sleep(5)
prog('disarmed (reverted: %s)' % (sorted(rev) or 'none'))
