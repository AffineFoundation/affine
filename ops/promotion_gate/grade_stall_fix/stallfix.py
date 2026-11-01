#!/usr/bin/env python3
"""SN120 grade-stall fix: installer + guard (RA/20261101-grade-stall-fix; Jacob: fixing the 2026-11-01 probe grade-pool stall is a
condition for the next gate switch).
  arm     : wait for a round boundary = right after a gate DECISION, with NO open held-out round (the last one closed, e.g. by a
            GREEN) -> take RA/LIFECYCLE-ACTIVATION.lock (non-blocking), verify the live files are the tested originals, back up,
            install new/probe_worker.py on the probe pod + new/gate_service.py here, restart the gate, then guard (below).
  guard   : auto-revert (restore the backups, close the round opened on the new code so no round changes mid-way, restart gate) if:
            gate inactive >= 60 s; >= 2 non-ssh gate error events / tracebacks after install; the first round opened after the
            install has no baseline 25 min after it opened; a fail-closed turn in it; an epoch TRAINING-RESULT +35 min without a
            DECISION. Exits OK once that round has its baseline AND has closed (full round watched), or after 8 h.
  revert  : the same revert by hand (WHY=...). status: print state.
Never touches the variance-levers watchdog, its state, the flag or weights; kills nothing."""
import fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
RA = Path('<VALIDATOR_STATE>/root-audits'); SF = RA / '20261101-grade-stall-fix'
GS = RA / '20261026-promotion-gate/state'; CODE = RA / '20261026-promotion-gate/code'; H = GS / 'heldout'
LE = Path('<VALIDATOR_REPO>/state/mimo-mainnet-v1/epochs')
PODD = '/root/affine-env-probe'; EVPY = '/root/affine-mimo-20261013/venv/bin/python'
PROBE_SSH = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
             '-o', 'UserKnownHostsFile=<PROBE_POD_KNOWN_HOSTS>', '-p', '<PROBE_POD_SSH_PORT>', 'root@<PROBE_POD_HOST>']   # [published copy: pod address redacted]
SHA = json.load(open(SF / 'SHA256.json')); ST = SF / 'state'; ST.mkdir(exist_ok=True)
os.environ['XDG_RUNTIME_DIR'] = '/run/user/%d' % os.getuid(); os.environ['TZ'] = 'America/Costa_Rica'; time.tzset()
def say(msg):
    line = '%s %s %s' % (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), time.strftime('%H:%M:%S CR'), msg)
    print(line, flush=True); open(ST / 'stallfix.log', 'a').write(line + '\n')
def rj(p):
    try: return json.load(open(p))
    except Exception: return None
def sha(b): return hashlib.sha256(b).hexdigest()
def pod(cmd, data=None, tries=3):
    for i in range(tries):
        p = subprocess.run(PROBE_SSH + ([] if data is not None else ['-n']) + [cmd], input=data, capture_output=True, timeout=180)
        if p.returncode == 0: return p.stdout
        say('pod ssh rc=%d (try %d): %s' % (p.returncode, i + 1, p.stderr.decode(errors='replace')[-160:].strip()))
        time.sleep(20)
    raise RuntimeError('probe pod unreachable')
def sysd(*a): return subprocess.run(['systemctl', '--user'] + list(a), capture_output=True, text=True).stdout.strip()
def open_rounds(): return sorted(p.name for p in H.glob('r2*') if (p / 'ROUND.private.json').exists() and not (p / 'CLOSED.json').exists())
FLAG = RA / '20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json'
def boundary_ok():
    """Round boundary: no open held-out round (the last one closed, e.g. by a GREEN), or the only open round has reached
    max_round_turns decided turns and every epoch sampled into it is decided (the gate closes it at the next sample draw and
    opens the next round, which is then the first round on the new code; the old round is never decided again)."""
    op = open_rounds()
    if not op: return 'no open round'
    if len(op) != 1: return None
    rid = op[0]; mt = int(((rj(FLAG) or {}).get('heldout') or {}).get('max_round_turns') or 8)
    try: turns = [json.loads(x) for x in open(H / rid / 'TURNS.private.jsonl')]
    except Exception: turns = []
    if sum(1 for t in turns if t.get('decision') in ('red', 'green')) < mt: return None
    eps = sorted((int(p.name) for p in (GS / 'epochs').iterdir() if p.name.isdigit()))[-20:]
    for e in eps:
        sm = rj(GS / 'epochs' / str(e) / 'SAMPLE.private.json') or {}
        if sm.get('round_id') == rid and not (GS / 'epochs' / str(e) / 'DECISION.json').exists(): return None
    return 'round %s reached %d decided turns, all its sampled epochs decided' % (rid, mt)
def last_decision():
    best = None
    for p in (GS / 'epochs').iterdir():
        q = p / 'DECISION.json'
        if p.name.isdigit() and q.exists() and (best is None or q.stat().st_mtime > best[1]): best = (int(p.name), q.stat().st_mtime)
    return best
def write(path, data, mode=0o644):
    t = Path(str(path) + '.tmp-sf'); t.write_bytes(data); os.chmod(t, mode); os.replace(t, path)

def install():
    lk = open(RA / 'LIFECYCLE-ACTIVATION.lock', 'w')
    try: fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: say('LIFECYCLE-ACTIVATION.lock held by another job: wait for the next boundary'); return False
    ts = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()); bk = SF / 'backups' / ('sf-' + ts); bk.mkdir(parents=True, mode=0o700)
    g = (CODE / 'gate_service.py').read_bytes()
    if sha(g) != SHA['orig_gate']: say('ABORT: live gate_service.py is not the tested original (%s)' % sha(g)[:16]); sys.exit(3)
    pw = pod('cd %s && cp -p probe_worker.py probe_worker.py.bak-pre-stallfix-%s && cat probe_worker.py.bak-pre-stallfix-%s' % (PODD, ts, ts))
    if sha(pw) != SHA['orig_pw']: say('ABORT: pod probe_worker.py is not the tested original (%s)' % sha(pw)[:16]); sys.exit(3)
    write(bk / 'gate_service.py', g, 0o600); write(bk / 'probe_worker.py', pw, 0o600)
    json.dump(dict(ts=ts, pod_backup='%s/probe_worker.py.bak-pre-stallfix-%s' % (PODD, ts)), open(bk / 'BACKUP.json', 'w'))
    say('backups: %s (gate_service.py, probe_worker.py) + pod %s/probe_worker.py.bak-pre-stallfix-%s' % (bk, PODD, ts))
    # short gate stop (< 90 s, the levers watchdog limit) so no round can open between the boundary check and the install
    sysd('stop', 'affine-promotion-gate'); t_inst = time.time(); why_b = boundary_ok()
    if not why_b:
        sysd('start', 'affine-promotion-gate'); say('boundary gone before the install (a round opened): window missed, gate restarted, wait'); return False
    try:
        out = pod('cd %s && cat > probe_worker.py.tmp-sf && %s -c "compile(open(\'probe_worker.py.tmp-sf\').read(),\'p\',\'exec\')" && mv -f probe_worker.py.tmp-sf probe_worker.py && sha256sum probe_worker.py' % (PODD, EVPY),
                  data=(SF / 'new' / 'probe_worker.py').read_bytes(), tries=1).decode()
    except Exception as x: out = 'ssh failed: %s' % x
    if SHA['new_pw'] not in out:
        sysd('start', 'affine-promotion-gate'); say('pod install check failed (%s): gate restarted on the old code, restoring pod file' % out[:120])
        revert_files(bk, 'pod install check failed'); sys.exit(4)
    write(CODE / 'gate_service.py', (SF / 'new' / 'gate_service.py').read_bytes(), 0o644)
    subprocess.run([sys.executable, '-m', 'py_compile', str(CODE / 'gate_service.py')], check=True)
    sysd('start', 'affine-promotion-gate'); time.sleep(6); a = sysd('is-active', 'affine-promotion-gate')
    ld = last_decision()
    inst = dict(installed_at=t_inst, installed_cr=time.strftime('%Y-%m-%d %H:%M:%S CR'), backup=str(bk), ts=ts, after_decision_epoch=ld[0],
                open_rounds=open_rounds(), boundary=why_b, gate=a)
    json.dump(inst, open(ST / 'INSTALLED.json', 'w'), indent=1)
    say('INSTALLED after DECISION e%d (%s): pod probe_worker.py %s, gate_service.py %s, gate %s' % (ld[0], why_b, SHA['new_pw'][:12], SHA['new_gate'][:12], a))
    open(SF / 'PROGRESS.md', 'a').write('- %s: installed after DECISION e%d, backup %s\n' % (time.strftime('%H:%M CR'), ld[0], bk))
    fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
    return True

def revert_files(bk, why):
    bk = Path(bk); ts = json.load(open(bk / 'BACKUP.json'))['ts']
    write(CODE / 'gate_service.py', (bk / 'gate_service.py').read_bytes(), 0o644)
    out = pod('cd %s && cp -p probe_worker.py.bak-pre-stallfix-%s probe_worker.py.tmp-sf && mv -f probe_worker.py.tmp-sf probe_worker.py && sha256sum probe_worker.py' % (PODD, ts)).decode()
    say('files restored from %s (pod sha ok: %s, gate sha ok: %s) [%s]' % (bk, SHA['orig_pw'] in out, sha((CODE / 'gate_service.py').read_bytes()) == SHA['orig_gate'], why))

def close_round(rid, why):
    p = H / rid / 'CLOSED.json'
    if not p.exists():
        t = p.with_name('.CLOSED.json.tmp'); t.write_text(json.dumps(dict(round_id=rid, closed_at=time.time(), closed_at_cr=time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)'),
            reason='grade-stall-fix revert: %s' % why, closing_epoch=None), indent=1, sort_keys=True) + '\n'); os.chmod(t, 0o644); os.replace(t, p)
    cur = rj(H / 'CURRENT.private.json') or {}
    if cur.get('id') == rid: (H / 'CURRENT.private.json').unlink(missing_ok=True)
    say('closed round %s opened on the new code (never revealed)' % rid)

def revert(why):
    inst = rj(ST / 'INSTALLED.json')
    if not inst: say('revert: nothing installed'); return
    say('REVERT: ' + why)
    sysd('stop', 'affine-promotion-gate')
    for rid in open_rounds():
        r = rj(H / rid / 'ROUND.private.json') or {}
        if float(r.get('opened_at') or 0) >= inst['installed_at']: close_round(rid, why)
    revert_files(inst['backup'], why)
    sysd('start', 'affine-promotion-gate'); time.sleep(5)
    say('REVERTED: gate %s' % sysd('is-active', 'affine-promotion-gate'))
    open(SF / 'PROGRESS.md', 'a').write('- %s: REVERT (%s)\n' % (time.strftime('%H:%M CR'), why))
    (ST / 'REVERTED').write_text(why + '\n')

BENIGN = ('Connection timed out', 'banner exchange', 'Connection closed', 'Connection reset', 'kex_exchange_identification')
def guard():
    inst = rj(ST / 'INSTALLED.json'); t0 = inst['installed_at']; end = t0 + 8 * 3600; inactive = 0; seen = set()
    while time.time() < end:
        if (ST / 'REVERTED').exists(): say('REVERTED marker: guard exit'); return
        now = time.time(); st = dict(at_cr=time.strftime('%Y-%m-%d %H:%M:%S CR'), installed_cr=inst['installed_cr'])
        inactive = 0 if sysd('is-active', 'affine-promotion-gate') == 'active' else inactive + 1
        if inactive >= 4: return revert('gate service inactive >= 60 s')
        errs = []; last_at = 0; ev = []
        try: lines = open(GS / 'gate.log', errors='replace').readlines()[-6000:]
        except Exception: lines = []
        for l in lines:
            try: d = json.loads(l); last_at = d.get('at', last_at)
            except Exception:
                if 'Traceback' in l and last_at >= t0: errs.append('traceback')
                continue
            if d.get('at', 0) < t0: continue
            e = str(d.get('event'))
            if 'error' in e and not any(b in str(d.get('error')) for b in BENIGN): errs.append('%s: %s' % (e, str(d.get('error'))[:160]))
            if e in ('heldout_baseline_stall_salvage', 'gate_run_stall_salvage', 'heldout_baseline_done', 'decision', 'heldout_fail_closed',
                     'heldout_baseline_partial_retry', 'gate_run_partial_retry', 'heldout_baseline_salvage', 'gate_run_salvage', 'heldout_baseline_stale'):
                ev.append(d)
                k = (e, d.get('at'))
                if k not in seen:
                    seen.add(k); say('event %s %s' % (e, json.dumps({x: d.get(x) for x in ('round', 'block', 'epoch', 'run', 'decision', 'z', 'graded', 'planned', 'info', 'total_s', 'got', 'need') if d.get(x) is not None})[:300]))
        st['errors'] = errs[-5:]
        if len(errs) >= 2: return revert('%d gate errors after install: %s' % (len(errs), errs[-1]))
        new = []
        for p in sorted(H.glob('r2*')):
            r = rj(p / 'ROUND.private.json') or {}
            if float(r.get('opened_at') or 0) < t0: continue
            b = rj(p / 'BASELINE.private.json'); closed = (p / 'CLOSED.json').exists()
            try: turns = [json.loads(x) for x in open(p / 'TURNS.private.jsonl')]
            except Exception: turns = []
            row = dict(id=r['id'], opened_cr=r.get('opened_at_cr'), baseline_s=((b or {}).get('timing') or {}).get('total_s'), closed=closed,
                       turns=[(t.get('epoch'), t.get('decision'), t.get('z'), t.get('fail_closed')) for t in turns])
            new.append(row)
            if not b and not closed and now - float(r['opened_at']) > 1500: return revert('round %s baseline not done %d min after open' % (r['id'], (now - float(r['opened_at'])) / 60))
            if any(t.get('fail_closed') for t in turns): return revert('fail-closed turn in round %s' % r['id'])
        st['new_rounds'] = new
        eps = sorted((int(p.name) for p in (GS / 'epochs').iterdir() if p.name.isdigit()))[-4:]
        for e in eps:
            tr = LE / str(e) / 'TRAINING-RESULT.json'
            if tr.exists() and tr.stat().st_mtime > t0 - 300 and not (GS / 'epochs' / str(e) / 'DECISION.json').exists() and now - tr.stat().st_mtime > 2100:
                return revert('epoch stall: e%d TRAINING-RESULT +%d min without DECISION' % (e, (now - tr.stat().st_mtime) / 60))
        t = ST / '.GUARD-STATUS.json.tmp'; t.write_text(json.dumps(st, indent=1)); os.replace(t, ST / 'GUARD-STATUS.json')
        if new and new[0]['baseline_s'] and new[0]['closed']:
            say('VERIFIED: first round on the new code %s: baseline %ss, turns %s, closed -> guard done' % (new[0]['id'], new[0]['baseline_s'], new[0]['turns']))
            (ST / 'VERIFIED').write_text(json.dumps(new[0]) + '\n'); return
        time.sleep(15)
    say('guard: 8 h elapsed, exit (no revert)')

if __name__ == '__main__':
    m = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if m == 'arm':
        if (ST / 'INSTALLED.json').exists(): say('already installed: guard only'); guard(); sys.exit(0)
        W = int(sys.argv[2]); say('armed: waiting for a round boundary after a DECISION of epoch >= %d (no open round, or a max-turns round fully decided)' % W)
        while True:
            ld = last_decision()
            if ld and ld[0] >= W and time.time() - ld[1] < 600 and boundary_ok():
                cur = rj(GS / 'GATE-BUSY.json') or {}
                if cur.get('who'): say('pod marked busy by %s: wait' % cur.get('who')); time.sleep(30); continue
                if install(): break
            time.sleep(3)
        guard()
    elif m == 'guard': guard()
    elif m == 'revert': revert(os.environ.get('WHY', 'manual'))
    else: print(json.dumps(dict(installed=rj(ST / 'INSTALLED.json'), guard=rj(ST / 'GUARD-STATUS.json'), reverted=(ST / 'REVERTED').exists(), verified=rj(ST / 'VERIFIED')), indent=1))
