#!/usr/bin/env python3
"""caps-apply: render every consumer of caps.json and roll the result out (the ONE command after editing caps.json).

  python3 tools/caps/caps_apply.py                     # plan: what would change (host files + live pods), nothing written
  python3 tools/caps/caps_apply.py --scope repo --write   # in a checkout of AffineFoundation/affine: rewrite the repo copies
  python3 tools/caps/caps_apply.py --apply             # on the validator host: wait for an idle epoch boundary, take the
                                                       # lifecycle locks, back up, write, push pods, restart, verify, REVERT.sh
Exit 0 = done / nothing to do; 1 = refused (see message); 2 = verification failed after apply (REVERT.sh printed).

--apply, in order:
  1. render every host consumer (consumers.py, render=True/'json') and every pod file (render='pod'); stop if the
     root-signed trainer constants would change (render='sign'): that needs a signed trainer release first (--stage-trainer
     writes the patched files to state/caps/stage/trainer/ for the signing step) - nothing is written in that case;
  2. wait (up to --wait s) for an idle decision gap: lifecycle active, no open training job, every finished turn decided,
     no gate restore running, no running train job in roles.sqlite3 (same rule as RA/20261102-12k-everywhere activate_12k.sh);
  3. flock state/mimo-mainnet-v1/lifecycle.lock + RA/LIFECYCLE-ACTIVATION.lock, stop affine-mimo-lifecycle + mining-discovery;
  4. back up every file it writes to state/caps/apply-<ts>/backup/ (+ the pod files), write, re-pin FF-PINS if ff_format moved,
     set the gate flag max_tokens (under its flock; rounds already open keep their cap), push the prior pod files and restart
     its sampler on a fresh pass;
  5. restart affine-ff / affine-mining-discovery / affine-training-params / affine-k1-transparency, start the lifecycle, release
     the locks; run caps_check --scope host,pods; write state/caps/ACTIVE.json and the REVERT.sh of this apply.
"""
import argparse, difflib, fcntl, hashlib, json, os, shutil, sqlite3, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capslib as L
import consumers as CS

STATE = Path(CS.REPO) / 'state/mimo-mainnet-v1'
GE = Path(CS.RA) / '20261026-promotion-gate/state'
CAPS_STATE = Path(CS.REPO) / 'state/caps'
UNITS_AFTER = ['affine-ff', 'affine-mining-discovery', 'affine-training-params', 'affine-k1-transparency']


def cr(t=None): return time.strftime('%Y-%m-%d %H:%M:%S CR', time.gmtime((t or time.time()) - 21600))
def say(*a): print(*a, flush=True)
def sh(*cmd, check=True): return subprocess.run(cmd, capture_output=True, text=True, check=check).stdout


def host_targets(C):
    """-> [(consumer, path, old, new)] for every renderable host file."""
    out = []
    for con in CS.CONSUMERS:
        if con.get('render') not in (True, 'json', 'sign') or not con.get('host'): continue
        for h in ([con['host']] if isinstance(con['host'], str) else con['host']):
            old = Path(h).read_text(); out.append((con, h, old, L.render_text(con, old, C)))
    return out


def repo_targets(C, root):
    out = []
    for con in CS.CONSUMERS:
        if con.get('render') not in (True, 'json') or not con.get('repo'): continue   # 'sign': only via a signed trainer release
        p = root / con['repo']; old = p.read_text(); out.append((con, str(p), old, L.render_text(con, old, C)))
    return out


def pod_targets(C):
    import caps_check as CC
    out = []
    for con in CS.CONSUMERS:
        if con.get('render') != 'pod': continue
        role, cmd = con['pod']; path = cmd.split()[-1]
        old = CC.pod_run(role, cmd); out.append((con, '%s:%s' % (role, path), old, L.render_text(con, old, C)))
    return out


def show(targets):
    n = 0
    for con, where, old, new in targets:
        if old == new: continue
        n += 1; say('CHANGE %s (%s, %s)' % (where, con['id'], con['cls']))
        for l in list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm='', n=0))[2:40]: say('   ' + l[:220])
    return n


# ---------------- epoch-boundary gate (same rule as activate_12k.sh idle()) ----------------
def idle():
    if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'affine-mimo-lifecycle.service']).returncode: return 'lifecycle not active'
    c = json.loads((STATE / 'CURRENT.json').read_text())['epoch_index']; E = STATE / 'epochs'
    for x in (c, c + 1, c + 2):
        if (E / str(x) / 'RECOVERY-REQUIRED.json').exists(): return 'recovery required e%d' % x
        if (E / str(x) / 'TRAINING-JOB.json').exists() and not (E / str(x) / 'TRAINING-RESULT.json').exists(): return 'open training job e%d' % x
    for x in (c - 1, c, c + 1, c + 2):
        d = GE / 'epochs' / str(x) / 'DECISION.json'
        if (E / str(x) / 'TRAINING-RESULT.json').exists() and not d.exists(): return 'e%d trained, gate undecided' % x
        if d.exists() and json.loads(d.read_text()).get('decision') == 'red' and not (GE / 'epochs' / str(x) / 'TRAINER-RESTORED.json').exists(): return 'e%d red, trainer not restored yet' % x
    if 'activ' in sh('systemctl', '--user', 'list-units', '--no-legend', 'affine-gate-restore-*', check=False): return 'gate restore running'
    try:
        db = sqlite3.connect('file:%s?mode=ro' % (STATE / 'roles.sqlite3'), uri=True, timeout=5)
        if db.execute("select count(*) from jobs where role='train' and status not in ('complete','failed','expired')").fetchone()[0]: return 'train job running'
    except Exception as e: return 'roles.sqlite3: %s' % e
    return None


class Locks:
    def __enter__(s):
        s.f9 = open(Path(CS.RA) / 'LIFECYCLE-ACTIVATION.lock', 'w'); s.f8 = open(STATE / 'lifecycle.lock', 'a+')
        for f in (s.f9, s.f8):
            t0 = time.time()
            while True:
                try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError:
                    if time.time() - t0 > 60: raise RuntimeError('lock busy: %s' % f.name)
                    time.sleep(1)
        return s
    def __exit__(s, *a):
        for f in (s.f8, s.f9): fcntl.flock(f, fcntl.LOCK_UN); f.close()


def backup(bdir, path):
    dst = bdir / ('host' + path); dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(path).exists(): shutil.copy2(path, dst)
    return dst


def atomic_write(path, text):
    p = Path(path); t = p.with_name('.%s.caps-new' % p.name); t.write_text(text)
    shutil.copymode(path, t) if p.exists() else None; os.replace(t, p)


def repin_ff(bdir, rev):
    pins_p = CS.FF + '/FF-PINS.json'; old = Path(pins_p).read_text(); pins = json.loads(old)
    pins['files']['ff_format.py'] = hashlib.sha256(Path(CS.FF + '/ff_format.py').read_bytes()).hexdigest()
    new = L._json_dump_like(old, pins)
    if new != old: rev.append(str(backup(bdir, pins_p))), atomic_write(pins_p, new)


def prior_rollout(pods_changed, bdir, log):
    """write the prior pod files, re-serve the current checkpoint, restart the sampler (exact PID) on a fresh pass."""
    import caps_check as CC
    roles = json.loads((Path(CS.REPO) / 'ops/pods/roles.json').read_text())['prior']['provision']
    serve = next(s['cmd'] for s in roles if s.get('type') == 'pod_cmd' and 'serve.sh {head}' in s.get('cmd', ''))
    worker = next(s['cmd'] for s in roles if s.get('type') == 'pod_cmd' and 'priors/worker.py' in s.get('cmd', ''))
    for con, where, old, new in pods_changed:
        path = where.split(':', 1)[1]; (bdir / 'pod-prior').mkdir(parents=True, exist_ok=True)
        (bdir / 'pod-prior' / Path(path).name).write_text(old)
        CC._PODS['prior'].run('cat > %s.caps-new && mv %s.caps-new %s' % (path, path, path), inp=new.encode(), timeout=60)
        log('prior pod %s written' % path)
    pid = CC.pod_run('prior', 'cat /opt/prior/runs/sampler/worker.pid').strip()
    if pid.isdigit():
        CC.pod_run('prior', 'touch /opt/prior/runs/sampler/STOP; kill -TERM %s 2>/dev/null; for i in $(seq 60); do kill -0 %s 2>/dev/null || break; sleep 2; done; '
                            'kill -0 %s 2>/dev/null && kill -KILL %s; rm -f /opt/prior/runs/sampler/STOP; true' % (pid, pid, pid, pid))
        log('prior worker %s stopped (exact pid)' % pid)
    CC.pod_run('prior', "python3 - <<'PY'\nimport json, os, time\np='/opt/prior/runs/sampler/PASS.json'; ps=json.load(open(p))\n"
                        "json.dump(dict(ps, finished=time.time(), ended='caps_apply: cap change'), open('/opt/prior/runs/sampler/pass-%06d.json' % ps['pass'], 'w'))\n"
                        "json.dump(dict(**{'pass': ps['pass'] + 1}, head=ps['head'], nonce=os.urandom(16).hex(), started=time.time(), note='caps_apply'), open(p, 'w'))\nPY")
    head = json.loads(CC.pod_run('prior', 'cat /opt/prior/serve/SERVING.json'))['ckpt']
    log('prior re-serve: %s' % CC._PODS['prior'].run(serve.replace('{head}', head), timeout=2400).strip()[-200:])
    log('prior worker: %s' % CC.pod_run('prior', worker).strip()[-120:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scope', default='host'); ap.add_argument('--root', default=str(L.ROOT)); ap.add_argument('--write', action='store_true')
    ap.add_argument('--apply', action='store_true'); ap.add_argument('--wait', type=int, default=12 * 3600)
    ap.add_argument('--locks-held', action='store_true', help='called by an activation script that already holds both locks')
    ap.add_argument('--stage-trainer', action='store_true'); ap.add_argument('--no-pods', action='store_true')
    a = ap.parse_args()
    C, _ = L.load_caps()
    if a.scope == 'repo':
        t = repo_targets(C, Path(a.root)); n = show(t)
        if a.write:
            for con, p, old, new in t:
                if old != new: Path(p).write_text(new)
            say('repo: %d file(s) rewritten from caps.json (max_output %d); run tools/caps/caps_check.py --scope repo' % (n, C['max_output_tokens']))
        else: say('repo: %d file(s) differ from caps.json (plan only; --write to rewrite)' % n)
        return
    ht = host_targets(C); pt = [] if a.no_pods else pod_targets(C)
    sign = [x for x in ht if x[0].get('render') == 'sign' and x[2] != x[3]]
    if a.stage_trainer:
        d = CAPS_STATE / 'stage/trainer'; d.mkdir(parents=True, exist_ok=True)
        for con, p, old, new in sign: (d / Path(p).name).write_text(new)
        say('staged %d trainer file(s) in %s for the signed-release step' % (len(sign), d)); return
    n = show([x for x in ht if x[0].get('render') != 'sign']) + show(pt)
    if sign:
        show(sign)
        say('REFUSED: the root-signed trainer constants differ from caps.json (%s). Build + sign a trainer release first '
            '(--stage-trainer writes the patched files), switch it, then re-run --apply.' % ', '.join(x[0]['id'] for x in sign)); sys.exit(1)
    if not a.apply:
        say('plan: %d host/pod file(s) would change (nothing written; --apply rolls out at an epoch boundary)' % n); return
    ts = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()); D = CAPS_STATE / ('apply-' + ts); bdir = D / 'backup'; bdir.mkdir(parents=True)
    LOGF = D / 'apply.log'
    def log(m): line = '%s %s' % (cr(), m); say(line); LOGF.open('a').write(line + '\n')
    prev = json.loads((CAPS_STATE / 'ACTIVE.json').read_text()) if (CAPS_STATE / 'ACTIVE.json').exists() else {}
    old_cap = int(prev.get('max_output_tokens') or json.loads(Path(CS.RA + '/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json').read_text())['heldout']['prior_v1']['max_tokens'])
    log('apply: max_output %d (was %d), %d file(s) to change' % (C['max_output_tokens'], old_cap, n))
    rev = []; changed_units = set()
    def body():
        for con, p, old, new in ht:
            if old == new: continue
            backup(bdir, p); rev.append(p)
            if con.get('lock'):
                with open(con['lock'], 'a+') as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX); atomic_write(p, new); fcntl.flock(lf, fcntl.LOCK_UN)
            else: atomic_write(p, new)
            log('wrote %s (%s)' % (p, con['id']))
            if con['id'] == 'ff_format': repin_ff(bdir, rev)
        pc = [x for x in pt if x[2] != x[3]]
        if pc: prior_rollout(pc, bdir, log)
    if a.locks_held:
        body()
    else:
        t0 = time.time()
        while True:
            why = idle()
            if why is None: break
            if time.time() - t0 > a.wait: log('REFUSED: no idle epoch boundary within %ds (%s)' % (a.wait, why)); sys.exit(1)
            time.sleep(5)
        with Locks():
            if idle() is not None: log('REFUSED: boundary closed while locking'); sys.exit(1)
            sh('systemctl', '--user', 'stop', 'affine-mimo-lifecycle.service', 'affine-mining-discovery.service')
            log('idle boundary at e%s: lifecycle + mining-discovery stopped, locks held' % json.loads((STATE / 'CURRENT.json').read_text())['epoch_index'])
            try: body()
            finally:
                for u in UNITS_AFTER: subprocess.run(['systemctl', '--user', 'restart', u + '.service'])
                sh('systemctl', '--user', 'start', 'affine-mimo-lifecycle.service'); log('units restarted, lifecycle started, locks released')
    revert = ['#!/bin/bash', '# caps_apply %s revert: restore every file this apply wrote (pod files: copy back by hand from backup/pod-prior)' % ts, 'set -e']
    revert += ['install -m $(stat -c %%a %s) %s %s' % (p, bdir / ('host' + p), p) for p in rev]
    revert += ['systemctl --user restart %s' % ' '.join(u + '.service' for u in UNITS_AFTER)]
    (D / 'REVERT.sh').write_text('\n'.join(revert) + '\n'); os.chmod(D / 'REVERT.sh', 0o700)
    v = subprocess.run([sys.executable, str(Path(__file__).with_name('caps_check.py')), '--scope', 'host,pods', '--quiet', '--json', str(D / 'check.json')],
                       capture_output=True, text=True)
    log('caps_check rc=%d: %s' % (v.returncode, v.stdout.strip().splitlines()[-1] if v.stdout.strip() else v.stderr[-300:]))
    (CAPS_STATE / 'ACTIVE.json').write_text(json.dumps(dict(version='sn120-caps-active-v1', max_output_tokens=C['max_output_tokens'],
                                                            max_context_tokens=C['max_context_tokens'], applied_at=time.time(), applied_cr=cr(),
                                                            apply_dir=str(D), check_rc=v.returncode), indent=1) + '\n')
    say('REVERT: bash %s' % (D / 'REVERT.sh'))
    sys.exit(0 if v.returncode == 0 else 2)


if __name__ == '__main__':
    main()
