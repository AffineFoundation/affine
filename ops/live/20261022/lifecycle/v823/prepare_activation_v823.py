#!/usr/bin/env python3
"""Prepare (do not install) v8.2.3 activation files: lifecycle drop-in, mining-observer config + drop-in.
Then dry-run the observer selector against a simulated v8.2.3 lifecycle process argv (no unit/proc of production touched)."""
import hashlib, json, os, sys, tempfile
from pathlib import Path
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); D = RA/'20261016-v823'; SG = D/'release-v823-signed'; A = D/'activation-v823'
S2 = RA/'20261015-mimo-v8.2-release-succession/release-v822-signed'; SRC = S2/'deployment-final-source-v822'
OBS2 = RA/'20261016-mining-discovery-rebind-v822/observer.v822.private.json'
UID_DIR = RA/'20261013-null-pow-v1/uid1000-host-prospective'
PY = '/home/const/subnet120-rewrite/.venv/bin/python'
hf = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok, why):
    if not ok: raise SystemExit('REFUSED: ' + why)
def put(p, raw, mode=0o600):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
r = json.loads((SG/'SIGNING-RESULT.json').read_text()); need(r['checks_passed'] is True, 'signing checks')
lp, cp, up = SG/'lifecycle.v8.2.3.ACTIVE.private.json', SG/'capture-operator.v8.2.3.ROOT-ADMITTED.private.json', SG/'uid-operator.v8.2.3.ROOT-ADMITTED.private.json'
A.mkdir(mode=0o700)
lc = ("# MiMo v8.2.3 (lifecycle-config succession: overlong_soft_start 7000->6000; policy dfe0cbf7 unchanged).\n"
      "# Installed by activate_v823.sh; removed by rollback_v823.sh. Overrides zzzz~~~~~~~v822s-release.conf (left in place).\n"
      "[Service]\nExecStart=\nExecStart=%s -B %s --operator-config %s\n" % (PY, UID_DIR/'mimo_uid_capacity_launcher.py', up))
put(A/'arbos/affine-mimo-lifecycle.service.d/zzzz~~~~~~~~v823-release.conf', lc.encode())
obs = json.loads(OBS2.read_text())
need(obs['lifecycle_config']['sha256'] == '943ed0779d4108076c6953ccf682aac21098112ceb84dce803759f7f9747dbfb', 'observer v822 baseline')
obs['lifecycle_config'] = dict(path=str(lp), sha256=hf(lp)); obs['host_capture_config'] = dict(path=str(cp), sha256=hf(cp))
obs['host_uid_config'] = dict(path=str(up), sha256=hf(up))
put(D/'observer.v823.private.json', (json.dumps(obs, indent=1, sort_keys=True) + '\n').encode())
oc = ("# v8.2.3: rebind public mining observer to lifecycle.v8.2.3 (see %s/AUDIT.md). Overrides zzzz~~mimo-observer-v822.conf.\n"
      "[Service]\nExecStart=\nExecStart=%s -B %s --config %s --watch\nWorkingDirectory=%s\nEnvironment=PYTHONPATH=%s\nEnvironment=PYTHONDONTWRITEBYTECODE=1\n"
      % (D, PY, UID_DIR/'publish_mimo_public_discovery.py', D/'observer.v823.private.json', SRC, SRC))
put(A/'arbos/affine-mining-discovery.service.d/zzzz~~~mimo-observer-v823.conf', oc.encode())
# --- observer dry run against a simulated v8.2.3 process (fake /proc; inspect_unit stub) ---
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(UID_DIR)); os.chdir(SRC)
import publish_mimo_public_discovery as P
fake = Path(tempfile.mkdtemp(prefix='v823-proc-')); (fake/'4242').mkdir()
argv = [PY, '-B', str(UID_DIR/'mimo_uid_capacity_launcher.py'), '--operator-config', str(up)]
(fake/'4242'/'cmdline').write_bytes(b'\0'.join(a.encode() for a in argv) + b'\0')
(fake/'4242'/'stat').write_text(Path('/proc/self/stat').read_text())
res = {}
try:
    sel = P.selector(obs, proc=fake, inspect_unit=lambda: 'MainPID=4242\nActiveState=active')
    res['v823_observer_with_v823_process'] = 'accepted'
except Exception as e:
    res['v823_observer_with_v823_process'] = 'REFUSED %s: %s' % (type(e).__name__, e)
obs2 = json.loads(OBS2.read_text())
try:
    P.selector(obs2, proc=fake, inspect_unit=lambda: 'MainPID=4242\nActiveState=active'); res['v822_observer_with_v823_process'] = 'accepted (unexpected)'
except Exception as e:
    res['v822_observer_with_v823_process'] = 'refused as expected: %s' % e
print(json.dumps(res))
need(res['v823_observer_with_v823_process'] == 'accepted', 'observer dry run')
files = sorted(str(p) for p in A.rglob('*') if p.is_file()) + [str(D/'observer.v823.private.json')]
put(A/'MANIFEST.sha256', ''.join('%s  %s\n' % (hf(f), f) for f in files).encode())
print(json.dumps(dict(prepared=True, files=files)))
