#!/usr/bin/env python3
"""Prepare (do not install) v8.2.4 activation files and dry-run them.
  Arbos: lifecycle drop-in (uid launcher from release-v824-signed), mining-observer config + drop-in, trainer drop-in.
  Pod:   owned-worker-v824s.json (launcher over the unchanged closure source-final-v822b), host files manifest.
Dry runs: observer selector vs a simulated v8.2.4 lifecycle argv (accept) and v8.2.3 observer (refuse);
          trainer roundtrip: e51 TRAINING-JOB trainer config vs v8.2.4 config (only settings.max_tasks differs;
          optimizer restore allowed); lifecycle-side v1/v2 receipt acceptance under the installed overrides."""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); D = RA/'20261017-v824'; SG = D/'release-v824-signed'; A = D/'activation-v824'
S2 = RA/'20261015-mimo-v8.2-release-succession/release-v822-signed'; SRC = S2/'deployment-final-source-v822'
OBS3 = RA/'20261016-v823/observer.v823.private.json'
UID_DIR_OLD = RA/'20261013-null-pow-v1/uid1000-host-prospective'
STATE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
PY = '/home/const/subnet120-rewrite/.venv/bin/python'
POD = '/workspace/affine-mimo-trainer-prod'; POD_HOST = POD + '/host-v824'
hf = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok, why):
    if not ok: raise SystemExit('REFUSED: ' + why)
def put(p, raw, mode=0o600):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
need(subprocess.run(['sha256sum', '--quiet', '-c', 'SIGNED-MANIFEST.sha256'], cwd=SG).returncode == 0, 'v8.2.4 signed manifest')
r = json.loads((SG/'SIGNING-RESULT.json').read_text()); need(r['checks_passed'] is True, 'signing checks')
tp = SG/'trainer-host-v824/TRAINER-HOST.v8.2.4.ROOT-ADMITTED.private.json'
A.mkdir(mode=0o700)
VARIANTS = {'v8.2.4-t128': 'max_tasks 128 (first: faster trainer + v2 receipts)', 'v8.2.4': 'max_tasks 256 (boundary re-stamp)'}
obs3 = json.loads(OBS3.read_text())
need(obs3['lifecycle_config']['sha256'] == 'b8c1a318139b5477f2cb1d4fa4fbc5aad2283a06427a4169065036efece5da3b', 'observer v823 baseline')
paths = {}
for tag, what in VARIANTS.items():
    lp, cp, up = (SG/('lifecycle.%s.ACTIVE.private.json' % tag), SG/('capture-operator.%s.ROOT-ADMITTED.private.json' % tag),
                  SG/('uid-operator.%s.ROOT-ADMITTED.private.json' % tag))
    lc = ("# MiMo %s: %s; ROOT-admitted host layers over policy dfe0cbf7 / closure d0b0475d (unchanged).\n"
          "# Installed by activate_v824.sh; removed by rollback_v824.sh. Overrides zzzz~~~~~~~~v823-release.conf (left in place).\n"
          "[Service]\nExecStart=\nExecStart=%s -B %s --operator-config %s\n" % (tag, what, PY, SG/'mimo_uid_capacity_launcher.py', up))
    put(A/tag/'affine-mimo-lifecycle.service.d/zzzz~~~~~~~~~v824-release.conf', lc.encode())
    obs = dict(obs3)
    obs['lifecycle_config'] = dict(path=str(lp), sha256=hf(lp)); obs['host_capture_config'] = dict(path=str(cp), sha256=hf(cp))
    obs['host_uid_config'] = dict(path=str(up), sha256=hf(up)); obs['host_uid_files_dir'] = str(SG)
    op = D/('observer.%s.private.json' % tag)
    put(op, (json.dumps(obs, indent=1, sort_keys=True) + '\n').encode())
    oc = ("# %s: rebind public mining observer to lifecycle.%s + v8.2.4 uid host layer. Overrides zzzz~~~mimo-observer-v823.conf.\n"
          "[Service]\nExecStart=\nExecStart=%s -B %s --config %s --watch\nWorkingDirectory=%s\nEnvironment=PYTHONPATH=%s\nEnvironment=PYTHONDONTWRITEBYTECODE=1\n"
          % (tag, tag, PY, UID_DIR_OLD/'publish_mimo_public_discovery.py', op, SRC, SRC))
    put(A/tag/'affine-mining-discovery.service.d/zzzz~~~~mimo-observer-v824.conf', oc.encode())
    paths[tag] = (lp, cp, up, op, obs)
# trainer: owned worker config (pod) + Arbos supervised-unit drop-in
base = json.loads(Path(D/'pod-owned-worker-v822s.json').read_text())   # copied read-only from the pod by activate preflight
need(base['command'][2:4] == ['-m', 'subnet.distributed_worker'] and base['cwd'] == POD + '/source-final-v822b', 'v822s owned worker baseline')
ow = dict(base); ow['command'] = [base['command'][0], '-B', POD_HOST + '/mimo_host_trainer_v824.py', '--operator-config',
                                  POD_HOST + '/TRAINER-HOST.v8.2.4.ROOT-ADMITTED.private.json', '--'] + base['command'][4:]
raw = (json.dumps(ow, indent=1) + '\n').encode()
put(A/'pod/owned-worker-v824s.json', raw); ow_sha = hashlib.sha256(raw).hexdigest()
unit = Path('/home/const/.config/systemd/user/affine-mimo-trainer-supervised.service.d/zzzz~~~~v822s-owned-worker.conf').read_text()
old_line = [l for l in unit.splitlines() if l.startswith('ExecStart="/usr/bin/ssh"')]; need(len(old_line) == 1, 'v822s trainer drop-in')
new_line = old_line[0].replace(POD + '/owned-worker-v822s.json', POD + '/owned-worker-v824s.json')
new_line = new_line[:new_line.rindex('--sha256 ') + 9] + ow_sha + '"'
need(new_line.count('owned-worker-v824s.json') == 1, 'trainer ExecStart rewrite')
put(A/'trainer/affine-mimo-trainer-supervised.service.d/zzzz~~~~~v824s-owned-worker.conf',
    ("# MiMo v8.2.4: trainer owned worker = ROOT-admitted host launcher over unchanged source-final-v822b.\n"
     "# Installed by activate_v824.sh; removed by rollback_v824.sh.\n[Service]\nExecStart=\n%s\n" % new_line).encode())
# --- observer dry run (fake /proc) ---
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(UID_DIR_OLD)); os.chdir(SRC)
import publish_mimo_public_discovery as P
res = {}
for tag, (lp, cp, up, op, obs) in paths.items():
    fake = Path(tempfile.mkdtemp(prefix='v824-proc-')); (fake/'4242').mkdir()
    argv = [PY, '-B', str(SG/'mimo_uid_capacity_launcher.py'), '--operator-config', str(up)]
    (fake/'4242'/'cmdline').write_bytes(b'\0'.join(a.encode() for a in argv) + b'\0')
    (fake/'4242'/'stat').write_text(Path('/proc/self/stat').read_text())
    unit = lambda: 'MainPID=4242\nActiveState=active'
    try:
        P.selector(obs, proc=fake, inspect_unit=unit); res['observer_%s_with_%s_process' % (tag, tag)] = 'accepted'
    except Exception as e:
        res['observer_%s_with_%s_process' % (tag, tag)] = 'REFUSED %s: %s' % (type(e).__name__, e)
    try:
        P.selector(obs3, proc=fake, inspect_unit=unit); res['observer_v823_with_%s_process' % tag] = 'accepted (unexpected)'
    except Exception as e:
        res['observer_v823_with_%s_process' % tag] = 'refused as expected: %s' % str(e)[:100]
lp = paths['v8.2.4'][0]
# --- trainer roundtrip (no secrets printed: only config documents are compared) ---
from subnet.mimo_grpo import TrainerConfig
from subnet.storage import canonical, sha
life = json.loads(lp.read_text())
job51 = json.loads((STATE/'epochs/51/TRAINING-JOB.json').read_bytes())
payload = job51.get('payload', job51)
doc51 = payload['training']['config']
doc824 = TrainerConfig(**life['trainer_config']).validate().document()
def flat(v, p=''):
    return {p: v} if not isinstance(v, dict) else {k2: v2 for k, x in v.items() for k2, v2 in flat(x, p + '/' + k).items()}
f0, f1 = flat(doc51), flat(doc824)
res['trainer_doc_delta_vs_e51_job'] = sorted(k for k in set(f0) | set(f1) if f0.get(k) != f1.get(k))
try:
    from subnet.mimo_grpo.optimizer import restorable_config
    res['optimizer_restore_e51_to_v824'] = bool(restorable_config(doc51, TrainerConfig(**life['trainer_config']).validate()))
except Exception as e:
    res['optimizer_restore_e51_to_v824'] = 'check: %s: %s' % (type(e).__name__, str(e)[:160])
res['e51_job_max_tasks'] = doc51['settings']['max_tasks']; res['v824_max_tasks'] = doc824['settings']['max_tasks']
print(json.dumps(res, indent=1))
need(all(res['observer_%s_with_%s_process' % (t, t)] == 'accepted' for t in VARIANTS), 'observer dry run')
need(res['trainer_doc_delta_vs_e51_job'] == ['/settings/max_tasks'], 'trainer doc delta')
files = sorted(str(p) for p in A.rglob('*') if p.is_file()) + [str(paths[t][3]) for t in VARIANTS]
put(A/'MANIFEST.sha256', ''.join('%s  %s\n' % (hf(f), f) for f in files).encode())
put(A/'PREPARE-RESULT.json', (json.dumps(dict(res, owned_worker_v824s_sha256=ow_sha), indent=1) + '\n').encode())
print(json.dumps(dict(prepared=True, files=files, owned_worker_v824s_sha256=ow_sha)))
