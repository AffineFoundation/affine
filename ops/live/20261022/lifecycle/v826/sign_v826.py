#!/usr/bin/env python3
"""SN120 MiMo v8.2.6 = v8.2.4 lifecycle composition + v8.2.6 host capture layer (ONLY change).
  capture launcher v2: post-pass deadline re-check (original tick re-run at once with a post-deadline time)
  CapturePass v826: per-miner fresh observation time, resolve_task memo, best-effort parallel read prefetch
  capture-operator workers 8 -> 16
Lifecycle config lifecycle.v8.2.4.ACTIVE (c38a9fcf) / policy dfe0cbf7 / closure / trainer / writer / v9 scorer / chain: UNCHANGED.
Follows sign_v824.py/sign_v825.py: ROOT 3301134b from the on-box authority seed named by the live lifecycle config (never
printed/copied), proposal shapes of the v8.2.4 capture/uid operators, new files only (O_EXCL). No services touched."""
import hashlib, inspect, json, os, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT_ID = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
PY = '/home/const/subnet120-rewrite/.venv/bin/python'
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
SRC = RA/'20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822'
V824 = RA/'20261017-v824/release-v824-signed'
D = RA/'20261021-v826-capture-deadline'; STAGE = D/'stage'; OUT = D/'release-v826-signed'
CAP824_SHA = 'baba0e55c607795672918d4bf6ab6fb0c5793101960a969968a0db1cde8ea882'
UID824_SHA = 'ef82a9f0f8d1c473a08c480073ab58d0ca1efef19c2e66ba9c603628d826928e'
LIFE824_SHA = 'c38a9fcf21a1d2a6ef8d762e72fb895e160aa893aba93aa11e509639dda07982'
WORKERS = 16
def need(ok, why):
    if not ok: raise SystemExit('REFUSED: ' + why)
hf = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p = Path(p); need(p.is_file() and not p.is_symlink(), 'ordinary file ' + str(p)); return json.loads(p.read_bytes())
pin = lambda p: dict(path=str(p), sha256=hf(p))
def main():
    need(subprocess.run(['sha256sum', '--quiet', '-c', 'SIGNED-MANIFEST.sha256'], cwd=V824).returncode == 0, 'v8.2.4 signed manifest')
    sys.path.insert(0, str(SRC))
    from subnet.storage import Identity, canonical, sha
    from subnet.backend_jobs import signed
    from subnet.mimo_release_policy import sign
    from subnet.mimo_lifecycle import Lifecycle
    from subnet.mimo_coordinator import Coordinator
    import subnet.forced_sampling as FS, subnet.protocol as PR, subnet.environments as EN, subnet.complete_group_admission as CA
    cp4, up4, lp4 = V824/'capture-operator.v8.2.4.ROOT-ADMITTED.private.json', V824/'uid-operator.v8.2.4.ROOT-ADMITTED.private.json', V824/'lifecycle.v8.2.4.ACTIVE.private.json'
    need(hf(cp4) == CAP824_SHA and hf(up4) == UID824_SHA and hf(lp4) == LIFE824_SHA, 'live v8.2.4 operator/lifecycle digests')
    cap4, uid4, life = read(cp4), read(up4), read(lp4)
    signed(cap4['admission'], ROOT_ID); signed(uid4['admission'], ROOT_ID)
    need(cap4['base']['lifecycle_config'] == pin(lp4), 'capture operator binds lifecycle v8.2.4 (unchanged)')
    msha = lambda f: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
    need(msha(Lifecycle._capture) == cap4['original_methods']['Lifecycle._capture']
         and msha(Coordinator.collect) == cap4['original_methods']['Coordinator.collect'], 'original capture pins unchanged')
    need(hf(STAGE/'mimo_host_path_launcher.py') == cap4['host_files']['mimo_host_path_launcher.py']
         == uid4['host_files']['mimo_host_path_launcher.py'], 'import housekeeping unchanged')
    for n in ('mimo_uid_capacity.py', 'mimo_uid_capacity_launcher.py'):
        need(hf(V824/n) == uid4['host_files'][n], 'uid host file unchanged ' + n)
    seed_path = Path(life['authority_seed_file'])
    need(seed_path.name == 'authority.seed' and 'live-controller-v1/controller-state' in str(seed_path), 'authority seed path')
    OUT.mkdir(mode=0o700)
    def put(name, raw):
        fd = os.open(OUT/name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        return OUT/name
    for n in ('mimo_host_path_launcher.py', 'mimo_host_capture_launcher.py', 'mimo_host_capture_cache.py'): put(n, (STAGE/n).read_bytes())
    for n in ('mimo_uid_capacity.py', 'mimo_uid_capacity_launcher.py'): put(n, (V824/n).read_bytes())
    CAP_VERSION = 'affine-host-capture-cache-launcher-v2'
    cap = {k: v for k, v in cap4.items() if k != 'admission'}
    cap.update(version=CAP_VERSION, workers=WORKERS,
               host_files={n: hf(OUT/n) for n in ('mimo_host_capture_cache.py', 'mimo_host_capture_launcher.py', 'mimo_host_path_launcher.py')},
               original_methods=dict(cap4['original_methods'], **{'Lifecycle.tick': msha(Lifecycle.tick), 'subnet.forced_sampling.binding': msha(FS.binding),
                                     'subnet.protocol.entries': msha(PR.entries), 'subnet.environments._source_hash': msha(EN._source_hash),
                                     'subnet.complete_group_admission.scope': msha(CA.scope)}))
    try: actor = Identity(bytes.fromhex(seed_path.read_text().strip()))
    except ValueError: raise SystemExit('REFUSED: authority seed file is not 32-byte hex')
    need(actor.id == ROOT_ID, 'key does not derive ROOT authority 3301134b')
    cap['admission'] = sign(actor, dict(version=CAP_VERSION, configuration_sha256=sha(canonical({k: v for k, v in cap.items() if k != 'admission'})),
                                        source_unchanged=True, scope='host-capture-scheduling-only'))
    capp = put('capture-operator.v8.2.6.ROOT-ADMITTED.private.json', canonical(cap))
    uid = {k: v for k, v in uid4.items() if k != 'admission'}
    uid.update(parent_entry=pin(OUT/'mimo_host_capture_launcher.py'), parent_operator=pin(capp),
               host_files={n: hf(OUT/n) for n in ('mimo_host_path_launcher.py', 'mimo_uid_capacity.py', 'mimo_uid_capacity_launcher.py')})
    need(uid['host_files'] == uid4['host_files'], 'uid host files byte-identical to v8.2.4')
    uid['admission'] = sign(actor, dict(version=uid4['version'], configuration_sha256=sha(canonical({k: v for k, v in uid.items() if k != 'admission'})),
                                        source_unchanged=True, scope='host-registration-capacity-only', capacity=1000))
    actor = None
    uidp = put('uid-operator.v8.2.6.ROOT-ADMITTED.private.json', canonical(uid))
    cpay, upay = signed(cap['admission'], ROOT_ID), signed(uid['admission'], ROOT_ID)
    need(set(cpay) == set(signed(cap4['admission'], ROOT_ID)) and set(upay) == set(signed(uid4['admission'], ROOT_ID)), 'admission shapes')
    cdelta = sorted(k for k in set(cap4) | set(cap) if k != 'admission' and cap4.get(k) != cap.get(k))
    udelta = sorted(k for k in set(uid4) | set(uid) if k != 'admission' and uid4.get(k) != uid.get(k))
    r = subprocess.run([PY, '-B', str(OUT/'mimo_uid_capacity_launcher.py'), '--operator-config', str(uidp), '--check'],
                       capture_output=True, text=True, timeout=1200, cwd=str(OUT))
    last = (r.stdout.strip().splitlines() or [''])[-1]
    ok = r.returncode == 0 and '"v824_override_pins_passed": true' in last and '"operator_admission_passed": true' in last
    summary = dict(version='mimo-v826-root-signing-result-v1', authority=ROOT_ID,
                   capture_operator=str(capp), capture_operator_sha256=hf(capp), capture_configuration_sha256=cpay['configuration_sha256'],
                   uid_operator=str(uidp), uid_operator_sha256=hf(uidp), uid_configuration_sha256=upay['configuration_sha256'],
                   lifecycle_config=str(lp4), lifecycle_sha256=LIFE824_SHA, lifecycle_changed=False,
                   change=dict(capture_operator_delta_vs_v824=cdelta, uid_operator_delta_vs_v824=udelta, workers=[cap4['workers'], WORKERS],
                               capture_launcher=['affine-host-capture-cache-launcher-v1', CAP_VERSION], capture_cache='affine-host-capture-cache-v826',
                               new_pinned_originals=['Lifecycle.tick', 'subnet.forced_sampling.binding', 'subnet.protocol.entries', 'subnet.environments._source_hash', 'subnet.complete_group_admission.scope']),
                   launcher_check=dict(ok=ok, out=last[:400], err='' if r.returncode == 0 else r.stderr.strip()[-800:]),
                   checks_passed=ok, services_changed=False, chain_calls=0)
    put('SIGNING-RESULT.json', (json.dumps(summary, indent=1, sort_keys=True) + '\n').encode())
    subprocess.run('find . -type f ! -name SIGNED-MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > SIGNED-MANIFEST.sha256 && chmod 600 SIGNED-MANIFEST.sha256',
                   shell=True, cwd=OUT, check=True)
    print(json.dumps(summary, indent=1))
    sys.exit(0 if ok else 2)
if __name__ == '__main__': main()
