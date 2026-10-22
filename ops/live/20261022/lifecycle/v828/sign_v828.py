#!/usr/bin/env python3
"""SN120 MiMo v8.2.8 = signed v8.2.7 composition (pinned, byte-identical, referenced in place) + one new top host launcher:
  1. original audit population built in a background thread; trainer job dispatched first; FINALIZED waits for it
  2. audit scheduler disabled (no audits scheduled/queued); lifecycle config + ops.mimo_lifecycle.run unchanged
  3. native grading workers 8 -> 32 (original ordered pool)
Follows sign_v827.py: ROOT 3301134b from the on-box authority seed named by the live lifecycle config (never printed or
copied), new files only (O_EXCL), --check of the composed launcher. No services touched, no chain calls."""
import hashlib, inspect, json, os, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT_ID = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
PY = '/home/const/subnet120-rewrite/.venv/bin/python'
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
SRC = RA/'20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822'
V827 = RA/'20261021-v827-incremental-capture/release-v827-signed'
D = RA/'20261021-v828-audit-off-path'; STAGE = D/'stage'; OUT = D/'release-v828-signed'
UID827_SHA = '4b8b081f0596ced3c70c4455482a14638bd4a6071bdcd84d164e1b030337d91d'
UIDLAUNCHER_SHA = '98fea1b63d896306382b7d533e9635f31b517f802bee15fd074fc386adf51af7'
LIFE824_SHA = 'c38a9fcf21a1d2a6ef8d762e72fb895e160aa893aba93aa11e509639dda07982'
VERSION = 'affine-host-v828-audit-off-path-launcher-v1'; SCOPE = 'host-audit-off-critical-path-only'
WORKERS = 32
def need(ok, why):
    if not ok: raise SystemExit('REFUSED: ' + why)
hf = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p = Path(p); need(p.is_file() and not p.is_symlink(), 'ordinary file ' + str(p)); return json.loads(p.read_bytes())
pin = lambda p: dict(path=str(p), sha256=hf(p))
def main():
    need(subprocess.run(['sha256sum', '--quiet', '-c', 'SIGNED-MANIFEST.sha256'], cwd=V827).returncode == 0, 'v8.2.7 signed manifest')
    need(read(V827/'SIGNING-RESULT.json')['checks_passed'] is True, 'v8.2.7 signing checks')
    uidp = V827/'uid-operator.v8.2.7.ROOT-ADMITTED.private.json'; uidl = V827/'mimo_uid_capacity_launcher.py'
    need(hf(uidp) == UID827_SHA and hf(uidl) == UIDLAUNCHER_SHA, 'v8.2.7 uid operator/launcher digests')
    need(hf(STAGE/'mimo_host_path_launcher.py') == hf(V827/'mimo_host_path_launcher.py'), 'import housekeeping byte-identical to v8.2.7')
    sys.path.insert(0, str(SRC))
    from subnet.storage import Identity, canonical, sha
    from subnet.backend_jobs import signed
    from subnet.mimo_release_policy import sign
    from subnet.mimo_lifecycle import Lifecycle
    import subnet.mimo_native_pool as NP
    import ops.mimo_lifecycle as OPS
    uid = read(uidp); signed(uid['admission'], ROOT_ID)
    cap = read(uid['parent_operator']['path']); need(cap['base']['lifecycle_config']['sha256'] == LIFE824_SHA, 'lifecycle v8.2.4 bound (unchanged)')
    life = read(cap['base']['lifecycle_config']['path']); need(hf(cap['base']['lifecycle_config']['path']) == LIFE824_SHA, 'lifecycle digest')
    need(life['release_policy']['signer'] == ROOT_ID, 'lifecycle authority')
    msha = lambda f: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
    methods = {'Lifecycle._audit_population': msha(Lifecycle._audit_population), 'Lifecycle._training': msha(Lifecycle._training),
               'Lifecycle.tick': msha(Lifecycle.tick), 'ops.mimo_lifecycle.archive_scoped_audit_scheduler': msha(OPS.archive_scoped_audit_scheduler),
               'subnet.mimo_native_pool.NativeGradingPool.__init__': msha(NP.NativeGradingPool.__init__),
               'subnet.mimo_native_pool.NativeGradingPool.__call__': msha(NP.NativeGradingPool.__call__)}
    need(methods['Lifecycle._training'] == '93a538d13ee04cabc0b9137ccea53dd07087e38f8442b4ab12bb7dd8308634c7', 'same original _training as v8.2.4 pin')
    seed_path = Path(life['authority_seed_file'])
    need(seed_path.is_file() and not seed_path.is_symlink() and seed_path.stat().st_mode & 0o077 == 0, 'owner-only authority seed file')
    OUT.mkdir(mode=0o700)
    def put(name, raw):
        fd = os.open(OUT/name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        return OUT/name
    for n in ('mimo_v828_launcher.py', 'mimo_host_path_launcher.py'): put(n, (STAGE/n).read_bytes())
    config = dict(version=VERSION, uid_entry=pin(uidl), uid_operator=pin(uidp),
                  host_files={n: hf(OUT/n) for n in ('mimo_v828_launcher.py', 'mimo_host_path_launcher.py')},
                  original_methods=methods, grading_workers=WORKERS)
    try: actor = Identity(bytes.fromhex(seed_path.read_text().strip()))
    except ValueError: raise SystemExit('REFUSED: authority seed file is not 32-byte hex')
    need(actor.id == ROOT_ID, 'key does not derive ROOT authority 3301134b')
    config['admission'] = sign(actor, dict(version=VERSION, configuration_sha256=sha(canonical(config)), source_unchanged=True,
                                           scope=SCOPE, grading_workers=WORKERS))
    actor = None
    opp = put('v828-operator.ROOT-ADMITTED.private.json', canonical(config))
    apay = signed(config['admission'], ROOT_ID)
    r = subprocess.run([PY, '-B', str(OUT/'mimo_v828_launcher.py'), '--operator-config', str(opp), '--check'],
                       capture_output=True, text=True, timeout=1200, cwd=str(OUT))
    last = (r.stdout.strip().splitlines() or [''])[-1]
    ok = (r.returncode == 0 and '"v824_override_pins_passed": true' in last and '"operator_admission_passed": true' in last
          and '"v828": true' in last and '"production_services_started": false' in last)
    summary = dict(version='mimo-v828-root-signing-result-v1', authority=ROOT_ID, operator=str(opp), operator_sha256=hf(opp),
                   configuration_sha256=apay['configuration_sha256'], launcher_sha256=hf(OUT/'mimo_v828_launcher.py'),
                   base=dict(uid_operator_v827=UID827_SHA, uid_launcher=UIDLAUNCHER_SHA, capture_operator_v827=uid['parent_operator']['sha256']),
                   lifecycle_config=cap['base']['lifecycle_config']['path'], lifecycle_sha256=LIFE824_SHA, lifecycle_changed=False,
                   change=dict(audit_population='background thread (original method) after DISPATCH; trainer job first; FINALIZED waits for it',
                               audit_scheduler='disabled (no audits scheduled); config audit_scheduler.enabled unchanged; run() unchanged',
                               grading_workers=[8, WORKERS], new_pinned_originals=sorted(methods)),
                   unchanged=['trainer', 'trainer job contents', 'weight writer', 'v9 scorer', 'policy', 'source closure', 'miner upload format', 'chain'],
                   launcher_check=dict(ok=ok, out=last[:600], err='' if r.returncode == 0 else r.stderr.strip()[-800:]),
                   checks_passed=ok, services_changed=False, chain_calls=0)
    put('SIGNING-RESULT.json', (json.dumps(summary, indent=1, sort_keys=True) + '\n').encode())
    subprocess.run('find . -type f ! -name SIGNED-MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > SIGNED-MANIFEST.sha256 && chmod 600 SIGNED-MANIFEST.sha256',
                   shell=True, cwd=OUT, check=True)
    print(json.dumps(summary, indent=1))
    sys.exit(0 if ok else 2)
if __name__ == '__main__': main()
