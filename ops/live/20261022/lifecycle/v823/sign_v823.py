#!/usr/bin/env python3
"""SN120 MiMo v8.2.3 = lifecycle-config succession on top of live v8.2.2 (policy dfe0cbf7, archive a2a03cb3, closure d0b0475d).
Exactly one change: trainer_config.overlong_soft_start 7000 -> 6000 (overlong_max_tokens 8192, overlong_penalty 0.5, KL/LR unchanged).
Public release policy, release succession, source archive, worker closure, archive_execution and writer are UNCHANGED, so
e40-e45 stay auditable (auditable_archives unchanged) and no host/worker/writer files change.
Signs (ROOT 3301134b, on-box authority seed named by the live lifecycle config; never printed/copied) only the two host
operator admissions that pin the new lifecycle config. Writes only NEW 0600 files (O_EXCL) into release-v823-signed/.
No services, units, CURRENT, R2 or chain touched."""
import copy, hashlib, json, os, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT_ID = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
PY = '/home/const/subnet120-rewrite/.venv/bin/python'
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
S2 = RA/'20261015-mimo-v8.2-release-succession/release-v822-signed'
SRC = S2/'deployment-final-source-v822'
OUT = RA/'20261016-v823/release-v823-signed'
UID_DIR = RA/'20261013-null-pow-v1/uid1000-host-prospective'
LIFE822_SHA = '943ed0779d4108076c6953ccf682aac21098112ceb84dce803759f7f9747dbfb'
CAP822_SHA = '0f0bfbd67545df5713f57b9309cf0364ca9979b3b6abf63d3ce825eead809a14'
UID822_SHA = 'a401bfb05f24fbf68a5ed9e0d4cdc9327c745887cdea13990aedbde929219717'
POLICY = 'dfe0cbf78bcf59c4b8ee2162bbd9f573a2e45d22b757c87d820ad7549a4b3eba'
NEW_SOFT = 6000
def need(ok, why):
    if not ok: raise SystemExit('REFUSED: ' + why)
def hf(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p = Path(p); need(p.is_file() and not p.is_symlink(), 'ordinary file ' + str(p)); return json.loads(p.read_bytes())
def pin(p): p = Path(p); return dict(path=str(p), sha256=hf(p))
log = []
def note(**k): log.append(k); print(json.dumps(k), flush=True)
def flat(v, p=''):
    return {p: v} if not isinstance(v, dict) else {k2: v2 for k, x in v.items() for k2, v2 in flat(x, p + '/' + k).items()}
def main():
    need(subprocess.run(['sha256sum', '--quiet', '-c', 'SIGNED-MANIFEST.sha256'], cwd=S2).returncode == 0, 'v8.2.2 signed manifest')
    sys.path.insert(0, str(SRC))
    from subnet.storage import Identity, canonical, sha
    from subnet.backend_jobs import signed
    from subnet.mimo_release_policy import ReleasePolicy, sign
    from subnet.mimo_release_succession import ReleaseChain
    from subnet.mimo_worker import source_inventory
    from subnet.mimo_grpo import TrainerConfig
    digest = lambda v: sha(canonical(v))
    lp2 = S2/'lifecycle.v8.2.2.ACTIVE.private.json'; need(hf(lp2) == LIFE822_SHA, 'live lifecycle v8.2.2 943ed077')
    cp2 = S2/'capture-operator.v8.2.2.ROOT-ADMITTED.private.json'; need(hf(cp2) == CAP822_SHA, 'capture operator v8.2.2')
    up2 = S2/'uid-operator.v8.2.2.ROOT-ADMITTED.private.json'; need(hf(up2) == UID822_SHA, 'uid operator v8.2.2')
    life2 = read(lp2); cap2 = read(cp2); uid2 = read(up2)
    signed(cap2['admission'], ROOT_ID); signed(uid2['admission'], ROOT_ID)
    need(digest(life2['release_policy']) == POLICY, 'release policy dfe0cbf7')
    chain = ReleaseChain(life2['release_succession'], ROOT_ID, allow_public=True)
    need(chain.active.policy_sha256 == POLICY, 'active chain member dfe0cbf7')
    need(source_inventory(SRC) == life2['adapters']['source_files'], 'source closure d0b0475d')
    t2 = life2['trainer_config']
    need((t2['overlong_soft_start'], t2['overlong_max_tokens'], t2['overlong_penalty'], t2['kl_coefficient'], t2['learning_rate'], t2['max_tasks'])
         == (7000, 8192, 0.5, 0.02, 1e-06, 128), 'v8.2.2 trainer knobs as expected')
    life = copy.deepcopy(life2); life['trainer_config']['overlong_soft_start'] = NEW_SOFT
    f0, f1 = flat(life2), flat(life)
    delta = sorted(k for k in set(f0) | set(f1) if f0.get(k) != f1.get(k))
    need(delta == ['/trainer_config/overlong_soft_start'], 'lifecycle delta ' + repr(delta))
    d0 = TrainerConfig(**life2['trainer_config']).validate().document(); d1 = TrainerConfig(**life['trainer_config']).validate().document()
    fd0, fd1 = flat(d0), flat(d1)
    ddelta = sorted(k for k in set(fd0) | set(fd1) if fd0.get(k) != fd1.get(k))
    need(ddelta == ['/settings/overlong_soft_start'], 'trainer document delta ' + repr(ddelta))
    c1 = TrainerConfig(**life['trainer_config'])
    need(c1.reward('correct', 6000) == 1.0 and abs(c1.reward('correct', 7096) - 0.75) < 1e-12 and c1.reward('correct', 8192) == 0.5
         and c1.reward('wrong', 8192) == -0.5 and c1.reward('correct', 5000) == 1.0, 'shaped reward at soft start 6000')
    note(step=0, verified=True, delta=delta, trainer_document_delta=ddelta, policy=POLICY, chain=chain.chain_sha256,
         archive_execution=sorted(life['archive_execution']), trainer_document_sha256=digest(d1), previous_trainer_document_sha256=digest(d0))
    OUT.mkdir(mode=0o700, parents=False, exist_ok=False)
    def put(name, value=None, raw=None):
        raw = raw if raw is not None else canonical(value); p = OUT/name
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        return p
    lp = put('lifecycle.v8.2.3.ACTIVE.private.json', life)
    need(read(lp) == life, 'lifecycle round-trip')
    seed_path = Path(life2['authority_seed_file'])
    need(seed_path.name == 'authority.seed' and 'live-controller-v1/controller-state' in str(seed_path), 'authority seed path')
    try: actor = Identity(bytes.fromhex(seed_path.read_text().strip()))
    except ValueError: raise SystemExit('REFUSED: authority seed file is not 32-byte hex')
    need(actor.id == ROOT_ID, 'key does not derive ROOT authority 3301134b')
    cap = {k: v for k, v in cap2.items() if k != 'admission'}; cap['base'] = dict(cap['base'])
    cap['base'].update(argv=['--config', str(lp), '--allow-public-mimo'], lifecycle_config=pin(lp))
    need(cap['base']['entry'] == pin(SRC/'ops/mimo_lifecycle.py') and cap['base']['source_root'] == str(SRC), 'entry/source unchanged')
    need(sha(canonical({k: v for k, v in cap2.items() if k != 'admission'})) == signed(cap2['admission'], ROOT_ID)['configuration_sha256']
         and signed(cap2['admission'], ROOT_ID)['version'] == cap2['version'], 'capture admission formula reproduces v8.2.2')
    need(sha(canonical({k: v for k, v in uid2.items() if k != 'admission'})) == signed(uid2['admission'], ROOT_ID)['configuration_sha256']
         and signed(uid2['admission'], ROOT_ID)['version'] == uid2['version'], 'uid admission formula reproduces v8.2.2')
    cap['admission'] = sign(actor, dict(version=cap2['version'],
        configuration_sha256=sha(canonical({k: v for k, v in cap.items() if k != 'admission'})),
        source_unchanged=True, scope='host-capture-scheduling-only'))
    p_old = signed(cap2['admission'], ROOT_ID); p_new = signed(cap['admission'], ROOT_ID)
    need(set(p_old) == set(p_new) and p_old['version'] == p_new['version'] and p_old['scope'] == p_new['scope'], 'capture admission shape')
    capp = put('capture-operator.v8.2.3.ROOT-ADMITTED.private.json', cap)
    uid = {k: v for k, v in uid2.items() if k != 'admission'}; uid['parent_operator'] = pin(capp)
    uid['admission'] = sign(actor, dict(version=uid2['version'],
        configuration_sha256=sha(canonical({k: v for k, v in uid.items() if k != 'admission'})),
        source_unchanged=True, scope='host-registration-capacity-only', capacity=1000))
    u_old = signed(uid2['admission'], ROOT_ID); u_new = signed(uid['admission'], ROOT_ID)
    need(set(u_old) == set(u_new) and u_old['version'] == u_new['version'] and u_new['capacity'] == 1000, 'uid admission shape')
    uidp = put('uid-operator.v8.2.3.ROOT-ADMITTED.private.json', uid)
    actor = None
    note(step=1, lifecycle=str(lp), lifecycle_sha256=hf(lp), capture=str(capp), capture_sha256=hf(capp), uid_operator=str(uidp), uid_operator_sha256=hf(uidp))
    r = subprocess.run([PY, '-B', str(UID_DIR/'mimo_uid_capacity_launcher.py'), '--operator-config', str(uidp), '--check'],
                       capture_output=True, text=True, timeout=1200, cwd=str(OUT))
    ok = r.returncode == 0
    note(check='lifecycle-launcher', ok=ok, out=(r.stdout.strip().splitlines() or [''])[-1][:400], err='' if ok else r.stderr.strip()[-800:])
    summary = dict(version='mimo-v823-root-signing-result-v1', authority=ROOT_ID, policy_sha256=POLICY, chain_sha256=chain.chain_sha256,
                   archive_sha256=life['release_policy']['payload']['scope']['immutable_manifest']['source_bundle']['sha256'],
                   change=dict(trainer_config=dict(overlong_soft_start=[7000, NEW_SOFT])), checks_passed=ok, steps=log,
                   services_changed=False, chain_calls=0)
    put('SIGNING-RESULT.json', raw=(json.dumps(summary, indent=1, sort_keys=True) + '\n').encode())
    subprocess.run('sha256sum *.json > SIGNED-MANIFEST.sha256 && chmod 600 SIGNED-MANIFEST.sha256', shell=True, cwd=OUT, check=True)
    print(json.dumps(dict(done=True, checks_passed=ok, out=str(OUT))))
    sys.exit(0 if ok else 2)
if __name__ == '__main__': main()
