#!/usr/bin/env python3
"""SN120 MiMo v8.2.4 = faster trainer (ROOT-admitted host overrides) + max_tasks 256, on top of live v8.2.3.
Policy dfe0cbf7, chain, source archive a2a03cb3, worker closure d0b0475d, archive_execution and the writer are UNCHANGED
(old-epoch audits keep verifying; no closure change). Changes:
  lifecycle config: lifecycle.max_tasks and trainer_config.max_tasks 128 -> 256 (only delta)
  lifecycle host layer (uid-operator dir = this release dir): v2 receipt acceptance (v1 unchanged), finalize completed_at
    clock fix, CURRENT max_tasks re-stamp only at an epoch boundary (current epoch without PLAN)
  trainer host layer: padded micro-batched engine (16384 tokens, 64 rows), pipelined CPUAdamW.step, fast staging +
    preupload, FastR2ModelPublisher (v2 receipts)
Signs with ROOT 3301134b (on-box authority seed named by the live lifecycle config; never printed/copied).
Writes only NEW files (O_EXCL) into release-v824-signed/. No services, units, CURRENT, R2 or chain touched."""
import copy, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT_ID = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
PY = '/home/const/subnet120-rewrite/.venv/bin/python'
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
S2 = RA/'20261015-mimo-v8.2-release-succession/release-v822-signed'
SRC = S2/'deployment-final-source-v822'
S3 = RA/'20261016-v823/release-v823-signed'
D = RA/'20261017-v824'; OUT = D/'release-v824-signed'; STAGE = Path('/home/const/sn120tb-v824')
LIFE823_SHA = 'b8c1a318139b5477f2cb1d4fa4fbc5aad2283a06427a4169065036efece5da3b'
CAP823_SHA = 'd07b9218e294ae68400f4544dd03bb8be1324a3ef9d5383e66f7b6110b187d4c'
UID823_SHA = '65155f1764d3830734b80c796ca8c9d4dd45b15dfe1a017fbff349bf660bcd68'
POLICY = 'dfe0cbf78bcf59c4b8ee2162bbd9f573a2e45d22b757c87d820ad7549a4b3eba'
POD_ROOT = '/workspace/affine-mimo-trainer-prod'
POD_SOURCE = POD_ROOT + '/source-final-v822b'
NEW_TASKS = 256
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
    need(subprocess.run(['sha256sum', '--quiet', '-c', 'SIGNED-MANIFEST.sha256'], cwd=S3).returncode == 0, 'v8.2.3 signed manifest')
    sys.path.insert(0, str(SRC))
    from subnet.storage import Identity, canonical, sha
    from subnet.backend_jobs import signed
    from subnet.mimo_release_policy import sign
    from subnet.mimo_release_succession import ReleaseChain
    from subnet.mimo_worker import source_inventory
    from subnet.mimo_grpo import TrainerConfig
    digest = lambda v: sha(canonical(v))
    lp3 = S3/'lifecycle.v8.2.3.ACTIVE.private.json'; need(hf(lp3) == LIFE823_SHA, 'live lifecycle v8.2.3 b8c1a318')
    cp3 = S3/'capture-operator.v8.2.3.ROOT-ADMITTED.private.json'; need(hf(cp3) == CAP823_SHA, 'capture operator v8.2.3')
    up3 = S3/'uid-operator.v8.2.3.ROOT-ADMITTED.private.json'; need(hf(up3) == UID823_SHA, 'uid operator v8.2.3')
    life3 = read(lp3); cap3 = read(cp3); uid3 = read(up3)
    signed(cap3['admission'], ROOT_ID); signed(uid3['admission'], ROOT_ID)
    need(digest(life3['release_policy']) == POLICY, 'release policy dfe0cbf7')
    chain = ReleaseChain(life3['release_succession'], ROOT_ID, allow_public=True)
    need(chain.active.policy_sha256 == POLICY, 'active chain member dfe0cbf7')
    need(source_inventory(SRC) == life3['adapters']['source_files'], 'source closure d0b0475d')
    need(life3['lifecycle']['max_tasks'] == 128 and life3['trainer_config']['max_tasks'] == 128, 'v8.2.3 max_tasks 128')
    life128 = copy.deepcopy(life3)   # variant v8.2.4-t128: v8.2.4 host layers, lifecycle config byte-identical in content
    life = copy.deepcopy(life3); life['lifecycle']['max_tasks'] = NEW_TASKS; life['trainer_config']['max_tasks'] = NEW_TASKS
    f0, f1 = flat(life3), flat(life)
    delta = sorted(k for k in set(f0) | set(f1) if f0.get(k) != f1.get(k))
    need(delta == ['/lifecycle/max_tasks', '/trainer_config/max_tasks'], 'lifecycle delta ' + repr(delta))
    d0 = TrainerConfig(**life3['trainer_config']).validate().document(); d1 = TrainerConfig(**life['trainer_config']).validate().document()
    fd0, fd1 = flat(d0), flat(d1)
    ddelta = sorted(k for k in set(fd0) | set(fd1) if fd0.get(k) != fd1.get(k))
    need(ddelta == ['/settings/max_tasks'], 'trainer document delta ' + repr(ddelta))
    # staged host layers (built + tested in /home/const/sn120tb-v824)
    uid_src = {'mimo_uid_capacity.py': STAGE/'rendered/mimo_uid_capacity.py',
               'mimo_uid_capacity_launcher.py': STAGE/'arbos/mimo_uid_capacity_launcher.py',
               'mimo_host_path_launcher.py': STAGE/'arbos/mimo_host_path_launcher.py'}
    need(hf(uid_src['mimo_host_path_launcher.py']) == uid3['host_files']['mimo_host_path_launcher.py'], 'host path launcher unchanged')
    note(step=0, verified=True, delta=delta, trainer_document_delta=ddelta, policy=POLICY, chain=chain.chain_sha256,
         trainer_document_sha256=digest(d1), previous_trainer_document_sha256=digest(d0))
    D.mkdir(mode=0o700, exist_ok=True); OUT.mkdir(mode=0o700, parents=False, exist_ok=False)
    def put(name, value=None, raw=None, mode=0o600):
        raw = raw if raw is not None else canonical(value); p = OUT/name; p.parent.mkdir(mode=0o700, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        return p
    for name, src in uid_src.items(): put(name, raw=src.read_bytes())
    tdir = STAGE/'trainer'
    sys.path.insert(0, str(STAGE/'tools'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('v824_launcher_probe', tdir/'mimo_host_trainer_v824.py')
    TL = importlib.util.module_from_spec(spec); spec.loader.exec_module(TL)
    for name in TL.HOST_FILES: put('trainer-host-v824/' + name, raw=(tdir/name).read_bytes())
    lp = put('lifecycle.v8.2.4.ACTIVE.private.json', life)
    need(read(lp) == life, 'lifecycle round-trip')
    lp128 = put('lifecycle.v8.2.4-t128.ACTIVE.private.json', life128)
    need(read(lp128) == life128 == life3, 'lifecycle t128 round-trip')
    # trainer host proposal (pins from the identical closure; source_root is the pod path)
    import trainer_host_config as THC
    prop, _ = THC.proposal(OUT/'trainer-host-v824', SRC, ROOT_ID)
    prop['source_root'] = POD_SOURCE
    seed_path = Path(life3['authority_seed_file'])
    need(seed_path.name == 'authority.seed' and 'live-controller-v1/controller-state' in str(seed_path), 'authority seed path')
    try: actor = Identity(bytes.fromhex(seed_path.read_text().strip()))
    except ValueError: raise SystemExit('REFUSED: authority seed file is not 32-byte hex')
    need(actor.id == ROOT_ID, 'key does not derive ROOT authority 3301134b')
    def operators(tag, lifecycle_path):
        cap = {k: v for k, v in cap3.items() if k != 'admission'}; cap['base'] = dict(cap['base'])
        cap['base'].update(argv=['--config', str(lifecycle_path), '--allow-public-mimo'], lifecycle_config=pin(lifecycle_path))
        need(cap['base']['entry'] == pin(SRC/'ops/mimo_lifecycle.py') and cap['base']['source_root'] == str(SRC), 'entry/source unchanged')
        cap['admission'] = sign(actor, dict(version=cap3['version'],
            configuration_sha256=sha(canonical({k: v for k, v in cap.items() if k != 'admission'})),
            source_unchanged=True, scope='host-capture-scheduling-only'))
        p_old = signed(cap3['admission'], ROOT_ID); p_new = signed(cap['admission'], ROOT_ID)
        need(set(p_old) == set(p_new) and p_old['version'] == p_new['version'] and p_old['scope'] == p_new['scope'], 'capture admission shape')
        capp = put('capture-operator.%s.ROOT-ADMITTED.private.json' % tag, cap)
        uid = {k: v for k, v in uid3.items() if k != 'admission'}; uid['parent_operator'] = pin(capp)
        uid['host_files'] = {name: hf(OUT/name) for name in uid_src}
        uid['admission'] = sign(actor, dict(version=uid3['version'],
            configuration_sha256=sha(canonical({k: v for k, v in uid.items() if k != 'admission'})),
            source_unchanged=True, scope='host-registration-capacity-only', capacity=1000))
        u_old = signed(uid3['admission'], ROOT_ID); u_new = signed(uid['admission'], ROOT_ID)
        need(set(u_old) == set(u_new) and u_old['version'] == u_new['version'] and u_new['capacity'] == 1000, 'uid admission shape')
        return capp, put('uid-operator.%s.ROOT-ADMITTED.private.json' % tag, uid), uid
    capp, uidp, uid = operators('v8.2.4', lp)
    capp128, uidp128, _ = operators('v8.2.4-t128', lp128)
    trainer = THC.admit(prop, TL, actor)
    tp = put('trainer-host-v824/TRAINER-HOST.v8.2.4.ROOT-ADMITTED.private.json', trainer)
    t_payload = signed(trainer['admission'], ROOT_ID)
    need(t_payload['configuration_sha256'] == sha(canonical(prop)) and t_payload['scope'] == TL.SCOPE, 'trainer admission')
    actor = None
    note(step=1, lifecycle_t128=str(lp128), lifecycle_t128_sha256=hf(lp128), capture_t128_sha256=hf(capp128), uid_operator_t128=str(uidp128),
         uid_operator_t128_sha256=hf(uidp128),
         lifecycle=str(lp), lifecycle_sha256=hf(lp), capture=str(capp), capture_sha256=hf(capp), uid_operator=str(uidp),
         uid_operator_sha256=hf(uidp), uid_host_files=uid['host_files'], trainer_host=str(tp), trainer_host_sha256=hf(tp),
         trainer_engine=prop['engine'], trainer_pins=prop['original_methods'])
    ok = True
    for tag, path in (('v8.2.4-t128', uidp128), ('v8.2.4', uidp)):
        r = subprocess.run([PY, '-B', str(OUT/'mimo_uid_capacity_launcher.py'), '--operator-config', str(path), '--check'],
                           capture_output=True, text=True, timeout=1200, cwd=str(OUT))
        last = (r.stdout.strip().splitlines() or [''])[-1]
        good = r.returncode == 0 and '"v824_override_pins_passed": true' in last and '"operator_admission_passed": true' in last
        ok = ok and good
        note(check='lifecycle-launcher ' + tag, ok=good, out=last[:400], err='' if r.returncode == 0 else r.stderr.strip()[-800:])
    summary = dict(version='mimo-v824-root-signing-result-v1', authority=ROOT_ID, policy_sha256=POLICY, chain_sha256=chain.chain_sha256,
                   archive_sha256=life['release_policy']['payload']['scope']['immutable_manifest']['source_bundle']['sha256'],
                   change=dict(max_tasks=[128, NEW_TASKS], variants=['v8.2.4-t128 (max_tasks 128, first)', 'v8.2.4 (max_tasks 256)'], trainer_engine=prop['engine'], receipts='v1+v2 accepted',
                               lifecycle_clock_fix=True, seal_gap_changes=False),
                   checks_passed=ok, steps=log, services_changed=False, chain_calls=0)
    put('SIGNING-RESULT.json', raw=(json.dumps(summary, indent=1, sort_keys=True) + '\n').encode())
    subprocess.run('find . -type f ! -name SIGNED-MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > SIGNED-MANIFEST.sha256 && chmod 600 SIGNED-MANIFEST.sha256',
                   shell=True, cwd=OUT, check=True)
    print(json.dumps(dict(done=True, checks_passed=ok, out=str(OUT))))
    sys.exit(0 if ok else 2)
if __name__ == '__main__': main()
