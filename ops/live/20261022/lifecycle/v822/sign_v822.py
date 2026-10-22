#!/usr/bin/env python3
"""ROOT signing for the MiMo v8.2.2 successor (chain v7 -> v8.2 -> v8.2.1 -> v8.2.2). Prepared, not activated.
* Authority key from the live v8.2 lifecycle config's authority_seed_file; never printed/copied; must derive 3301134b.
* Writes only NEW 0600 files (O_EXCL) under release-v821-signed/. No services, units, CURRENT, R2 or chain touched.
* v8.2.1 = v8.2 + 4 trainer files (archive 1aa805b5, inventory d0b0475d); lifecycle/trainer max_tasks 80; job_seconds 7200.
* Writer-reader keeps v7 69ae058b + v8.2 8ae233ad inventories unchanged and ADDS 1aa805b5 -> d0b0475d.
* --dry-run: throwaway key (v7/v8.2 members re-signed by it), temp dir, removed afterwards.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
ROOT_ID = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
PY = '/home/const/subnet120-rewrite/.venv/bin/python'
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
D = RA/'20261015-mimo-v8.2-release-succession'; ST = D/'staged-v822-successor'; S82 = D/'release-v821-signed'
UID_DIR = RA/'20261013-null-pow-v1/uid1000-host-prospective'
HOST_DIR = RA/'20261014-community-weights-check-v1/parallel-evidence-v6'
_B = json.loads((D/'staged-v822-successor/SUCCESSOR-BINDING.json').read_bytes())
SCOPE = _B['scope_sha256']; ARCHIVE = _B['archive_sha256']  # v8.2.1 incl. audit/eval archive-inventory fix 547573dd
A82 = '8ae233ad397a72ecf35416c9c751bf40cc62acb5d5aa3fbb21d74594b3fba375'
P82 = json.loads((D/'release-v821-signed/SIGNING-RESULT.json').read_bytes())['policy_sha256']; assert P82.startswith('4f017106')
A821 = _B['previous_archive_sha256']; assert A821.startswith('e6bba768')
KINDS = {'cross-device-sampling','heldout-evaluation','model-publication','mutation-controls','native-grading','runtime','trainer-update'}
def need(ok, why):
    if not ok: raise SystemExit('REFUSED: ' + why)
def hf(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p = Path(p); need(p.is_file() and not p.is_symlink(), 'ordinary file ' + str(p)); return json.loads(p.read_bytes())
def pin(p): p = Path(p); return dict(path=str(p), sha256=hf(p))
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--first-round', type=int, required=True)
    a = ap.parse_args(); need(a.first_round > 229, 'v8.2.2 first_round after v8.2.1 (229)')
    OUT = Path(tempfile.mkdtemp(prefix='v822-dryrun-')) if a.dry_run else D/'release-v822-signed'
    OUT.mkdir(mode=0o700, exist_ok=True); SRC = OUT/'deployment-final-source-v822'; log = []
    def note(**k): log.append(k); print(json.dumps(k), flush=True)
    binding = read(ST/'SUCCESSOR-BINDING.json')
    need(binding['archive_sha256'] == ARCHIVE and binding['scope_sha256'] == SCOPE, 'staged binding')
    release = read(ST/'source-release-v822/RELEASE.json')
    need(release['archive_sha256'] == ARCHIVE == hf(ST/'source-release-v822/source.tar.gz'), 'v8.2.1 archive')
    for name, row in release['source'].items():
        need(hf(ST/'deployment-final-source-v822'/name) == row['sha256'], 'staged source differs: ' + name)
    if not SRC.exists():
        tmp = Path(tempfile.mkdtemp(prefix='.src-', dir=OUT))
        for name in release['source']:
            d = tmp/name; d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ST/'deployment-final-source-v822'/name, d)
        tmp.rename(SRC)
    need({str(p.relative_to(SRC)) for p in SRC.rglob('*') if p.is_file()} == set(release['source']), 'final source tree')
    for name, row in release['source'].items(): need(hf(SRC/name) == row['sha256'], 'final source copy differs: ' + name)
    SRC = SRC.resolve(); sys.path.insert(0, str(SRC))
    from subnet.storage import Identity, canonical, sha
    from subnet.backend_jobs import signed
    from subnet.mimo_release_policy import ReleasePolicy, QUALIFICATIONS, QUALIFICATION_VERSION, sign
    from subnet.mimo_release_succession import ReleaseChain, SUCCESSION_VERSION
    from subnet.mimo_worker import source_inventory
    from subnet.mimo_grpo import TrainerConfig
    digest = lambda v: sha(canonical(v))
    need(QUALIFICATIONS == KINDS, 'qualification categories')
    inv = source_inventory(); need(digest(inv) == binding['source_inventory_sha256'], 'source inventory d0b0475d')
    draft = read(ST/'PUBLIC-POLICY.v8.2.2.UNSIGNED.private.json')
    need(draft['version'] == 'mimo-qualified-release-policy-v1' and draft['mode'] == 'public', 'policy version/mode')
    need(draft['scope_sha256'] == SCOPE == digest(draft['scope']), 'scope 32526ff1')
    need(draft['scope']['immutable_manifest']['source_bundle']['sha256'] == ARCHIVE, 'scope binds v8.2.1 archive')
    ReleasePolicy._validate_public_protocol(draft['scope'])
    quals = read(ST/'QUALIFICATIONS.v8.2.2.UNSIGNED.private.json')
    need(quals['scope_sha256'] == SCOPE and quals['candidate_archive_sha256'] == ARCHIVE and quals['ready'] == 7, 'packet 7/7')
    payloads = quals['payloads']; need(len(payloads) == 7 and {p['kind'] for p in payloads} == KINDS, 'seven kinds')
    for p in payloads:
        need(set(p) == {'version','scope_sha256','kind','passed','evidence_sha256'} and p['version'] == QUALIFICATION_VERSION
             and p['scope_sha256'] == SCOPE and p['passed'] is True, 'payload shape ' + p['kind'])
        need(hf(RA/quals['status'][p['kind']]['evidence']) == p['evidence_sha256'], 'evidence digest ' + p['kind'])
    life = read(ST/'lifecycle.v8.2.2.UNSIGNED.private.json')
    need(life['lifecycle']['max_tasks'] == _B['max_tasks'] == life['trainer_config']['max_tasks'] and life['adapters']['job_seconds'] == 7200 and life['audit_scheduler']['max_execution_retries'] == 3, 'v8.2.2 limits')
    TrainerConfig(**life['trainer_config']).validate()
    need(inv == life['adapters']['source_files'] and ARCHIVE in life['adapters']['source_bundle_key'], 'lifecycle source/archive')
    life82 = read(S82/'lifecycle.v8.2.1.ACTIVE.private.json'); need(digest(life82['release_policy']) == P82, 'live v8.2 policy')
    chain82 = life82['release_succession']; c82 = signed(chain82, ROOT_ID)
    need([e['first_round'] for e in c82['entries']] == [189, 228, 229] and digest(c82['entries'][2]['policy']) == P82, 'v8.2.1 chain')
    cap82 = read(S82/'capture-operator.v8.2.1.ROOT-ADMITTED.private.json'); uid82 = read(S82/'uid-operator.v8.2.1.ROOT-ADMITTED.private.json')
    reader82 = signed(read(S82/'writer-reader.v8.2.1.ROOT-SIGNED.private.json'), ROOT_ID)
    comp82 = signed(read(S82/'writer-composition.v8.2.1.ROOT-SIGNED.private.json'), ROOT_ID)
    host82 = signed(read(S82/'HOST-COMPOSITION.v8.2.1.ROOT-SIGNED.private.json'), ROOT_ID)
    hacc82 = signed(read(S82/'HOST-COMPOSITION.v8.2.1-accel.ROOT-SIGNED.private.json'), ROOT_ID)
    need(hf(SRC/'ops/mimo_hourly_writer.py') == comp82['entry']['sha256'], 'hourly writer entry unchanged vs v8.2.1')
    need(hf(SRC/'ops/mimo_lifecycle.py') == hf(D/'v822-tree/ops/mimo_lifecycle.py') != cap82['base']['entry']['sha256'], 'lifecycle entry = archive-scoped audits a7bc7f77')
    ACC = ST/'writer-accel-v822'
    note(step=0, verified=True, scope=SCOPE, archive=ARCHIVE, files=len(release['source']), first_round=a.first_round, out=str(OUT))
    if a.dry_run: actor = Identity(); A = actor.id
    else:
        seed_path = Path(life82['authority_seed_file']); need(seed_path == Path(life['authority_seed_file']), 'same authority_seed_file')
        try: actor = Identity(bytes.fromhex(seed_path.read_text().strip()))
        except ValueError: raise SystemExit('REFUSED: authority seed file is not 32-byte hex')
        A = ROOT_ID; need(actor.id == ROOT_ID, 'key does not derive ROOT authority 3301134b')
    def put(name, value, raw=None):
        raw = raw if raw is not None else canonical(value); p = OUT/name
        if p.exists(): need(p.read_bytes() == raw, 'existing output differs: ' + name)
        else:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        return p
    receipts = [sign(actor, p) for p in sorted(payloads, key=lambda p: p['kind'])]
    qp = put('QUALIFICATIONS.v8.2.2.ROOT-SIGNED.private.json', receipts); note(step=1, file=str(qp), sha256=hf(qp))
    pol = dict(draft); pol.update(enabled=True, qualifications=receipts)
    pdoc = sign(actor, pol); pp = put('PUBLIC-POLICY.v8.2.2.ROOT-SIGNED.private.json', pdoc)
    policy = ReleasePolicy(pdoc, A, allow_public=True); need(policy.payload['scope_sha256'] == SCOPE, 'signed policy scope')
    note(step=2, file=str(pp), sha256=hf(pp), policy_sha256=policy.policy_sha256)
    members = [e['policy'] for e in c82['entries']]
    if a.dry_run:
        def resign(doc):
            p = dict(signed(doc, ROOT_ID)); p['qualifications'] = [sign(actor, signed(q, ROOT_ID)) for q in p['qualifications']]
            return sign(actor, p)
        members = [resign(m) for m in members]
    chain_payload = dict(version=SUCCESSION_VERSION, run_id=policy.run_id, entries=[
        dict(policy=members[0], first_round=189), dict(policy=members[1], first_round=228), dict(policy=members[2], first_round=229), dict(policy=pdoc, first_round=a.first_round)])
    cdoc = sign(actor, chain_payload); chp = put('RELEASE-SUCCESSION.v7-v8.2-v8.2.1-v8.2.2.ROOT-SIGNED.private.json', cdoc)
    chain = ReleaseChain(cdoc, A, allow_public=True)
    need(chain.active.policy_sha256 == policy.policy_sha256 and chain.succeeds(digest(members[2]), policy.policy_sha256) and chain.succeeds(digest(members[1]), digest(members[2]))
         and chain.succeeds(digest(members[0]), digest(members[1])), 'chain order')
    note(step=3, file=str(chp), sha256=hf(chp), chain_sha256=chain.chain_sha256, members=[digest(m) for m in members] + [policy.policy_sha256],
         first_rounds=[189, 228, 229, a.first_round])
    reader = dict(reader82); reader['approved_execution'] = dict(reader82['approved_execution'])
    need(set(reader['approved_execution']) == {A82, A821, '69ae058be865ab47ccf7f4decbe855c4826bc87c96d87aa1173d69ef6fbab2f9'}, 'v8.2.1 reader archives')
    need(reader['approved_execution'][A821]['source_files'] == inv, 'v8.2.1 archive inventory == running closure')
    reader['approved_execution'][ARCHIVE] = dict(runtime_versions=dict(reader82['approved_execution'][A82]['runtime_versions']), source_files=inv)
    reader['release_policy'] = cdoc
    need(life['archive_execution'] == reader['approved_execution'], 'lifecycle archive_execution == writer approved_execution')
    rp = put('writer-reader.v8.2.2.ROOT-SIGNED.private.json', sign(actor, reader))
    from subnet.mimo_assessment_bridge import EvidenceSource
    def not_called(*x, **k): raise AssertionError('schema-only')
    es = EvidenceSource(read(rp), A, snapshot=not_called, current_estimates=not_called, observation_reader=not_called, allow_public=True)
    need(type(es.release).__name__ == 'ReleaseChain', 'reader release is the chain')
    note(step=4, file=str(rp), sha256=hf(rp), approved_archives=sorted(reader['approved_execution']))
    comp = dict(comp82); comp.update(entry=pin(SRC/'ops/mimo_hourly_writer.py'), new_source_root=str(SRC), new_source_files=inv, reader_policy=pin(rp))
    cp = put('writer-composition.v8.2.2.ROOT-SIGNED.private.json', sign(actor, comp))
    host = dict(host82); host.update(parent_composition=pin(cp), parent_entry=pin(SRC/'ops/mimo_hourly_writer.py'))
    hp = put('HOST-COMPOSITION.v8.2.2.ROOT-SIGNED.private.json', sign(actor, host))
    accdir = OUT/'writer-accel-v822'
    if not accdir.exists():
        accdir.mkdir(mode=0o700)
        for n in ('mimo_population_cache.py', 'population_derivations.py'): shutil.copy2(ACC/n, accdir/n); os.chmod(accdir/n, 0o600)
    for n in ('mimo_population_cache.py', 'population_derivations.py'): need(hf(accdir/n) == hf(ACC/n), 'accel copy ' + n)
    hacc = dict(host); hacc.update(mimo_module=pin(accdir/'mimo_population_cache.py'), derivation_module=pin(accdir/'population_derivations.py'))
    need(set(hacc) == set(hacc82), 'accel host composition fields')
    hap = put('HOST-COMPOSITION.v8.2.2-accel.ROOT-SIGNED.private.json', sign(actor, hacc))
    note(step=5, composition_sha256=hf(cp), host_sha256=hf(hp), host_accel=str(hap), host_accel_sha256=hf(hap))
    life['release_policy'] = pdoc; life['release_succession'] = cdoc; life['enabled'] = True
    life['lifecycle'].update(first_round=life82['lifecycle']['first_round'], previous_discovery_sha256=life82['lifecycle']['previous_discovery_sha256'])
    lp = put('lifecycle.v8.2.2.ACTIVE.private.json', life)
    cap = {k: v for k, v in cap82.items() if k != 'admission'}; cap['base'] = dict(cap['base'])
    cap['base'].update(argv=['--config', str(lp), '--allow-public-mimo'], entry=pin(SRC/'ops/mimo_lifecycle.py'), lifecycle_config=pin(lp), source_root=str(SRC))
    cap['admission'] = sign(actor, dict(version=cap82['version'], configuration_sha256=sha(canonical({k: v for k, v in cap.items() if k != 'admission'})),
                                        source_unchanged=True, scope='host-capture-scheduling-only'))
    capp = put('capture-operator.v8.2.2.ROOT-ADMITTED.private.json', cap)
    uid = {k: v for k, v in uid82.items() if k != 'admission'}; uid['parent_operator'] = pin(capp)
    uid['admission'] = sign(actor, dict(version=uid82['version'], configuration_sha256=sha(canonical({k: v for k, v in uid.items() if k != 'admission'})),
                                        source_unchanged=True, scope='host-registration-capacity-only', capacity=1000))
    uidp = put('uid-operator.v8.2.2.ROOT-ADMITTED.private.json', uid)
    trp = put('TRAINER-ROLE-RELEASE-SUCCESSION.v8.2.2.private.json', dict(release_succession=cdoc))
    actor = None
    note(step=6, lifecycle_sha256=hf(lp), capture_sha256=hf(capp), uid_operator=str(uidp), uid_operator_sha256=hf(uidp), trainer_role_field_sha256=hf(trp))
    def check(argv, label):
        r = subprocess.run([PY, '-B', *map(str, argv)], capture_output=True, text=True, timeout=1200, cwd=str(OUT))
        ok = r.returncode == 0; note(check=label, ok=ok, out=(r.stdout.strip().splitlines() or [''])[-1][:400], err='' if ok else r.stderr.strip()[-800:]); return ok
    ok = check([UID_DIR/'mimo_uid_capacity_launcher.py', '--operator-config', uidp, '--check'], 'lifecycle-launcher')
    if not a.dry_run:
        ok &= check(['-I', HOST_DIR/'hourly_population_index_entry_v6.py', '--policy', hap, '--authority', A, '--check'], 'hourly-writer-accel')
    summary = dict(version='mimo-v822-root-signing-result-v1', dry_run=a.dry_run, authority=A, scope_sha256=SCOPE, archive_sha256=ARCHIVE,
                   policy_sha256=policy.policy_sha256, chain_sha256=chain.chain_sha256, first_round=a.first_round, checks_passed=bool(ok),
                   steps=log, services_changed=False, chain_calls=0)
    if not (OUT/'SIGNING-RESULT.json').exists(): put('SIGNING-RESULT.json', summary, raw=(json.dumps(summary, indent=1, sort_keys=True) + '\n').encode())
    print(json.dumps(dict(done=True, dry_run=a.dry_run, checks_passed=bool(ok), out=str(OUT), policy_sha256=policy.policy_sha256, chain_sha256=chain.chain_sha256)))
    if a.dry_run: shutil.rmtree(OUT)
    sys.exit(0 if ok else 2)
if __name__ == '__main__': main()
