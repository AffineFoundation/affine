#!/usr/bin/env python3
"""SN120 MiMo v8.2.5 = v8.2.4 trainer host layer + engine.drop_unresolved=true (ONLY change).
Lifecycle config / trainer_config / policy dfe0cbf7 / closure / writer / scorer / chain: UNCHANGED.
Follows sign_v824.py: ROOT 3301134b, on-box authority seed named by the live lifecycle config (never printed/copied),
proposal shape of trainer_host_config.proposal, new files only (O_EXCL). No services touched."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT_ID = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
SRC = RA/'20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822'
V824 = RA/'20261017-v824/release-v824-signed'
D = RA/'20261021-v825-drop-unresolved'; STAGE = D/'stage/trainer-host-v825'; OUT = D/'release-v825-signed'
POD = '/workspace/affine-mimo-trainer-prod'; POD_SOURCE = POD + '/source-final-v822b'
def need(ok, why):
    if not ok: raise SystemExit('REFUSED: ' + why)
hf = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    need(subprocess.run(['sha256sum', '--quiet', '-c', 'SIGNED-MANIFEST.sha256'], cwd=V824).returncode == 0, 'v8.2.4 signed manifest')
    sys.path.insert(0, str(SRC))
    from subnet.storage import Identity, canonical, sha
    from subnet.backend_jobs import signed
    from subnet.mimo_release_policy import sign
    import importlib.util
    spec = importlib.util.spec_from_file_location('v825_launcher_probe', STAGE/'mimo_host_trainer_v825.py')
    L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
    old = json.loads((V824/'trainer-host-v824/TRAINER-HOST.v8.2.4.ROOT-ADMITTED.private.json').read_text())
    signed(old['admission'], ROOT_ID)
    # unchanged host files must be byte-identical to v8.2.4
    for n in L.HOST_FILES:
        if n in old['host_files']: need(hf(STAGE/n) == old['host_files'][n], 'unchanged v8.2.4 host file ' + n)
    import subnet.mimo_grpo.transaction as T, subnet.mimo_grpo.publication as PB, subnet.mimo_grpo.service as SV
    import subnet.mimo_grpo.optimizer as OP, subnet.mimo_training_worker as W
    objs = {'mimo_grpo.transaction._receipt': T._receipt, 'mimo_grpo.transaction.Transaction.run': T.Transaction.run,
            'mimo_grpo.publication.validate_publication': PB.validate_publication,
            'mimo_grpo.service.execute_transaction': SV.execute_transaction,
            'mimo_training_worker.validate_result': W.validate_result, 'mimo_grpo.optimizer.CPUAdamW.step': OP.CPUAdamW.step,
            'mimo_grpo.publication.R2ModelPublisher.__init__': PB.R2ModelPublisher.__init__}
    pins = {k: L.method_sha(v) for k, v in objs.items()}
    need(pins == old['original_methods'], 'original method pins identical to v8.2.4')
    engine = dict(old['engine'], drop_unresolved=True)
    prop = dict(version=L.VERSION, scope=L.SCOPE, authority=ROOT_ID, source_root=POD_SOURCE,
                host_files={n: hf(STAGE/n) for n in L.HOST_FILES}, original_methods=pins, engine=engine)
    life = json.loads((V824/'lifecycle.v8.2.4.ACTIVE.private.json').read_text())
    seed_path = Path(life['authority_seed_file'])
    need(seed_path.name == 'authority.seed' and 'live-controller-v1/controller-state' in str(seed_path), 'authority seed path')
    try: actor = Identity(bytes.fromhex(seed_path.read_text().strip()))
    except ValueError: raise SystemExit('REFUSED: authority seed file is not 32-byte hex')
    need(actor.id == ROOT_ID, 'authority id')
    cfg = dict(prop, admission=sign(actor, dict(version=L.VERSION, configuration_sha256=sha(canonical(prop)),
                                                source_unchanged=True, scope=L.SCOPE)))
    actor = None
    payload = signed(cfg['admission'], ROOT_ID)
    need(payload['configuration_sha256'] == sha(canonical(prop)), 'admission binds proposal')
    OUT.mkdir(mode=0o700); (OUT/'trainer-host-v825').mkdir(mode=0o700); (OUT/'trainer-host-v825/patched').mkdir(mode=0o700)
    def put(rel, raw):
        fd = os.open(OUT/rel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        return OUT/rel
    for n in L.HOST_FILES: put('trainer-host-v825/' + n, (STAGE/n).read_bytes())
    tp = put('trainer-host-v825/TRAINER-HOST.v8.2.5.ROOT-ADMITTED.private.json', canonical(cfg))
    delta = sorted(k for k in set(old) | set(cfg) if k != 'admission' and old.get(k) != cfg.get(k))
    summary = dict(version='mimo-v825-root-signing-result-v1', authority=ROOT_ID, trainer_host=str(tp),
                   trainer_host_sha256=hf(tp), configuration_sha256=payload['configuration_sha256'],
                   change=dict(engine_drop_unresolved=[None, True], new_host_files=['mimo_host_trainer_v825.py', 'v825_batched.py'],
                               config_delta_vs_v824=delta, lifecycle_changed=False, trainer_config_changed=False),
                   original_method_pins_unchanged=True, checks_passed=True, services_changed=False, chain_calls=0)
    put('SIGNING-RESULT.json', (json.dumps(summary, indent=1, sort_keys=True) + '\n').encode())
    subprocess.run('find . -type f ! -name SIGNED-MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > SIGNED-MANIFEST.sha256 && chmod 600 SIGNED-MANIFEST.sha256',
                   shell=True, cwd=OUT, check=True)
    print(json.dumps(summary, indent=1))
if __name__ == '__main__': main()
