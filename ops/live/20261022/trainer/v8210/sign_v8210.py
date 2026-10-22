#!/usr/bin/env python3
"""SN120 MiMo trainer host v8.2.10 = v8.2.5 + engine.optimizer_persistence (ONLY change).
Authorized by Jacob (owner) 2026-10-22: "saving the optimizer state should not be important can we remove this
requirement"; standing order "Push forward with any changes you need to make the learning stable and working".
Main config: optimizer_persistence='none'. Variant -sync: 'synchronous' (= exact v8.2.5 staging/recovery).
Lifecycle config / trainer_config / policy / closure / writer / scorer / chain: UNCHANGED.
Follows sign_v825.py: ROOT 3301134b, on-box authority seed named by the live lifecycle config (never printed/copied),
new files only (O_EXCL). No services touched."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT_ID = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
SRC = RA/'20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822'
V824 = RA/'20261017-v824/release-v824-signed'
V825 = RA/'20261021-v825-drop-unresolved/release-v825-signed'
D = RA/'20261022-async-optimizer-save'; STAGE = D/'stage/trainer-host-v8210'; OUT = D/'release-v8210-signed'
POD = '/workspace/affine-mimo-trainer-prod'; POD_SOURCE = POD + '/source-final-v822b'
NEW_SITE = 'mimo_grpo.recovery.restore_transaction_state'
def need(ok, why):
    if not ok: raise SystemExit('REFUSED: ' + why)
hf = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    for rel in (V824, V825):
        need(subprocess.run(['sha256sum', '--quiet', '-c', 'SIGNED-MANIFEST.sha256'], cwd=rel).returncode == 0, 'signed manifest ' + rel.name)
    need(not OUT.exists(), 'output exists (never overwrite)')
    sys.path.insert(0, str(SRC))
    from subnet.storage import Identity, canonical, sha
    from subnet.backend_jobs import signed
    from subnet.mimo_release_policy import sign
    import importlib.util
    spec = importlib.util.spec_from_file_location('v8210_launcher_probe', STAGE/'mimo_host_trainer_v8210.py')
    L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
    old = json.loads((V825/'trainer-host-v825/TRAINER-HOST.v8.2.5.ROOT-ADMITTED.private.json').read_text())
    signed(old['admission'], ROOT_ID)
    for n in L.HOST_FILES:
        if n in old['host_files']: need(hf(STAGE/n) == old['host_files'][n], 'unchanged v8.2.5 host file ' + n)
    new_files = sorted(set(L.HOST_FILES) - set(old['host_files']))
    need(new_files == ['mimo_host_trainer_v8210.py', 'v8210_optimizer_offpath.py'], 'exactly two new host files')
    import subnet.mimo_grpo.transaction as T, subnet.mimo_grpo.publication as PB, subnet.mimo_grpo.service as SV
    import subnet.mimo_grpo.optimizer as OP, subnet.mimo_grpo.recovery as RC, subnet.mimo_training_worker as W
    objs = {'mimo_grpo.transaction._receipt': T._receipt, 'mimo_grpo.transaction.Transaction.run': T.Transaction.run,
            'mimo_grpo.publication.validate_publication': PB.validate_publication,
            'mimo_grpo.service.execute_transaction': SV.execute_transaction,
            'mimo_training_worker.validate_result': W.validate_result, 'mimo_grpo.optimizer.CPUAdamW.step': OP.CPUAdamW.step,
            'mimo_grpo.publication.R2ModelPublisher.__init__': PB.R2ModelPublisher.__init__,
            NEW_SITE: RC.restore_transaction_state}
    need(set(objs) == set(L.SITES), 'site inventory')
    pins = {k: L.method_sha(v) for k, v in objs.items()}
    need({k: v for k, v in pins.items() if k != NEW_SITE} == old['original_methods'], 'original method pins identical to v8.2.5')
    life = json.loads((V824/'lifecycle.v8.2.4.ACTIVE.private.json').read_text())
    seed_path = Path(life['authority_seed_file'])
    need(seed_path.name == 'authority.seed' and 'live-controller-v1/controller-state' in str(seed_path), 'authority seed path')
    try: actor = Identity(bytes.fromhex(seed_path.read_text().strip()))
    except ValueError: raise SystemExit('REFUSED: authority seed file is not 32-byte hex')
    need(actor.id == ROOT_ID, 'authority id')
    cfgs = {}
    for label, mode in (('', 'none'), ('-sync', 'synchronous')):
        engine = dict(old['engine'], optimizer_persistence=mode)
        prop = dict(version=L.VERSION, scope=L.SCOPE, authority=ROOT_ID, source_root=POD_SOURCE,
                    host_files={n: hf(STAGE/n) for n in L.HOST_FILES}, original_methods=pins, engine=engine)
        cfg = dict(prop, admission=sign(actor, dict(version=L.VERSION, configuration_sha256=sha(canonical(prop)),
                                                    source_unchanged=True, scope=L.SCOPE)))
        payload = signed(cfg['admission'], ROOT_ID)
        need(payload['configuration_sha256'] == sha(canonical(prop)), 'admission binds proposal')
        cfgs[label] = (cfg, payload)
    actor = None
    OUT.mkdir(mode=0o700); (OUT/'trainer-host-v8210').mkdir(mode=0o700); (OUT/'trainer-host-v8210/patched').mkdir(mode=0o700)
    def put(rel, raw):
        fd = os.open(OUT/rel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        return OUT/rel
    for n in L.HOST_FILES: put('trainer-host-v8210/' + n, (STAGE/n).read_bytes())
    written = {}
    for label, (cfg, payload) in cfgs.items():
        tp = put(f'trainer-host-v8210/TRAINER-HOST.v8.2.10{label}.ROOT-ADMITTED.private.json', canonical(cfg))
        written[label or 'main'] = dict(path=str(tp), sha256=hf(tp), configuration_sha256=payload['configuration_sha256'],
                                        optimizer_persistence=cfg['engine']['optimizer_persistence'])
    main_cfg = cfgs[''][0]
    delta = sorted(k for k in set(old) | set(main_cfg) if k != 'admission' and old.get(k) != main_cfg.get(k))
    summary = dict(version='mimo-v8210-root-signing-result-v1', authority=ROOT_ID, trainer_host=written,
                   change=dict(engine_optimizer_persistence=[None, 'none'], new_host_files=new_files,
                               new_original_method_pin={NEW_SITE: pins[NEW_SITE]},
                               config_delta_vs_v825=delta, lifecycle_changed=False, trainer_config_changed=False,
                               published_model_bytes_changed=False),
                   original_method_pins_v825_unchanged=True, checks_passed=True, services_changed=False, chain_calls=0)
    put('SIGNING-RESULT.json', (json.dumps(summary, indent=1, sort_keys=True) + '\n').encode())
    subprocess.run('find . -type f ! -name SIGNED-MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > SIGNED-MANIFEST.sha256 && chmod 600 SIGNED-MANIFEST.sha256',
                   shell=True, cwd=OUT, check=True)
    print(json.dumps(summary, indent=1))
if __name__ == '__main__': main()
