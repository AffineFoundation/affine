#!/usr/bin/env python3
"""SN120 MiMo trainer host v8.3.2 = v8.3.1 + optimizer drift control (launcher only):
  drift_cap_rel=2e-5 (per-tensor and global ||dW||/||W|| after each optimizer step) and restart LR x0.5 for the first 3 optimizer
  steps after fresh AdamW moments. Both are optional engine keys, OFF when absent (code default); the main config turns them ON,
  the companion '-off' config is the same v8.3.2 launcher with the exact v8.3.1 engine (flags absent = v8.3.1 step bit-for-bit).
Authorized by Jacob (owner) 2026-10-26 13:06 CR: "Ship the drift cap and gentler restarts to prod now".
Grad clip 1.0, loss, data, scoring, weight writer, TOPLOC, nonce, lifecycle, trainer_config: UNCHANGED.
Follows sign_v831.py: ROOT 3301134b, on-box authority seed named by the live lifecycle config (never printed/copied),
new files only (O_EXCL). No services touched.   usage: sign_v832.py <evidence.json>"""
import hashlib, json, os, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT_ID = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
SRC = RA/'20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822'
V824 = RA/'20261017-v824/release-v824-signed'
V831 = RA/'20261025-shadow-trainer/release-v831-signed'
D = RA/'20261026-gradnorm'; STAGE = D/'stage/trainer-host-v832'; OUT = D/'release-v832-signed'
POD_SOURCE = '/workspace/affine-mimo-trainer-prod/source-final-v822b'
DRIFT = dict(drift_cap_rel=2e-5, restart_lr_warm_steps=3, restart_lr_scale=0.5)
def need(ok, why):
    if not ok: raise SystemExit('REFUSED: ' + why)
hf = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    evidence = Path(sys.argv[1]); need(evidence.is_file(), 'evidence')
    for rel in (V824, V831):
        need(subprocess.run(['sha256sum', '--quiet', '-c', 'SIGNED-MANIFEST.sha256'], cwd=rel).returncode == 0, 'signed manifest ' + rel.name)
    need(not OUT.exists(), 'output exists (never overwrite)')
    sys.path.insert(0, str(SRC))
    from subnet.storage import Identity, canonical, sha
    from subnet.backend_jobs import signed
    from subnet.mimo_release_policy import sign
    import importlib.util
    spec = importlib.util.spec_from_file_location('v832_launcher_probe', STAGE/'mimo_host_trainer_v83.py')
    L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
    need(L.VERSION == 'affine-mimo-trainer-host-v832' and L.SCOPE == 'sn120-mimo-trainer-host-overrides-v8.3.2', 'v832 identity')
    old = json.loads((V831/'trainer-host-v831/TRAINER-HOST.v8.3.1.ROOT-ADMITTED.private.json').read_text())
    signed(old['admission'], ROOT_ID)
    need(old['version'] == 'affine-mimo-trainer-host-v831', 'base is v8.3.1')
    need(set(L.HOST_FILES) == set(old['host_files']), 'same host file inventory as v8.3.1')
    changed = sorted(n for n in L.HOST_FILES if hf(STAGE/n) != old['host_files'][n])
    need(changed == ['mimo_host_trainer_v83.py'], 'only the launcher changed vs v8.3.1: %r' % changed)
    need(set(L.SITES) == set(old['original_methods']), 'site inventory unchanged')
    pins = dict(old['original_methods'])
    import subnet.mimo_grpo.transaction as T, subnet.mimo_grpo.publication as PB, subnet.mimo_grpo.service as SV
    import subnet.mimo_grpo.optimizer as OP, subnet.mimo_grpo.recovery as RC, subnet.mimo_training_worker as W
    objs = {'mimo_grpo.transaction._receipt': T._receipt, 'mimo_grpo.transaction.Transaction.run': T.Transaction.run,
            'mimo_grpo.publication.validate_publication': PB.validate_publication,
            'mimo_grpo.service.execute_transaction': SV.execute_transaction,
            'mimo_training_worker.validate_result': W.validate_result, 'mimo_grpo.optimizer.CPUAdamW.step': OP.CPUAdamW.step,
            'mimo_grpo.publication.R2ModelPublisher.__init__': PB.R2ModelPublisher.__init__,
            'mimo_grpo.recovery.restore_transaction_state': RC.restore_transaction_state,
            'mimo_training_worker.prepare_signed_dispatch': W.prepare_signed_dispatch}
    need(set(objs) == set(L.SITES), 'site objects')
    need({k: L.method_sha(v) for k, v in objs.items()} == pins, 'original method pins identical to v8.3.1')
    need(set(old['engine']) == L.ENGINE_KEYS and not set(old['engine']) & L.DRIFT_KEYS, 'v8.3.1 engine')
    engines = {'on': dict(old['engine'], **DRIFT), 'off': dict(old['engine'])}
    need(L.drift_settings(engines['on']) == (2e-5, 3, 0.5) and L.drift_settings(engines['off']) == (None, 0, 1.0), 'drift settings')
    for e in engines.values():
        need({k: v for k, v in e.items() if k not in L.DRIFT_KEYS} == old['engine'], 'every v8.3.1 engine key unchanged')
    life = json.loads((V824/'lifecycle.v8.2.4.ACTIVE.private.json').read_text())
    seed_path = Path(life['authority_seed_file'])
    need(seed_path.name == 'authority.seed' and 'live-controller-v1/controller-state' in str(seed_path), 'authority seed path')
    need(seed_path.is_file() and not seed_path.is_symlink() and seed_path.stat().st_mode & 0o077 == 0, 'seed mode')
    try: actor = Identity(bytes.fromhex(seed_path.read_text().strip()))
    except ValueError: raise SystemExit('REFUSED: authority seed file is not 32-byte hex')
    need(actor.id == ROOT_ID, 'authority id')
    cfgs = {}
    for kind, engine in engines.items():
        prop = dict(version=L.VERSION, scope=L.SCOPE, authority=ROOT_ID, source_root=POD_SOURCE,
                    host_files={n: hf(STAGE/n) for n in L.HOST_FILES}, original_methods=pins, engine=engine)
        cfg = dict(prop, admission=sign(actor, dict(version=L.VERSION, configuration_sha256=sha(canonical(prop)),
                                                    source_unchanged=True, scope=L.SCOPE)))
        need(signed(cfg['admission'], ROOT_ID)['configuration_sha256'] == sha(canonical(prop)), 'admission binds proposal')
        cfgs[kind] = cfg
    actor = None
    OUT.mkdir(mode=0o700); (OUT/'trainer-host-v832').mkdir(mode=0o700); (OUT/'trainer-host-v832/patched').mkdir(mode=0o700)
    def put(rel, raw):
        fd = os.open(OUT/rel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        return OUT/rel
    for n in L.HOST_FILES: put('trainer-host-v832/' + n, (STAGE/n).read_bytes())
    put('EVIDENCE.json', evidence.read_bytes())
    paths = {'on': put('trainer-host-v832/TRAINER-HOST.v8.3.2.ROOT-ADMITTED.private.json', canonical(cfgs['on'])),
             'off': put('trainer-host-v832/TRAINER-HOST.v8.3.2-off.ROOT-ADMITTED.private.json', canonical(cfgs['off']))}
    summary = dict(version='mimo-v832-root-signing-result-v1', authority=ROOT_ID,
                   trainer_host={k: dict(path=str(p), sha256=hf(p), configuration_sha256=sha(canonical({x: y for x, y in cfgs[k].items() if x != 'admission'})),
                                         engine=engines[k]) for k, p in paths.items()},
                   change=dict(changed_host_files=['mimo_host_trainer_v83.py'], launcher_sha256=hf(STAGE/'mimo_host_trainer_v83.py'),
                               engine_new_optional_keys=sorted(L.DRIFT_KEYS), main_engine_flags=DRIFT, off_engine='exact v8.3.1 engine',
                               grad_clip='1.0 unchanged', lifecycle_changed=False, trainer_config_changed=False,
                               loss_data_scoring_writer_toploc_nonce_changed=False),
                   base='release-v831-signed', original_method_pins_v831_unchanged=True,
                   evidence_sha256=hf(evidence), checks_passed=True, services_changed=False, chain_calls=0,
                   authorized="Jacob 2026-10-26 13:06 CR 'Ship the drift cap and gentler restarts to prod now' via parent agent")
    put('SIGNING-RESULT.json', (json.dumps(summary, indent=1, sort_keys=True) + '\n').encode())
    subprocess.run('find . -type f ! -name SIGNED-MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > SIGNED-MANIFEST.sha256 && chmod 600 SIGNED-MANIFEST.sha256',
                   shell=True, cwd=OUT, check=True)
    print(json.dumps(summary, indent=1))
if __name__ == '__main__': main()
