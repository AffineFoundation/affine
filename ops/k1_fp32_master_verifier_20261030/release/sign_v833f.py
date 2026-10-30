#!/usr/bin/env python3
"""SN120 MiMo trainer host v8.3.3f = live v8.3.3u + persistent fp32 head master (engine.sgd_fp32_master=true).
Only the launcher mimo_host_trainer_v83.py changes (adds the optional key; flag-off path byte-for-byte the v8.3.3 step); every other
host file, the source root (source-final-v823u), the original-method pins and every other engine key are identical to v8.3.3u.
Step size 1x (lr 5e-7), per-tensor rel cap 2e-5, grad clip 1.0, fresh zero moments every step: UNCHANGED.
Authorized by Jacob 2026-10-30 (t821u-t823u, t829u design; t831u/pt831u "push the changes ahorita"). ROOT 3301134b from the live lifecycle
config's seed file (never printed). O_EXCL new files only. No services touched.  --dry-run: throwaway key, temp dir."""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, importlib.util
from pathlib import Path
sys.dont_write_bytecode = True
ROOT_ID = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
RA = Path('/home/const/subnet120-rewrite/state/root-audits'); D = RA/'20261030-fp32-verifier'
SRC = RA/'20261029-v829u-125task/deployment-final-source-v823u'
B = RA/'20261027-k1-per-uid/trainer-v833/release-v833u-signed'
STAGE = D/'stage/mimo_host_trainer_v83.py'
LIFE = RA/'20261017-v824/release-v824-signed/lifecycle.v8.2.4.ACTIVE.private.json'
def need(ok, why):
    if not ok: raise SystemExit('REFUSED: ' + why)
hf = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
    need(subprocess.run(['sha256sum', '--quiet', '-c', 'SIGNED-MANIFEST.sha256'], cwd=B).returncode == 0, 'v833u manifest')
    sys.path.insert(0, str(SRC))
    from subnet.storage import Identity, canonical, sha
    from subnet.backend_jobs import signed
    from subnet.mimo_release_policy import sign
    H = B/'trainer-host-v833'
    spec = importlib.util.spec_from_file_location('v833f_launcher_probe', STAGE)
    L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
    old = json.loads((H/'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json').read_text()); signed(old['admission'], ROOT_ID)
    need(old['source_root'] == '/workspace/affine-mimo-trainer-prod/source-final-v823u', 'base = v833u')
    need(L.VERSION == old['version'] and L.SCOPE == old['scope'] and set(L.HOST_FILES) == set(old['host_files']), 'same identity/inventory')
    files = {n: (hf(STAGE) if n == 'mimo_host_trainer_v83.py' else hf(H/n)) for n in L.HOST_FILES}
    need([n for n in files if files[n] != old['host_files'][n]] == ['mimo_host_trainer_v83.py'], 'only the launcher changed')
    import subnet.mimo_grpo.transaction as T, subnet.mimo_grpo.publication as PB, subnet.mimo_grpo.service as SV
    import subnet.mimo_grpo.optimizer as OP, subnet.mimo_grpo.recovery as RC, subnet.mimo_training_worker as W
    objs = {'mimo_grpo.transaction._receipt': T._receipt, 'mimo_grpo.transaction.Transaction.run': T.Transaction.run,
            'mimo_grpo.publication.validate_publication': PB.validate_publication, 'mimo_grpo.service.execute_transaction': SV.execute_transaction,
            'mimo_training_worker.validate_result': W.validate_result, 'mimo_grpo.optimizer.CPUAdamW.step': OP.CPUAdamW.step,
            'mimo_grpo.publication.R2ModelPublisher.__init__': PB.R2ModelPublisher.__init__,
            'mimo_grpo.recovery.restore_transaction_state': RC.restore_transaction_state,
            'mimo_training_worker.prepare_signed_dispatch': W.prepare_signed_dispatch}
    need(set(objs) == set(L.SITES) == set(old['original_methods']), 'site inventory')
    need({k: L.method_sha(v) for k, v in objs.items()} == old['original_methods'], 'original method pins identical')
    engine = dict(old['engine'], sgd_fp32_master=True)
    need(L.fp32_settings(engine) is True and L.sgd_settings(engine) == (5e-07, 1.0, 'fresh-adam', ()) and engine['sgd_tensor_rel_cap'] == 2e-05, 'engine')
    need({k: v for k, v in engine.items() if k != 'sgd_fp32_master'} == old['engine'], 'every other engine key unchanged')
    if a.dry_run: actor = Identity(); A = actor.id
    else:
        life = json.loads(LIFE.read_text()); seed_path = Path(life['authority_seed_file'])
        need(seed_path.is_file() and not seed_path.is_symlink() and seed_path.stat().st_mode & 0o077 == 0, 'seed mode')
        try: actor = Identity(bytes.fromhex(seed_path.read_text().strip()))
        except ValueError: raise SystemExit('REFUSED: authority seed file is not 32-byte hex')
        A = ROOT_ID; need(actor.id == ROOT_ID, 'authority id')
    prop = {k: v for k, v in old.items() if k != 'admission'}; prop['host_files'] = files; prop['engine'] = engine
    pay = dict(signed(old['admission'], ROOT_ID)); pay['configuration_sha256'] = sha(canonical(prop))
    cfg = dict(prop, admission=sign(actor, pay)); actor = None
    need(signed(cfg['admission'], A)['configuration_sha256'] == sha(canonical(prop)), 'admission binds proposal')
    OUT = Path(tempfile.mkdtemp(prefix='v833f-dry-')) if a.dry_run else D/'release-v833f-signed'
    if not a.dry_run: need(not OUT.exists(), 'output exists'); OUT.mkdir(mode=0o700)
    (OUT/'trainer-host-v833f').mkdir(mode=0o700); (OUT/'trainer-host-v833f/patched').mkdir(mode=0o700)
    def put(rel, raw):
        fd = os.open(OUT/rel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        return OUT/rel
    for n in L.HOST_FILES: put('trainer-host-v833f/' + n, (STAGE if n == 'mimo_host_trainer_v83.py' else H/n).read_bytes())
    p = put('trainer-host-v833f/TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json', canonical(cfg))
    summary = dict(version='mimo-v833f-root-signing-result-v1', dry_run=a.dry_run, authority=A, base='release-v833u-signed',
                   trainer_host=dict(path=str(p), sha256=hf(p), configuration_sha256=sha(canonical(prop))),
                   change=dict(changed_host_files=['mimo_host_trainer_v83.py'], launcher_sha256=files['mimo_host_trainer_v83.py'],
                               engine_new_keys={'sgd_fp32_master': True}, step_size='1x unchanged (lr 5e-7)', tensor_rel_cap='2e-5 unchanged',
                               optimizer_state='none (fresh every step) unchanged', original_method_pins_changed=False, source_root_changed=False),
                   checks_passed=True, services_changed=False, chain_calls=0,
                   authorized="Jacob 2026-10-30 t821u-t823u/t829u (fp32 head master design), t831u/pt831u 'push the changes ahorita' via parent agent")
    put('SIGNING-RESULT.json', (json.dumps(summary, indent=1, sort_keys=True) + '\n').encode())
    if not a.dry_run:
        subprocess.run('find . -type f ! -name SIGNED-MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > SIGNED-MANIFEST.sha256 && chmod 600 SIGNED-MANIFEST.sha256', shell=True, cwd=OUT, check=True)
    print(json.dumps(dict(done=True, dry_run=a.dry_run, out=str(OUT), config_sha256=hf(p), launcher=files['mimo_host_trainer_v83.py'])))
    if a.dry_run: shutil.rmtree(OUT)
if __name__ == '__main__': main()
