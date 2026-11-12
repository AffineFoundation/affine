#!/usr/bin/env python3
"""SN120 trainer host-v834f = the ROOT-admitted host-v834e + a sharded (ZeRO-3) execution of the same update for
8 x 32 GiB GPUs (stage/v835_zero3.py and the launcher's install() hook; no other host file, engine key, source
root or original-method pin changes). Modelled on ops/ff_train_params/sign_v833j.py.

The signature covers exactly the reviewed inventory: stage/ must equal SIGNED-MANIFEST.sha256 and
PROPOSAL.unsigned.json file for file (nothing missing, nothing extra, no bytecode), every host file other than
the two changed ones must equal the base admission, and --sign needs the manifest digest that was reviewed.
The result is verified with the staged launcher's own checked() on the written release.

ROOT seed: read in this process from the lifecycle config's authority_seed_file only with --sign, used once to
sign, never printed, logged or written. A dry run signs with a throwaway key and opens no seed or lifecycle file.

usage:
  sign_host_v834f.py                     check stage / manifest / proposal (no key, no base)
  sign_host_v834f.py --dry-run --base-private HOST_V834E.private.json [--source-root DIR]
  sign_host_v834f.py --sign --base-private HOST_V834E.private.json --lifecycle LIFECYCLE.private.json \\
                     --source-root ADMITTED_SOURCE --manifest-sha256 REVIEWED_HEX --out NEW_RELEASE_DIR
  sign_host_v834f.py --write-proposal    regenerate PROPOSAL / manifest / public copy from stage/ (before review)
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
STAGE = HERE / 'stage'
CHECKOUT = HERE.parents[3]
ROOT_ID = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
HOST_VERSION = 'host-v834f'
RELEASE_DIR = 'trainer-host-v834f'
PREFIX = './' + RELEASE_DIR + '/'
CONFIG_NAME = 'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json'
ADDED = ('v835_zero3.py',)
CHANGED = ('mimo_host_trainer_v83.py', 'v835_zero3.py')
PUBLIC = ('trainer-config.public.json',)
ADMITTED_V834E = HERE.parent / 'host_v834e_admitted.sha256.json'

# Runs the written release's own launcher.checked() in a fresh interpreter. checked() also requires the process
# to stand in the config's source_root, which is a path on the trainer box. When that directory does not exist
# here, only that one precondition is emulated (reported as cwd_emulated); every other check runs unchanged.
VERIFY = r'''
import importlib.util, json, os, sys
from pathlib import Path
sys.dont_write_bytecode = True
host_dir, config_path, closure = sys.argv[1:4]
sys.path.insert(0, closure)
spec = importlib.util.spec_from_file_location('mimo_host_trainer_v83', os.path.join(host_dir, 'mimo_host_trainer_v83.py'))
L = importlib.util.module_from_spec(spec); sys.modules[spec.name] = L; spec.loader.exec_module(L)
root = json.loads(Path(config_path).read_bytes())['source_root']
emulated = not Path(root).is_dir()
if emulated:
    class BoxPath(type(Path())):
        @classmethod
        def cwd(cls):
            return cls(root)
        def is_dir(self):
            return True if str(self) == root else super().is_dir()
        def resolve(self, strict=False):
            return self if str(self) == root else super().resolve(strict)
    L.Path = BoxPath
else:
    os.chdir(root)
config = L.checked(config_path)
print(json.dumps(dict(host_checked=True, cwd_emulated=emulated, host_files=len(config['host_files']))))
'''


def need(ok, why):
    if not ok:
        raise SystemExit('REFUSED: ' + why)


def hf(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_launcher(path):
    """The staged launcher as a module, compiled from source (no bytecode file is written next to it)."""
    module = types.ModuleType('v834f_launcher_probe')
    module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def stage_files():
    return sorted(p.relative_to(STAGE).as_posix() for p in STAGE.rglob('*') if p.is_file())


def manifest_text(digests):
    return ''.join('%s  %s%s\n' % (digests[rel], PREFIX, rel) for rel in sorted(digests))


def read_manifest():
    rows = {}
    for line in (HERE / 'SIGNED-MANIFEST.sha256').read_text().splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition('  ')
        need(rel.startswith(PREFIX) and len(digest) == 64, 'bad manifest line: %s' % line[:120])
        need(rel[len(PREFIX):] not in rows, 'duplicate manifest path: %s' % rel)
        rows[rel[len(PREFIX):]] = digest
    return rows


def check_stage():
    """Fail closed unless stage/, the manifest and the proposal describe the same files with the same bytes."""
    need(STAGE.is_dir(), 'stage/ missing')
    proposal = json.loads((HERE / 'PROPOSAL.unsigned.json').read_text())
    launcher = load_launcher(STAGE / 'mimo_host_trainer_v83.py')
    host_files, public_files = proposal['host_files'], proposal['public_files']
    listed = dict(host_files, **public_files)
    for rel in sorted(listed):
        need(not rel.endswith('.pyc') and '__pycache__' not in rel, 'bytecode in proposal: %s' % rel)
        need((STAGE / rel).is_file() and not (STAGE / rel).is_symlink(), 'host file missing (fail closed): %s' % rel)
        need(hf(STAGE / rel) == listed[rel], 'digest mismatch %s' % rel)
    manifest = read_manifest()
    for rel, digest in sorted(manifest.items()):
        need(not rel.endswith('.pyc') and '__pycache__' not in rel, 'bytecode in manifest: %s' % rel)
        need((STAGE / rel).is_file(), 'manifest file missing (fail closed): %s' % rel)
        need(hf(STAGE / rel) == digest, 'manifest digest mismatch: %s' % rel)
    need(manifest == listed, 'manifest and proposal list different files')
    need(set(host_files) == set(launcher.HOST_FILES) and not set(host_files) & set(public_files),
         'proposal host_files != the staged launcher HOST_FILES')
    need(set(public_files) == set(PUBLIC), 'public files')
    extra = [rel for rel in stage_files() if rel not in listed]
    need(not extra, 'unlisted file in stage (fail closed): %s' % ', '.join(extra[:5]))
    manifest_sha = hashlib.sha256((HERE / 'SIGNED-MANIFEST.sha256').read_bytes()).hexdigest()
    need(proposal['manifest_sha256'] == manifest_sha, 'proposal names another manifest')
    need(proposal['host_version'] == HOST_VERSION and proposal['version'] == launcher.VERSION
         and proposal['scope'] == launcher.SCOPE and sorted(proposal['added_host_files']) == sorted(ADDED)
         and sorted(proposal['changed_host_files']) == sorted(CHANGED), 'proposal identity / change list')
    admitted = json.loads(ADMITTED_V834E.read_text())['host_files']
    need(set(host_files) == set(admitted) | set(ADDED) and not set(ADDED) & set(admitted), 'inventory = host-v834e + added')
    need(sorted(n for n in host_files if host_files[n] != admitted.get(n)) == sorted(CHANGED),
         'only the declared host files differ from the admitted host-v834e')
    public = json.loads((STAGE / PUBLIC[0]).read_text())
    need(public.get('host_version') == HOST_VERSION and public.get('host_files_sha256') == host_files
         and 'engine' not in public and 'trainer_config' not in public, 'public copy must describe this stage and restate no signed setting')
    return proposal, launcher, manifest_sha


def public_copy(launcher, digests):
    zero3 = load_launcher(STAGE / 'v835_zero3.py')
    return dict(
        about=('Public copy of trainer host-v834f (no secrets). host-v834f is the ROOT-admitted host-v834e plus a '
               'sharded (ZeRO-3, torch FSDP) execution of the same update for GPUs below 100 GiB. It is a separate '
               'ROOT admission. This file restates no signed setting: the engine block, the source root and the '
               'original-method pins are those of the host-v834e admission, unchanged.'),
        host_version=HOST_VERSION, base_host_version='host-v834e', version=launcher.VERSION, scope=launcher.SCOPE,
        added_host_files=list(ADDED), changed_host_files=list(CHANGED), host_files_sha256=digests,
        zero3=dict(env=zero3.ZERO3_ENV, below_gpu_gib=zero3.LOW_MEM_GPU_GIB,
                   # the settings of the sharded update: one named set, chosen whole (never a single value)
                   profile_env=zero3.PROFILE_ENV, default_profile=zero3.DEFAULT_PROFILE,
                   profiles={name: profile._asdict() for name, profile in sorted(zero3.PROFILES.items())},
                   token_budget=zero3.PROFILES[zero3.DEFAULT_PROFILE].token_budget,
                   logit_chunk=zero3.LOGIT_CHUNK, target_peak_gib=zero3.TARGET_PEAK_GIB,
                   sharding='torch.distributed.fsdp.fully_shard, reshard_after_forward=True (FULL_SHARD)'))


def write_proposal():
    """Regenerate the unsigned review artifacts from stage/. Review the diff before anything is signed."""
    launcher = load_launcher(STAGE / 'mimo_host_trainer_v83.py')
    digests = {rel: hf(STAGE / rel) for rel in launcher.HOST_FILES}
    (STAGE / PUBLIC[0]).write_text(json.dumps(public_copy(launcher, digests), indent=1, sort_keys=True) + '\n')
    public = {rel: hf(STAGE / rel) for rel in PUBLIC}
    text = manifest_text(dict(digests, **public))
    (HERE / 'SIGNED-MANIFEST.sha256').write_text(text)
    proposal = dict(
        host_version=HOST_VERSION, base_host_version='host-v834e', version=launcher.VERSION, scope=launcher.SCOPE,
        added_host_files=list(ADDED), changed_host_files=list(CHANGED), host_files=digests, public_files=public,
        manifest_sha256=hashlib.sha256(text.encode()).hexdigest(),
        note=('UNSIGNED proposal for ROOT signing with sign_host_v834f.py against the ROOT-admitted host-v834e '
              'private config. Do not bootstrap or override ROOT.'))
    (HERE / 'PROPOSAL.unsigned.json').write_text(json.dumps(proposal, indent=2, sort_keys=True) + '\n')
    return proposal


def closure_api(source_root):
    """Signing / verification functions of the source closure the trainer host itself imports."""
    sys.path.insert(0, str(source_root))
    # imported here: `subnet` must resolve to the closure given on the command line
    from subnet.backend_jobs import signed
    from subnet.mimo_release_policy import sign
    from subnet.storage import Identity, canonical, sha
    return Identity, canonical, sha, signed, sign


def read_root_identity(Identity, lifecycle):
    """ROOT identity from the lifecycle's authority_seed_file. The seed never leaves this function."""
    try:
        seed_path = Path(json.loads(Path(lifecycle).read_text())['authority_seed_file'])
        need(seed_path.is_file() and not seed_path.is_symlink() and seed_path.stat().st_mode & 0o077 == 0, 'seed mode')
        actor = Identity(bytes.fromhex(seed_path.read_text().strip()))
    except SystemExit:
        raise
    except Exception as x:      # never echo the exception: it may quote file content
        raise SystemExit('REFUSED: ROOT seed unreadable (%s)' % type(x).__name__) from None
    need(actor.id == ROOT_ID, 'authority id')
    return actor


def put(root, rel, raw):
    path = Path(root) / rel
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def host_checked(host_dir, config_path, source_root):
    proc = subprocess.run([sys.executable, '-B', '-c', VERIFY, str(host_dir), str(config_path), str(source_root)],
                          capture_output=True, text=True)
    need(proc.returncode == 0, 'host verifier rejected the release: ' + (proc.stderr.strip().splitlines() or ['?'])[-1][:300])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true', help='sign with a throwaway key; never opens a seed (default)')
    ap.add_argument('--sign', action='store_true', help='operator only: ROOT-sign the reviewed stage')
    ap.add_argument('--write-proposal', action='store_true')
    ap.add_argument('--base-private', type=Path, help='ROOT-admitted host-v834e TRAINER-HOST...private.json')
    ap.add_argument('--lifecycle', type=Path, help='lifecycle private config naming authority_seed_file (--sign only)')
    ap.add_argument('--source-root', type=Path, help='admitted source closure (default for a dry run: this checkout)')
    ap.add_argument('--manifest-sha256', help='sha256 of the reviewed SIGNED-MANIFEST.sha256 (--sign only)')
    ap.add_argument('--out', type=Path, help='new release directory, outside this checkout (--sign only)')
    a = ap.parse_args(list(argv) if argv is not None else None)
    need(not (a.sign and a.dry_run), 'choose --sign or --dry-run')
    if a.write_proposal:
        need(not a.sign, '--write-proposal never signs')
        proposal = write_proposal()
        print(json.dumps(dict(written=True, manifest_sha256=proposal['manifest_sha256'], host_files=len(proposal['host_files']))))
        return
    proposal, launcher, manifest_sha = check_stage()
    host_files = proposal['host_files']
    out = dict(ok=True, dry_run=not a.sign, host_version=HOST_VERSION, host_file_count=len(host_files),
               manifest_sha256=manifest_sha, changed_host_files=list(CHANGED), root_id=ROOT_ID, signed=False)
    if a.base_private is None:
        need(not a.sign, '--base-private required for --sign')
        out['next'] = 'stage, manifest and proposal agree; pass --base-private for a dry-run signature'
        print(json.dumps(out, indent=2, sort_keys=True))
        return
    if a.sign:
        need(a.lifecycle is not None and a.source_root is not None and a.out is not None and a.manifest_sha256,
             '--sign needs --lifecycle, --source-root, --manifest-sha256 and --out')
        need(a.manifest_sha256 == manifest_sha, 'this manifest is not the reviewed one')
        need(not a.out.exists(), 'output exists')
        need(CHECKOUT not in a.out.resolve().parents and a.out.resolve() != CHECKOUT, 'release directory must be outside the checkout')
    source_root = (a.source_root or CHECKOUT).resolve()
    Identity, canonical, sha, signed, sign = closure_api(source_root)
    old = json.loads(a.base_private.read_text())
    try:
        base_payload = signed(old['admission'], ROOT_ID)
    except Exception:
        raise SystemExit('REFUSED: the base admission is not ROOT-signed') from None
    need(base_payload.get('configuration_sha256') == sha(canonical({k: v for k, v in old.items() if k != 'admission'})),
         'the base admission does not bind the base configuration')
    need(launcher.VERSION == old['version'] and launcher.SCOPE == old['scope'], 'same identity as the base')
    admitted = json.loads(ADMITTED_V834E.read_text())['host_files']
    need(old['host_files'] == admitted, 'base is not the admitted host-v834e')
    need(set(old['original_methods']) == set(launcher.SITES), 'original-method sites')
    prop = {k: v for k, v in old.items() if k != 'admission'}
    prop['host_files'] = dict(host_files)
    need(prop['engine'] == old['engine'] and prop['original_methods'] == old['original_methods']
         and prop['source_root'] == old['source_root'] and prop['authority'] == old['authority'],
         'engine / pins / source / authority unchanged')
    for validate in (launcher.drift_settings, launcher.sgd_settings, launcher.fp32_settings,
                     launcher.direction_settings, launcher.support_settings):
        validate(prop['engine'])
    actor = read_root_identity(Identity, a.lifecycle) if a.sign else Identity()
    authority = ROOT_ID if a.sign else actor.id
    root_configuration = sha(canonical(prop))            # what the ROOT admission of this stage binds
    payload = dict(base_payload)
    payload['configuration_sha256'] = root_configuration
    need(payload == dict(version=launcher.VERSION, configuration_sha256=root_configuration, source_unchanged=True,
                         scope=launcher.SCOPE), 'admission payload shape')
    if not a.sign:      # the dry run is signed by a throwaway key: its config names that key, so the host verifier can run
        prop = dict(prop, authority=authority)
        payload['configuration_sha256'] = sha(canonical(prop))
    cfg = dict(prop, admission=sign(actor, payload))
    actor = None
    need(signed(cfg['admission'], authority)['configuration_sha256'] == sha(canonical(prop)), 'admission binds proposal')
    release = Path(tempfile.mkdtemp(prefix='v834f-dry-')) if not a.sign else a.out
    try:
        if a.sign:
            release.mkdir(mode=0o700, parents=True)
        for rel in sorted(dict(host_files, **proposal['public_files'])):
            put(release, RELEASE_DIR + '/' + rel, (STAGE / rel).read_bytes())
        config_path = put(release, RELEASE_DIR + '/' + CONFIG_NAME, canonical(cfg))
        verified = host_checked(release / RELEASE_DIR, config_path, source_root)
        result = dict(version='mimo-v834f-root-signing-result-v1', dry_run=not a.sign, authority=authority, base='host-v834e',
                      trainer_host=dict(sha256=hf(config_path), configuration_sha256=sha(canonical(prop)),
                                        root_configuration_sha256=root_configuration),
                      changed_host_files=list(CHANGED), added_host_files=list(ADDED),
                      host_file_sha256={n: host_files[n] for n in CHANGED}, reviewed_manifest_sha256=manifest_sha,
                      engine_changed=False, services_changed=False, verified=verified)
        put(release, 'SIGNING-RESULT.json', (json.dumps(result, indent=1, sort_keys=True) + '\n').encode())
        if a.sign:
            digests = {p.relative_to(release).as_posix(): hf(p) for p in sorted(release.rglob('*')) if p.is_file()}
            need(all(not rel.endswith('.pyc') and '__pycache__' not in rel for rel in digests), 'bytecode in the release')
            put(release, 'SIGNED-MANIFEST.sha256', ''.join('%s  ./%s\n' % (digests[rel], rel) for rel in sorted(digests)).encode())
        out.update(signed=True, authority=authority, config_sha256=hf(config_path),
                   root_configuration_sha256=root_configuration, verified=verified, out=str(release) if a.sign else None)
    finally:
        if not a.sign:
            shutil.rmtree(release, ignore_errors=True)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
