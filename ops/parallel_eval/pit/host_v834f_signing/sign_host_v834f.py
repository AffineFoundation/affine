#!/usr/bin/env python3
"""ROOT-sign host-v834f (ZeRO-3 / FSDP) — modelled on ops/ff_train_params/sign_v833j.py.

Reads the ROOT seed in-process from the lifecycle authority_seed_file, never
prints or logs it, signs exactly the reviewed manifest / host_files inventory,
and verifies with the host launcher's verifier (``signed`` + configuration
sha binding). Default is ``--dry-run``: uses a fresh Identity and never touches
the seed.

Usage:
  python3 sign_host_v834f.py --dry-run
  python3 sign_host_v834f.py --sign --base-private /path/to/host-v834e/...ROOT-ADMITTED.private.json \\
      --lifecycle /path/to/lifecycle...ACTIVE.private.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
STAGE = HERE / 'stage'
ROOT_ID = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
CHANGED = ('mimo_host_trainer_v83.py', 'v83_dp.py', 'v835_zero3.py')


def need(ok: bool, why: str) -> None:
    if not ok:
        raise SystemExit('REFUSED: ' + why)


def hf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_stage_launcher():
    spec = importlib.util.spec_from_file_location(
        'v834f_launcher_probe', STAGE / 'mimo_host_trainer_v83.py',
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _verify_stage_inventory(proposal: dict, launcher) -> dict:
    """Fail-closed: every HOST_FILES entry exists, digests match, no .pyc."""
    host_files = proposal['host_files']
    need(set(host_files) == set(launcher.HOST_FILES), 'proposal host_files != launcher.HOST_FILES')
    for rel, expect in host_files.items():
        path = STAGE / rel
        need(path.is_file(), 'host file missing (fail closed): %s' % rel)
        need(not rel.endswith('.pyc') and '/__pycache__/' not in rel, 'pyc in proposal: %s' % rel)
        got = hf(path)
        need(got == expect, 'digest mismatch %s' % rel)
    # Manifest: every listed path must exist; inventory must equal proposal keys
    # (trainer-config.public.json is informational only and must NOT appear).
    man_lines = (HERE / 'SIGNED-MANIFEST.sha256').read_text().splitlines()
    man_rels = set()
    for line in man_lines:
        line = line.strip()
        if not line:
            continue
        digest, _, rel = line.partition('  ')
        need(rel.startswith('./trainer-host-v834f/'), 'bad manifest path: %s' % rel)
        need(not rel.endswith('.pyc') and '/__pycache__/' not in rel, 'pyc in manifest: %s' % rel)
        local_rel = rel[len('./trainer-host-v834f/'):]
        man_rels.add(local_rel)
        local = STAGE / local_rel
        need(local.is_file(), 'manifest file missing (fail closed): %s' % rel)
        need(hf(local) == digest, 'manifest digest mismatch: %s' % rel)
    need(man_rels == set(host_files), 'manifest paths != proposal host_files')
    need('trainer-config.public.json' not in man_rels, 'public overlay must not be in signed manifest')
    need('v835_zero3.py' in host_files, 'v835_zero3.py required in host-v834f')
    return host_files


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', default=False,
                    help='never touch ROOT seed (safe; uses ephemeral Identity)')
    ap.add_argument('--sign', action='store_true',
                    help='operator-only: perform ROOT sign (requires --base-private + --lifecycle)')
    ap.add_argument('--base-private', type=Path, default=None,
                    help='prior ROOT-admitted TRAINER-HOST…private.json (host-v834e)')
    ap.add_argument('--lifecycle', type=Path, default=None,
                    help='lifecycle…ACTIVE.private.json carrying authority_seed_file')
    ap.add_argument('--source-root', type=Path, default=None,
                    help='admitted worker source root (defaults to cwd / PYTHONPATH probe)')
    args = ap.parse_args(list(argv) if argv is not None else None)
    # Default to dry-run when neither flag is given.
    dry = (not args.sign) or args.dry_run
    if args.sign and args.dry_run:
        need(False, 'pass only one of --sign / --dry-run')

    need(STAGE.is_dir(), 'stage/ missing — regenerate host_v834f_signing artifact')
    proposal_path = HERE / 'PROPOSAL.unsigned.json'
    need(proposal_path.is_file(), 'PROPOSAL.unsigned.json missing')
    proposal = json.loads(proposal_path.read_text())
    launcher = _load_stage_launcher()
    host_files = _verify_stage_inventory(proposal, launcher)

    # Import verifier pieces from the admitted source closure.
    # Load Identity / canonical / sha / sign via file specs so a missing boto3
    # (storage.py top-level import) cannot block dry-run in lean test envs.
    src = args.source_root
    if src is None:
        repo_root = HERE.parents[3]
        src = repo_root
    sys.path.insert(0, str(src))

    def _load_attr(module_path: Path, module_name: str, attr: str):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        mod = importlib.util.module_from_spec(spec)
        # Provide a tiny boto3 stub if storage imports it and boto3 is absent.
        if module_name.endswith('storage') and 'boto3' not in sys.modules:
            try:
                import boto3  # noqa: F401
            except ModuleNotFoundError:
                sys.modules['boto3'] = type(sys)('boto3')
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return getattr(mod, attr)

    Identity = _load_attr(src / 'subnet' / 'storage.py', 'v834f_storage', 'Identity')
    canonical = _load_attr(src / 'subnet' / 'storage.py', 'v834f_storage', 'canonical')
    sha = _load_attr(src / 'subnet' / 'storage.py', 'v834f_storage', 'sha')
    # backend_jobs.signed and mimo_release_policy.sign may pull more deps; prefer
    # normal imports once storage is loaded, with the same stubbing safety net.
    from subnet.backend_jobs import signed
    from subnet.mimo_release_policy import sign

    out = {
        'ok': True,
        'dry_run': dry,
        'host_version': 'host-v834f',
        'host_file_count': len(host_files),
        'manifest_sha256': proposal.get('manifest_sha256'),
        'changed_host_files': list(CHANGED),
        'root_id': ROOT_ID,
    }

    if dry:
        # Ephemeral identity — never reads a real seed.
        actor = Identity()
        authority = actor.id
        # Build a dry proposal binding without a base private when absent.
        if args.base_private is not None and args.base_private.is_file():
            old = json.loads(args.base_private.read_text())
            signed(old['admission'], ROOT_ID)
            prop = {k: v for k, v in old.items() if k != 'admission'}
            prop['host_files'] = host_files
            need(prop['version'] == launcher.VERSION and prop['scope'] == launcher.SCOPE,
                 'base version/scope mismatch')
        else:
            prop = dict(
                version=launcher.VERSION,
                scope=launcher.SCOPE,
                authority=ROOT_ID,
                source_root=str(src),
                host_files=host_files,
                original_methods={s: '0' * 64 for s in launcher.SITES},
                engine=dict(
                    batched_engine='packed', token_budget=69632, max_rows=64,
                    fast_stage=True, stage_workers=8, part_concurrency=16,
                    shard_concurrency=8, optimizer_step_threads=14,
                    drop_unresolved=False, optimizer_persistence='none',
                    loss_agg_mode='seq-mean-token-mean', loss_scale=0.5635,
                    dp_world=8, dp_port=29583, dp_token_budget=69632, dp_max_rows=256,
                    training_kernels={},
                    sgd_fp32_master=True, sgd_mode='fresh-adam', sgd_lr=5e-7,
                    sgd_update_norm_cap=1.0, sgd_direction='normgrad',
                    sgd_tensor_rel_cap=2e-5, sgd_freeze=[],
                ),
            )
        pay = dict(
            version=launcher.VERSION,
            configuration_sha256=sha(canonical(prop)),
            source_unchanged=True,
            scope=launcher.SCOPE,
        )
        cfg = dict(prop, admission=sign(actor, pay))
        actor = None  # drop seed material
        # Verify with the same binder the host launcher uses.
        need(
            signed(cfg['admission'], authority)['configuration_sha256'] == sha(canonical(prop)),
            'admission binds proposal',
        )
        tmp = Path(tempfile.mkdtemp(prefix='v834f-dry-'))
        try:
            (tmp / 'trainer-host-v834f').mkdir(mode=0o700)
            for n in launcher.HOST_FILES:
                dest = tmp / 'trainer-host-v834f' / n
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((STAGE / n).read_bytes())
                os.chmod(dest, 0o600)
            cfg_path = tmp / 'trainer-host-v834f' / 'TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json'
            cfg_path.write_bytes(canonical(cfg))
            os.chmod(cfg_path, 0o600)
            out['config_sha256'] = hf(cfg_path)
            out['verified'] = True
            out['next'] = (
                'Hand PROPOSAL.unsigned.json + stage/ + SIGNED-MANIFEST.sha256 to Jacob. '
                'Re-run with --sign --base-private <v834e ROOT-ADMITTED> '
                '--lifecycle <ACTIVE.private.json> on the operator host.'
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    # ---- real ROOT sign ----
    need(args.base_private is not None and args.base_private.is_file(),
         '--base-private required for --sign')
    need(args.lifecycle is not None and args.lifecycle.is_file(),
         '--lifecycle required for --sign')
    old = json.loads(args.base_private.read_text())
    signed(old['admission'], ROOT_ID)
    need(launcher.VERSION == old['version'] and launcher.SCOPE == old['scope'],
         'same identity')
    need(set(launcher.HOST_FILES) == set(host_files), 'HOST_FILES inventory')
    need(set(old['original_methods']) == set(launcher.SITES), 'original_methods sites')
    prop = {k: v for k, v in old.items() if k != 'admission'}
    prop['host_files'] = host_files
    need(prop['engine'] == old['engine'] and prop['original_methods'] == old['original_methods']
         and prop['source_root'] == old['source_root'], 'engine/pins/source unchanged')
    launcher.direction_settings(prop['engine'])
    launcher.support_settings(prop['engine'])
    launcher.fp32_settings(prop['engine'])
    launcher.sgd_settings(prop['engine'])

    life = json.loads(args.lifecycle.read_text())
    seed_path = Path(life['authority_seed_file'])
    need(seed_path.is_file() and not seed_path.is_symlink() and seed_path.stat().st_mode & 0o077 == 0,
         'seed mode')
    # Read seed in-process; never print or log it.
    actor = Identity(bytes.fromhex(seed_path.read_text().strip()))
    need(actor.id == ROOT_ID, 'authority id')
    pay = dict(signed(old['admission'], ROOT_ID))
    pay['configuration_sha256'] = sha(canonical(prop))
    cfg = dict(prop, admission=sign(actor, pay))
    actor = None
    need(signed(cfg['admission'], ROOT_ID)['configuration_sha256'] == sha(canonical(prop)),
         'admission binds proposal')

    out_dir = HERE / 'release-v834f-signed'
    need(not out_dir.exists(), 'output exists')
    out_dir.mkdir(mode=0o700)
    host_out = out_dir / 'trainer-host-v834f'
    host_out.mkdir(mode=0o700)
    (host_out / 'patched').mkdir(mode=0o700)

    def put(rel: str, raw: bytes) -> Path:
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        return dest

    for n in launcher.HOST_FILES:
        put('trainer-host-v834f/' + n, (STAGE / n).read_bytes())
    cfg_path = put(
        'trainer-host-v834f/TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json',
        canonical(cfg),
    )
    # Fail-closed: verify with the staged launcher.checked against the written tree.
    # checked() requires cwd == source_root; probe the binder only here.
    need(signed(cfg['admission'], ROOT_ID)['configuration_sha256'] == sha(canonical(prop)),
         'post-write admission verify')
    result = dict(
        version='mimo-v834f-root-signing-result-v1',
        dry_run=False,
        authority=ROOT_ID,
        base='host-v834e',
        trainer_host=dict(sha256=hf(cfg_path), configuration_sha256=sha(canonical(prop))),
        changed_host_files=list(CHANGED),
        host_file_sha256={n: host_files[n] for n in CHANGED},
        engine_changed=False,
        services_changed=False,
        note='host-v834f ZeRO-3 / FSDP FULL_SHARD; separate ROOT admission from host-v834e',
    )
    put('SIGNING-RESULT.json', (json.dumps(result, indent=1, sort_keys=True) + '\n').encode())
    # Refresh signed manifest over the release tree (no .pyc).
    import subprocess
    subprocess.run(
        "find . -type f ! -name SIGNED-MANIFEST.sha256 -print0 | sort -z | "
        "xargs -0 sha256sum > SIGNED-MANIFEST.sha256 && chmod 600 SIGNED-MANIFEST.sha256",
        shell=True, cwd=out_dir, check=True,
    )
    out.update(done=True, out=str(out_dir), config_sha256=hf(cfg_path), verified=True)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
