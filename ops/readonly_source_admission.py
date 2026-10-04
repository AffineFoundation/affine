"""Prepare immutable source/tokenizer/native evidence without holding mining.

This is a CPU preparation tool, not GPU qualification or deployment approval.
It reads an already installed source and checkpoint, uses isolated grader
sessions, and writes one new private receipt. It does not install dependencies
into the role interpreter, change a live config, or claim that GPUs are idle.
"""
import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from nacl.signing import VerifyKey


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def approve(envelope, authority, role, now=None):
    if envelope.get('signer') != authority:
        raise ValueError('original admission signer')
    VerifyKey(bytes.fromhex(authority)).verify(
        canonical(envelope['payload']), base64.b64decode(envelope['signature'], validate=True))
    p = envelope['payload']; now = time.time() if now is None else now
    if p.get('revision') != 'immutable-readonly-source-admission-v1' or p.get('role') != role:
        raise ValueError('exact CPU preparation role')
    if (type(p.get('created_at')) not in (int, float)
            or type(p.get('expires_at')) not in (int, float)
            or not p['created_at'] <= now < p['expires_at']
            or not 0 < p['expires_at'] - p['created_at'] <= 3600):
        raise ValueError('bounded original CPU preparation window')
    for name in ('GPU_jobs', 'chain_transactions', 'live_configuration_writes'):
        if type(p.get(name)) is not int or p[name] != 0:
            raise ValueError('read-only CPU authorization')
    if p.get('allows_concurrent_mining') is not True or p.get('helper_sha256') != digest(__file__):
        raise ValueError('explicit original helper and concurrent-read scope')
    cp = p['checkpoint']; files = cp['files']
    if (not isinstance(files, dict) or not 1 <= len(files) <= 32
            or 'config.json' not in files or not any(n.endswith('.safetensors') for n in files)
            or hashlib.sha256(canonical(files)).hexdigest() != cp['id']):
        raise ValueError('complete checkpoint identity')
    for name, sha in files.items():
        if (re.fullmatch('[A-Za-z0-9_.-]+', name) is None or name.startswith('.')
                or re.fullmatch('[0-9a-f]{64}', sha) is None):
            raise ValueError('exact safe checkpoint inventory')
    indices = p['context_indices']; controls = p['native_control_indices']
    if (not isinstance(indices, list) or not 1 <= len(indices) <= 128
            or any(type(i) is not int or i < 0 for i in indices)
            or len(set(indices)) != len(indices) or not isinstance(controls, list)
            or not controls or any(type(i) is not int for i in controls)
            or len(set(controls)) != len(controls) or not set(controls) <= set(indices)):
        raise ValueError('bounded explicit native/context controls')
    return p


def checkpoint_inventory(path, files):
    path = Path(path)
    if (not path.is_absolute() or not path.is_dir() or path.is_symlink()
            or path.resolve() != path or any(p.is_symlink() or not p.is_file() for p in path.iterdir())
            or {p.name for p in path.iterdir()} != set(files)):
        raise ValueError('immutable exact checkpoint directory')
    objects = {n: dict(sha256=digest(path/n), bytes=(path/n).stat().st_size) for n in files}
    if {n: m['sha256'] for n, m in objects.items()} != files:
        raise ValueError('complete checkpoint byte readback')
    return objects


def run(plan, authority, role, source, checkpoint, archive, output):
    p = approve(plan, authority, role)
    source = Path(source); checkpoint = Path(checkpoint); output = Path(output)
    if (not source.is_absolute() or source.resolve() != source or source.is_symlink()
            or output.exists() or output.is_symlink() or not output.is_absolute()
            or output.parent.resolve() != output.parent
            or any(parent == source or parent == checkpoint for parent in output.parents)):
        raise ValueError('isolated new receipt outside immutable inputs')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU child must hide CUDA devices')
    if digest(archive) != p['source']['descriptor']['sha256']:
        raise ValueError('exact admitted source archive')
    installed = list(source.rglob('*'))
    if any(f.is_symlink() for f in installed):
        raise ValueError('regular immutable source members')
    observed = {str(f.relative_to(source)): digest(f) for f in installed if f.is_file()}
    if observed != p['source']['source_files']:
        raise ValueError('full source readback before candidate imports')
    # Imports come from the immutable candidate; never a running role checkout.
    sys.path.insert(0, str(source)); sys.dont_write_bytecode = True
    from subnet.source_bootstrap import admitted_files, verify_cache
    members = admitted_files(Path(archive).read_bytes(), p['source']['descriptor'])
    verify_cache(source, members)
    if {n: hashlib.sha256(v).hexdigest() for n, v in members.items()} != p['source']['source_files']:
        raise ValueError('approved full source inventory')
    objects = checkpoint_inventory(checkpoint, p['checkpoint']['files'])
    versions = {n: importlib.metadata.version(n) for n in p['runtime_versions']}
    if versions != p['runtime_versions']:
        raise ValueError('actual pinned role packages')
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('CPU admission initialized CUDA')
    from transformers import AutoTokenizer
    from subnet.harness import render
    from subnet.environments import EnvironmentSpec, create_session, _source_hash
    spec_dict = p['math_control_spec']
    if _source_hash(type('BoundSpec', (), spec_dict)()) != spec_dict['source_hash']:
        raise ValueError('exact original native environment')
    spec = EnvironmentSpec.from_dict(spec_dict)
    native_versions = {n: importlib.metadata.version(n) for n in spec.config['dependency_versions']}
    if native_versions != spec.config['dependency_versions']:
        raise ValueError('actual original environment packages')
    snapshot = source/spec.config['task_snapshot']
    if digest(snapshot) != p['original_MATH_snapshot_sha256']:
        raise ValueError('original dataset snapshot')
    rows = json.loads(snapshot.read_text())
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    config = json.loads((checkpoint/'config.json').read_text())
    if len(tokenizer) != p['tokenizer_length'] or config['vocab_size'] != p['model_output_width']:
        raise ValueError('approved tokenizer and model widths')
    contexts = []; controls = []; isolated_versions = None
    for index in p['context_indices']:
        session = create_session(spec)
        try:
            initial = session.reset(index, 20261003)
            ids = render(tokenizer, initial['messages'], initial.get('tools', []), p['math_control_harness'])
            if (not ids or any(type(i) is not int or not 0 <= i < config['vocab_size'] for i in ids)
                    or len(ids)+p['math_control_harness']['max_output_tokens'] > config['max_position_embeddings']):
                raise ValueError('exact context and output token budget')
            contexts.append(dict(index=index, task_hash=initial['task_hash'], prompt_tokens=len(ids),
                                 prompt_ids_sha256=hashlib.sha256(canonical(ids)).hexdigest()))
            if index in p['native_control_indices']:
                from affine_math_v1.taskset import VERIFY
                session._run(session.runtime.prepare_uv_script(VERIFY))
                interpreter = next(iter(session.runtime._uv_interpreters.values()))
                script = "import json,importlib.metadata;print(json.dumps({n:importlib.metadata.version(n) for n in ['math-verify','sympy','antlr4-python3-runtime']}))"
                isolated_versions = json.loads(subprocess.check_output([interpreter, '-B', '-c', script], text=True))
                if isolated_versions != p['original_native_uv_versions']:
                    raise ValueError('isolated original grader package versions')
                for label, text in [('positive', '\\boxed{'+str(rows[index]['data']['answer'])+'}'),
                                    ('negative', 'No boxed answer.')]:
                    if session.done:
                        session.close(); session = create_session(spec); session.reset(index, 20261003)
                    result = session.step({'text': text})
                    if result['classification'] != label or result['reward'] != (1.0 if label == 'positive' else 0.0):
                        raise ValueError('original positive/negative native control')
                    controls.append(dict(index=index, task_hash=initial['task_hash'], classification=label, reward=result['reward']))
        finally:
            session.close()
    if torch.cuda.is_initialized() or checkpoint_inventory(checkpoint, p['checkpoint']['files']) != objects:
        raise ValueError('CPU scope or immutable model changed')
    verify_cache(source, members)
    report = dict(revision=p['revision'], plan_sha256=hashlib.sha256(canonical(plan)).hexdigest(),
                  helper_sha256=digest(__file__), source_sha256=p['source']['descriptor']['sha256'],
                  source_members=len(members), role=role, checkpoint=p['checkpoint']['id'],
                  actual_local_cache=str(checkpoint), learned_objects=objects, runtime_versions=versions,
                  contexts=contexts, native_controls=controls, isolated_grader_versions=isolated_versions,
                  CPU_only=True, CUDA_initialized=False, model_loaded=False, GPU_jobs=0,
                  global_idle_claimed=False, controller_held=False, chain_transactions=False,
                  live_configuration_writes=False, GPU_qualification=False, completed_at=time.time())
    with output.open('xb') as stream:
        stream.write(canonical(report))
    output.chmod(0o600)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'authority', 'role', 'source', 'checkpoint', 'archive', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(); os.umask(0o077)
    if digest(args.plan) != args.plan_sha256:
        raise ValueError('original signed preparation file digest')
    report = run(json.loads(Path(args.plan).read_text()), args.authority, args.role,
                 args.source, args.checkpoint, args.archive, args.output)
    print(json.dumps({k: report[k] for k in ('CPU_only', 'source_members', 'GPU_jobs', 'global_idle_claimed')}))


if __name__ == '__main__':
    main()
