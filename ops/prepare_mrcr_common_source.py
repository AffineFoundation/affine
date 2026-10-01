"""Prepare MRCR public-shell candidate support in a NEW trusted source tree.

No model, state, wallet, service configuration or active source is changed.
The input is an operator-owned approved package, not an uploaded miner tree.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

BASE = {'harness.py': '4b565259163fc1bf6df3775a89d10fd953a65cccd9423910b56a398f183bec04',
        'gpu_runtime.py': 'dff049849c3cfd05edabedf1637f2e51c57d2f4085ff6ff1339b1afe69c9d357'}
POLICY_SHA = 'e3e8ef38f69eb04d840b0a1299ccd9274c8149e34d641a91dcb1041eb150f3b2'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('qualified patch context')
    return text.replace(old, new, 1)


def transformed(source, policy_file):
    source = Path(source); policy_file = Path(policy_file)
    for name, digest in BASE.items():
        path = source/'subnet'/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('qualified MRCR base source')
    if policy_file.is_symlink() or hashlib.sha256(policy_file.read_bytes()).hexdigest() != POLICY_SHA:
        raise ValueError('qualified public MRCR policy')
    harness = (source/'subnet/harness.py').read_text()
    harness = replace_once(harness, "('autoregressive', 'candidates', 'visible-copy-candidates')",
        "('autoregressive', 'candidates', 'visible-copy-candidates', 'public-mrcr-shell-candidates')")
    harness = replace_once(harness, "    if value['policy'] == 'candidates':\n",
        "    if value['policy'] == 'public-mrcr-shell-candidates':\n"
        "        from .native_mrcr_public_policy import REVISION\n"
        "        digest=hashlib.sha256((Path(__file__).parent/'native_mrcr_public_policy.py').read_bytes()).hexdigest()\n"
        "        if value.get('public_policy_revision')!=REVISION or value.get('public_policy_sha256')!=digest:\n"
        "            raise ValueError('public MRCR policy pin')\n"
        "    if value['policy'] == 'candidates':\n")
    helper = '''def mrcr_candidates(messages):
    from .native_mrcr_public_policy import shell_command,parse_public_question
    questions=[m['content'].split('\\n\\n',1)[0].strip() for m in messages
        if m.get('role')=='user' and isinstance(m.get('content'),str) and m['content'].startswith('Prepend ')]
    if len(questions)!=1:raise ValueError('exact public MRCR question required')
    prefix,_,_=parse_public_question(questions[0])
    wrong_prefix=('1' if prefix[0]=='0' else '0')+prefix[1:]
    wrong_question=questions[0].replace(prefix,wrong_prefix,1)
    commands=[shell_command(questions[0]),shell_command(wrong_question)]
    return [json.dumps({'tool_call':{'name':'bash','arguments':{'command':c}}},separators=(',',':')) for c in commands]


'''
    harness = replace_once(harness, 'def source_hash():\n', helper+'def source_hash():\n')
    harness = replace_once(harness, 'return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()',
        "return hashlib.sha256(Path(__file__).read_bytes()+(Path(__file__).parent/'native_mrcr_public_policy.py').read_bytes()).hexdigest()")
    harness = replace_once(harness, "    if config['policy'] == 'visible-copy-candidates':\n",
        "    if config['policy'] == 'public-mrcr-shell-candidates':\n"
        "        config=dict(config,policy='candidates',candidates=mrcr_candidates(messages))\n"
        "    if config['policy'] == 'visible-copy-candidates':\n")
    gpu = (source/'subnet/gpu_runtime.py').read_text()
    gpu = replace_once(gpu, "        if config['policy']=='visible-copy-candidates':\n",
        "        if config['policy']=='public-mrcr-shell-candidates':\n"
        "            config={**config,'policy':'candidates','candidates':policy.mrcr_candidates(messages)}\n"
        "        if config['policy']=='visible-copy-candidates':\n")
    return {'harness.py': harness, 'gpu_runtime.py': gpu}


def prepare(source, destination, policy_file):
    source = Path(source); destination = Path(destination)
    if source.is_symlink() or destination.is_symlink():
        raise ValueError('regular source and destination required')
    source = source.resolve(); destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('new separate source destination required')
    inventory = source/'subnet'
    if inventory.resolve()!=inventory or any(p.is_symlink() for p in inventory.rglob('*')):
        raise ValueError('trusted regular source inventory required')
    files=[p for p in inventory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    allowed={'.py','.toml','.txt','.md','.json','.gz','.csv','.pdf','.sh','.xlsx','.gitignore',''}
    for path in files:
        parts=path.relative_to(inventory).parts
        if path.suffix not in allowed or any(part.lower() in {'state','wallet','wallets','models','.git','.env'} for part in parts) or path.name.startswith('.env'):
            raise ValueError('source-only inventory required')
    if sum(p.stat().st_size for p in files)>64*1024*1024:
        raise ValueError('bounded approved source inventory required')
    patches = transformed(source, policy_file)
    destination.mkdir(parents=True)
    shutil.copytree(inventory, destination/'subnet', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(policy_file, destination/'subnet/native_mrcr_public_policy.py')
    for name, text in patches.items():
        (destination/'subnet'/name).write_text(text)
    record = {'base_source_files': BASE, 'public_policy_sha256': POLICY_SHA,
        'copied_file_hashes': {str(p.relative_to(inventory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'patched_source_files': {n:hashlib.sha256((destination/'subnet'/n).read_bytes()).hexdigest() for n in patches},
        'state_copied': False, 'models_copied': False, 'active_source_changed': False,
        'model_or_native_execution': False, 'scope': 'operator-trusted-prospective-MRCR-common-source-v1'}
    (destination/'source-preparation.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'destination', 'policy-file'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(prepare(a.source, a.destination, a.policy_file)))


if __name__ == '__main__':
    main()
