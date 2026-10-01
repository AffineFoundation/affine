"""Prepare a NEW balanced-replay source from an approved MRCR-v7c tree.

This copies source only and changes no services, state, credentials or models.
Use an operator-owned clean source tree; never accept miner-supplied code.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

PINS_SHA = '68ff484d5be8e7437216c92c07e86cb6ee556912f5aa3b0dcc4011f35ccb76d3'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, destination, helper_source=None):
    source = Path(source).absolute(); destination = Path(destination).absolute()
    helpers = Path(helper_source or Path(__file__).resolve().parent.parent).absolute()
    if source.resolve() != source or helpers.resolve() != helpers or destination.is_symlink():
        raise ValueError('regular unaliased source roots required')
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(source) or destination.is_relative_to(helpers):
        raise ValueError('new separate destination required')
    inventory = source/'subnet'
    if inventory.resolve() != inventory or any(p.is_symlink() for p in inventory.rglob('*')):
        raise ValueError('regular source inventory required')
    patch_root = Path(__file__).resolve().parent/'replay_source_patches'
    if digest(patch_root/'pins.json') != PINS_SHA:
        raise ValueError('qualified replay patch inventory')
    pins = json.loads((patch_root/'pins.json').read_bytes())
    for name, expected in pins['base'].items():
        if digest(inventory/name) != expected:
            raise ValueError('qualified MRCR-v7c base source')
    for name, expected in pins['patches'].items():
        if (patch_root/name).is_symlink() or digest(patch_root/name) != expected:
            raise ValueError('qualified replay patch bytes')
    for name, entry in pins['helpers'].items():
        path = helpers/name
        if path.resolve() != path or digest(path) != entry['sha256']:
            raise ValueError('qualified replay helper bytes')
    files = [p for p in inventory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    allowed = {'.py','.toml','.txt','.md','.json','.gz','.csv','.pdf','.sh','.xlsx','.gitignore',''}
    for path in files:
        parts = path.relative_to(inventory).parts
        if path.suffix not in allowed or any(v.lower() in {'state','wallet','wallets','models','.git','private','.env'} for v in parts) or path.name.startswith('.env'):
            raise ValueError('source-only inventory required')
    if sum(p.stat().st_size for p in files) > 64*1024*1024:
        raise ValueError('bounded source inventory required')
    destination.mkdir(parents=True)
    shutil.copytree(inventory, destination/'subnet', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name, entry in pins['helpers'].items():
        shutil.copyfile(helpers/name, destination/'subnet'/entry['destination'])
    for name in sorted(pins['patches']):
        subprocess.run(['git','apply','--check',str(patch_root/name)], cwd=destination, check=True, capture_output=True)
        subprocess.run(['git','apply',str(patch_root/name)], cwd=destination, check=True, capture_output=True)
    for name, expected in pins['outputs'].items():
        if digest(destination/'subnet'/name) != expected:
            raise ValueError('qualified replay output source')
    record = {'revision':'prospective-balanced-replay-controller-v1','base_source_files':pins['base'],
        'patched_source_files':pins['outputs'],'helper_files':pins['helpers'],
        'copied_source_files':{str(p.relative_to(inventory)):digest(p) for p in files},
        'active_source_changed':False,'state_copied':False,'models_copied':False,
        'model_execution':False,'chain_transactions':False,'deployed':False}
    (destination/'source-preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--helper-source',type=Path)
    a = p.parse_args(); print(json.dumps(prepare(a.source,a.destination,a.helper_source)))


if __name__ == '__main__': main()
