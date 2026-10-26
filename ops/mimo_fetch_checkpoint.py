#!/usr/bin/env python3
"""Fetch the selected public MiMo revision into a dedicated node workspace.

No upload, release activation, wallet or private credential operation occurs.
HF's immutable revision and LFS hashes are checked before producing the local
content-addressed checkpoint descriptor consumed by all worker roles.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subnet.mimo_release import MODEL_ID, MODEL_REVISION, canonical, digest

REQUIRED = {'config.json', 'generation_config.json', 'chat_template.jinja',
            'tokenizer_config.json', 'tokenizer.json', 'merges.txt', 'vocab.json',
            'model.safetensors.index.json'}


def fetch(workspace):
    from huggingface_hub import HfApi, snapshot_download
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise ValueError('upstream model revision changed')
    selected = {r.rfilename: r for r in info.siblings
                if r.rfilename in REQUIRED or r.rfilename.endswith('.safetensors')}
    if not REQUIRED <= set(selected) or len(selected) > 32:
        raise ValueError('unexpected selected checkpoint inventory')
    required = sum(r.size for r in selected.values())
    incoming = workspace / ('incoming-' + MODEL_REVISION)
    if not incoming.exists() and shutil.disk_usage(workspace).free < required + 2 * 1024**3:
        raise ValueError('insufficient space for pinned model download')
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=incoming,
                      allow_patterns=sorted(selected), max_workers=4)
    files = {}
    for name, row in selected.items():
        path = incoming / name
        if Path(name).name != name or path.is_symlink() or path.stat().st_size != row.size:
            raise ValueError('model file name/size mismatch')
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
                hasher.update(chunk)
        files[name] = hasher.hexdigest()
        if row.lfs is not None and files[name] != row.lfs.sha256:
            raise ValueError('upstream LFS hash mismatch: ' + name)
    identifier = digest(files)
    # Download metadata has no role in runtime computation. Keep it outside
    # the model directory, whose exact inventory is authenticated on load.
    metadata = incoming / '.cache'
    if metadata.exists():
        shutil.rmtree(metadata)
    if set(p.name for p in incoming.iterdir()) != set(files):
        raise ValueError('unexpected file in checkpoint directory')
    target = workspace / identifier
    if target.exists():
        raise FileExistsError('checkpoint already installed; use its existing descriptor')
    incoming.rename(target)
    descriptor = dict(id=identifier, files=files)
    record = dict(model=MODEL_ID, revision=MODEL_REVISION, directory=str(target),
                  bytes=required, descriptor=descriptor)
    path = workspace / 'FETCHED.json'
    with path.open('xb') as stream:
        stream.write(canonical(record))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    args = parser.parse_args()
    record = fetch(args.workspace)
    print(json.dumps(dict(checkpoint=record['descriptor']['id'], bytes=record['bytes'],
                          revision=record['revision'], directory=record['directory'])))


if __name__ == '__main__':
    main()
