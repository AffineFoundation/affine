#!/usr/bin/env python3
"""Stage the pinned original model immutably; never opens or advances a run.

This is an initial asset readback receipt, not a training or qualification
receipt. Checkpoint publication after training uses the signed trainer path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def stage(bucket, fetched, *, namespace, expected_checkpoint):
    from subnet.mimo_runtime import validate_descriptor, file_hash
    from subnet.mimo_release import MODEL_ID, MODEL_REVISION
    from subnet.mimo_grpo.publication import R2ModelPublisher, _namespace
    from subnet.storage import canonical
    checkpoint = validate_descriptor(fetched['descriptor'])
    if (checkpoint['id'] != expected_checkpoint or fetched.get('model') != MODEL_ID
            or fetched.get('revision') != MODEL_REVISION):
        raise ValueError('exact original MiMo checkpoint and revision required')
    namespace = _namespace(namespace)
    if not namespace.startswith('private/'):
        raise ValueError('staging uses an isolated private namespace')
    root = Path(fetched['directory']).absolute()
    if root.resolve() != root or not root.is_dir():
        raise ValueError('ordinary original checkpoint directory required')
    if set(p.name for p in root.iterdir()) != set(checkpoint['files']):
        raise ValueError('exact original checkpoint inventory required')
    rows = {}
    for name, expected in sorted(checkpoint['files'].items()):
        path = root / name
        if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise ValueError('original checkpoint file changed')
        rows[name] = dict(size=path.stat().st_size, sha256=expected)
    if sum(r['size'] for r in rows.values()) != fetched['bytes']:
        raise ValueError('original checkpoint total size changed')
    # Reuse the tested conditional multipart implementation, with no signer and
    # no fake optimizer update. Only these explicit original model files exist
    # in this operation's namespace; no CURRENT/discovery object is touched.
    transfer = R2ModelPublisher(bucket, None, namespace=namespace,
        source_closure_sha256=hashlib.sha256(canonical(checkpoint)).hexdigest(),
        assert_trainer_lease=lambda: None)
    prefix = namespace + '/checkpoints/' + checkpoint['id']
    objects = {}
    for name, row in rows.items():
        key = prefix + '/model/' + name
        if not transfer._verify(key, row, missing_ok=True):
            transfer._upload(root/name, key, row['size'])
            transfer._verify(key, row)
        objects[name] = dict(key=key, **row)
    transfer._immutable_json(prefix+'/descriptor.json', checkpoint)
    receipt = dict(version='mimo-initial-checkpoint-readback-v1',
        model=MODEL_ID, revision=MODEL_REVISION, descriptor=checkpoint, objects=objects,
        model_full_get_verified=True, optimizer_state_uploaded=False,
        training_update=False, discovery_changed=False, qualified=False)
    transfer._immutable_json(prefix+'/initial-readback.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bucket-config', required=True, type=Path)
    parser.add_argument('--fetched', required=True, type=Path)
    parser.add_argument('--expected-checkpoint', required=True)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    from subnet.storage import Bucket, canonical
    receipt = stage(Bucket(json.loads(args.bucket_config.read_bytes())),
        json.loads(args.fetched.read_bytes()), namespace=args.namespace,
        expected_checkpoint=args.expected_checkpoint)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.receipt.exists():
        if args.receipt.read_bytes() != canonical(receipt):
            raise ValueError('original receipt differs')
    else:
        with args.receipt.open('xb') as stream:
            stream.write(canonical(receipt)); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(status='private-assets-staged', checkpoint=receipt['descriptor']['id'],
        files=len(receipt['objects']), bytes=sum(x['size'] for x in receipt['objects'].values()),
        full_get_verified=True, activated=False)))


if __name__ == '__main__':
    main()
