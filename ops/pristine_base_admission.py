"""Prospective, read-only admission of the byte-exact upstream Qwen math base.

Run with ``python -m ops.pristine_base_admission --config proposed.json
--descriptor checkpoint.json --authority TRUSTED_PUBLIC_KEY``. The authority
must come from the caller's trust anchor, never from the descriptor itself.
Optionally pass ``--checkpoint /absolute/cache`` to read every local file byte.

This helper does not launch or reset a service. Future restart callers must
explicitly invoke it before authorizing a pristine-base reset; historical
completed_math_restart_service v1 policies and signed records are unchanged.
A successful descriptor check alone does not verify a local or remote cache.

The independent filemap comes from ROOT-ORIGIN-AUTHENTICATION.json, version
immutable-upstream-checkpoint-public-origin-authentication-v1, SHA256
ee065c97fa5d4ffd63393fa2adaee9f536a4587b5e82a40620baaf09c3f6b286.
That receipt binds the immutable upstream revision's small-file byte hashes
and public Hugging Face LFS SHA256 values for its four tensor shards. A different
serialization, even of equivalent tensors, does not meet this byte-exact claim.
"""
import argparse
import base64
import json
from pathlib import Path
from types import MappingProxyType

from nacl.signing import VerifyKey

from subnet.storage import canonical, sha


VERSION = 'pristine-upstream-base-admission-v1'
MODEL_ID = 'Qwen/Qwen2.5-Math-7B-Instruct'
MODEL_REVISION = 'ef9926d75ab1d54532f6a30dd5e760355eb9aa4d'
CHECKPOINT_ID = '2173b3e0119380ab813c9588bedff6b3edd5d3dcb01565490fa33ac67b2debf6'
# These pins are independent of the candidate config and its signed descriptor.
PRISTINE_FILES = MappingProxyType({
    'config.json': '3a6dd7a4b3e6dd81c05ec943d837166508ceb953e1ce2baa57c84611d4765ef0',
    'generation_config.json': '071d157e94ad482562138f323df1ac3355d8bde9efce56f0a82ac2c333ee6525',
    'merges.txt': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
    'model-00001-of-00004.safetensors': '11f73294487dca1614bd7495e2c4e50372f5794f4cd175ce62d9ec2a8e97873a',
    'model-00002-of-00004.safetensors': '2b992a00b09340ac3f8e54db70509b709e2eef9f601e679682e3a50b496a7b08',
    'model-00003-of-00004.safetensors': '2460a589b1d606998574e0d84162e94307a15e2eea2aac6f4ea35f0361657660',
    'model-00004-of-00004.safetensors': '9e7925045deba3bb6ac461f1c2e7eabbe2a189549156dff1ec08b6c750a82bc5',
    'model.safetensors.index.json': '624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028',
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': 'd743761ff02297cc5cdf38a612b0f78d5cdf5ceb3f251ab1d25ad353d24b1887',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
})


def validate_pristine_base(config, descriptor, authority, *, checkpoint_path=None):
    """Check proposed base identity; return an unsigned, non-execution result.

    Transport metadata in initial_checkpoint is allowed, but its id and complete
    files must match the authenticated descriptor and independent upstream pins.
    This does not authenticate config or grant any reset/optimizer authorization.
    """
    if (config.get('model_id') != MODEL_ID
            or config.get('model_revision') != MODEL_REVISION):
        raise ValueError('exact pristine model name and upstream revision required')
    if descriptor.get('signer') != authority:
        raise ValueError('checkpoint descriptor authority mismatch')
    VerifyKey(bytes.fromhex(authority)).verify(
        canonical(descriptor['payload']),
        base64.b64decode(descriptor['signature'], validate=True))
    payload = descriptor['payload']
    if (not isinstance(payload, dict) or set(payload) != {'id', 'files'}
            or not isinstance(payload['files'], dict)
            or sha(canonical(payload['files'])) != payload['id']):
        raise ValueError('checkpoint descriptor filemap identity mismatch')
    expected = dict(PRISTINE_FILES)
    if (sha(canonical(expected)) != CHECKPOINT_ID
            or payload['id'] != CHECKPOINT_ID or payload['files'] != expected):
        raise ValueError('checkpoint is not the independently pinned pristine base')
    candidate = config.get('initial_checkpoint')
    if (not isinstance(candidate, dict) or candidate.get('id') != payload['id']
            or candidate.get('files') != payload['files']):
        raise ValueError('configured initial checkpoint differs from pristine descriptor')
    if checkpoint_path is not None:
        if str(checkpoint_path) != config.get('initial_checkpoint_path'):
            raise ValueError('local readback must use configured initial checkpoint path')
        from ops.readonly_source_admission import checkpoint_inventory
        checkpoint_inventory(checkpoint_path, expected)
    return dict(version=VERSION, model_id=MODEL_ID, model_revision=MODEL_REVISION,
                checkpoint_id=CHECKPOINT_ID, descriptor_authority=authority,
                pristine_descriptor_verified=True,
                local_bytes_verified=checkpoint_path is not None,
                execute_allowed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--descriptor', required=True)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--checkpoint')
    args = parser.parse_args(argv)
    result = validate_pristine_base(
        json.loads(Path(args.config).read_bytes()),
        json.loads(Path(args.descriptor).read_bytes()), args.authority,
        checkpoint_path=args.checkpoint)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
