"""Prospective cache transport only: existing restore validates all tensor contents."""
import shutil
from pathlib import Path
from .distributed_roles import authenticate
from .persistent_cpu_adamw import sha
from .persistent_training_state import validate_descriptor, _hash_file

VERSION = 'verified-current-parent-cache-v1'


def admit(document, authority, descriptor, checkpoint, inventory, resource_admission):
    policy = authenticate(document, authority)
    if policy.get('version') != VERSION:
        raise ValueError('explicit signed current-parent cache policy')
    validate_descriptor(descriptor, policy['descriptor_sha256'], checkpoint, inventory)
    bindings = dict(checkpoint=checkpoint, genesis_sha256=descriptor['genesis_sha256'],
                    optimizer_steps=descriptor['optimizer_steps'], parameters_sha256=sha(inventory))
    if any(policy.get(k) != v for k, v in bindings.items()):
        raise ValueError('current parent cache lineage')
    if policy.get('independent_durable_readback_verified') is not True or policy.get('authority_committed') is not True:
        raise ValueError('original durable adoption and independent readback required')
    needed = sum(s['size'] for s in descriptor['shards'])
    if (type(policy.get('retained_cache_bytes')) is not int or policy['retained_cache_bytes'] != needed or
            resource_admission.get('current_parent_cache_bytes') != needed or
            resource_admission.get('admitted') is not True):
        raise ValueError('exact additional full-state disk admission')
    base = resource_admission.get('additional_disk_required_bytes')
    if (type(base) is not int or resource_admission.get('cache_additional_disk_required_bytes') != base + needed):
        raise ValueError('canonical fresh export plus retained cache disk reserve')
    return policy


def fetcher(cache, descriptor, remote_fetch, *, document, authority, checkpoint, inventory, resource_admission):
    """No authority in local files. Caller must admit exact signed parent first.

    Copy only hash-verified bytes into the original bounded restore workspace;
    original restore_state then checks tensor allowlist/spans/FP32/finite values.
    Corrupt/missing cache falls back to the identical approved R2 shard.
    """
    admit(document, authority, descriptor, checkpoint, inventory, resource_admission)
    root = Path(cache)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('regular current cache directory')
    needed = sum(s['size'] for s in descriptor['shards'])
    if resource_admission.get('current_parent_cache_bytes') != needed:
        raise ValueError('cache reserve binding')
    shards = {s['name']: s for s in descriptor['shards']}
    telemetry = dict(local_verified_bytes=0, remote_requested_bytes=0)
    def fetch(name, destination):
        row = shards[name]; source = root / name
        if source.is_file() and not source.is_symlink() and _hash_file(source) == (row['sha256'], row['size']):
            shutil.copyfile(source, destination)
            telemetry['local_verified_bytes'] += row['size']
        else:
            remote_fetch(name, destination)
            telemetry['remote_requested_bytes'] += row['size']
    return fetch, telemetry
