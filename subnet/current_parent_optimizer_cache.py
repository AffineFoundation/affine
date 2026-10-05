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


def pending_retention(cache_root, signed_policy, authority, *, current_bytes, pending_bytes,
                      fresh_disk_bytes):
    """Export publish callback wrapper, opt-in only; never declares adoption."""
    import os
    policy = authenticate(signed_policy, authority)
    if policy.get('version') != 'retain-pending-parent-cache-v1':
        raise ValueError('signed pending retention policy')
    for key,value in dict(current_bytes=current_bytes,pending_bytes=pending_bytes,
                          fresh_disk_bytes=fresh_disk_bytes).items():
        if type(value) is not int or value < 0 or policy.get(key) != value:
            raise ValueError('canonical simultaneous cache disk admission')
    root=Path(cache_root)
    if not root.is_dir() or root.is_symlink():raise ValueError('regular cache root')
    required=current_bytes+pending_bytes+fresh_disk_bytes
    if policy.get('total_disk_required_bytes') != required or shutil.disk_usage(root).free < pending_bytes+fresh_disk_bytes:
        raise ValueError('actual pending plus fresh filesystem reserve')
    # One pending namespace, named by the original signed job hash. Never reuse
    # an interrupted namespace or accumulate additional history.
    from .persistent_training_state import checkpoint_id
    checkpoint_id(policy['job_sha256'])
    destination=root/('pending-'+policy['job_sha256'])
    destination.mkdir(mode=0o700,exist_ok=False)
    import threading
    retained={}; lock=threading.Lock()
    def wrap(publish):
        def retain(name,path):
            with lock:
                if Path(name).name!=name or name in retained:raise ValueError('pending shard collision')
                digest,size=_hash_file(Path(path))
                if sum(v['size']for v in retained.values())+size>pending_bytes:
                    raise ValueError('pending payload exceeds admission')
                target=destination/name
                with target.open('xb')as output,Path(path).open('rb')as source:
                    shutil.copyfileobj(source,output,1024**2);output.flush();os.fsync(output.fileno())
                target.chmod(0o600)
                if _hash_file(target)!=(digest,size):raise ValueError('pending copy digest')
                retained[name]=dict(sha256=digest,size=size)
            publish(name,path)
        return retain
    return destination,wrap,retained


def promote_pending(pending, *, signed_policy, authority, adoption, readback_request,
                    readback_receipt, now):
    """ROOT authorizes exact original evidence; no unattested readback booleans."""
    from .remote_optimizer_readback import validate_receipt
    policy=authenticate(signed_policy,authority)
    if policy.get('version')!='promote-current-parent-cache-v1':raise ValueError('signed promotion policy')
    for field,envelope in [('adoption_sha256',adoption),('readback_request_sha256',readback_request),
                           ('readback_receipt_sha256',readback_receipt)]:
        if policy.get(field)!=sha(envelope):raise ValueError('original evidence digest '+field)
    adopted=authenticate(adoption,authority)
    descriptor=adopted['descriptor']
    if adopted.get('descriptor_sha256')!=sha(descriptor) or adopted.get('job_sha256')!=policy['job_sha256']:
        raise ValueError('original adoption job/descriptor')
    validate_descriptor(descriptor,policy['descriptor_sha256'],policy['checkpoint'],policy['inventory'])
    if (descriptor['genesis_sha256']!=policy['genesis_sha256'] or
            descriptor['optimizer_steps']!=policy['optimizer_steps']):raise ValueError('promotion genesis/counter')
    binding=policy['approved_readback_binding']
    if binding.get('descriptor_sha256')!=sha(descriptor) or binding.get('job_sha256')!=policy['job_sha256']:
        raise ValueError('readback parent binding')
    validate_receipt(readback_receipt,readback_request,authority,approved_binding=binding,
        approved_objects=policy['approved_readback_objects'],qualified_reader=policy['qualified_reader'],now=now)
    root=Path(pending)
    if root.is_symlink() or root.name!='pending-'+policy['job_sha256']:raise ValueError('original pending namespace')
    if set(p.name for p in root.iterdir())!={s['name']for s in descriptor['shards']}:
        raise ValueError('partial pending inventory')
    for shard in descriptor['shards']:
        path=root/shard['name']
        if path.is_symlink() or not path.is_file() or _hash_file(path)!=(shard['sha256'],shard['size']):
            raise ValueError('pending complete full hash')
    # Do not replace an existing CURRENT or delete history. The operator must
    # arrange an authorized lifecycle transition before another promotion.
    target=root.parent/'current'
    if target.exists() or target.is_symlink():raise ValueError('current cache collision requires lifecycle transition')
    root.rename(target)
    return target
