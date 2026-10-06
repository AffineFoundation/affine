"""Default-off validation renewal for an already authorized, unchanged API.

These envelopes deliberately use versions rejected by API execution launchers.
Only a separately ROOT-approved metadata consumer may use their locator.
"""
import base64
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from nacl.signing import SigningKey, VerifyKey


POLICY = 'running-api-validation-renewal-policy-v1'
VALIDATION = 'running-api-metadata-validation-v1'
LOCATOR = 'running-api-validation-locator-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_bytes(path):
    path = Path(path)
    if not path.is_absolute() or path != path.resolve() or not path.is_file():
        raise ValueError('owned absolute non-symlink file')
    return path.read_bytes()


def authenticate(envelope, authority):
    if envelope.get('signer') != authority:
        raise ValueError('validation authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(envelope['payload']),
                                             base64.b64decode(envelope['signature'], validate=True))
    return envelope['payload']


def inspect_api(unit, pid):
    status = dict(row.split('=', 1) for row in subprocess.check_output(
        ['systemctl', '--user', 'show', unit, '-p', 'MainPID', '-p', 'InvocationID', '-p', 'ActiveState'],
        text=True).splitlines())
    proc = Path('/proc', str(pid))
    ticks = int((proc/'stat').read_text().rsplit(')', 1)[1].split()[19])
    boot = int(next(row.split()[1] for row in Path('/proc/stat').read_text().splitlines()
                    if row.startswith('btime ')))
    return dict(systemd=status, pid=pid, ticks=ticks,
                argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x],
                started_at=boot+ticks/os.sysconf('SC_CLK_TCK'))


def validate_dependency(policy, authority, now, inspect=inspect_api):
    if (policy.get('version') != POLICY or policy.get('enabled') is not True or
            policy.get('validation_only') is not True or policy.get('API_restart_allowed') is not False or
            policy.get('created_at', now+1) > now):
        raise ValueError('explicit validation-only renewal policy')
    ttl = policy['validity_seconds']
    renew = policy['renew_before_seconds']
    if type(ttl) is not int or not 600 <= ttl <= 86400 or type(renew) is not int or not 60 <= renew < min(3601, ttl):
        raise ValueError('bounded validation lifetime')
    raw = file_bytes(policy['execution_scope_path'])
    if hashlib.sha256(raw).hexdigest() != policy['execution_scope_sha256']:
        raise ValueError('immutable original API execution scope')
    api = authenticate(json.loads(raw), authority)
    if api['version'] != 'independent-queue-source-aware-api173-v1' or api['execute_allowed'] is not True:
        raise ValueError('original API execution authorization')
    original = policy['process']
    if inspect(policy['unit'], original['pid']) != original:
        raise ValueError('same live original API only')
    if (original['systemd'] != dict(MainPID=str(original['pid']), InvocationID=policy['invocation'], ActiveState='active') or
            not api['created_at'] <= original['started_at'] < api['expires_at'] or
            original['argv'][original['argv'].index('--scope')+1] != policy['execution_scope_path']):
        raise ValueError('original API started under original scope')
    parent_raw = file_bytes(api['parent_scope_path'])
    if hashlib.sha256(parent_raw).hexdigest() != api['parent_scope_sha256']:
        raise ValueError('original parent scope pin')
    parent = authenticate(json.loads(parent_raw), authority)
    if not parent['created_at'] <= original['started_at'] < parent['expires_at']:
        raise ValueError('original parent authorization at startup')
    config_raw = file_bytes(policy['config_path'])
    if (digest(json.loads(config_raw)) != api['config_sha256'] or
            original['argv'][original['argv'].index('--config')+1] != policy['config_path']):
        raise ValueError('unchanged original API config')
    required = {str(Path(api['operator_tree'])/name): pin for name, pin in api['operator_files'].items()}
    if len(api['operator_files']) != 173 or len(parent['operator_files']) != 172:
        raise ValueError('full original CPU operator maps')
    required.update({str(Path(parent['operator_tree'])/name): pin for name, pin in parent['operator_files'].items()})
    required.update({policy['execution_scope_path']: policy['execution_scope_sha256'],
                     api['parent_scope_path']: api['parent_scope_sha256'],
                     policy['config_path']: hashlib.sha256(config_raw).hexdigest()})
    registry_raw = file_bytes(api['registry_path'])
    registry = authenticate(json.loads(registry_raw), authority)
    required[api['registry_path']] = hashlib.sha256(registry_raw).hexdigest()
    for source, files in registry['approved_sources'].items():
        root = Path(api['source_trees'][source])
        required.update({str(root/name): pin for name, pin in files.items()})
    if any(policy['files'].get(path) != pin for path, pin in required.items()):
        raise ValueError('complete operator/config/registry/source pins')
    for path, pin in policy['files'].items():
        if hashlib.sha256(file_bytes(path)).hexdigest() != pin:
            raise ValueError('validation file changed')
    queue = Path(policy['database'])
    st = queue.stat()
    if queue != queue.resolve() or str(queue) != api['database'] or [st.st_dev, st.st_ino] != policy['queue_inode']:
        raise ValueError('same original queue inode')
    # Recheck identity after hashing; validation never opens or writes the queue.
    if inspect(policy['unit'], original['pid']) != original:
        raise ValueError('API changed during validation')
    return api, parent


def sign(key, value):
    return dict(payload=value, signer=key.verify_key.encode().hex(),
                signature=base64.b64encode(key.sign(canonical(value)).signature).decode())


def write_exclusive(path, value):
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def ensure_validation(policy_envelope, authority, seed_path, *, now=None, inspect=inspect_api):
    """Renew only under a pre-existing explicit ROOT policy; no API execution."""
    now = time.time() if now is None else now
    policy = authenticate(policy_envelope, authority)
    validate_dependency(policy, authority, now, inspect)
    directory = Path(policy['validation_directory'])
    if not directory.is_absolute() or directory != directory.resolve():
        raise ValueError('private validation namespace')
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.stat().st_mode & 0o077:
        raise ValueError('private validation namespace permissions')
    locator_path = directory/'latest.json'
    fd = os.open(directory/'renewal.lock', os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if locator_path.exists():
            locator = authenticate(json.loads(file_bytes(locator_path)), authority)
            if locator['version'] != LOCATOR or locator['policy_sha256'] != digest(policy):
                raise ValueError('same signed validation locator policy')
            path = Path(locator['validation_path'])
            if path.parent != directory or hashlib.sha256(file_bytes(path)).hexdigest() != locator['validation_sha256']:
                raise ValueError('owned immutable validation record')
            value = authenticate(json.loads(file_bytes(path)), authority)
            if value['version'] != VALIDATION or value['policy_sha256'] != digest(policy):
                raise ValueError('validation record policy')
            if value['created_at'] <= now < value['expires_at']-policy['renew_before_seconds']:
                return locator
        seed = Path(seed_path)
        raw_seed = file_bytes(seed)
        if seed.stat().st_mode & 0o077:
            raise ValueError('private local authority')
        key = SigningKey(bytes.fromhex(raw_seed.decode().strip()))
        if key.verify_key.encode().hex() != authority:
            raise ValueError('same local authority')
        value = dict(version=VALIDATION, policy_sha256=digest(policy), created_at=now,
                     expires_at=now+policy['validity_seconds'], validation_only=True,
                     API_restart_allowed=False, original_execution_scope_sha256=policy['execution_scope_sha256'],
                     original_process=policy['process'], queue_inode=policy['queue_inode'],
                     files_sha256=digest(policy['files']))
        path = directory/('validation-'+str(time.time_ns())+'.json')
        write_exclusive(path, sign(key, value))
        locator = dict(version=LOCATOR, policy_sha256=digest(policy), validation_path=str(path),
                       validation_sha256=hashlib.sha256(file_bytes(path)).hexdigest())
        temporary = directory/('latest-'+str(time.time_ns())+'.tmp')
        write_exclusive(temporary, sign(key, locator))
        if locator_path.is_symlink():
            raise ValueError('locator symlink refused')
        os.replace(temporary, locator_path)
        return locator
    finally:
        os.close(fd)
