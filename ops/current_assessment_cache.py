"""ROOT-bound optional local cache integration; no chain, network or key creation.

Missing/corrupt optional cache state runs the original admission path. The
writer retains its unchanged evidence timeout, fallback, cursor and source
trust. A missing private producer key permits authenticated read-only hits.
"""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import stat
from nacl.signing import SigningKey
from ops.current_assessment_admission_cache import authenticate, need, valid_digest
from ops.current_assessment_cache_storage import RetainingAdmissionCache as AdmissionCache

VERSION = 'hourly-authenticated-admission-cache-binding-v1'


def validate_scope(scope):
    need(type(scope) is dict and set(scope) == {'version', 'policy_path', 'policy_sha256', 'seed_path'}
         and scope['version'] == VERSION and valid_digest(scope['policy_sha256']),
         'exact ROOT-bound optional admission cache configuration')
    for key in ('policy_path', 'seed_path'):
        need(type(scope[key]) is str and Path(scope[key]).is_absolute(), 'absolute cache configuration path')
    return scope


def private_bytes(filename, limit):
    path = Path(filename)
    need(path.is_absolute() and path.resolve() == path, 'non-symlink private cache configuration')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1
             and stat.S_IMODE(before.st_mode) in (0o400, 0o600) and before.st_size <= limit,
             'bounded private owned cache configuration file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(fd)
        current = path.lstat()
        def identity(info):
            return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink,
                    info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        need(identity(before) == identity(after) == identity(current) and len(raw) == before.st_size,
             'same original cache configuration bytes')
        return raw
    finally:
        os.close(fd)


def policy_and_key(scope, authority):
    validate_scope(scope)
    raw = private_bytes(scope['policy_path'], 1024 * 1024)
    need(hashlib.sha256(raw).hexdigest() == scope['policy_sha256'], 'exact ROOT-pinned cache policy bytes')
    document = json.loads(raw)
    policy = authenticate(document, authority)
    key = None
    try:
        seed = private_bytes(scope['seed_path'], 128).decode('ascii').strip()
        need(valid_digest(seed), 'exact private cache seed encoding')
        candidate = SigningKey(bytes.fromhex(seed))
        need(candidate.verify_key.encode().hex() == policy['cache_signer'], 'ROOT-admitted dedicated cache signer')
        key = candidate
    except TimeoutError:
        raise
    except Exception:
        # The public ROOT-pinned cache remains useful without a producer key.
        pass
    return document, key


def factory(scope, expected_authority):
    """Construct under the existing evidence deadline, once context is known."""
    validate_scope(scope)
    def open_cache(*, authority, context):
        need(authority == expected_authority, 'same original admission authority')
        try:
            document, key = policy_and_key(scope, authority)
            return AdmissionCache(document, authority=authority, context=context, signing_key=key)
        except TimeoutError:
            raise
        except Exception:
            # Optional cache initialization cannot turn valid reports into
            # scientific refusals. None selects the unchanged loader branch.
            return nullcontext(None)
    return open_cache
