"""Default-off job-bound source validation reuse; fresh native replay is retained.

This caches authenticated source/snapshot bytes, never task, runtime, trace,
observations, native readiness checks or grading results. Production callers do
not opt in merely by importing this module.
"""
import base64
import hashlib
import json
import stat
import time
from pathlib import Path
from nacl.signing import VerifyKey

VERSION = 'immutable-job-native-source-validation-v1'

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def fingerprint(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ValueError('immutable source requires regular unlinked file')
    return tuple(getattr(value, k) for k in ('st_dev', 'st_ino', 'st_uid', 'st_mode',
                                           'st_size', 'st_mtime_ns', 'st_ctime_ns'))

class JobSourceValidation:
    def __init__(self, signed_scope, *, authority, job_id, spec):
        from . import environments as e
        VerifyKey(bytes.fromhex(authority)).verify(canonical(signed_scope['payload']),
                                                  base64.b64decode(signed_scope['signature'], validate=True))
        if signed_scope['signer'] != authority:
            raise ValueError('source validation authority')
        scope = signed_scope['payload']
        self.expires = scope['expires_at']
        if (scope['version'] != VERSION or scope['execute_allowed'] is not True or
                scope['job_id'] != job_id or not scope['created_at'] <= time.time() < self.expires or
                not 0 < self.expires - scope['created_at'] <= 3600):
            raise ValueError('finite exact native job')
        self.spec_bytes = canonical(spec.to_dict())
        if scope['environment_sha256'] != hashlib.sha256(self.spec_bytes).hexdigest():
            raise ValueError('exact environment binding')
        if (spec.adapter != 'prime_v1' or spec.id != 'affine_math' or not spec.config.get('task_snapshot') or
                any(spec.config.get(k) is not None for k in ('prolog_session_revision', 'rcore_terminal_revision'))):
            raise ValueError('snapshot native MATH only')
        self.root = Path(scope['source_root'])
        if not self.root.is_dir() or self.root != self.root.resolve():
            raise ValueError('immutable source root')
        inventory = scope['source_files']
        if not isinstance(inventory, dict) or not 1 <= len(inventory) <= 20000:
            raise ValueError('bounded complete source inventory')
        self.files = {}
        for name, digest in inventory.items():
            path = self.root / name
            if Path(name).is_absolute() or '..' in Path(name).parts or path != path.resolve():
                raise ValueError('source path binding')
            before = fingerprint(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest or fingerprint(path) != before:
                raise ValueError('original source fingerprint')
            self.files[path] = before
        legacy, research = e._roots(spec.config)
        self.roots = [legacy / 'rollouts/envs']
        if research.exists():
            self.roots.append(research / 'environments')
        required = [Path(__file__), Path(e.__file__), e.PACKAGE_ROOT / 'native_math_grader.py', e._snapshot_path(spec.config)]
        for root in self.roots:
            if not root.exists() or root != root.resolve():
                raise ValueError('environment source root')
            required.extend(p for p in root.rglob('*') if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts)
        if any(p not in self.files for p in required):
            raise ValueError('environment source closure missing')
        self.members = {p for root in self.roots for p in root.rglob('*')
                        if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts}
        if e._source_hash(spec) != spec.source_hash:
            raise ValueError('original environment source hash')
        self.snapshot = e._snapshot_path(spec.config)
        if self.files[self.snapshot][4] > 128 * 1024**2:
            raise ValueError('bounded native snapshot')
        self.rows = tuple(canonical(row) for row in json.loads(self.snapshot.read_bytes()))
        if len(self.rows) != spec.num_samples:
            raise ValueError('snapshot task count')
        self.validate(spec)

    def validate(self, spec):
        if time.time() >= self.expires or canonical(spec.to_dict()) != self.spec_bytes:
            raise ValueError('expired or cross-environment native validation')
        members = {p for root in self.roots for p in root.rglob('*')
                   if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts}
        if members != self.members or any(p != p.resolve() or fingerprint(p) != value
                                          for p, value in self.files.items()):
            raise ValueError('authenticated environment source changed')

    def snapshot_row(self, spec, index):
        self.validate(spec)
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise ValueError('environment index')
        # A new decoded dictionary prevents session/task mutation from leaking.
        return json.loads(self.rows[index])
