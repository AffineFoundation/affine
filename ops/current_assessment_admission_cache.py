"""Prospective signed memoization of exact original completed-report admission.

Opt-in only. The caller still authenticates the current audit config/populations,
reads the authoritative queue snapshot, applies completion cutoff exclusions,
resolves conflicts/adjudications, and computes every assessment. No SQLite,
network, model, writer, or Coordinator is constructed here.
"""
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import time
from nacl.signing import VerifyKey
from subnet.continuous_audit_service import admit_completed_reports

POLICY_VERSION = 'exact-original-report-admission-cache-policy-v1'
ENTRY_VERSION = 'exact-original-report-admission-cache-entry-v1'
KEY_VERSION = 'complete-queue-bytes-selected-records-and-context-v1'
CONTEXT_FIELDS = {'workers', 'approved_sources', 'execution_evidence_policy',
                  'deferral_policy', 'historical_report_admission'}
POLICY_FIELDS = {'version', 'admission_authority', 'directory', 'cache_signer', 'implementation_sha256',
                 'baseline_module_hashes', 'maximum_entry_bytes', 'maximum_directory_bytes',
                 'maximum_entries', 'minimum_free_bytes'}
REQUIRED_MODULES = ('subnet.continuous_audit_service', 'subnet.continuous_audit_policy',
                    'subnet.distributed_roles', 'subnet.storage')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False, ensure_ascii=True).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def valid_digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def authenticate(document, identity):
    need(type(document) is dict and set(document) == {'payload', 'signer', 'signature'}
         and document['signer'] == identity and valid_digest(identity),
         'exact cache signature envelope and admitted signer')
    VerifyKey(bytes.fromhex(identity)).verify(canonical(document['payload']),
        base64.b64decode(document['signature'], validate=True))
    return document['payload']


def cutoff_phase(execution, cutoff):
    """Return an exact cutoff equivalence class, or None to use the original.

The pinned _admit_report_batch uses cutoff only for the global execution-policy
activation and per-source resource-enforcement activation. The complete policy
bytes are separately in the context key. All source predicates are retained,
including sources unrelated to this job (conservative extra invalidation).
Malformed/unusual contexts use the unchanged original, preserving error order.
Report completion cutoff is still applied by the loader before this API.
"""
    if execution is None:
        return {'execution_policy': None}
    try:
        if (type(execution) is not dict or set(execution) != {'version', 'effective_cutoff', 'sources'}
                or execution['version'] not in ('explicit-backend-execution-evidence-v1',
                                               'explicit-backend-execution-evidence-v2')):
            return None
        effective = execution['effective_cutoff']
        if (type(effective) not in (int, float) or not math.isfinite(effective)
                or not 0 <= effective <= 2**53
                or type(cutoff) not in (int, float) or not math.isfinite(cutoff)
                or cutoff < effective or type(execution['sources']) is not dict):
            return None
        phases = []
        for source, entry in sorted(execution['sources'].items()):
            if type(source) is not str or type(entry) is not dict:
                return None
            threshold = entry.get('effective_cutoff', effective)
            if (type(threshold) not in (int, float) or not math.isfinite(threshold)
                    or not effective <= threshold <= 2**53):
                return None
            phases.append([source, cutoff >= threshold])
        return {'execution_policy_active': True, 'sources': phases}
    except (KeyError, TypeError, ValueError):
        return None


def queue_fingerprint(queue):
    """Cover every SQLite scalar field; large original JSON strings hash raw.

Non-SQLite shapes fall back to original admission, never a weaker digest.
No row id/digest/report_digest is trusted as a substitute for original bytes.
"""
    if type(queue) is not dict or any(type(k) is not str for k in queue):
        return None
    if any(type(queue.get(k)) is not str for k in ('envelope', 'report', 'report_request')):
        return None
    result = {}
    for key, value in queue.items():
        if type(value) is str:
            try:
                raw = value.encode('utf-8')
            except UnicodeEncodeError:
                return None
            result[key] = {'type': 'str', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        elif value is None or type(value) in (int, bool):
            result[key] = {'type': type(value).__name__, 'value': value}
        elif type(value) is float and math.isfinite(value):
            result[key] = {'type': 'float', 'value': value}
        else:
            return None
    return result


class AdmissionCache:
    """One invocation's frozen admission context and an explicitly trusted cache.

ROOT signs policy including exact implementation/baseline bytes and cache
signer. Tests use generated fixture keys. Production key loading/activation is
outside this module and requires a separately admitted integration.
"""
    def __init__(self, policy_document, *, authority, context, signing_key=None, policy_authority=None):
        # Production uses the independently authenticated ROOT trust anchor.
        # Isolated controls may explicitly supply their generated fixture anchor.
        policy = authenticate(policy_document, authority if policy_authority is None else policy_authority)
        need(type(policy) is dict and set(policy) == POLICY_FIELDS
             and policy['version'] == POLICY_VERSION and policy['admission_authority'] == authority,
             'exact admission cache policy and admitted original authority')
        need(valid_digest(policy['cache_signer'])
             and policy['implementation_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'ROOT-pinned cache implementation and signer')
        need(type(policy['maximum_entry_bytes']) is int
             and 1024 <= policy['maximum_entry_bytes'] <= 16 * 1024**2,
             'bounded signed admission cache entry')
        need(type(policy['maximum_directory_bytes']) is int
             and policy['maximum_entry_bytes'] <= policy['maximum_directory_bytes'] <= 4 * 1024**3
             and type(policy['maximum_entries']) is int and 1 <= policy['maximum_entries'] <= 20000
             and type(policy['minimum_free_bytes']) is int
             and 1024**3 <= policy['minimum_free_bytes'] <= 1024**4,
             'bounded cache directory quota and filesystem reserve')
        pins = policy['baseline_module_hashes']
        need(type(pins) is dict and pins, 'explicit original admission module pins')
        actual = {str(Path(importlib.import_module(name).__file__).resolve()) for name in REQUIRED_MODULES}
        need(actual <= set(pins), 'actual admission modules covered by ROOT pins')
        for filename, sha in pins.items():
            path = Path(filename)
            need(path.is_absolute() and valid_digest(sha)
                 and hashlib.sha256(path.read_bytes()).hexdigest() == sha,
                 'unchanged original admission module bytes')
        need(type(context) is dict and set(context) == CONTEXT_FIELDS, 'complete frozen admission context')
        self._context_bytes = canonical(dict(authority=authority, **context))
        self._context = json.loads(self._context_bytes)
        self._context_sha = hashlib.sha256(self._context_bytes).hexdigest()
        self._policy_sha = digest(policy_document)
        self._policy = json.loads(canonical(policy))
        self._key = signing_key
        if signing_key is not None:
            need(signing_key.verify_key.encode().hex() == policy['cache_signer'], 'admitted cache signing key')
        self._path = Path(policy['directory'])
        need(self._path.is_absolute() and self._path.resolve() == self._path, 'original absolute cache directory')
        self._directory_fd = os.open(self._path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            self._directory_identity = self._check_directory(initial=True)
        except BaseException:
            os.close(self._directory_fd)
            raise
        self.stats = dict(hits=0, misses=0, uncacheable=0, writes=0, baseline_calls=0,
                          rejected_entries=0, write_errors=0, skipped_storage=0)

    def close(self):
        if self._directory_fd is not None:
            os.close(self._directory_fd)
            self._directory_fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _check_directory(self, initial=False):
        current = self._path.lstat()
        opened = os.fstat(self._directory_fd)
        need(stat.S_ISDIR(current.st_mode) and current.st_uid == os.getuid()
             and stat.S_IMODE(current.st_mode) == 0o700
             and (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino),
             'private owned original admission cache directory')
        identity = (current.st_dev, current.st_ino)
        if not initial:
            need(identity == self._directory_identity, 'cache directory was not replaced')
        return identity

    def _read(self, key):
        self._check_directory()
        try:
            fd = os.open(key + '.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=self._directory_fd)
        except FileNotFoundError:
            return None
        try:
            before = os.fstat(fd)
            need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
                 and stat.S_IMODE(before.st_mode) == 0o600 and before.st_nlink == 1
                 and before.st_size <= self._policy['maximum_entry_bytes'],
                 'bounded private owned original admission cache file')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                raw = stream.read(before.st_size + 1)
            after = os.fstat(fd)
            need(stat.S_ISREG(after.st_mode) and after.st_uid == os.getuid()
                 and stat.S_IMODE(after.st_mode) == 0o600 and after.st_nlink == 1,
                 'cache file remains private, owned and unlinked')
            need(len(raw) == before.st_size and
                 (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                 (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                 'admission cache entry unchanged during read')
        finally:
            os.close(fd)
        entry = authenticate(json.loads(raw), self._policy['cache_signer'])
        need(type(entry) is dict and set(entry) == {'version', 'key', 'policy_sha256',
             'context_sha256', 'created_at', 'result'} and entry['version'] == ENTRY_VERSION
             and entry['key'] == key and entry['policy_sha256'] == self._policy_sha
             and entry['context_sha256'] == self._context_sha,
             'exact signed original admission cache binding')
        need(type(entry['created_at']) in (int, float) and math.isfinite(entry['created_at'])
             and entry['created_at'] >= 0, 'finite original cache creation time')
        result = entry['result']
        need(type(result) is dict and set(result) == {'admitted', 'deferred'}
             and type(result['admitted']) is dict and type(result['deferred']) is list,
             'exact signed original admission result')
        return result['admitted'], result['deferred']

    @contextmanager
    def _publication_lock(self):
        # Multiple admitted producers serialize budget checks and publication.
        # Busy/unsafe optional storage never delays original admission.
        self._check_directory()
        fd = os.open('.cache-write.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                     0o600, dir_fd=self._directory_fd)
        try:
            info = os.fstat(fd)
            need(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                 and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1,
                 'private owned cache publication lock')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            yield True
        finally:
            os.close(fd)

    def _storage_available(self, incoming):
        self._check_directory()
        total, entries = 0, 0
        # Count actual owned files under the held publication lock. No deletion,
        # estimated usage counter, trusted mutable index, or broad cleanup.
        with os.scandir(self._directory_fd) as listing:
            for entry in listing:
                info = entry.stat(follow_symlinks=False)
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                    return False
                total += info.st_size
                entries += entry.name != '.cache-write.lock'
                if (total + incoming > self._policy['maximum_directory_bytes']
                        or entries + 1 > self._policy['maximum_entries']):
                    return False
        free = os.fstatvfs(self._directory_fd)
        return free.f_bavail * free.f_frsize - incoming >= self._policy['minimum_free_bytes']

    def _write(self, key, result):
        if self._key is None:
            return
        payload = dict(version=ENTRY_VERSION, key=key, policy_sha256=self._policy_sha,
                       context_sha256=self._context_sha, created_at=time.time(),
                       result=dict(admitted=result[0], deferred=result[1]))
        document = dict(payload=payload, signer=self._policy['cache_signer'],
                        signature=base64.b64encode(self._key.sign(canonical(payload)).signature).decode())
        raw = canonical(document)
        # An unusually large result remains a correctly computed original result.
        if len(raw) > self._policy['maximum_entry_bytes']:
            return
        with self._publication_lock() as acquired:
            if not acquired or not self._storage_available(len(raw)):
                self.stats['skipped_storage'] += 1
                return
            self._check_directory()
            name = '.entry-' + secrets.token_hex(16)
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600,
                         dir_fd=self._directory_fd)
            try:
                with os.fdopen(fd, 'wb', closefd=False) as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(fd)
                self._check_directory()
                # A concurrent trusted producer may publish the same exact result.
                existing = self._read(key)
                if existing is not None:
                    need(canonical(existing) == canonical(result), 'same original cache result on concurrent publication')
                    return
                os.replace(name, key + '.json', src_dir_fd=self._directory_fd, dst_dir_fd=self._directory_fd)
                os.fsync(self._directory_fd)
                self.stats['writes'] += 1
            finally:
                os.close(fd)
                try:
                    os.unlink(name, dir_fd=self._directory_fd)
                except FileNotFoundError:
                    pass

    def _original(self, queue, selected_records, cutoff):
        self.stats['baseline_calls'] += 1
        c = self._context
        return admit_completed_reports([queue], selected_records, c['authority'], c['workers'],
            c['approved_sources'], execution_evidence_policy=c['execution_evidence_policy'],
            cutoff=cutoff, deferral_policy=c['deferral_policy'],
            historical_report_admission=c['historical_report_admission'])

    def admit(self, queue, selected_records, *, cutoff):
        phase = cutoff_phase(self._context['execution_evidence_policy'], cutoff)
        fingerprint = queue_fingerprint(queue)
        if phase is None or fingerprint is None:
            self.stats['uncacheable'] += 1
            return self._original(queue, selected_records, cutoff)
        key = digest(dict(version=KEY_VERSION, policy_sha256=self._policy_sha,
                          context_sha256=self._context_sha, cutoff_phase=phase,
                          queue=fingerprint, selected_records_sha256=digest(selected_records)))
        try:
            cached = self._read(key)
        except TimeoutError:
            raise  # Never swallow the enclosing original evidence deadline.
        except Exception:
            # Cache state is not scientific evidence. Refuse reuse, preserve
            # the original bad entry, and run every original admission check.
            self.stats['rejected_entries'] += 1
            return self._original(queue, selected_records, cutoff)
        if cached is not None:
            self.stats['hits'] += 1
            return cached
        self.stats['misses'] += 1
        result = self._original(queue, selected_records, cutoff)
        try:
            self._write(key, result)
        except TimeoutError:
            raise
        except Exception:
            # A failed optional cache publication cannot alter the result.
            self.stats['write_errors'] += 1
        return result
