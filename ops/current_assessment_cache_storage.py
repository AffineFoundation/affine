"""Bounded optional cache retention; original evidence admission is unchanged.

The original class, signed cache policy, entry schema, key derivation and read
authentication remain byte-identical. This wrapper changes only optional local
publication. One nonblocking original publication lock covers this invocation's
inventory and own immutable writes. Other cooperating writers skip storage.
No persisted usage counter, new authority, transaction or evidence index.
"""
from contextlib import contextmanager
import heapq
import json
import math
import os
import stat

from ops.current_assessment_admission_cache import (
    AdmissionCache, ENTRY_VERSION, authenticate, need, valid_digest,
)

MAXIMUM_EVICTIONS = 1024
MAXIMUM_RETENTION_CHECKS = 2048


def identity(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_nlink,
            s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class RetainingAdmissionCache(AdmissionCache):
    def __init__(self, *args, **kwargs):
        self._held_lock = None
        self._lock_tried = False
        self._lock_acquired = False
        self._inventory = None
        self._directory_version = None
        self._storage_disabled = False
        self.storage_stats = dict(inventories=0, metadata_entries=0, evictions=0,
                                  retention_checks=0, preserved_invalid=0,
                                  lock_busy=0, accounting_resets=0)
        super().__init__(*args, **kwargs)

    def close(self):
        if self._held_lock is not None:
            self._held_lock.__exit__(None, None, None)
            self._held_lock = None
        self._lock_acquired = False
        super().close()

    @contextmanager
    def _publication_lock(self):
        if not self._lock_tried:
            self._lock_tried = True
            manager = super()._publication_lock()
            acquired = manager.__enter__()
            if acquired:
                self._held_lock = manager
                self._lock_acquired = True
            else:
                manager.__exit__(None, None, None)
                self.storage_stats['lock_busy'] += 1
        self._check_directory()
        yield self._lock_acquired

    def _version(self):
        s = os.fstat(self._directory_fd)
        return (s.st_mtime_ns, s.st_ctime_ns)

    def _safe_file(self, s):
        return (stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid()
                and stat.S_IMODE(s.st_mode) == 0o600 and s.st_nlink == 1)

    def _scan(self):
        self._check_directory()
        need(self._lock_acquired, 'original publication lock held for inventory')
        total, entries, heap = 0, 0, []
        self.storage_stats['inventories'] += 1
        before = self._version()
        with os.scandir(self._directory_fd) as listing:
            for n, entry in enumerate(listing):
                # Refuse optional writes rather than traverse unbounded debris.
                if n > self._policy['maximum_entries'] + 64:
                    self._storage_disabled = True
                    return False
                s = entry.stat(follow_symlinks=False)
                self.storage_stats['metadata_entries'] += 1
                if not self._safe_file(s):
                    self._storage_disabled = True
                    return False
                total += s.st_size
                entries += entry.name != '.cache-write.lock'
                if (entry.name.endswith('.json') and valid_digest(entry.name[:-5])
                        and s.st_size <= self._policy['maximum_entry_bytes']):
                    heap.append((s.st_mtime_ns, entry.name, identity(s)))
        if self._version() != before:
            self._storage_disabled = True
            return False
        heapq.heapify(heap)
        self._inventory = dict(total=total, entries=entries, heap=heap)
        self._directory_version = before
        return True

    def _retire_one(self):
        while self._inventory['heap']:
            if (self.storage_stats['evictions'] >= MAXIMUM_EVICTIONS
                    or self.storage_stats['retention_checks'] >= MAXIMUM_RETENTION_CHECKS):
                return False
            _, name, expected = heapq.heappop(self._inventory['heap'])
            self.storage_stats['retention_checks'] += 1
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=self._directory_fd)
                try:
                    before = os.fstat(fd)
                    need(identity(before) == expected and self._safe_file(before),
                         'unchanged private retention candidate')
                    with os.fdopen(fd, 'rb', closefd=False) as stream:
                        raw = stream.read(self._policy['maximum_entry_bytes'] + 1)
                    need(len(raw) == before.st_size and identity(os.fstat(fd)) == expected,
                         'unchanged bounded retention bytes')
                    p = authenticate(json.loads(raw), self._policy['cache_signer'])
                    need(type(p) is dict and set(p) == {'version', 'key', 'policy_sha256',
                         'context_sha256', 'created_at', 'result'}
                         and p['version'] == ENTRY_VERSION and p['key'] == name[:-5]
                         and p['policy_sha256'] == self._policy_sha
                         and valid_digest(p['context_sha256'])
                         and type(p['created_at']) in (int, float)
                         and math.isfinite(p['created_at']) and p['created_at'] >= 0
                         and type(p['result']) is dict
                         and set(p['result']) == {'admitted', 'deferred'}
                         and type(p['result']['admitted']) is dict
                         and type(p['result']['deferred']) is list,
                         'valid signed optional cache entry only')
                    self._check_directory()
                    need(identity(os.stat(name, dir_fd=self._directory_fd,
                                          follow_symlinks=False)) == expected,
                         'same closed publication before retention')
                    os.unlink(name, dir_fd=self._directory_fd)
                finally:
                    os.close(fd)
                self._inventory['total'] -= before.st_size
                self._inventory['entries'] -= 1
                self._directory_version = self._version()
                self.storage_stats['evictions'] += 1
                return True
            except TimeoutError:
                raise
            except Exception:
                # Bad original cache files are preserved, never repaired here.
                self.storage_stats['preserved_invalid'] += 1
                self._inventory = None
                self._storage_disabled = True
                return False
        return False

    def _storage_available(self, incoming):
        self._check_directory()
        if self._storage_disabled:
            return False
        if self._inventory is None or self._version() != self._directory_version:
            if not self._scan():
                return False
        while (self._inventory['total'] + incoming > self._policy['maximum_directory_bytes']
               or self._inventory['entries'] + 1 > self._policy['maximum_entries']):
            if not self._retire_one():
                self._storage_disabled = True
                return False
        free = os.fstatvfs(self._directory_fd)
        return free.f_bavail * free.f_frsize - incoming >= self._policy['minimum_free_bytes']

    def _write(self, key, result):
        before_writes = self.stats['writes']
        try:
            super()._write(key, result)
            if self.stats['writes'] != before_writes and self._inventory is not None:
                s = os.stat(key + '.json', dir_fd=self._directory_fd, follow_symlinks=False)
                need(self._safe_file(s) and s.st_size <= self._policy['maximum_entry_bytes'],
                     'own completed immutable cache publication')
                self._inventory['total'] += s.st_size
                self._inventory['entries'] += 1
                # Do not retire this invocation's new entries before old ones.
                self._directory_version = self._version()
        except BaseException:
            self._inventory = None
            self.storage_stats['accounting_resets'] += 1
            raise
