"""Conformance checks for an ``ObjectStore`` adapter.

The controller is only as safe as the store under it. It depends on a few
exact rules: a create-only write, a compare-and-swap write, a copy that is
tied to the etag of its source, a digest of the bytes that are really
stored, and a listing that is complete.

The R2 adapter is not in this package. It proves these rules in its own
tests, against the real store, in a scratch prefix::

    check_store(adapter, 'shadow/batchctl/<run_id>/contract/', big_bytes=64 << 20)

``check_store`` raises ``ContractBroken`` with the rule that failed.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Any, Callable, List, Tuple

from .interfaces import NotFound, ObjectStore, PreconditionFailed, StoreError, TooLarge

STALE_ETAG = '"an-etag-that-no-object-has"'


class ContractBroken(AssertionError):
    """The store does not keep a rule that the controller depends on."""


def _need(ok: Any, rule: str) -> None:
    if not ok:
        raise ContractBroken(rule)


def _raises(kind: type, call: Callable[[], Any], rule: str) -> None:
    try:
        call()
    except kind:
        return
    except Exception as exc:
        raise ContractBroken('%s (it raised %r)' % (rule, exc)) from exc
    raise ContractBroken('%s (it raised nothing)' % rule)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def missing_key(store: ObjectStore, p: str) -> None:
    key = p + 'missing'
    _raises(NotFound, lambda: store.get(key, max_bytes=10), 'get of a missing key must raise NotFound')
    _need(store.head(key) is None, 'head of a missing key must return None')
    _raises(NotFound, lambda: store.digest(key), 'digest of a missing key must raise NotFound')
    try:
        store.delete(key)
    except Exception as exc:
        raise ContractBroken('delete of a missing key must not be an error') from exc
    _need(list(store.list(key)) == [], 'list of a prefix without keys must be empty')


def read_after_write(store: ObjectStore, p: str) -> None:
    key = p + 'doc'
    for data in (b'one', b'', b'\x00\xff' * 40000):
        etag = store.put(key, data)
        blob = store.get(key, max_bytes=len(data))
        _need(blob.data == data, 'get must return the bytes of the last put')
        info = store.head(key)
        _need(
            info is not None and info.key == key and info.size == len(data),
            'head must return the key and the size of the last put',
        )
        digest = store.digest(key)
        _need(
            digest.size == len(data) and digest.sha256 == _sha(data),
            'digest must be the size and the sha256 of the stored bytes',
        )
        _need(
            isinstance(etag, str) and etag and etag == blob.etag == info.etag == digest.etag,
            'put, get, head and digest must give the same etag for the same bytes',
        )


def size_limit(store: ObjectStore, p: str) -> None:
    key = p + 'limit'
    store.put(key, b'12345')
    _raises(TooLarge, lambda: store.get(key, max_bytes=4), 'get of an object above max_bytes must raise TooLarge')
    _need(store.get(key, max_bytes=5).data == b'12345', 'get of an object of exactly max_bytes must work')


def etag_follows_bytes(store: ObjectStore, p: str) -> None:
    key = p + 'etag'
    first = store.put(key, b'one')
    second = store.put(key, b'two')
    _need(first != second, 'the etag must change when the bytes change')
    listed = {info.key: info for info in store.list(key)}
    _need(
        key in listed and listed[key].etag == second and listed[key].size == 3,
        'list must give the size and the etag of the last put',
    )


def create_only(store: ObjectStore, p: str) -> None:
    key = p + 'create'
    etag = store.put(key, b'first', if_none_match=True)
    _raises(
        PreconditionFailed, lambda: store.put(key, b'second', if_none_match=True),
        'a create-only put on a key that exists must raise PreconditionFailed',
    )
    blob = store.get(key, max_bytes=16)
    _need(blob.data == b'first' and blob.etag == etag, 'a refused create-only put must write nothing')


def compare_and_swap(store: ObjectStore, p: str) -> None:
    key = p + 'pointer'
    _raises(
        PreconditionFailed, lambda: store.put(key, b'v1', if_match=STALE_ETAG),
        'a put with if_match on a missing key must raise PreconditionFailed',
    )
    _need(store.head(key) is None, 'a refused put with if_match must write nothing')
    first = store.put(key, b'v1', if_none_match=True)
    second = store.put(key, b'v2', if_match=first)
    _need(store.get(key, max_bytes=16).data == b'v2', 'a put with the current etag must replace the object')
    for stale in (first, STALE_ETAG):
        _raises(
            PreconditionFailed, lambda: store.put(key, b'v3', if_match=stale),
            'a put with a stale etag must raise PreconditionFailed',
        )
    blob = store.get(key, max_bytes=16)
    _need(blob.data == b'v2' and blob.etag == second, 'a refused put with if_match must write nothing')


def copy_by_etag(store: ObjectStore, p: str) -> None:
    src, dst = p + 'copy/src', p + 'copy/dst'
    _raises(
        NotFound, lambda: store.copy(src, dst, src_etag=STALE_ETAG),
        'copy of a missing source must raise NotFound',
    )
    old = store.put(src, b'old bytes')
    new = store.put(src, b'new bytes!')
    for stale in (old, STALE_ETAG):
        _raises(
            PreconditionFailed, lambda: store.copy(src, dst, src_etag=stale),
            'copy with a stale source etag must raise PreconditionFailed',
        )
    _need(store.head(dst) is None, 'a refused copy must write nothing')
    store.copy(src, dst, src_etag=new)
    copied = store.digest(dst)
    _need(
        (copied.size, copied.sha256) == (10, _sha(b'new bytes!'))
        and store.get(dst, max_bytes=16).data == b'new bytes!',
        'copy must write the bytes of the source',
    )
    _need(store.get(src, max_bytes=16).data == b'new bytes!', 'copy must not change the source')


def delete(store: ObjectStore, p: str) -> None:
    key = p + 'gone'
    store.put(key, b'x')
    store.delete(key)
    _need(
        store.head(key) is None and not list(store.list(key)),
        'a deleted key must be gone at once, for head and for list',
    )
    _raises(NotFound, lambda: store.get(key, max_bytes=1), 'get of a deleted key must raise NotFound')
    store.put(key, b'y', if_none_match=True)
    _need(store.get(key, max_bytes=1).data == b'y', 'a create-only put must work after a delete')


def complete_listing(store: ObjectStore, p: str, many: int) -> None:
    base = p + 'list/'
    keys = [base + 'k%06d' % i for i in range(many)]
    for key in keys:
        store.put(key, key.encode())
    outside = [p + 'list-sibling', p + 'lisz']
    for key in outside:
        store.put(key, b'x')
    listed = list(store.list(base))
    names = [info.key for info in listed]
    _need(
        sorted(names) == keys,
        'list must return every key under the prefix (also past one page), each once, and no other key',
    )
    _need(all(info.size == len(info.key) for info in listed), 'list must give the size of each key')
    _need(
        sorted(info.key for info in store.list(p + 'lis')) == sorted(keys + outside),
        'a prefix is a plain string prefix, not a directory',
    )


def big_object(store: ObjectStore, p: str, big_bytes: int) -> None:
    src, dst = p + 'big/src', p + 'big/dst'
    block = hashlib.sha256(b'storecheck').digest()
    data = (block * (big_bytes // len(block) + 1))[:big_bytes]
    store.put(src, data)
    digest = store.digest(src)
    _need(
        digest.size == big_bytes and digest.sha256 == _sha(data),
        'digest of a large object must be the sha256 of all its bytes',
    )
    _raises(
        PreconditionFailed, lambda: store.copy(src, dst, src_etag=STALE_ETAG),
        'copy of a large object with a stale source etag must raise PreconditionFailed',
    )
    _need(store.head(dst) is None, 'a refused copy of a large object must write nothing')
    store.copy(src, dst, src_etag=digest.etag)
    copied = store.digest(dst)
    _need(
        (copied.size, copied.sha256) == (digest.size, digest.sha256),
        'copy of a large object must keep every byte',
    )
    _raises(
        TooLarge, lambda: store.get(src, max_bytes=big_bytes - 1),
        'get of a large object above max_bytes must raise TooLarge',
    )


def _race(racers: int, attempt: Callable[[int], Any]) -> List[Tuple[str, int]]:
    """Start all writers at the same moment. Returns (outcome, writer)."""
    barrier = threading.Barrier(racers)
    results: List[Tuple[str, int]] = []
    lock = threading.Lock()

    def writer(index: int) -> None:
        barrier.wait()
        try:
            attempt(index)
            outcome = 'won'
        except PreconditionFailed:
            outcome = 'lost'
        except StoreError:
            outcome = 'unknown'
        except Exception:
            outcome = 'broken'
        with lock:
            results.append((outcome, index))

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(racers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def one_winner(store: ObjectStore, p: str, racers: int) -> None:
    for name, prepare in (('create', None), ('swap', b'start')):
        key = p + 'race/' + name
        etag = store.put(key, prepare) if prepare is not None else None

        def attempt(index: int, key: str = key, etag: Any = etag) -> None:
            body = b'writer-%d' % index
            if etag is None:
                store.put(key, body, if_none_match=True)
            else:
                store.put(key, body, if_match=etag)

        results = _race(racers, attempt)
        _need(
            not [index for outcome, index in results if outcome == 'broken'],
            'a store error must be a StoreError',
        )
        winners = [index for outcome, index in results if outcome == 'won']
        _need(len(winners) <= 1, 'of many conditional puts on one key, at most one may succeed (%s)' % name)
        data = store.get(key, max_bytes=64).data
        if winners:
            _need(
                data == b'writer-%d' % winners[0],
                'the stored bytes must be those of the conditional put that succeeded (%s)' % name,
            )
        else:
            maybe = [b'writer-%d' % index for outcome, index in results if outcome == 'unknown']
            _need(
                data in maybe or (prepare is not None and data == prepare),
                'conditional puts that were all refused must write nothing (%s)' % name,
            )


def check_store(
    store: ObjectStore, prefix: str, *, many: int = 1100, big_bytes: int = 0, racers: int = 8,
) -> List[str]:
    """Run every check under ``prefix`` and return the names of the checks.

    ``prefix`` must be empty at the start; it is emptied again at the end.
    ``many``: number of keys for the listing check. Use more than one page
    of the store (R2 and S3: 1000 keys). ``big_bytes``: size of the large
    object, or 0 to skip it. Use a size above the multipart limit of the
    adapter. ``racers``: writers in the race for one key.
    """
    if not prefix.endswith('/'):
        raise ValueError('prefix must end with "/"')
    if list(store.list(prefix)):
        raise ValueError('prefix is not empty; the checks need a scratch prefix')
    checks: List[Tuple[str, Callable[[], None]]] = [
        ('missing_key', lambda: missing_key(store, prefix)),
        ('read_after_write', lambda: read_after_write(store, prefix)),
        ('size_limit', lambda: size_limit(store, prefix)),
        ('etag_follows_bytes', lambda: etag_follows_bytes(store, prefix)),
        ('create_only', lambda: create_only(store, prefix)),
        ('compare_and_swap', lambda: compare_and_swap(store, prefix)),
        ('copy_by_etag', lambda: copy_by_etag(store, prefix)),
        ('delete', lambda: delete(store, prefix)),
        ('complete_listing', lambda: complete_listing(store, prefix, many)),
        ('one_winner', lambda: one_winner(store, prefix, racers)),
    ]
    if big_bytes > 0:
        checks.append(('big_object', lambda: big_object(store, prefix, big_bytes)))
    done = []
    try:
        for name, check in checks:
            check()
            done.append(name)
    finally:
        for info in list(store.list(prefix)):
            store.delete(info.key)
    _need(not list(store.list(prefix)), 'after a delete of every key the prefix must be empty')
    return done


__all__ = ['ContractBroken', 'check_store']
