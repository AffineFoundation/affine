"""The store contract, and the checks that an adapter must pass.

The fake store of the fault harness must keep the same contract as the real
one, or the harness proves nothing. Each broken store below breaks one rule
and must be found.
"""
from __future__ import annotations

import threading
import time

import pytest

from .guard import GuardedStore, shadow_prefix
from .interfaces import Digest, NotFound, StoreError
from .sim import RawStore
from .storecheck import ContractBroken, check_store

RUN = 'run0001'


class AtomicStore(RawStore):
    """The fake store with each request atomic, as a real store is."""

    def __init__(self):
        super().__init__()
        self._guard = threading.RLock()

    def get(self, key, *, max_bytes):
        with self._guard:
            return super().get(key, max_bytes=max_bytes)

    def head(self, key):
        with self._guard:
            return super().head(key)

    def digest(self, key):
        with self._guard:
            return super().digest(key)

    def put(self, key, data, *, if_none_match=False, if_match=None, who='box'):
        with self._guard:
            return super().put(key, data, if_none_match=if_none_match, if_match=if_match, who=who)

    def copy(self, src, dst, *, src_etag, who='box'):
        with self._guard:
            return super().copy(src, dst, src_etag=src_etag, who=who)

    def delete(self, key, *, who='box'):
        with self._guard:
            return super().delete(key, who=who)

    def list(self, prefix):
        with self._guard:
            return super().list(prefix)


def test_the_fake_store_of_the_harness_keeps_the_contract():
    store = AtomicStore()
    done = check_store(store, 'scratch/', big_bytes=3 << 20)
    assert done == [
        'missing_key', 'read_after_write', 'size_limit', 'etag_follows_bytes', 'create_only',
        'compare_and_swap', 'copy_by_etag', 'delete', 'complete_listing', 'one_winner', 'big_object',
    ]
    assert store.objs == {}


def test_the_guarded_store_keeps_the_contract():
    raw = AtomicStore()
    raw.put('someone-else/key', b'x')
    done = check_store(GuardedStore(raw, RUN), shadow_prefix(RUN) + 'contract/')
    assert 'big_object' not in done and len(done) == 10
    assert raw.objs == {'someone-else/key': b'x'}


def test_the_checks_need_an_empty_scratch_prefix():
    store = AtomicStore()
    store.put('scratch/keep-me', b'x')
    with pytest.raises(ValueError):
        check_store(store, 'scratch/')
    with pytest.raises(ValueError):
        check_store(store, 'other')
    assert store.objs == {'scratch/keep-me': b'x'}


class CreateTwice(AtomicStore):
    def put(self, key, data, *, if_none_match=False, if_match=None, who='box'):
        return super().put(key, data, if_match=if_match, who=who)


class SwapAnything(AtomicStore):
    def put(self, key, data, *, if_none_match=False, if_match=None, who='box'):
        return super().put(key, data, if_none_match=if_none_match, who=who)


class SwapCreates(AtomicStore):
    def put(self, key, data, *, if_none_match=False, if_match=None, who='box'):
        if if_match is not None and key not in self.objs:
            if_match = None
        return super().put(key, data, if_none_match=if_none_match, if_match=if_match, who=who)


class CopyAnything(AtomicStore):
    """A store that does not tie a copy to the etag of its source."""

    def copy(self, src, dst, *, src_etag, who='box'):
        with self._guard:
            if src not in self.objs:
                raise NotFound(src)
            self._set(who, dst, self.objs[src])


class ClaimedDigest(AtomicStore):
    """A store that answers with a checksum that the uploader claimed."""

    def digest(self, key):
        real = super().digest(key)
        return Digest(real.size, 'c' * 64, real.etag)


class OnePage(AtomicStore):
    def list(self, prefix):
        return super().list(prefix)[:1000]


class NoSizeLimit(AtomicStore):
    def get(self, key, *, max_bytes):
        return super().get(key, max_bytes=1 << 60)


class DeleteComplains(AtomicStore):
    def delete(self, key, *, who='box'):
        if key not in self.objs:
            raise StoreError('no such key')
        return super().delete(key, who=who)


class ConstantEtag(AtomicStore):
    @staticmethod
    def etag(data):
        return 'same'


class SlowDelete(AtomicStore):
    """An eventually consistent store: a delete shows up later."""

    def delete(self, key, *, who='box'):
        return None


class DirectoryPrefix(AtomicStore):
    def list(self, prefix):
        if not prefix.endswith('/'):
            prefix += '/'
        return super().list(prefix)


class RacyCreate(AtomicStore):
    """Check, wait, then write: two creators can both pass the check."""

    def put(self, key, data, *, if_none_match=False, if_match=None, who='box'):
        if not if_none_match or key in self.objs:
            return super().put(key, data, if_none_match=if_none_match, if_match=if_match, who=who)
        time.sleep(0.02)
        return super().put(key, data, who=who)


class TruncatedBigDigest(AtomicStore):
    """A digest that reads only the first part of a large object."""

    def digest(self, key):
        real = super().digest(key)
        if real.size <= 1 << 20:
            return real
        return Digest(real.size, 'd' * 64, real.etag)


BROKEN = {
    'create-only is not enforced': (CreateTwice, 'create-only put on a key that exists'),
    'compare-and-swap is not enforced': (SwapAnything, 'if_match'),
    'compare-and-swap creates a missing key': (SwapCreates, 'if_match on a missing key'),
    'copy ignores the source etag': (CopyAnything, 'copy with a stale source etag'),
    'digest is a claimed value': (ClaimedDigest, 'digest must be the size and the sha256'),
    'listing stops after one page': (OnePage, 'list must return every key'),
    'no size limit on get': (NoSizeLimit, 'above max_bytes'),
    'delete of a missing key fails': (DeleteComplains, 'delete of a missing key'),
    'etag does not follow the bytes': (ConstantEtag, 'the etag must change'),
    'delete is not immediate': (SlowDelete, 'a deleted key must be gone at once'),
    'prefix is a directory': (DirectoryPrefix, 'list must'),
    'two creators both win': (RacyCreate, 'at most one may succeed'),
    'digest of a large object is wrong': (TruncatedBigDigest, 'digest of a large object'),
}


@pytest.mark.parametrize('name', sorted(BROKEN))
def test_a_store_that_breaks_a_rule_is_found(name):
    make, rule = BROKEN[name]
    with pytest.raises(ContractBroken, match=rule):
        check_store(make(), 'scratch/', big_bytes=3 << 20)
