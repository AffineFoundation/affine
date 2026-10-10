"""Inactive CPU fixtures: all HTTP bodies are synthetic, all keys test-only."""
import ast
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.request

from ops import native_task_representative_selection as native
from types import FunctionType
ORIGINAL = 'serial-reference'
CANDIDATE = 'candidate'

def serial_reference(selector, root, objects, expires):
    paths=[]
    # Serial bounded reads keep decoded input residency out of the parent.
    for obj in objects:
        if time.time()>=expires:raise TimeoutError('representative fixed native deadline')
        path=document_path(root,obj['sha256'])
        if not path.exists():
            with urllib.request.urlopen(obj['url'],timeout=max(.1,min(30,expires-time.time()))) as response:
                raw=response.read(obj['size']+1)
            if len(raw)!=obj['size'] or hashlib.sha256(raw).hexdigest()!=obj['sha256']:
                raise ValueError('original representative GET size/hash')
            install_document_bundle(selector.controller,root,obj['sha256'],raw)
        verify_owned_document(selector.controller,root,obj['sha256'])
        paths.append(path)
    return paths

def prepare(which, namespace):
    function = serial_reference if which == ORIGINAL else native.prepare_documents
    return FunctionType(function.__code__, dict(function.__globals__, **namespace))


class OwnedPath:
    def __init__(self, fixture, sha):
        self.fixture, self.sha = fixture, sha

    def exists(self):
        return self.sha in self.fixture.owned


class Fixture:
    def __init__(self, count=12, *, cached=(), failures=None, corrupt=(), delays=None):
        self.raw = {str(i): json.dumps({'fixture': i}, sort_keys=True).encode() for i in range(count)}
        self.objects = [dict(url=str(i), size=len(self.raw[str(i)]),
                             sha256=hashlib.sha256(self.raw[str(i)]).hexdigest()) for i in range(count)]
        self.position = {o['sha256']: i for i, o in enumerate(self.objects)}
        self.owned = {self.objects[i]['sha256']: self.raw[str(i)] for i in cached}
        self.failures, self.corrupt, self.delays = failures or {}, set(corrupt), delays or {}
        self.calls, self.installed, self.verified, self.thread_ids = [], [], [], []
        self.active = self.peak = self.opened = self.closed = 0
        self.lock = threading.Lock()
        self.now = 100.0
        self.expires = 700.0
        self.before_get = self.before_install = self.before_verify = None
        self.caller = threading.get_ident()

    def get(self, url, timeout):
        i = int(url)
        with self.lock:
            self.calls.append(i)
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.opened += 1
        try:
            if self.before_get:
                self.before_get(i, timeout)
            time.sleep(self.delays.get(i, 0))
            if i in self.failures:
                raise self.failures[i]
            body = self.raw[url] + (b'!' if i in self.corrupt else b'')
            fixture = self

            class Response(io.BytesIO):
                def close(self):
                    if not self.closed:
                        with fixture.lock:
                            fixture.closed += 1
                            fixture.active -= 1
                    super().close()

            return Response(body)
        except BaseException:
            with self.lock:
                self.closed += 1
                self.active -= 1
            raise

    def install(self, controller, root, sha, raw):
        self.thread_ids.append(threading.get_ident())
        i = self.position[sha]
        if self.before_install:
            self.before_install(i)
        self.installed.append(i)
        self.owned[sha] = raw

    def verify(self, controller, root, sha):
        self.thread_ids.append(threading.get_ident())
        i = self.position[sha]
        if self.before_verify:
            self.before_verify(i)
        if hashlib.sha256(self.owned[sha]).hexdigest() != sha:
            raise ValueError('native owned copy SHA')
        self.verified.append(i)

    def invoke(self, source=CANDIDATE):
        namespace = dict(Path=Path, hashlib=hashlib,
                         time=SimpleNamespace(time=lambda: self.now),
                         urllib=SimpleNamespace(request=SimpleNamespace(urlopen=self.get)),
                         document_path=lambda root, sha: OwnedPath(self, sha),
                         install_document_bundle=self.install,
                         verify_owned_document=self.verify)
        f = prepare(source, namespace)
        try:
            return [self.position[p.sha] for p in f(SimpleNamespace(controller=object()), '/unused', self.objects, self.expires)]
        finally:
            assert self.active == 0, 'owned GET survived return/error'
            assert self.closed == self.opened, 'HTTP response leak'
            assert all(i == self.caller for i in self.thread_ids), 'ownership work escaped selecting thread'


class PrefetchTests(unittest.TestCase):

    def test_order_and_raw_outcome_equal_even_when_completion_reversed(self):
        fixtures = [Fixture(delays={0: .035, 1: .020, 2: .010}) for _ in range(2)]
        results = [f.invoke(p) for f, p in zip(fixtures, (ORIGINAL, CANDIDATE))]
        self.assertEqual(results[0], results[1])
        self.assertEqual(fixtures[0].owned, fixtures[1].owned)
        self.assertEqual(fixtures[0].installed, fixtures[1].installed)
        self.assertEqual(fixtures[0].verified, fixtures[1].verified)
        self.assertGreater(fixtures[1].peak, 1)
        self.assertLessEqual(fixtures[1].peak, 4)

    def test_cached_documents_skip_get_but_keep_verification(self):
        a, b = Fixture(cached=(0, 2, 6)), Fixture(cached=(0, 2, 6))
        self.assertEqual(a.invoke(ORIGINAL), b.invoke())
        self.assertEqual(sorted(a.calls), sorted(b.calls))
        self.assertEqual(a.verified, b.verified)
        self.assertEqual(a.installed, b.installed)

    def test_transport_error_is_earliest_in_original_order_and_drained(self):
        for source in (ORIGINAL, CANDIDATE):
            f = Fixture(failures={1: ConnectionError('earlier'), 2: OSError('later')}, delays={1: .025, 3: .040})
            with self.assertRaisesRegex(ConnectionError, 'earlier'):
                f.invoke(source)
            self.assertEqual(f.installed, [0])
            self.assertEqual(f.verified, [0])
            self.assertLessEqual(max(f.calls), 4)

    def test_hash_or_size_failure_matches_original_error(self):
        for source in (ORIGINAL, CANDIDATE):
            f = Fixture(corrupt=(2,))
            with self.assertRaisesRegex(ValueError, 'original representative GET size/hash'):
                f.invoke(source)
            self.assertEqual(f.installed, [0, 1])

    def test_owned_cache_drift_stops_serial_admission(self):
        for source in (ORIGINAL, CANDIDATE):
            f = Fixture(cached=(1,))
            f.owned[f.objects[1]['sha256']] = b'changed'
            with self.assertRaisesRegex(ValueError, 'native owned copy SHA'):
                f.invoke(source)
            self.assertEqual(f.installed, [0])
            self.assertEqual(f.verified, [0])

    def test_install_failure_drains_and_no_later_install(self):
        f = Fixture(delays={1: .025, 2: .025, 3: .025})
        def fail(i):
            raise OSError('synthetic install failure')
        f.before_install = fail
        with self.assertRaisesRegex(OSError, 'synthetic install failure'):
            f.invoke()
        self.assertEqual(f.installed, [])
        self.assertLessEqual(len(f.calls), 4)

    def test_initial_expiry_no_get(self):
        for source in (ORIGINAL, CANDIDATE):
            f = Fixture()
            f.now = f.expires
            with self.assertRaisesRegex(TimeoutError, 'representative fixed native deadline'):
                f.invoke(source)
            self.assertEqual(f.calls, [])
            self.assertEqual(f.installed, [])

    def test_budget_expiry_keeps_original_installed_prefix(self):
        for source in (ORIGINAL, CANDIDATE):
            f = Fixture()
            def verify(i):
                if i == 0:
                    f.now = f.expires
            f.before_verify = verify
            with self.assertRaisesRegex(TimeoutError, 'representative fixed native deadline'):
                f.invoke(source)
            self.assertEqual(f.installed, [0])
            self.assertEqual(f.verified, [0])

    def test_get_timeout_uses_same_remaining_budget_cap(self):
        f = Fixture(count=4)
        f.now = f.expires - 7
        def inspect(i, timeout):
            self.assertEqual(timeout, 7)
        f.before_get = inspect
        self.assertEqual(f.invoke(), [0, 1, 2, 3])

    def test_four_slots_no_unbounded_submission_when_first_is_blocked(self):
        f = Fixture(count=256)
        release, four_started = threading.Event(), threading.Event()
        started = []
        def block(i, timeout):
            with f.lock:
                started.append(i)
                if len(started) == 4:
                    four_started.set()
            if i == 0:
                self.assertTrue(release.wait(2))
        f.before_get = block
        result = {}
        def work():
            f.caller = threading.get_ident()
            try:
                result['paths'] = f.invoke()
            except BaseException as error:
                result['error'] = error
        thread = threading.Thread(target=work)
        thread.start()
        try:
            self.assertTrue(four_started.wait(2))
            time.sleep(.02)
            self.assertEqual(sorted(started), [0, 1, 2, 3])
            self.assertEqual(f.installed, [])
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        if 'error' in result:
            raise result['error']
        self.assertEqual(result['paths'], list(range(256)))
        self.assertLessEqual(f.peak, 4)

    def test_empty_wave(self):
        self.assertEqual(Fixture(count=0).invoke(), [])

    def test_completed_futures_cannot_retain_consumed_raw_buffers(self):
        futures, slots = [], []
        class ObservedPool(ThreadPoolExecutor):
            def submit(self, fn, obj, slot):
                # Keep deliberate extra references to EVERY completed slot and
                # Future. Previously admitted raw bytes must still be released.
                if len(slots) >= 4:
                    self_test.assertEqual(slots[-4], {})
                slots.append(slot)
                future = super().submit(fn, obj, slot)
                futures.append(future)
                return future
        self_test = self
        with patch('concurrent.futures.ThreadPoolExecutor', ObservedPool):
            self.assertEqual(Fixture(count=256).invoke(), list(range(256)))
        self.assertTrue(all(future.result() is None for future in futures))
        self.assertTrue(all(slot == {} for slot in slots))

    def test_extra_oversized_artifact_rejected_before_get(self):
        f = Fixture(count=1)
        f.objects[0]['size'] = 2_000_001
        with self.assertRaisesRegex(ValueError, 'bounded original native document'):
            f.invoke()
        self.assertEqual(f.calls, [])



if __name__ == "__main__":
    with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
        unittest.main(verbosity=2)
