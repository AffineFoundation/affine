#!/usr/bin/env python3
"""The box fetch (``pe_box.fetch`` and ``gate_fetch.py``): retries with a pause, and the real error on the host.

Real parts: the host function ``pe_box.fetch``, the box program ``gate_fetch.py`` as a process, the shell command
between them (``ShellBox`` runs it in a local shell), and an HTTP server on this machine that serves byte ranges.
The server can fail a request the way a network does: it closes the connection, gives HTTP 503, or sends a short
body. A port with no server gives the socket error "connection refused" (errno 111). That is the same code path
as "network unreachable" (errno 101): an ``OSError`` under ``URLError``.
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest import mock

from ops.parallel_eval.live import gate_fetch
from ops.parallel_eval.live import pe_box
from ops.parallel_eval.pit.r2_fixture import ShellBox

LIVE = Path(pe_box.__file__).resolve().parent
FAST = dict(GATE_FETCH_ATTEMPTS='4', GATE_FETCH_BACKOFF_BASE='0.05', GATE_FETCH_BACKOFF_MAX='0.2')
FILES = {'model-00001-of-00002.safetensors': os.urandom(300_000), 'model-00002-of-00002.safetensors': os.urandom(70_000),
         'config.json': b'{"layers": 2}'}
CID = hashlib.sha256(b'checkpoint').hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class RangeServer:
    """An HTTP server on this machine that serves byte ranges. ``faults`` makes the next requests of a file fail."""

    def __init__(self, files: Dict[str, bytes], port: int = 0) -> None:
        self.files = dict(files)
        self.faults: Dict[str, List[str]] = {}
        self.requests: List[Tuple[str, Optional[str]]] = []
        self.delay = 0.0
        self.lock = threading.Lock()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args) -> None:
                return None

            def do_GET(self) -> None:
                name = self.path.split('?')[0].strip('/')
                span = self.headers.get('Range')
                with outer.lock:
                    outer.requests.append((name, span))
                    queue = outer.faults.get(name) or []
                    fault = queue.pop(0) if queue else None
                if outer.delay:
                    time.sleep(outer.delay)
                self.close_connection = True
                if fault == 'drop':
                    return
                if fault in ('503', '403'):
                    self.send_response(int(fault))
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                body = outer.files[name]
                first, last = (int(x) for x in span.split('=')[1].split('-'))
                data = body[first:last + 1]
                self.send_response(206)
                self.send_header('Content-Range', 'bytes %d-%d/%d' % (first, first + len(data) - 1, len(body)))
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data[:len(data) // 2] if fault == 'short' else data)

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> 'RangeServer':
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def count(self, name: str, span: Optional[str] = None) -> int:
        return sum(1 for got, got_span in self.requests if got == name and span in (None, got_span))


class BoxFetchCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.pst = self.tmp / 'pst'                    # the helper directory of the box
        self.pst.mkdir()
        shutil.copy2(LIVE / 'gate_fetch.py', self.pst / 'gate_fetch.py')
        self.dest = self.tmp / 'ckpt'
        self.box = ShellBox()
        for patch in (mock.patch.object(pe_box, 'PST', str(self.pst)), mock.patch.dict(os.environ, FAST)):
            patch.start()
            self.addCleanup(patch.stop)
        self.files = {name: hashlib.sha256(body).hexdigest() for name, body in FILES.items()}

    def serve(self, port: int = 0) -> RangeServer:
        server = RangeServer(FILES, port).start()
        self.addCleanup(server.stop)
        return server

    def urls(self, port: int) -> Dict[str, str]:
        return {name: 'http://127.0.0.1:%d/%s?X-Amz-Signature=secret-of-the-url' % (port, name) for name in FILES}

    def fetch(self, port: int, **kw) -> tuple:
        return pe_box.fetch(self.box, CID, kw.get('files') or self.files, self.urls(port), str(self.dest), threads=4)

    def log(self) -> str:
        return (self.dest / (CID + '.fetch.log')).read_text()

    def assert_materialized(self) -> None:
        self.assertTrue((self.dest / CID / '.verified').is_file())
        for name, body in FILES.items():
            self.assertEqual((self.dest / CID / name).read_bytes(), body, name)

    def assert_no_url_left(self, text: str = '') -> None:
        self.assertFalse((self.dest / '.work' / (CID + '.urls.private.json')).exists(), 'the URL file is still on the box')
        self.assertNotIn('secret-of-the-url', text)
        self.assertNotIn('http://', text)


class TestRetries(BoxFetchCase):
    def test_requests_that_fail_are_tried_again_and_the_fetch_passes(self) -> None:
        server = self.serve()
        big = 'model-00001-of-00002.safetensors'
        # The big file: its size request fails 2 times, then passes. Its range request fails 2 times, then passes.
        server.faults = {big: ['drop', '503', None, 'short', 'drop'], 'config.json': ['503', 'drop']}
        seconds, size = self.fetch(server.port)
        self.assert_materialized()
        self.assertGreaterEqual(size, sum(len(body) for body in FILES.values()))
        self.assertEqual(server.count(big), 6)
        self.assertEqual(server.count('config.json'), 4)
        self.assertEqual(server.count('model-00002-of-00002.safetensors'), 2)
        log = self.log()
        self.assertIn('retry size of %s (try 1 of 4)' % big, log)
        self.assertIn('HTTP Error 503', log)
        self.assertIn('"status": "materialized"', log.strip().splitlines()[-1])
        self.assert_no_url_left(log)

    def test_no_route_to_the_store_at_the_start_then_it_comes_back(self) -> None:
        """The socket error of the first tries is "connection refused". The pause gives the server time to come up."""
        port = free_port()
        done: Dict[str, object] = {}

        def run() -> None:
            try:
                done['out'] = self.fetch(port)
            except Exception as exc:        # the test reads it from ``done``
                done['error'] = exc

        with mock.patch.dict(os.environ, dict(GATE_FETCH_ATTEMPTS='12', GATE_FETCH_BACKOFF_BASE='0.1', GATE_FETCH_BACKOFF_MAX='0.4')):
            worker = threading.Thread(target=run)
            worker.start()
            time.sleep(0.6)
            self.serve(port)
            worker.join(60)
        self.assertNotIn('error', done, done.get('error'))
        self.assert_materialized()
        self.assertIn('[Errno 111] Connection refused', self.log())
        self.assertRegex(self.log(), r'retry size of \S+ \(try 1 of 12\): URLError: ConnectionRefusedError')

    def test_two_fetches_of_one_id_at_one_time_download_it_one_time(self) -> None:
        server = self.serve()
        server.delay = 0.2
        results: List[object] = []

        def run() -> None:
            try:
                results.append(self.fetch(server.port))
            except Exception as exc:
                results.append(exc)

        workers = [threading.Thread(target=run) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(60)
        self.assertEqual([type(r) for r in results], [tuple, tuple], results)
        self.assertEqual(results[0][1], results[1][1])
        self.assert_materialized()
        for name, body in FILES.items():
            self.assertEqual(server.count(name, 'bytes=0-%d' % (len(body) - 1)), 1, name)
        statuses = sorted(json.loads(line)['status'] for line in self.log().splitlines() if line.startswith('{'))
        self.assertEqual(statuses, ['materialized', 'present'])


class TestTheRealErrorOnTheHost(BoxFetchCase):
    def test_a_fetch_that_fails_for_good_gives_the_socket_error_to_the_host(self) -> None:
        with self.assertRaises(pe_box.BoxIOError) as caught:
            self.fetch(free_port())
        text = str(caught.exception)
        self.assertIn('box fetch %s failed (rc=1)' % CID[:12], text)
        self.assertIn('URLError: ConnectionRefusedError: [Errno 111] Connection refused', text)
        self.assertIn('(errno ECONNREFUSED) after 4 tries; a new fetch can pass', text)
        self.assertFalse((self.dest / CID / '.verified').exists())
        self.assert_no_url_left(text)
        record = json.loads(self.log().strip().splitlines()[-1])
        self.assertEqual((record['status'], record['errno'], record['errno_name'], record['attempts'], record['retryable']),
                         ('failed', 111, 'ECONNREFUSED', 4, True))

    def test_an_http_answer_that_a_new_try_cannot_change_stops_at_once(self) -> None:
        server = self.serve()
        server.faults = {'config.json': ['403'] * 9}
        with self.assertRaises(pe_box.BoxIOError) as caught:
            self.fetch(server.port)
        text = str(caught.exception)
        self.assertIn('size of config.json: HTTPError: HTTP Error 403: Forbidden after 1 tries', text)
        self.assertNotIn('a new fetch can pass', text)
        self.assertEqual(server.count('config.json'), 1)
        self.assert_no_url_left(text)

    def test_a_range_that_fails_for_good_is_named_and_the_other_threads_stop(self) -> None:
        server = self.serve()
        big = 'model-00001-of-00002.safetensors'
        server.faults = {big: [None] + ['drop'] * 20}          # the size request passes, each range request fails
        with self.assertRaises(pe_box.BoxIOError) as caught:
            self.fetch(server.port)
        text = str(caught.exception)
        self.assertIn('range %s 0-%d' % (big, len(FILES[big]) - 1), text)
        self.assertIn('after 4 tries', text)
        self.assertNotIn('Stopped', text)
        self.assert_no_url_left(text)

    def test_a_file_with_the_wrong_sha256_is_an_error_with_its_name(self) -> None:
        server = self.serve()
        wrong = dict(self.files, **{'config.json': 'f' * 64})
        with self.assertRaises(pe_box.BoxIOError) as caught:
            self.fetch(server.port, files=wrong)
        self.assertIn("sha256 mismatch: ['config.json']", str(caught.exception))
        self.assertFalse((self.dest / CID / '.verified').exists())
        self.assert_no_url_left(str(caught.exception))

    def test_a_fetch_program_that_dies_gives_the_end_of_its_log(self) -> None:
        (self.pst / 'gate_fetch.py').write_text('import sys\nsys.stderr.write("Killed by the box\\n")\nsys.exit(137)\n')
        with self.assertRaises(pe_box.BoxIOError) as caught:
            self.fetch(free_port())
        self.assertIn('failed (rc=137): Killed by the box', str(caught.exception))
        self.assert_no_url_left(str(caught.exception))

    def test_a_fetch_that_is_there_is_not_done_again(self) -> None:
        server = self.serve()
        self.fetch(server.port)
        before = len(server.requests)
        self.assertEqual(self.fetch(server.port)[0], 0.0)
        self.assertEqual(len(server.requests), before)


class TestErrorText(unittest.TestCase):
    """The record of the error that the live run had: errno 101, network unreachable."""

    def test_network_unreachable_is_reported_with_its_errno(self) -> None:
        error = gate_fetch.FetchError('size of doc.json', urllib.error.URLError(OSError(101, 'Network is unreachable')), 10)
        record = gate_fetch.failure_record('c' * 64, error)
        self.assertEqual(record, dict(
            ckpt='c' * 64, status='failed', errno=101, errno_name='ENETUNREACH', attempts=10, retryable=True,
            error='size of doc.json: URLError: OSError: [Errno 101] Network is unreachable'))
        text = pe_box.fetch_error_text(['retry ...', json.dumps(record)])
        self.assertEqual(text, 'size of doc.json: URLError: OSError: [Errno 101] Network is unreachable '
                               '(errno ENETUNREACH) after 10 tries; a new fetch can pass')

    def test_an_error_text_has_no_url(self) -> None:
        error = ValueError('unknown url type: https://store.example/key?X-Amz-Signature=abc')
        self.assertEqual(gate_fetch.failure_record(None, error)['error'], 'ValueError: unknown url type: <url>')
        self.assertEqual(pe_box.fetch_error_text(['Traceback', 'x https://store.example/k?sig=1 y']), 'Traceback | x <url> y')
        self.assertEqual(pe_box.fetch_error_text([]), 'no output in the fetch log')

    def test_the_pause_doubles_and_has_a_maximum(self) -> None:
        fetcher = gate_fetch.Fetcher({}, 1)
        with mock.patch.object(gate_fetch.random, 'random', return_value=1.0):
            self.assertEqual([fetcher.pause(n) for n in range(7)], [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0])
        with mock.patch.object(gate_fetch.random, 'random', return_value=0.0):
            self.assertEqual(fetcher.pause(0), 1.0)


if __name__ == '__main__':
    unittest.main()
