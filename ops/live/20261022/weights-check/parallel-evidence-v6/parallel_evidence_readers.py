"""Run two authenticated evidence readers concurrently; never run a writer.

Linux fork children call only the supplied, source-pinned reader functions. Each
invocation creates its own pipes and workers, and merges only two complete JSON
results. The caller's original evidence alarm remains authoritative. No result
survives a failed reader, timeout, or malformed/oversized child response.
"""
import ctypes
import json
import multiprocessing
from multiprocessing.connection import wait
import os
import signal
import sys
import threading
import time

MAX_RESULT_BYTES = 256 * 1024**2


def _child(connection, reader, args, kwargs, parent_pid, timeout_seconds):
    # A stopped/killed parent cannot leave an evidence scan retaining its locks.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent_pid:
        os._exit(71)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(max(1, int(timeout_seconds)))
    try:
        try:
            value = dict(ok=True, value=reader(*args, **kwargs))
            raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
            if len(raw) > MAX_RESULT_BYTES:
                raise ValueError('bounded evidence child response exceeded')
        except BaseException as error:
            raw = json.dumps(dict(ok=False, error=type(error).__name__, message=str(error)[:4096]),
                             separators=(',', ':'), allow_nan=False).encode()
        connection.send_bytes(raw)
    finally:
        connection.close()


def _result(raw):
    doc = json.loads(raw)
    if type(doc) is not dict or type(doc.get('ok')) is not bool:
        raise ValueError('invalid evidence child response')
    if doc['ok']:
        if set(doc) != {'ok', 'value'} or type(doc['value']) is not dict:
            raise ValueError('invalid complete evidence child result')
        return doc['value']
    if set(doc) != {'ok', 'error', 'message'} or not all(type(doc[k]) is str for k in ('error', 'message')):
        raise ValueError('invalid evidence child error')
    # Preserve common failures for the existing writer's authenticated fallback;
    # no remote class names are imported or executed.
    exceptions = {'TimeoutError': TimeoutError, 'ValueError': ValueError,
                  'TypeError': TypeError, 'KeyError': KeyError,
                  'AttributeError': AttributeError, 'RuntimeError': RuntimeError,
                  'OSError': OSError, 'AssertionError': AssertionError}
    raise exceptions.get(doc['error'], RuntimeError)(doc['message'])


def _safe_to_fork():
    if sys.platform != 'linux' or threading.active_count() != 1:
        return False
    try:
        return len(os.listdir('/proc/self/task')) == 1
    except OSError:
        return False


def compose_loader(legacy_reader, mimo_reader, *, merge, transition_policy,
                   timeout_seconds=480, statistics=None):
    """Use the original reducers and merger without changing their arguments.

    Enter admission-cache managers only inside the legacy child. Preparation
    must supply unopened factories, not live DB connections or locked managers.
    A multithreaded/non-Linux host safely uses the original sequential behavior.
    """
    if not all(callable(v) for v in (legacy_reader, mimo_reader, merge)):
        raise ValueError('two readers and original merge required')
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 600:
        raise ValueError('bounded evidence deadline exceeds ten minutes')

    def load(*args, **kwargs):
        if not _safe_to_fork() or 'fork' not in multiprocessing.get_all_start_methods():
            old = legacy_reader(*args, **kwargs)
            new = mimo_reader(authority=kwargs['authority'], cutoff=kwargs['cutoff'], verifiers=kwargs['verifiers'])
            return merge(old, new, transition_policy=transition_policy)
        started = time.monotonic()
        deadline = started + timeout_seconds
        context = multiprocessing.get_context('fork')
        workers, readers, senders, answers = [], [], [], {}
        try:
            calls = [(legacy_reader, args, kwargs),
                     (mimo_reader, (), {k: kwargs[k] for k in ('authority', 'cutoff', 'verifiers')})]
            for index, (reader, call_args, call_kwargs) in enumerate(calls):
                receiving, sending = context.Pipe(duplex=False)
                readers.append(receiving); senders.append(sending)
                process = context.Process(target=_child,
                    args=(sending, reader, call_args, call_kwargs, os.getpid(), timeout_seconds),
                    name='affine-evidence-' + str(index), daemon=True)
                workers.append(process)
                process.start()
                sending.close()
            pending = dict(zip(readers, range(2)))
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('original evidence reader deadline')
                ready = wait(list(pending), timeout=min(remaining, .5))
                for connection in ready:
                    index = pending.pop(connection)
                    try:
                        answers[index] = connection.recv_bytes(maxlength=MAX_RESULT_BYTES)
                    except (EOFError, OSError) as error:
                        raise RuntimeError('evidence child ended without complete bounded result') from error
                    connection.close()
                for index, process in enumerate(workers):
                    if index not in answers and process.exitcode is not None and process.exitcode != 0:
                        raise RuntimeError('evidence child exited unsuccessfully')
            # Preserve original reader ordering and error precedence.
            old, new = _result(answers[0]), _result(answers[1])
            return merge(old, new, transition_policy=transition_policy)
        finally:
            for connection in readers + senders:
                connection.close()
            cleanup_failures = []
            for process in workers:
                if process.pid is None:
                    continue
                if process.is_alive():
                    process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill(); process.join(timeout=1)
                if process.is_alive():
                    cleanup_failures.append(process.pid)
                else:
                    process.close()
            if cleanup_failures:
                raise RuntimeError('evidence children failed to exit: ' + str(cleanup_failures))
            if statistics is not None:
                statistics.update(elapsed=time.monotonic()-started, children=len(workers), completed=len(answers))
    return load
