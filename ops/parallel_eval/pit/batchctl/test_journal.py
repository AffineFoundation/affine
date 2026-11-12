"""Journal: hash chain, fsync-before-return, torn-tail repair, fail-closed."""
from __future__ import annotations

import os

import pytest

from ops.parallel_eval.pit.batchctl import journal as J


def _make(tmp_path, n=3):
    path = tmp_path / 'journal.jsonl'
    jr = J.Journal.create(path, kind='init', body={'run': 'r1'}, at=1.0)
    jr.replay()
    for i in range(n):
        jr.append('note', {'n': i}, 2.0 + i)
    jr.close()
    return path


def test_create_append_replay_roundtrip(tmp_path):
    path = _make(tmp_path, n=4)
    jr = J.Journal.open(path)
    records = jr.replay()
    assert [r['t'] for r in records] == ['init', 'note', 'note', 'note', 'note']
    assert [r['i'] for r in records] == [0, 1, 2, 3, 4]
    assert records[0]['p'] == J.GENESIS_HASH
    for prev, rec in zip(records, records[1:]):
        assert rec['p'] == prev['h']
    assert jr.count == 5 and jr.last_hash == records[-1]['h']
    assert jr.replay() == []
    rec = jr.append('note', {'n': 99}, 9.0)
    assert rec['i'] == 5 and rec['p'] == records[-1]['h']
    jr.close()
    assert J.verify_file(path) == (6, rec['h'])


def test_create_refuses_to_overwrite(tmp_path):
    path = _make(tmp_path)
    with pytest.raises(J.JournalError):
        J.Journal.create(path, kind='init', body={}, at=1.0)


def test_create_leaves_no_file_without_a_valid_first_record(tmp_path):
    path = tmp_path / 'journal.jsonl'

    class Boom(J.FileOps):
        def replace(self, src, dst):
            raise OSError('crash before rename')

    with pytest.raises(OSError):
        J.Journal.create(path, kind='init', body={}, at=1.0, ops=Boom())
    assert not path.exists()


def test_every_cut_of_the_last_record_is_repaired(tmp_path):
    path = _make(tmp_path, n=2)
    whole = path.read_bytes()
    lines = whole.splitlines(keepends=True)
    durable = b''.join(lines[:-1])
    last = lines[-1]
    for cut in range(len(last)):
        path.write_bytes(durable + last[:cut])
        jr = J.Journal.open(path)
        assert [r['i'] for r in jr.replay()] == [0, 1]
        assert path.read_bytes() == durable
        rec = jr.append('note', {'again': cut}, 5.0)
        jr.close()
        assert rec['i'] == 2
        assert J.verify_file(path)[0] == 3


def test_garbage_tail_is_cut(tmp_path):
    path = _make(tmp_path, n=1)
    durable = path.read_bytes()
    path.write_bytes(durable + b'\x00\xff{"not":"sealed"}\n{"i":5}\n')
    jr = J.Journal.open(path)
    assert len(jr.replay()) == 2
    jr.close()
    assert path.read_bytes() == durable


def test_valid_record_after_a_bad_region_fails_closed(tmp_path):
    path = _make(tmp_path, n=3)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + lines[1] + b'garbage\n' + lines[3])
    with pytest.raises(J.JournalCorrupt):
        J.Journal.open(path)


def test_removed_record_before_the_last_one_fails_closed(tmp_path):
    path = _make(tmp_path, n=3)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + lines[1] + lines[3])
    with pytest.raises(J.JournalCorrupt):
        J.Journal.open(path)


@pytest.mark.parametrize('mutate', ['edit', 'drop', 'swap'])
def test_edit_drop_or_swap_in_the_middle_fails_closed(tmp_path, mutate):
    path = _make(tmp_path, n=3)
    lines = path.read_bytes().splitlines(keepends=True)
    if mutate == 'edit':
        lines[1] = lines[1].replace(b'"n":0', b'"n":7')
    elif mutate == 'drop':
        del lines[1]
    else:
        lines[1], lines[2] = lines[2], lines[1]
    path.write_bytes(b''.join(lines))
    with pytest.raises(J.JournalCorrupt):
        J.Journal.open(path)
    with pytest.raises(J.JournalCorrupt):
        J.verify_file(path)


def test_empty_or_headless_file_fails_closed(tmp_path):
    path = tmp_path / 'journal.jsonl'
    path.write_bytes(b'')
    with pytest.raises(J.JournalCorrupt):
        J.Journal.open(path)
    path.write_bytes(b'{"i":0')
    with pytest.raises(J.JournalCorrupt):
        J.Journal.open(path)


def test_verify_file_reports_a_torn_tail_without_repair(tmp_path):
    path = _make(tmp_path, n=1)
    torn = path.read_bytes() + b'{"i":2,'
    path.write_bytes(torn)
    with pytest.raises(J.JournalCorrupt):
        J.verify_file(path)
    assert path.read_bytes() == torn


def test_append_returns_only_after_fsync(tmp_path):
    path = _make(tmp_path, n=0)
    calls = []

    class Spy(J.FileOps):
        def write(self, fd, data):
            calls.append('write')
            return super().write(fd, data)

        def fsync(self, fd):
            calls.append('fsync')
            super().fsync(fd)

    jr = J.Journal.open(path, ops=Spy())
    jr.append('note', {'a': 1}, 2.0)
    assert calls and calls[-1] == 'fsync' and 'write' in calls
    jr.close()


def test_short_writes_are_completed(tmp_path):
    path = _make(tmp_path, n=0)

    class Short(J.FileOps):
        def write(self, fd, data):
            return super().write(fd, bytes(data[:7]))

    jr = J.Journal.open(path, ops=Short())
    rec = jr.append('note', {'payload': 'x' * 100}, 2.0)
    jr.close()
    assert J.verify_file(path) == (2, rec['h'])


@pytest.mark.parametrize('fail_at', ['write', 'fsync'])
def test_failed_append_poisons_the_journal(tmp_path, fail_at):
    path = _make(tmp_path, n=1)

    class Fail(J.FileOps):
        armed = False

        def write(self, fd, data):
            if self.armed and fail_at == 'write':
                super().write(fd, bytes(data[:5]))
                raise OSError('disk error')
            return super().write(fd, data)

        def fsync(self, fd):
            if self.armed and fail_at == 'fsync':
                raise OSError('fsync error')
            super().fsync(fd)

    ops = Fail()
    jr = J.Journal.open(path, ops=ops)
    jr.replay()
    ops.armed = True
    with pytest.raises(OSError):
        jr.append('note', {'bad': True}, 3.0)
    ops.armed = False
    with pytest.raises(J.JournalPoisoned):
        jr.append('note', {'after': True}, 4.0)
    assert jr.count == 2
    jr.close()
    again = J.Journal.open(path)
    kinds = [r['b'] for r in again.replay()]
    # The failed record is either fully there or fully gone, never half.
    assert kinds in ([{'run': 'r1'}, {'n': 0}], [{'run': 'r1'}, {'n': 0}, {'bad': True}])
    again.append('note', {'ok': True}, 5.0)
    again.close()
    J.verify_file(path)


def test_closed_journal_refuses_appends(tmp_path):
    path = _make(tmp_path, n=0)
    jr = J.Journal.open(path)
    jr.close()
    with pytest.raises(J.JournalPoisoned):
        jr.append('note', {}, 1.0)


def test_records_are_canonical_single_lines(tmp_path):
    path = _make(tmp_path, n=2)
    for line in path.read_bytes().splitlines():
        assert b'\n' not in line and b' ' not in line
        assert J._self_consistent(line) is not None


def test_root_lock_is_exclusive(tmp_path):
    lock_path = tmp_path / 'controller.lock'
    first = J.RootLock(lock_path).acquire()
    with pytest.raises(J.JournalLocked):
        J.RootLock(lock_path).acquire()
    first.release()
    second = J.RootLock(lock_path).acquire()
    second.release()
    assert os.path.exists(lock_path)
