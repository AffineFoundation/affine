"""An uninterrupted owner may reuse hashes; recovery and changed bytes may not."""
from pathlib import Path
import os
from types import SimpleNamespace

import pytest

from subnet.mimo_grpo import transaction as t


def fixture(root):
    pin = dict(epoch='fixture', checkpoint='a' * 64, optimizer_step=0)
    config = {'objective': 'fixture-only'}
    job = t.Job('fixture-job', 'fixture', 'b' * 64, 'c' * 64,
                t.digest(config), t.digest(pin), 'a' * 64, ('d' * 64,), 0)
    report = dict(version=t.TRAINER, input=pin, optimizer_step=1,
                  input_documents_sha256=['d' * 64], config=config,
                  runtime_context={'manifest_sha256': 'b' * 64})
    calls = {'update': 0, 'stage': 0, 'publish': 0}
    def update():
        calls['update'] += 1
        return report
    def stage(report, path):
        calls['stage'] += 1
        files = {'model/config.json': b'{}', 'model/model.safetensors': b'model',
                 'optimizer/state.bin': b'optimizer'}
        for name, data in files.items():
            p = path / name
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(data)
        return list(files)
    def publish(request):
        calls['publish'] += 1
        assert not any(n.startswith('optimizer/') for n in request['model_files'])
        return dict(job_id=job.job_id, output_checkpoint=request['output_checkpoint'],
                    model_full_get_verified=True, optimizer_state_uploaded=False,
                    receipt_sha256='e' * 64)
    return t.Transaction(root, job), dict(update=update, stage=stage, publish_model=publish), calls


def count_hashes(monkeypatch):
    original = t.hashlib.sha256
    calls = []
    def counted(*args, **kwargs):
        if not args and not kwargs:
            calls.append(True)  # Only streaming artifact hashes use no input.
        return original(*args, **kwargs)
    monkeypatch.setattr(t.hashlib, 'sha256', counted)
    return calls


def expose_changed_ctime(monkeypatch, original):
    """Model a changed FS stamp without assuming wall-clock tick granularity.

    Actual byte writes and mtime restoration still happen in each test. A
    real filesystem may assign both writes the same ctime tick; the owner-lock
    optimization explicitly does not claim to detect hostile concurrent writes
    in that case. This deterministic stat view tests the promised behavior:
    once ctime differs, reuse must stop even if size and mtime are identical.
    """
    fstat, lstat = os.fstat, Path.lstat
    def changed(value):
        if (value.st_dev, value.st_ino) != (original.st_dev, original.st_ino):
            return value
        names = ('st_mode', 'st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        result = {name: getattr(value, name) for name in names}
        result['st_ctime_ns'] = max(value.st_ctime_ns, original.st_ctime_ns + 1)
        return SimpleNamespace(**result)
    monkeypatch.setattr(os, 'fstat', lambda fd: changed(fstat(fd)))
    monkeypatch.setattr(Path, 'lstat', lambda path, **kwargs: changed(lstat(path, **kwargs)))


def test_uninterrupted_owner_hashes_once_and_preserves_exact_journal(tmp_path, monkeypatch):
    first, callbacks, _ = fixture(tmp_path / 'memo')
    hashes = count_hashes(monkeypatch)
    result = first.run(**callbacks)
    assert len(hashes) == 3
    original = t._hash_file
    monkeypatch.setattr(t, '_hash_file', lambda path, **kwargs: original(path))
    baseline, callbacks, _ = fixture(tmp_path / 'cold')
    without_memo = baseline.run(**callbacks)
    assert len(hashes) == 9  # Baseline reads each of the three artifacts twice.
    assert t.canonical(result) == t.canonical(without_memo)
    for name in result['artifacts']:
        assert (first.root / name).read_bytes() == (baseline.root / name).read_bytes()


@pytest.mark.parametrize('change', ['content_restore_mtime', 'replace_inode', 'symlink', 'parent_symlink'])
def test_change_after_staging_is_never_hidden_by_memo(tmp_path, monkeypatch, change):
    transaction, callbacks, calls = fixture(tmp_path / 'job')
    original_write = t._write
    original_stat = []
    hashes = count_hashes(monkeypatch)
    def mutate_after_staged_record(path, value):
        original_write(path, value)
        if value['state'] != 'staged':
            return
        artifact = transaction.root / 'optimizer/state.bin'
        before = artifact.stat()
        original_stat.append(before)
        if change == 'content_restore_mtime':
            artifact.write_bytes(b'OPTIMIZER')
            os.utime(artifact, ns=(before.st_atime_ns, before.st_mtime_ns))
            assert artifact.stat().st_mtime_ns == before.st_mtime_ns
            expose_changed_ctime(monkeypatch, before)
        elif change == 'replace_inode':
            replacement = artifact.with_name('replacement')
            replacement.write_bytes(b'OPTIMIZER')
            os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
            replacement.replace(artifact)
            assert artifact.stat().st_ino != before.st_ino
        elif change == 'symlink':
            target = artifact.with_name('replacement')
            target.write_bytes(artifact.read_bytes())
            artifact.unlink()
            artifact.symlink_to(target)
        else:
            directory = artifact.parent
            moved = directory.with_name('replacement-directory')
            directory.rename(moved)
            directory.symlink_to(moved, target_is_directory=True)
    monkeypatch.setattr(t, '_write', mutate_after_staged_record)
    with pytest.raises(ValueError, match='staged artifact integrity|ordinary local staged file'):
        transaction.run(**callbacks)
    assert calls == {'update': 1, 'stage': 1, 'publish': 0}
    assert len(hashes) == (4 if change in ('content_restore_mtime', 'replace_inode') else 3)


def test_same_bytes_new_inode_rehashes_instead_of_trusting_replacement(tmp_path, monkeypatch):
    path = tmp_path / 'artifact'
    path.write_bytes(b'same contents')
    memo = {}
    hashes = count_hashes(monkeypatch)
    expected = t._hash_file(path, memo=memo)
    old = path.stat()
    replacement = tmp_path / 'replacement'
    replacement.write_bytes(path.read_bytes())
    os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
    replacement.replace(path)
    assert t._hash_file(path, memo=memo) == expected
    assert len(hashes) == 2


@pytest.mark.parametrize('new_object', [False, True])
def test_publication_retry_always_rehashes_even_on_same_object(tmp_path, monkeypatch, new_object):
    transaction, callbacks, calls = fixture(tmp_path / 'job')
    hashes = count_hashes(monkeypatch)
    def lose_ack(request):
        raise TimeoutError('publication reply lost')
    with pytest.raises(TimeoutError):
        transaction.run(**dict(callbacks, publish_model=lose_ack))
    assert len(hashes) == 3
    if new_object:
        transaction = t.Transaction(transaction.root, transaction.job)
    result = transaction.run(**callbacks)
    assert result['state'] == 'published' and len(hashes) == 6
    assert calls == {'update': 1, 'stage': 1, 'publish': 1}


def test_failed_publication_then_modified_artifact_cannot_resume(tmp_path, monkeypatch):
    transaction, callbacks, calls = fixture(tmp_path / 'job')
    def lose_ack(request):
        raise TimeoutError('publication reply lost')
    with pytest.raises(TimeoutError):
        transaction.run(**dict(callbacks, publish_model=lose_ack))
    path = transaction.root / 'optimizer/state.bin'
    before = path.stat()
    path.write_bytes(b'OPTIMIZER')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    cold = t.Transaction(transaction.root, transaction.job)
    with pytest.raises(ValueError, match='staged artifact integrity'):
        cold.run(**callbacks)
    assert calls == {'update': 1, 'stage': 1, 'publish': 0}


def test_stage_interruption_never_retrains_or_uses_partial_memo(tmp_path, monkeypatch):
    transaction, callbacks, calls = fixture(tmp_path / 'job')
    def fail_stage(report, path):
        callbacks['stage'](report, path)
        raise OSError('interrupted local export')
    with pytest.raises(OSError):
        transaction.run(**dict(callbacks, stage=fail_stage))
    with pytest.raises(t.RecoveryRequired):
        transaction.run(**callbacks)
    assert calls == {'update': 1, 'stage': 1, 'publish': 0}


def test_same_size_mutation_during_hash_with_restored_mtime_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'artifact'
    path.write_bytes(b'original')
    before = path.stat()
    original = t.hashlib.sha256
    class MutatingDigest:
        def __init__(self): self.value = original()
        def update(self, data):
            self.value.update(data)
            path.write_bytes(b'MODIFIED')
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            expose_changed_ctime(monkeypatch, before)
        def hexdigest(self): return self.value.hexdigest()
    monkeypatch.setattr(t.hashlib, 'sha256', MutatingDigest)
    memo = {}
    with pytest.raises(ValueError, match='changed during hashing'):
        t._hash_file(path, memo=memo)
    assert memo == {}


def test_memo_hit_still_fsyncs_and_checks_descriptor(tmp_path, monkeypatch):
    path = tmp_path / 'artifact'
    path.write_bytes(b'original')
    memo = {}
    fsyncs = []
    original = t.os.fsync
    def fsync(fd):
        fsyncs.append(fd)
        original(fd)
    monkeypatch.setattr(t.os, 'fsync', fsync)
    expected = t._hash_file(path, memo=memo)
    assert t._hash_file(path, memo=memo) == expected
    assert len(fsyncs) == 2
