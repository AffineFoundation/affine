"""The live queue as a batch source: what it lists, what it skips, and that it only reads."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from . import feed as F
from .interfaces import BatchRef, SourceError

HOTKEY = '5F' + 'a' * 46
CHECKPOINT = 'c' * 64
DOC = 'd' * 64
NONE_SKIPPED = dict.fromkeys(F.REASONS, 0)
FILES = {'turn': 'K1-TURN.json', 'job': 'TRAINING-JOB.json', 'dispatch': 'TRAIN-DISPATCH-FF.json'}


def write(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def turn(n: int, **over: Any) -> Dict[str, Any]:
    return dict({'epoch': 'run-e%06d' % n, 'epoch_index': n, 'uid': 7, 'hotkey': HOTKEY, 'status': 'turn'}, **over)


def job(checkpoint: Any = CHECKPOINT) -> Dict[str, Any]:
    return {'payload': {'manifest': {'payload': {'checkpoint': {'id': checkpoint, 'files': {}}}}, 'training': {}}}


def two_ids(manifest_id: str) -> Dict[str, Any]:
    """A job that names one id in its manifest and ``CHECKPOINT`` beside the manifest."""
    doc = job(manifest_id)
    doc['payload']['checkpoint'] = {'id': CHECKPOINT}
    return doc


def dispatch(*shas: str, **over: Any) -> Dict[str, Any]:
    rows = [{'document': {'sha256': sha, 'key': 'docs/%s.json' % sha[:8]}} for sha in shas or (DOC,)]
    return dict({'signature': 'sig', 'signer': 'root', 'payload': {'selected': rows}}, **over)


def life(tmp_path: Path, floor: Any = 10) -> Path:
    """A life root of the test, away from the directories that ``conftest`` seeds for the environment."""
    root = tmp_path / 'own' / 'life'
    (root / 'epochs').mkdir(parents=True)
    write(root / 'CURRENT.json', {'epoch_index': floor, 'checkpoint': {'id': CHECKPOINT}})
    return root


def epoch(root: Path, n: int, *marks: str, **docs: Any) -> Path:
    """A complete epoch directory. ``marks`` are marker files. A document that is None is not written."""
    for kind, doc in dict({'turn': turn(n), 'job': job(), 'dispatch': dispatch()}, **docs).items():
        if doc is not None:
            write(root / 'epochs' / str(n) / FILES[kind], doc)
    for mark in marks:
        write(root / 'epochs' / str(n) / mark, {})
    return root / 'epochs' / str(n)


def snapshot(top: Path) -> Dict[str, Tuple[int, int, int]]:
    """Mode, size and mtime of every file and directory. A directory changes when an entry comes or goes."""
    stats = {str(path): path.lstat() for path in [top, *top.rglob('*')]}
    return {name: (stat.st_mode, stat.st_size, stat.st_mtime_ns) for name, stat in stats.items()}


def test_a_batch_id_is_the_letter_e_and_the_epoch_number():
    assert (F.batch_id_of(0), F.batch_id_of(1050)) == ('e0', 'e1050')
    assert [F.epoch_of(F.batch_id_of(n)) for n in (0, 7, 999999999)] == [0, 7, 999999999]
    for bad in ('', 'e', 'e01', 'e-1', 'E12', '12', ' e12', 'e12\n', 'e1000000000', 'e12/../13', 'e\u0661\u0662',
                None, 12):
        assert F.epoch_of(bad) is None


def test_the_listing_is_by_epoch_and_names_id_miner_and_reference(tmp_path):
    root = life(tmp_path)
    for n in (12, 10, 11):
        epoch(root, n, turn=turn(n, uid=n))
    queue = F.LiveQueue(root)
    found = queue.batches()
    assert [batch.epoch for batch in found] == [10, 11, 12] and queue.skipped == NONE_SKIPPED
    assert found[2] == F.LiveBatch(12, 12, HOTKEY, root / 'epochs' / '12', CHECKPOINT, (DOC,), 'k1-turn')
    ref = hashlib.sha256(('{"documents":["%s"],"epoch":12}' % DOC).encode()).hexdigest()
    listed = queue.list()
    assert [row.batch_id for row in listed] == ['e10', 'e11', 'e12']
    assert listed[2] == found[2].as_ref() == BatchRef('e12', '12:5F' + 'a' * 14, ref)


def test_the_inbox_wins_and_gives_the_same_batch_as_the_turn(tmp_path):
    root = life(tmp_path)
    epoch(root, 12)
    inbox = tmp_path / 'own' / 'inbox'
    queue = F.LiveQueue(root, inboxes=[tmp_path / 'own' / 'no-inbox', inbox])
    (from_turn,) = queue.batches()
    write(inbox / 'ticket.json', {'epoch': 12, 'uid': '7', 'hotkey_prefix': HOTKEY[:16]})
    (from_inbox,) = queue.batches()
    assert (from_turn.source, from_inbox.source) == ('k1-turn', 'inbox')
    assert (from_turn.hotkey, from_inbox.hotkey) == (HOTKEY, HOTKEY[:16])
    assert from_inbox.as_ref() == from_turn.as_ref() == BatchRef('e12', '7:' + HOTKEY[:16], from_turn.ref)


def test_the_reference_names_the_documents_and_not_their_order(tmp_path):
    root = life(tmp_path)
    a, b, c = ('%064x' % n for n in (1, 2, 3))
    epoch(root, 12, dispatch=dispatch(b, a))
    queue = F.LiveQueue(root)
    (first,) = queue.batches()
    assert first.documents == (a, b)
    epoch(root, 12, dispatch=dispatch(a, b, a))
    assert queue.batches() == [first]
    epoch(root, 12, dispatch=dispatch(a, c))
    (changed,) = queue.batches()
    assert changed.ref != first.ref and (changed.batch_id, changed.miner) == (first.batch_id, first.miner)
    epoch(root, 13, dispatch=dispatch(a, c))
    assert queue.batch('e13').ref != changed.ref


def test_below_the_floor_only_an_epoch_with_an_open_pit_dispatch_is_listed(tmp_path):
    root = life(tmp_path, floor=20)
    epoch(root, 12)
    epoch(root, 13, 'FINALIZED.json', 'PIT-DISPATCHED.json')
    epoch(root, 14, 'PIT-DISPATCHED.json', 'PIT-RESULT.json')
    epoch(root, 20)
    queue = F.LiveQueue(root)
    assert [batch.batch_id for batch in queue.batches()] == ['e13', 'e20']
    assert queue.skipped == dict(NONE_SKIPPED, below_floor=2)


@pytest.mark.parametrize('reason, marks, docs', [
    ('not_ready', ['TRAINING-RESULT.json'], {}),
    ('not_ready', ['PIT-RESULT.json'], {}),
    ('not_ready', ['RERUN-ACTIVE.json'], {}),
    ('not_ready', ['FINALIZED.json'], {}),
    ('not_ready', [], {'job': None}),
    ('not_ready', [], {'dispatch': None}),
    ('job', [], {'job': job('C' * 64)}),
    ('job', [], {'job': {'payload': {'training': {}}}}),
    ('dispatch', ['TRAIN-DISPATCH-K1.json'], {'dispatch': None}),
    ('dispatch', [], {'dispatch': dispatch(signature='')}),
    ('dispatch', [], {'dispatch': dispatch(signer='SHADOW-UNSIGNED')}),
    ('dispatch', [], {'dispatch': dispatch(payload={'selected': []})}),
    ('dispatch', [], {'dispatch': dispatch(payload={'selected': [{'document': {'sha256': DOC, 'key': ''}}]})}),
    ('dispatch', [], {'dispatch': dispatch(payload={'selected': [{'document': {'sha256': 'D' * 64, 'key': 'k'}}]})}),
    ('dispatch', [], {'dispatch': [dispatch()]}),
    ('no_miner', [], {'turn': turn(12, uid=None)}),
    ('no_miner', [], {'turn': turn(12, uid=True)}),
    ('no_miner', [], {'turn': turn(12, uid=-1)}),
    ('no_miner', [], {'turn': turn(12, hotkey='')}),
])
def test_an_epoch_that_is_not_complete_is_skipped_with_its_reason(tmp_path, reason, marks, docs):
    root = life(tmp_path)
    epoch(root, 12, *marks, **docs)
    epoch(root, 13)
    queue = F.LiveQueue(root)
    assert [batch.epoch for batch in queue.batches()] == [13]
    assert queue.skipped == dict(NONE_SKIPPED, **{reason: 1})


@pytest.mark.parametrize('doc, found', [
    (job(), CHECKPOINT),
    ({'payload': {'manifest': {'checkpoint': {'id': CHECKPOINT}}}}, CHECKPOINT),
    ({'manifest': {'payload': {'checkpoint': {'id': CHECKPOINT}}}}, CHECKPOINT),
    ({'payload': {'manifest': {'payload': {}}, 'checkpoint': {'id': CHECKPOINT}}}, CHECKPOINT),
    (two_ids('a' * 64), 'a' * 64),
    # The live queue reads the id of the manifest here. It is not a sha256, so the job names no checkpoint.
    (two_ids('G' * 64), None),
    ({'payload': {'checkpoint': {'id': CHECKPOINT[:63]}}}, None),
    ([job()], None),
])
def test_the_checkpoint_of_a_job_is_found_in_the_order_of_the_live_queue(tmp_path, doc, found):
    assert F.job_checkpoint(write(tmp_path / 'own' / 'TRAINING-JOB.json', doc)) == found
    assert F.job_checkpoint(tmp_path / 'own' / 'none.json') is None


@pytest.mark.parametrize('current', [
    None, 'not json', '[12]', '{}', '{"epoch_index": null}', '{"epoch_index": "x"}', '{"epoch_index": true}',
    '{"epoch_index": -1}', '{"epoch_index": 12.5}',
])
def test_without_a_floor_the_listing_fails(tmp_path, current):
    root = life(tmp_path)
    epoch(root, 12)
    if current is None:
        (root / 'CURRENT.json').unlink()
    else:
        (root / 'CURRENT.json').write_text(current)
    queue = F.LiveQueue(root)
    with pytest.raises(SourceError):
        queue.batches()
    with pytest.raises(SourceError):
        queue.list()


def test_the_turns_log_and_the_pending_directory_are_sources_after_the_turn(tmp_path):
    root, own = life(tmp_path), tmp_path / 'own'
    epoch(root, 12)
    for n in (15, 16, 17, 18):
        epoch(root, n, turn=None)
    # 203 lines: the turn of epoch 18 is not in the last 200.
    lines = [json.dumps(turn(18)), *['{}'] * 199, 'not json', '[1]', json.dumps(turn(15, uid=5))]
    (own / 'K1-TURNS.jsonl').write_text('\n'.join(lines) + '\n')
    write(own / 'pending' / 'e12.json', {'uid': 9, 'hotkey': HOTKEY})
    write(own / 'pending' / 'e16.json', {'uid': 6, 'hotkey_prefix': 'prefix'})
    write(own / 'pending' / 'e17.json', {'uid': 9, 'hotkey': HOTKEY, 'pit_claimed': True})
    write(own / 'pending' / 'x18.json', {'epoch': 18, 'uid': 9, 'hotkey': HOTKEY})
    queue = F.LiveQueue(root, turns_log=own / 'K1-TURNS.jsonl', pending=own / 'pending')
    assert [(batch.epoch, batch.uid, batch.source) for batch in queue.batches()] == [
        (12, 7, 'k1-turn'), (15, 5, 'k1-turns-jsonl'), (16, 6, 'pending')]
    assert queue.skipped == NONE_SKIPPED and queue.batch('e16').miner == '6:prefix'
    assert [batch.epoch for batch in F.LiveQueue(root).batches()] == [12]


def test_batch_reads_an_admitted_epoch_again_when_the_floor_moved(tmp_path):
    root = life(tmp_path)
    epoch_dir = epoch(root, 12)
    queue = F.LiveQueue(root)
    (admitted,) = queue.batches()
    write(root / 'CURRENT.json', {'epoch_index': 20})
    write(epoch_dir / 'FINALIZED.json', {})
    assert queue.batches() == [] and queue.batch('e12') == admitted
    (root / 'CURRENT.json').unlink()
    assert queue.batch('e12') == admitted
    for bad in ('e13', 'e012', '12', '', 'e12/../13', None):
        assert queue.batch(bad) is None


@pytest.mark.parametrize('docs', [
    {'turn': None}, {'turn': turn(12, uid=None)}, {'job': job(None)}, {'dispatch': dispatch(signer='SHADOW-UNSIGNED')},
    {'dispatch': None},
])
def test_batch_is_none_when_a_part_of_the_epoch_is_missing(tmp_path, docs):
    root = life(tmp_path)
    epoch(root, 12, **docs)
    assert F.LiveQueue(root).batch('e12') is None


def test_one_bad_file_does_not_fail_the_listing(tmp_path):
    root, own = life(tmp_path), tmp_path / 'own'
    epoch(root, 12)
    (epoch(root, 10, turn=None) / 'K1-TURN.json').mkdir()
    (epoch(root, 11, turn=None) / 'K1-TURN.json').write_text('{"epoch_index": 11')
    epoch(root, 13, turn=turn(13, epoch_index=float('inf')))
    write(own / 'inbox' / 'list.json', [12])
    (own / 'inbox' / 'bytes.json').write_bytes(b'\xff\xfe\x00')
    queue = F.LiveQueue(root, inboxes=[own / 'inbox'])
    assert [batch.batch_id for batch in queue.batches()] == ['e12'] and queue.skipped == NONE_SKIPPED


@pytest.mark.skipif(os.geteuid() == 0, reason='root can read every directory')
@pytest.mark.parametrize('name', ['inbox', 'pending', 'life/epochs'])
def test_a_directory_that_cannot_be_read_fails_the_call(tmp_path, name):
    root, own = life(tmp_path), tmp_path / 'own'
    epoch(root, 12)
    (own / 'inbox').mkdir()
    (own / 'pending').mkdir()
    queue = F.LiveQueue(root, inboxes=[own / 'inbox'], pending=own / 'pending')
    assert queue.batch('e12') is not None and len(queue.batches()) == 1
    os.chmod(own / name, 0)
    try:
        with pytest.raises(SourceError):
            queue.batches()
        with pytest.raises(SourceError):
            queue.batch('e12')
    finally:
        os.chmod(own / name, 0o755)


def test_a_scan_writes_nothing(tmp_path):
    root, own = life(tmp_path), tmp_path / 'own'
    epoch(root, 9)
    epoch(root, 12)
    epoch(root, 13, 'TRAINING-RESULT.json')
    write(own / 'inbox' / 'ticket.json', {'epoch': 12, 'uid': 7, 'hotkey': HOTKEY})
    write(own / 'pending' / 'e12.json', {'uid': 7, 'hotkey': HOTKEY})
    (own / 'K1-TURNS.jsonl').write_text(json.dumps(turn(12)) + '\n')
    queue = F.LiveQueue(root, inboxes=[own / 'inbox'], turns_log=own / 'K1-TURNS.jsonl', pending=own / 'pending')
    before = snapshot(own)
    listed = queue.list()
    assert [row.batch_id for row in listed] == ['e12'] and queue.batch('e12').as_ref() == listed[0]
    assert queue.skipped == dict(NONE_SKIPPED, below_floor=1, not_ready=1)
    assert snapshot(own) == before

    paths = [own, *own.rglob('*')]
    try:
        for path in paths:
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        locked = snapshot(own)
        if os.geteuid() != 0:
            with pytest.raises(OSError):
                (root / 'epochs' / '12' / 'probe').write_text('x')
        assert queue.list() == listed and queue.batch('e12').as_ref() == listed[0]
        assert snapshot(own) == locked
    finally:
        for path in paths:
            os.chmod(path, before[str(path)][0] & 0o7777)


def test_the_paths_come_from_the_caller_and_not_from_the_environment(tmp_path, monkeypatch):
    root, decoy = life(tmp_path), tmp_path / 'decoy'
    epoch(root, 12)
    write(decoy / 'CURRENT.json', {'epoch_index': 0})
    epoch(decoy, 99)
    write(decoy / 'inbox' / 'ticket.json', {'epoch': 12, 'uid': 1, 'hotkey': 'other'})
    for name in ('LIFE_ROOT', 'PE_LIFE', 'PE_STATE', 'PIT_STATE', 'PE_LIFE_STATE'):
        monkeypatch.setenv(name, str(decoy))
    assert [batch.as_ref().miner for batch in F.LiveQueue(root).batches()] == ['7:' + HOTKEY[:16]]
    with pytest.raises(TypeError):
        F.LiveQueue(root, inboxes=str(decoy / 'inbox'))


def test_the_source_text_can_only_read():
    text = Path(F.__file__).read_text()
    for word in ('write_text', 'write_bytes', 'os.replace', 'unlink', 'mkdir', 'open(', 'os.environ', 'getenv',
                 'subprocess', 'socket'):
        assert word not in text, word
    assert set(re.findall(r'\bQ\.(\w+)', text)) == {'_epoch_index_of', '_above_floor_or_pit_open', '_epoch_dir_ready'}
    tree = ast.parse(text)
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert imports and all(node in tree.body for node in imports)
    local = {(node.level, node.module) for node in imports if isinstance(node, ast.ImportFrom) and node.level}
    assert local == {(3, 'durable'), (2, None), (1, 'interfaces'), (1, 'layout')}
    for node in imports:
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or '']
        if not getattr(node, 'level', 0):
            assert all(name.split('.')[0] in sys.stdlib_module_names for name in names), names
