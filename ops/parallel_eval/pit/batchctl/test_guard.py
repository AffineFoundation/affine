"""Shadow-mode hard denies.

Each test names one thing that the controller must never reach: the live
HEAD, live R2 keys, the ledger, weights and extrinsics, ``mining.json``,
``llms.txt``, the website, Discord, a box of someone else.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from . import BOX_PREFIX_LIVE, BOX_PREFIX_SHADOW, MODE_LIVE, MODE_SHADOW, R2_LIVE_ROOT, R2_SHADOW_ROOT
from . import guard as G
from . import interfaces as I
from .guard import (
    GuardedStore, LiveModeRefused, StateRoot, box_name, check_key, check_mode, check_outside_live,
    own_prefix, parse_box_name, run_prefix, shadow_prefix,
)
from ..shadow import ShadowWriteError
from ..shadow_guard import R2_PREFIX as PIT_SHADOW_PREFIX
from ..shadow_guard import ShadowGuard
from ..shadow_guard import ShadowWriteError as PitShadowWriteError

RUN = 'run0001'
PREFIX = R2_SHADOW_ROOT + RUN + '/'
LIVE_PREFIX = R2_LIVE_ROOT + RUN + '/'
PACKAGE = Path(__file__).parent

LIVE_KEYS = [
    'HEAD.json',
    'heads/current',
    'pit/HEAD-CHAIN.json',
    'pit/ckpts/CURRENT.json',
    'ledger.json',
    'mining.json',
    'llms.txt',
    'weights.json',
    'index.html',
    'shadow/HEAD.json',
    'shadow/batchctl/HEAD.json',
    'shadow/batchctl/other0001/HEAD.json',
    'shadow/batchctl/run0001/HEAD.json',
    'pit-shadow/HEAD.json',
    'pit-shadow/batchctl/HEAD.json',
    'pit-shadow/batchctl/other0001/HEAD.json',
    'pit-shadow/other/run0001/HEAD.json',
    PREFIX,
    PREFIX[:-1],
    PREFIX + '../other0001/HEAD.json',
    PREFIX + '../../../HEAD.json',
    PREFIX + './HEAD.json',
    PREFIX + 'states//manifest.json',
    PREFIX + 'a\\b',
    PREFIX + 'a\x00b',
    '/' + PREFIX + 'HEAD.json',
    'x' + PREFIX + 'HEAD.json',
    None,
    7,
]

LIVE_NAMES = [
    'mining.json', 'llms.txt', 'llms-full.txt', 'ledger.json', 'gate-ledger.json',
    'weights.json', 'WEIGHTS.json', 'weights.pt', 'extrinsics.json', 'index.html',
    'DECISION.json', 'HEAD-CHAIN.json', 'CURRENT.json', 'FINALIZED.json',
    'R2_HEAD_POINTER', 'r2-head-pointer', 'Mining.JSON', 'LLMS.TXT',
]


class Recorder:
    """A store that only records. No call may reach it in a deny test."""

    def __init__(self):
        self.calls = []

    def get(self, key, *, max_bytes):
        self.calls.append(('get', key))
        return I.Blob(b'', 'e')

    def head(self, key):
        self.calls.append(('head', key))
        return None

    def digest(self, key):
        self.calls.append(('digest', key))
        return I.Digest(0, '0' * 64, 'e')

    def put(self, key, data, *, if_none_match=False, if_match=None):
        self.calls.append(('put', key))
        return 'e'

    def copy(self, src, dst, *, src_etag):
        self.calls.append(('copy', src, dst))

    def delete(self, key):
        self.calls.append(('delete', key))

    def list(self, prefix):
        self.calls.append(('list', prefix))
        return [I.ObjectInfo(prefix + 'a', 1, 'e')]


def every_call(store: GuardedStore, key):
    good = store.prefix + 'states/' + 'a' * 64 + '/manifest.json'
    return [
        lambda: store.get(key, max_bytes=10),
        lambda: store.head(key),
        lambda: store.digest(key),
        lambda: store.put(key, b'x'),
        lambda: store.put(key, b'x', if_none_match=True),
        lambda: store.put(key, b'x', if_match='e'),
        lambda: store.copy(key, good, src_etag='e'),
        lambda: store.copy(good, key, src_etag='e'),
        lambda: store.delete(key),
    ]


# ----------------------------------------------------------------------------
# Mode
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('mode', ['live', 'LIVE', 'shadow ', 'Shadow', '', None, 'promote'])
def test_every_mode_but_shadow_is_refused(mode):
    with pytest.raises(LiveModeRefused):
        check_mode(mode)
    check_mode(MODE_SHADOW)
    assert issubclass(LiveModeRefused, ShadowWriteError)


@pytest.mark.parametrize('mode', ['LIVE', 'Live', 'live ', 'shadow ', 'prod', 'promote', '', None, 7, True, ['live']])
def test_an_unknown_mode_is_refused_with_and_without_the_live_flag(mode):
    for armed in (False, True):
        with pytest.raises(LiveModeRefused):
            check_mode(mode, armed)


@pytest.mark.parametrize('flag', [False, None, 0, 1, 'true', 'yes', [True]], ids=repr)
def test_live_mode_needs_its_explicit_flag(flag):
    with pytest.raises(LiveModeRefused, match='live mode needs its explicit flag'):
        check_mode(MODE_LIVE)
    with pytest.raises(LiveModeRefused, match='live mode needs its explicit flag'):
        check_mode(MODE_LIVE, flag)
    with pytest.raises(LiveModeRefused, match='live mode needs its explicit flag'):
        check_mode(MODE_LIVE, live_armed=flag)
    check_mode(MODE_LIVE, True)
    check_mode(MODE_LIVE, live_armed=True)
    # Shadow needs no flag, and the flag does not change it.
    check_mode(MODE_SHADOW)
    check_mode(MODE_SHADOW, live_armed=True)


# ----------------------------------------------------------------------------
# R2
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('key', LIVE_KEYS)
def test_r2_key_outside_the_shadow_prefix_is_refused_for_every_call(key):
    inner = Recorder()
    store = GuardedStore(inner, RUN)
    for call in every_call(store, key):
        with pytest.raises(ShadowWriteError):
            call()
    assert inner.calls == []


@pytest.mark.parametrize('name', LIVE_NAMES)
def test_r2_key_named_like_a_live_artifact_is_refused_inside_the_prefix(name):
    inner = Recorder()
    store = GuardedStore(inner, RUN)
    for key in (PREFIX + name, PREFIX + 'states/' + name, PREFIX + 'staging/b1/000000000001/obj/' + name):
        for call in every_call(store, key):
            with pytest.raises(ShadowWriteError):
                call()
    assert inner.calls == []


def test_r2_listing_outside_the_shadow_prefix_is_refused():
    inner = Recorder()
    store = GuardedStore(inner, RUN)
    for prefix in ('', 'shadow/', R2_SHADOW_ROOT, R2_SHADOW_ROOT + 'other0001/', PREFIX + '../', PREFIX[:-1]):
        with pytest.raises(ShadowWriteError):
            store.list(prefix)
    assert inner.calls == []
    assert [info.key for info in store.list(PREFIX + 'staging/')] == [PREFIX + 'staging/a']

    class Leaky(Recorder):
        def list(self, prefix):
            return [I.ObjectInfo('HEAD.json', 1, 'e')]

    with pytest.raises(ShadowWriteError):
        GuardedStore(Leaky(), RUN).list(PREFIX)


def test_shadow_keys_pass_through_unchanged():
    inner = Recorder()
    store = GuardedStore(inner, RUN)
    assert store.prefix == PREFIX == shadow_prefix(RUN)
    key = PREFIX + 'HEAD.json'
    store.get(key, max_bytes=5)
    store.put(key, b'x', if_match='e')
    store.copy(PREFIX + 'staging/b/000000000001/obj/w', PREFIX + 'states/' + 'a' * 64 + '/obj/w', src_etag='e')
    store.delete(PREFIX + 'staging/b/000000000001/obj/w')
    assert [call[0] for call in inner.calls] == ['get', 'put', 'copy', 'delete']
    assert check_key(PREFIX, key) == key


@pytest.mark.parametrize('run_id', [
    '', 'abc', 'A' * 8, 'run/0001', 'run-0001', '../x', 'x' * 33, 'run0001\n', ' run0001', None, 5,
])
def test_bad_run_id_is_refused(run_id):
    for call in (lambda: shadow_prefix(run_id), lambda: own_prefix(run_id), lambda: GuardedStore(Recorder(), run_id)):
        with pytest.raises(ShadowWriteError):
            call()


def test_the_prefix_of_a_run_is_inside_the_one_shadow_prefix_of_the_pit_package():
    assert R2_SHADOW_ROOT == 'pit-shadow/batchctl/' and R2_SHADOW_ROOT.startswith(PIT_SHADOW_PREFIX)
    assert shadow_prefix(RUN) == 'pit-shadow/batchctl/%s/' % RUN
    guard = ShadowGuard(state_root=Path('/nonexistent-batchctl-root'))
    guard.check_s3('PutObject', {'Bucket': 'b', 'Key': shadow_prefix(RUN) + 'HEAD.json'})
    with pytest.raises(PitShadowWriteError):
        guard.check_s3('PutObject', {'Bucket': 'b', 'Key': 'shadow/batchctl/%s/HEAD.json' % RUN})


def test_a_prefix_that_is_not_a_shadow_prefix_is_refused():
    for prefix in ('', 'live/', 'pit/', 'pit-shadow/', 'pit-shadow/batchctl', 'pit-shadow/other/run0001/',
                   'shadow/batchctl/run0001/', 'xpit-shadow/batchctl/run0001/'):
        with pytest.raises(ShadowWriteError):
            check_key(prefix, prefix + 'HEAD.json')


def test_the_prefix_of_a_live_run_is_outside_the_shadow_prefix():
    assert R2_LIVE_ROOT == 'pit/batchctl/' and not R2_LIVE_ROOT.startswith(PIT_SHADOW_PREFIX)
    assert run_prefix(RUN, MODE_LIVE) == 'pit/batchctl/%s/' % RUN == LIVE_PREFIX
    assert run_prefix(RUN) == run_prefix(RUN, MODE_SHADOW) == shadow_prefix(RUN) == PREFIX
    assert not LIVE_PREFIX.startswith(PIT_SHADOW_PREFIX) and not PREFIX.startswith(R2_LIVE_ROOT)
    # The shadow guard of the PIT package refuses a write to the prefix of a live run.
    guard = ShadowGuard(state_root=Path('/nonexistent-batchctl-root'))
    with pytest.raises(PitShadowWriteError):
        guard.check_s3('PutObject', {'Bucket': 'b', 'Key': LIVE_PREFIX + 'HEAD.json'})
    for mode in ('LIVE', 'prod', '', None):
        with pytest.raises(LiveModeRefused):
            run_prefix(RUN, mode)
        with pytest.raises(LiveModeRefused):
            GuardedStore(Recorder(), RUN, mode)
    for prefix in ('pit/', 'pit/batchctl', 'pit/other/run0001/', 'xpit/batchctl/run0001/', 'live/batchctl/run0001/'):
        with pytest.raises(ShadowWriteError):
            check_key(prefix, prefix + 'HEAD.json')


@pytest.mark.parametrize('key', LIVE_KEYS + [
    PREFIX + 'HEAD.json',
    PREFIX + 'RUN.json',
    PREFIX + 'states/' + 'a' * 64 + '/manifest.json',
    'pit/batchctl/HEAD.json',
    'pit/batchctl/other0001/HEAD.json',
    'pit-shadow/' + LIVE_PREFIX + 'HEAD.json',
    LIVE_PREFIX,
    LIVE_PREFIX[:-1],
    LIVE_PREFIX + '../other0001/HEAD.json',
    LIVE_PREFIX + '../../../' + PREFIX + 'HEAD.json',
    LIVE_PREFIX + './HEAD.json',
    LIVE_PREFIX + 'states//manifest.json',
    LIVE_PREFIX + 'a\\b',
    LIVE_PREFIX + 'ledger.json',
    LIVE_PREFIX + 'states/WEIGHTS.json',
    '/' + LIVE_PREFIX + 'HEAD.json',
    'x' + LIVE_PREFIX + 'HEAD.json',
])
def test_a_live_store_refuses_every_shadow_key_and_every_key_outside_its_prefix(key):
    inner = Recorder()
    store = GuardedStore(inner, RUN, MODE_LIVE)
    assert store.mode == MODE_LIVE and store.prefix == LIVE_PREFIX
    for call in every_call(store, key):
        with pytest.raises(ShadowWriteError):
            call()
    assert inner.calls == []


def test_a_live_store_reaches_its_own_prefix_and_a_shadow_store_refuses_it():
    inner = Recorder()
    live = GuardedStore(inner, RUN, MODE_LIVE)
    key = LIVE_PREFIX + 'HEAD.json'
    for prefix in ('', 'pit/', R2_LIVE_ROOT, R2_LIVE_ROOT + 'other0001/', R2_SHADOW_ROOT, PREFIX, PREFIX + 'staging/',
                   LIVE_PREFIX + '../', LIVE_PREFIX[:-1]):
        with pytest.raises(ShadowWriteError):
            live.list(prefix)
    assert inner.calls == []
    live.get(key, max_bytes=5)
    live.put(key, b'x', if_match='e')
    live.copy(LIVE_PREFIX + 'staging/b/000000000001/obj/w', LIVE_PREFIX + 'states/' + 'a' * 64 + '/obj/w', src_etag='e')
    live.delete(LIVE_PREFIX + 'staging/b/000000000001/obj/w')
    assert [info.key for info in live.list(LIVE_PREFIX + 'staging/')] == [LIVE_PREFIX + 'staging/a']
    assert [call[0] for call in inner.calls] == ['get', 'put', 'copy', 'delete', 'list']
    assert check_key(LIVE_PREFIX, key) == key

    inner = Recorder()
    shadow = GuardedStore(inner, RUN)
    assert shadow.mode == MODE_SHADOW and shadow.prefix == PREFIX
    for live_key in (key, LIVE_PREFIX + 'RUN.json', LIVE_PREFIX + 'states/' + 'a' * 64 + '/manifest.json',
                     LIVE_PREFIX, R2_LIVE_ROOT + 'HEAD.json'):
        for call in every_call(shadow, live_key):
            with pytest.raises(ShadowWriteError):
                call()
    for prefix in (R2_LIVE_ROOT, LIVE_PREFIX, LIVE_PREFIX + 'staging/'):
        with pytest.raises(ShadowWriteError):
            shadow.list(prefix)
    assert inner.calls == []


# ----------------------------------------------------------------------------
# Local state root
# ----------------------------------------------------------------------------

def test_state_root_inside_or_around_a_live_root_is_refused(tmp_path, monkeypatch):
    live = tmp_path / 'live-state'
    live.mkdir()
    for key in ('PE_STATE', 'PIT_STATE', 'PE_LIFE_STATE', 'HG_GATE_STATE', 'PIT_FP32_MASTER_INDEX'):
        env = {key: str(live)}
        for bad in (live, live / 'batchctl', live / 'a' / 'b', tmp_path):
            with pytest.raises(ShadowWriteError):
                check_outside_live(bad, env)
            with pytest.raises(ShadowWriteError):
                StateRoot.create(bad, RUN, env)
        assert check_outside_live(tmp_path / 'ok', env) == (tmp_path / 'ok').resolve()
    assert not (live / 'batchctl').exists()

    # A symlink into a live root is the live root.
    link = tmp_path / 'link'
    link.symlink_to(live)
    with pytest.raises(ShadowWriteError):
        StateRoot.create(link / 'x', RUN, {'PE_STATE': str(live)})

    # With no environment given, the live roots of the process environment count. They are set
    # here: a shell that names its own PE_STATE must not change what this test proves.
    monkeypatch.setenv('PE_STATE', str(tmp_path / 'pe'))
    monkeypatch.setenv('PIT_STATE', str(tmp_path / 'pe' / 'pit'))
    with pytest.raises(ShadowWriteError):
        StateRoot.create(tmp_path, RUN)
    with pytest.raises(ShadowWriteError):
        StateRoot.create(tmp_path / 'pe' / 'pit' / 'x', RUN)


def test_state_root_needs_its_own_marker(tmp_path):
    path = tmp_path / 'root'
    root = StateRoot.create(path, RUN)
    assert sorted(p.name for p in path.iterdir()) == [G.MARKER]
    assert StateRoot.open(path, RUN).path == root.path

    with pytest.raises(ShadowWriteError):
        StateRoot.open(path, 'run0002')
    with pytest.raises(ShadowWriteError):
        StateRoot.create(path, RUN)
    with pytest.raises(ShadowWriteError):
        StateRoot.open(tmp_path / 'missing', RUN)
    plain = tmp_path / 'plain'
    plain.mkdir()
    with pytest.raises(ShadowWriteError):
        StateRoot.open(plain, RUN)

    # A directory that holds any other file is not a state root.
    (path / 'mining.json').write_text('{}')
    with pytest.raises(ShadowWriteError):
        StateRoot.open(path, RUN)


@pytest.mark.parametrize('name', LIVE_NAMES + ['notes.txt', 'journal.jsonl.bak', '../journal.jsonl', 'a/b'])
def test_state_root_refuses_every_file_name_it_does_not_own(tmp_path, name):
    root = StateRoot.create(tmp_path / 'root', RUN)
    with pytest.raises(ShadowWriteError):
        root.file(name)
    with pytest.raises(ShadowWriteError):
        root.check(root.path / name)


def test_state_root_owns_four_files_and_their_temp_names(tmp_path):
    root = StateRoot.create(tmp_path / 'root', RUN)
    for name in (G.MARKER, G.JOURNAL, G.LOCK, G.STATUS, G.JOURNAL + '.create.tmp', G.JOURNAL + '.repair.tmp',
                 G.STATUS + '.tmp'):
        assert root.file(name) == root.path / name
        assert root.check(root.path / name) == root.path / name
    with pytest.raises(ShadowWriteError):
        root.check(tmp_path / 'elsewhere' / G.JOURNAL)
    assert G.FORBIDDEN_NAMES.isdisjoint(name.lower() for name in G.STATE_FILES)


def test_a_live_state_root_has_its_own_marker(tmp_path):
    assert G.MARKERS == {MODE_SHADOW: 'BATCHCTL-SHADOW-ROOT.json', MODE_LIVE: 'BATCHCTL-LIVE-ROOT.json'}
    assert G.MARKER == G.MARKERS[MODE_SHADOW] and G.MARKERS[MODE_LIVE] not in G.STATE_FILES
    path = tmp_path / 'live-root'
    root = StateRoot.create(path, RUN, mode=MODE_LIVE)
    assert root.mode == MODE_LIVE and root.marker == 'BATCHCTL-LIVE-ROOT.json'
    assert sorted(p.name for p in path.iterdir()) == ['BATCHCTL-LIVE-ROOT.json']
    doc = json.loads((path / root.marker).read_text())
    assert doc['mode'] == MODE_LIVE and doc['run_id'] == RUN
    again = StateRoot.open(path, RUN, mode=MODE_LIVE)
    assert (again.path, again.run_id, again.mode) == (root.path, RUN, MODE_LIVE)

    # It owns the four files of its mode, and not the marker of a shadow root.
    assert root.state_files == {root.marker, G.JOURNAL, G.LOCK, G.STATUS}
    assert G.FORBIDDEN_NAMES.isdisjoint(name.lower() for name in root.state_files)
    for name in (root.marker, G.JOURNAL, G.LOCK, G.STATUS, G.JOURNAL + '.create.tmp', G.STATUS + '.tmp'):
        assert root.file(name) == root.path / name
        assert root.check(root.path / name) == root.path / name
    for name in [G.MARKER, G.MARKER + '.tmp'] + LIVE_NAMES:
        with pytest.raises(ShadowWriteError):
            root.file(name)

    # The default mode is shadow, and a shadow root does not own the marker of a live root.
    shadow = StateRoot.create(tmp_path / 'shadow-root', RUN)
    assert shadow.mode == MODE_SHADOW and shadow.marker == G.MARKER and shadow.state_files == G.STATE_FILES
    assert json.loads((shadow.path / G.MARKER).read_text())['mode'] == MODE_SHADOW
    with pytest.raises(ShadowWriteError):
        shadow.file(G.MARKERS[MODE_LIVE])


def test_a_state_root_of_one_mode_is_not_a_state_root_of_the_other_mode(tmp_path):
    shadow, live = tmp_path / 'shadow-root', tmp_path / 'live-root'
    StateRoot.create(shadow, RUN)
    StateRoot.create(live, RUN, mode=MODE_LIVE)
    with pytest.raises(ShadowWriteError):
        StateRoot.open(shadow, RUN, mode=MODE_LIVE)
    with pytest.raises(ShadowWriteError):
        StateRoot.open(live, RUN)
    with pytest.raises(ShadowWriteError):
        StateRoot.create(shadow, RUN, mode=MODE_LIVE)
    with pytest.raises(ShadowWriteError):
        StateRoot.create(live, RUN)

    # A marker with the name of one mode and the document of the other mode is refused.
    forged = tmp_path / 'forged'
    forged.mkdir()
    (forged / G.MARKERS[MODE_LIVE]).write_text((shadow / G.MARKER).read_text())
    with pytest.raises(ShadowWriteError, match='state root marker is for another run or mode'):
        StateRoot.open(forged, RUN, mode=MODE_LIVE)

    # A root that holds the markers of both modes is a root of neither mode.
    (live / G.MARKER).write_text((shadow / G.MARKER).read_text())
    for mode in (MODE_LIVE, MODE_SHADOW):
        with pytest.raises(ShadowWriteError, match='state root holds a file that the controller does not own'):
            StateRoot.open(live, RUN, mode=mode)

    for mode in ('LIVE', 'prod', '', None):
        with pytest.raises(LiveModeRefused):
            StateRoot.create(tmp_path / 'unknown', RUN, mode=mode)
        with pytest.raises(LiveModeRefused):
            StateRoot.open(shadow, RUN, mode=mode)
    assert not (tmp_path / 'unknown').exists()


def test_a_live_state_root_must_be_outside_every_live_root_too(tmp_path):
    lane = tmp_path / 'live-state'
    lane.mkdir()
    env = {'PE_STATE': str(lane)}
    for bad in (lane, lane / 'batchctl', tmp_path):
        with pytest.raises(ShadowWriteError):
            StateRoot.create(bad, RUN, env, MODE_LIVE)
    assert not (lane / 'batchctl').exists()
    root = StateRoot.create(tmp_path / 'ok', RUN, env, MODE_LIVE)
    assert StateRoot.open(root.path, RUN, env, MODE_LIVE).mode == MODE_LIVE
    with pytest.raises(ShadowWriteError):
        StateRoot.open(root.path, RUN, {'PE_STATE': str(root.path)}, MODE_LIVE)


# ----------------------------------------------------------------------------
# Boxes
# ----------------------------------------------------------------------------

def test_box_names_are_exact_and_do_not_look_like_live_boxes():
    assert box_name(RUN, 7, 0) == BOX_PREFIX_SHADOW + RUN + '-f7-t0'
    assert not BOX_PREFIX_SHADOW.startswith('affine-pe-')
    assert parse_box_name(RUN, box_name(RUN, 7, 3)) == (7, 3)
    assert parse_box_name(RUN, box_name(RUN, 123456789012, 123456789)) == (123456789012, 123456789)
    for fence, try_no in ((0, 0), (-1, 0), (1, -1)):
        with pytest.raises(ValueError):
            box_name(RUN, fence, try_no)


@pytest.mark.parametrize('name', [
    'affine-pe-pit-9-a1',
    'affine-pe-fakepass-1',
    BOX_PREFIX_SHADOW + 'run0002-f1-t0',
    BOX_PREFIX_SHADOW + RUN,
    BOX_PREFIX_SHADOW + RUN + '-',
    BOX_PREFIX_SHADOW + RUN + '-f1',
    BOX_PREFIX_SHADOW + RUN + '-f1-t',
    BOX_PREFIX_SHADOW + RUN + '-f1-t0-x',
    BOX_PREFIX_SHADOW + RUN + '-f1-t0 ',
    BOX_PREFIX_SHADOW + RUN + '-f1-t0\n',
    BOX_PREFIX_SHADOW + RUN + '-f01-t0',
    BOX_PREFIX_SHADOW + RUN + '-f0-t0',
    BOX_PREFIX_SHADOW + RUN + '-f1-t00',
    BOX_PREFIX_SHADOW + RUN + '-f-1-t0',
    BOX_PREFIX_SHADOW + RUN + '-F1-T0',
    BOX_PREFIX_SHADOW + RUN + 'x-f1-t0',
    'x' + BOX_PREFIX_SHADOW + RUN + '-f1-t0',
    (BOX_PREFIX_SHADOW + RUN + '-f1-t0').upper(),
    '',
    None,
    7,
])
def test_a_box_name_that_this_run_did_not_make_is_not_ours(name):
    assert parse_box_name(RUN, name) is None


def test_the_box_names_of_the_two_modes_are_never_mixed():
    live, shadow = box_name(RUN, 7, 3, MODE_LIVE), box_name(RUN, 7, 3)
    assert live == BOX_PREFIX_LIVE + RUN + '-f7-t3' == 'affine-bctl-live-run0001-f7-t3'
    assert shadow == box_name(RUN, 7, 3, MODE_SHADOW) == BOX_PREFIX_SHADOW + RUN + '-f7-t3'
    assert not BOX_PREFIX_LIVE.startswith('affine-pe-')
    assert not BOX_PREFIX_LIVE.startswith(BOX_PREFIX_SHADOW) and not BOX_PREFIX_SHADOW.startswith(BOX_PREFIX_LIVE)
    assert own_prefix(RUN, MODE_LIVE) == BOX_PREFIX_LIVE + RUN + '-'
    assert own_prefix(RUN) == own_prefix(RUN, MODE_SHADOW) == BOX_PREFIX_SHADOW + RUN + '-'

    assert parse_box_name(RUN, live, MODE_LIVE) == (7, 3)
    assert parse_box_name(RUN, shadow) == parse_box_name(RUN, shadow, MODE_SHADOW) == (7, 3)
    # A shadow run never takes a live name as its own, and a live run never takes a shadow name.
    assert parse_box_name(RUN, live) is None and parse_box_name(RUN, live, MODE_SHADOW) is None
    assert parse_box_name(RUN, shadow, MODE_LIVE) is None
    for name in (box_name('run0002', 7, 3, MODE_LIVE), live + '-x', live + '\n', live.upper(), 'x' + live,
                 BOX_PREFIX_LIVE + RUN, BOX_PREFIX_LIVE + RUN + '-f0-t0', 'affine-pe-pit-9-a1', '', None):
        assert parse_box_name(RUN, name, MODE_LIVE) is None
    for mode in ('LIVE', 'prod', '', None):
        for call in (lambda: own_prefix(RUN, mode), lambda: box_name(RUN, 1, 0, mode),
                     lambda: parse_box_name(RUN, live, mode)):
            with pytest.raises(LiveModeRefused):
                call()
    for fence, try_no in ((0, 0), (-1, 0), (1, -1)):
        with pytest.raises(ValueError):
            box_name(RUN, fence, try_no, MODE_LIVE)


# ----------------------------------------------------------------------------
# No code path to the chain, the website or Discord
# ----------------------------------------------------------------------------

# Modules of this package that run in production (not tests, not the fakes).
RUNTIME = sorted(
    path for path in PACKAGE.glob('*.py')
    if not path.name.startswith('test_') and path.name not in ('sim.py', 'modelcheck.py')
)
# Third-party code that one runtime module may import.
THIRD_PARTY = {'r2.py': {'boto3', 'botocore', 'urllib3'}}
# Modules outside this package that the controller may import, by the
# number of dots of the relative import.
OUTSIDE = {2: {'fakepass_sandbox', 'shadow', 'gpu_policy', ''}, 3: {'durable'}}
# Modules of the PIT package that one runtime module may import more. Each is
# the existing code of the live lane that the module must reuse, not copy.
OUTSIDE_FOR = {
    'feed.py': {'queue_ingest'},
    'judge.py': {'probe_fingerprint', 'pe_score'},
    'agent.py': {'probe_layout', 'probe_serve'},
    'settle.py': {'gate_writer'},
    'run.py': {'shadow_guard', 'gate_writer'},
}
# Standard-library modules that can talk to something. Only the injected
# interfaces may do I/O, so none of these is imported.
NETWORK = {
    'socket', 'ssl', 'http', 'urllib', 'ftplib', 'smtplib', 'poplib', 'imaplib', 'telnetlib',
    'xmlrpc', 'asyncio', 'subprocess', 'multiprocessing', 'webbrowser', 'socketserver',
}
# Words that must not be in the source of the runtime: there is no code for
# the chain, the ledger, the website or Discord to hide behind.
WORDS = (
    'bittensor', 'substrate', 'extrinsic', 'set_weights', 'discord', 'webhook', 'wallet',
    'mining.json', 'llms.txt', 'payout.json', 'requests.', 'httpx', 'aiohttp',
)


def imports_of(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield 0, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 2 and not node.module:
            # ``from .. import name``: a module of the PIT package, or a constant of its ``__init__``.
            for alias in node.names:
                is_module = (PACKAGE.parent / (alias.name + '.py')).is_file() or (PACKAGE.parent / alias.name).is_dir()
                yield node.level, alias.name if is_module else ''
        elif isinstance(node, ast.ImportFrom):
            yield node.level, node.module or ''


@pytest.mark.parametrize('path', RUNTIME, ids=lambda path: path.name)
def test_runtime_module_imports_nothing_that_can_reach_the_live_system(path):
    stdlib = set(sys.stdlib_module_names)
    allowed_third_party = THIRD_PARTY.get(path.name, set())
    local = {p.stem for p in PACKAGE.glob('*.py')}
    for level, module in imports_of(path):
        top = module.split('.')[0]
        if level == 0:
            assert top in stdlib or top in allowed_third_party, '%s imports %s' % (path.name, module)
            assert top not in NETWORK, '%s imports %s' % (path.name, module)
        elif level == 1:
            assert top in local or top == '', '%s imports .%s' % (path.name, module)
        else:
            allowed = set(OUTSIDE.get(level, ()))
            if level == 2:
                allowed |= OUTSIDE_FOR.get(path.name, set())
            assert top in allowed, '%s imports %s%s' % (path.name, '.' * level, module)


@pytest.mark.parametrize('path', RUNTIME, ids=lambda path: path.name)
def test_runtime_module_has_no_word_of_the_chain_the_website_or_discord(path):
    if path.name == 'guard.py':
        # The guard names the live artifacts in order to refuse them.
        return
    text = path.read_text().lower()
    for word in WORDS:
        assert word not in text, '%s mentions %s' % (path.name, word)


def test_the_batch_source_interface_has_no_write_call():
    public = {name for name in vars(I.BatchSource) if not name.startswith('_')}
    assert public == {'list'}
