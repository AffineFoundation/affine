"""Shadow-mode hard denies.

Each test names one thing that the controller must never reach: the live
HEAD, live R2 keys, the ledger, weights and extrinsics, ``mining.json``,
``llms.txt``, the website, Discord, a box of someone else.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from . import BOX_PREFIX_SHADOW, MODE_SHADOW, R2_SHADOW_ROOT
from . import guard as G
from . import interfaces as I
from .guard import (
    GuardedStore, LiveModeRefused, StateRoot, box_name, check_key, check_mode, check_outside_live,
    own_prefix, parse_box_name, shadow_prefix,
)
from ..shadow import ShadowWriteError

RUN = 'run0001'
PREFIX = R2_SHADOW_ROOT + RUN + '/'
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
    good = PREFIX + 'states/' + 'a' * 64 + '/manifest.json'
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


def test_a_prefix_that_is_not_a_shadow_prefix_is_refused():
    for prefix in ('', 'live/', 'pit/', 'shadow/batchctl', 'shadow/other/run0001/'):
        with pytest.raises(ShadowWriteError):
            check_key(prefix, prefix + 'HEAD.json')


# ----------------------------------------------------------------------------
# Local state root
# ----------------------------------------------------------------------------

def test_state_root_inside_or_around_a_live_root_is_refused(tmp_path):
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

    # The test environment itself has live roots under tmp_path.
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


# ----------------------------------------------------------------------------
# No code path to the chain, the website or Discord
# ----------------------------------------------------------------------------

# Modules of this package that run in production (not tests, not the fakes).
RUNTIME = sorted(
    path for path in PACKAGE.glob('*.py')
    if not path.name.startswith('test_') and path.name not in ('sim.py', 'modelcheck.py')
)
# Third-party code that one runtime module may import.
THIRD_PARTY = {'s3store.py': {'boto3', 'botocore'}}
# Modules outside this package that the controller may import, by the
# number of dots of the relative import.
OUTSIDE = {2: {'fakepass_sandbox', 'shadow', 'gpu_policy', ''}, 3: {'durable'}}
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
            assert top in OUTSIDE.get(level, ()), '%s imports %s%s' % (path.name, '.' * level, module)


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
