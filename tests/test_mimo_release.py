import io
import json
import tarfile

import pytest

from subnet.mimo_release import MODEL_ID, MODEL_REVISION, VERL_REVISION, build, verify


def fixture(tmp_path):
    root = tmp_path / 'worktree'
    (root / 'subnet').mkdir(parents=True)
    (root / 'subnet/core.py').write_text('def answer(): return 42\n')
    (root / 'vendor/mimo_verl_pinned').mkdir(parents=True)
    (root / 'vendor/mimo_verl_pinned/LICENSE').write_text('Apache-2.0 fixture\n')
    (root / 'state').mkdir()
    (root / 'state/authority.seed').write_text('never-package-this')
    settings = dict(model=dict(id=MODEL_ID, revision=MODEL_REVISION), reference_revision=VERL_REVISION)
    return root, settings


def test_reproducible_archive_excludes_private_state(tmp_path):
    root, settings = fixture(tmp_path)
    a = build(root, tmp_path / 'a', settings=settings)
    b = build(root, tmp_path / 'b', settings=settings)
    assert a == b == verify(tmp_path / 'a')
    assert set(a['source']) == {'subnet/core.py', 'vendor/mimo_verl_pinned/LICENSE'}
    assert not (tmp_path / 'worktree/current.json').exists()


@pytest.mark.parametrize('extra', ['state/authority.seed', '../outside.py', '/etc/passwd', 'subnet/a.private.json'])
def test_private_and_escaping_paths_rejected(tmp_path, extra):
    root, settings = fixture(tmp_path)
    with pytest.raises(ValueError):
        build(root, tmp_path / 'a', settings=settings, extra=[extra])


def test_symlink_cannot_smuggle_credentials(tmp_path):
    root, settings = fixture(tmp_path)
    (root / 'subnet/included.py').symlink_to(root / 'state/authority.seed')
    with pytest.raises(ValueError, match='symlink'):
        build(root, tmp_path / 'a', settings=settings)


def test_staged_release_is_immutable(tmp_path):
    root, settings = fixture(tmp_path)
    build(root, tmp_path / 'a', settings=settings)
    (root / 'subnet/core.py').write_text('changed\n')
    with pytest.raises(FileExistsError):
        build(root, tmp_path / 'a', settings=settings)


def test_only_explicit_pinned_task_assets_ship_with_source(tmp_path):
    import hashlib
    root,settings=fixture(tmp_path)
    (root/'assets').mkdir()
    approved=b'{"tasks":[{"question":"1+1","answer":"2"}]}\n'
    (root/'assets/math.tasks.json').write_bytes(approved)
    (root/'assets/unapproved.json').write_text('must not appear')
    settings['task_assets']={'assets/math.tasks.json':dict(sha256=hashlib.sha256(approved).hexdigest(),bytes=len(approved))}
    record=build(root,tmp_path/'release',settings=settings)
    assert record==verify(tmp_path/'release')
    assert 'assets/math.tasks.json'in record['source'] and 'assets/unapproved.json'not in record['source']
    with tarfile.open(tmp_path/'release/source.tar.gz')as archive:
        archive.extractall(tmp_path/'installed',filter='data')
    assert (tmp_path/'installed/assets/math.tasks.json').read_bytes()==approved
    (root/'assets/math.tasks.json').write_bytes(b'changed')
    with pytest.raises(ValueError,match='assets differ'):
        build(root,tmp_path/'wrong',settings=settings)


def test_assets_cannot_bypass_inventory_via_extra_or_symlink(tmp_path):
    import hashlib
    root,settings=fixture(tmp_path)
    (root/'assets').mkdir();(root/'assets/tasks.json').write_bytes(b'public fixture')
    with pytest.raises(ValueError,match='explicit pinned'):
        build(root,tmp_path/'not-pinned',settings=settings,extra=['assets/tasks.json'])
    (root/'assets/link').symlink_to(root/'state',target_is_directory=True)
    raw=(root/'state/authority.seed').read_bytes()
    settings['task_assets']={'assets/link/authority.seed':dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))}
    with pytest.raises(ValueError):build(root,tmp_path/'linked',settings=settings)


@pytest.mark.parametrize('bad',[
    {'../tasks.json':{'sha256':'a'*64,'bytes':1}},
    {'assets/task.json':{'sha256':'a'*64,'bytes':True}},
    {'assets/task.json':{'sha256':'a'*64,'bytes':16777217}},
    {'assets/task.json':{'sha256':'A'*64,'bytes':1}},
    {'state/task.json':{'sha256':'a'*64,'bytes':1}},
])
def test_task_asset_declarations_fail_closed(tmp_path,bad):
    root,settings=fixture(tmp_path);settings['task_assets']=bad
    with pytest.raises(ValueError):build(root,tmp_path/'bad',settings=settings)


def test_verifier_requires_settings_and_task_inventory_agree(tmp_path):
    import hashlib
    root,settings=fixture(tmp_path);(root/'assets').mkdir()
    raw=b'{"public":"task"}';(root/'assets/task.json').write_bytes(raw)
    settings['task_assets']={'assets/task.json':dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))}
    build(root,tmp_path/'release',settings=settings)
    path=tmp_path/'release/RELEASE.json';record=json.loads(path.read_bytes())
    record['settings']['task_assets']={};path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='assets differ'):verify(tmp_path/'release')


def test_native_environment_hash_survives_release_extraction(tmp_path):
    import hashlib
    root, settings = fixture(tmp_path)
    relative = 'subnet/vendor/legacy/rollouts/envs/native_math'
    native = root / relative
    native.mkdir(parents=True)
    for name, data in [('grader.py', b'print(1)'), ('pyproject.toml', b'[project]'),
                       ('rules.md', b'grading rules'), ('tasks.json.gz', b'compressed fixture')]:
        (native / name).write_bytes(data)
    (native / 'discard.pyc').write_bytes(b'not source')
    def inventory(directory):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.iterdir() if p.suffix != '.pyc'}
    build(root, tmp_path / 'release', settings=settings)
    verify(tmp_path / 'release')
    with tarfile.open(tmp_path / 'release/source.tar.gz') as archive:
        archive.extractall(tmp_path / 'installed', filter='data')
    installed = tmp_path / 'installed' / relative
    assert inventory(installed) == inventory(native)
    assert not (installed / 'discard.pyc').exists()


def test_verifier_rejects_rehashed_archive_with_unlisted_extra(tmp_path):
    import hashlib
    root, settings = fixture(tmp_path)
    build(root, tmp_path / 'a', settings=settings)
    p = tmp_path / 'a/source.tar.gz'
    with tarfile.open(p) as archive:
        originals = [(m.name, archive.extractfile(m).read()) for m in archive.getmembers()]
    with tarfile.open(p, 'w:gz') as archive:
        for name, raw in originals + [('state/authority.seed', b'smuggled')]:
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
    record = json.loads((p.parent / 'RELEASE.json').read_bytes())
    record.update(archive_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), archive_bytes=p.stat().st_size)
    (p.parent / 'RELEASE.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='membership'):
        verify(p.parent)


def test_floating_model_revision_not_accepted(tmp_path):
    root, settings = fixture(tmp_path)
    settings['model']['revision'] = 'main'
    with pytest.raises(ValueError, match='immutable'):
        build(root, tmp_path / 'a', settings=settings)
