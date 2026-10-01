"""Isolated trusted package/resource enforcement for the NEXT environment version.

Callers must obtain these descriptors from an authenticated operator manifest.
This module is deliberately not imported by the current live environment adapter.
"""
from __future__ import annotations
import csv
import io
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.parse import urlsplit, urlunsplit
import zipfile

SCHEMA = 'trusted-environment-resources-v1'
ENV_DEPENDENCIES = {
    'affine_wiki': ('verifiers', 'wiki-search-v1', 'chromadb'),
    'affine_deshuffle': ('verifiers', 'deshuffle-papers'),
    'affine_rcore': ('verifiers', 'reasoning-core'),
    'affine_autobench': ('verifiers', 'automation-bench'),
    'affine_tau2': ('verifiers', 'tau2'),
    'affine_tau2_synth': ('verifiers', 'tau2'),
    'affine_tau2_gen': ('verifiers', 'tau2'),
    'affine_kb_synth': ('verifiers', 'tau2'),
    'swesmith': ('verifiers', 'swesmith', 'swebench'),
    'affine_tmax': ('verifiers', 'harbor'),
    'terminal_bench_2': ('verifiers', 'harbor'),
}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def safe_relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(s in ('', '.', '..') for s in p.parts) or '\\' in name:
        raise ValueError('unsafe resource path')
    return p.as_posix()

def _origin(distribution):
    raw = json.loads(distribution.read_text('direct_url.json') or '{}')
    if raw.get('vcs_info'):
        url = urlsplit(raw['url'])
        # Authentication/query fragments never enter public manifests.
        url = urlunsplit((url.scheme, url.hostname or '', url.path, '', ''))
        return {'kind': 'git', 'url': url, 'commit': raw['vcs_info']['commit_id']}
    return {'kind': 'installed-file-footprint'}

def collect_dependencies(names):
    """Pin exact installed versions and every regular Python source bytefile.

    Non-Python native execution artifacts are a distinct numerical/platform pin.
    Explicitly pass the reviewed transitive package set; no mutable resolver runs.
    """
    rows = []
    for name in sorted(set(names)):
        dist = metadata.distribution(name)
        files = {}
        modules = {}
        # RECORD identifies ownership roots, but is not an exhaustive import
        # allowlist. Scan every Python file under those actual namespace roots.
        for entry in dist.files or []:
            if not str(entry).endswith('.py'):
                continue
            relative = safe_relative(str(entry));parts=PurePosixPath(relative).parts
            top=parts[0];name=top[:-3] if len(parts)==1 else top
            if not name.isidentifier():
                raise ValueError('unresolved provider module root: '+top)
            modules[name] = {'kind':'module' if len(parts)==1 else 'package',
                             'path':top}
        for name, module in modules.items():
            root = Path(dist.locate_file(module['path']))
            if root.is_symlink() or (module['kind']=='package' and any(p.is_symlink() for p in root.rglob('*'))):
                raise ValueError('provider resource symlink unsupported: '+name)
            # Provider code can read packaged prompts/templates at import time.
            # Carry their exact bytes too; Python caches are deliberately NOT
            # transported, and the isolated worker rejects any added cache.
            paths = [root] if module['kind']=='module' else sorted(p for p in root.rglob('*')
                if p.is_file() and '__pycache__' not in p.relative_to(root).parts and p.suffix not in ('.pyc','.pyo'))
            for path in paths:
                if not path.is_file():
                    raise ValueError('missing declared package source: '+str(path))
                relative = module['path'] if module['kind']=='module' else module['path']+'/'+path.relative_to(root).as_posix()
                files[safe_relative(relative)] = digest(path.read_bytes())
            spec = util.find_spec(name)
            if spec is None:
                raise ValueError('provider module not importable: '+name)
            if module['kind']=='module':
                matched=spec.origin is not None and Path(spec.origin).resolve()==root.resolve()
            elif (root/'__init__.py').exists():
                matched=spec.origin is not None and Path(spec.origin).resolve()==(root/'__init__.py').resolve()
            else:
                matched=spec.origin is None and {Path(p).resolve() for p in (spec.submodule_search_locations or [])}=={root.resolve()}
            if not matched:
                raise ValueError('provider import resolution mismatch: '+name)
        if not files:
            raise ValueError('no Python source footprint: '+name)
        rows.append({'name': dist.metadata['Name'], 'version': dist.version,
                     'origin': _origin(dist), 'modules':modules, 'files': files})
    body = {'schema': SCHEMA, 'kind': 'dependencies', 'packages': rows,
            'native_binaries_bound': False}
    return dict(body, id=digest(canonical(body)))

def _descriptor(manifest, kind):
    body = {k:v for k,v in manifest.items() if k != 'id'}
    if body.get('schema') != SCHEMA or body.get('kind') != kind or digest(canonical(body)) != manifest.get('id'):
        raise ValueError('resource descriptor identity mismatch')
    return body

def verify_dependencies(manifest):
    """Verify installed Python source footprints, not cached/in-memory code.

    The next adapter must additionally use adapter_resources' fresh-worker
    guard: exact isolated materialization, no bytecode, -B and no preloaded
    providers. A source footprint alone cannot attest actual Python execution.
    """
    body = _descriptor(manifest, 'dependencies')
    for row in body['packages']:
        dist = metadata.distribution(row['name'])
        if dist.version != row['version']:
            raise ValueError('environment dependency version mismatch: '+row['name'])
        actual = collect_dependencies([row['name']])['packages'][0]
        if actual['files'] != row['files'] or actual['origin'] != row['origin'] or actual['modules'] != row['modules']:
            raise ValueError('environment dependency source mismatch: '+row['name'])
    return True

def export_dependencies(manifest, archive):
    verify_dependencies(manifest)
    entries = {}
    for row in manifest['packages']:
        dist = metadata.distribution(row['name'])
        for name, sha in row['files'].items():
            data = Path(dist.locate_file(name)).read_bytes()
            if digest(data) != sha:
                raise ValueError('dependency changed during export')
            if name in entries and entries[name] != data:
                raise ValueError('conflicting shared namespace bytefile')
            entries[name] = data
        entries.update(_package_metadata(row))
    return _export(manifest, entries, archive)

def _package_metadata(row):
    distdir = row['name'].replace('-', '_')+'-'+row['version']+'.dist-info'
    entries = {distdir+'/METADATA': ('Metadata-Version: 2.1\nName: '+row['name']+'\nVersion: '+row['version']+'\n').encode()}
    if row['origin'].get('kind') == 'git':
        entries[distdir+'/direct_url.json'] = canonical({'url':row['origin']['url'], 'vcs_info':{'vcs':'git','commit_id':row['origin']['commit']}})
    record = io.StringIO();writer=csv.writer(record, lineterminator='\n')
    for name in sorted(set(row['files'])|set(entries)|{distdir+'/RECORD'}):
        writer.writerow((name,'',''))
    entries[distdir+'/RECORD'] = record.getvalue().encode()
    return entries

def collect_task_resources(directory, *, origin, bindings=None, audience='verifier'):
    """Content-address original task tests/fixtures/build context, never host paths.

    Whole Harbor task bundles may contain private grader tests; their default
    audience is verifier. Export a separate environment/ build-context descriptor
    for miner setup rather than publishing the whole verifier bundle.
    """
    if audience not in ('verifier', 'miner-setup'):
        raise ValueError('resource audience')
    root = Path(directory).resolve()
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('task resource symlink unsupported')
        if path.is_file():
            files[safe_relative(path.relative_to(root).as_posix())] = digest(path.read_bytes())
    if not files:
        raise ValueError('empty original task resource bundle')
    body = {'schema': SCHEMA, 'kind': 'task-resources', 'origin': origin,
            'audience': audience, 'bindings': bindings or {'task_dir': '.'}, 'files': files}
    return dict(body, id=digest(canonical(body)))

def export_task_resources(manifest, directory, archive):
    _descriptor(manifest, 'task-resources')
    root = Path(directory).resolve()
    entries = {}
    for name, sha in manifest['files'].items():
        path = root/safe_relative(name)
        if path.is_symlink() or not path.is_file() or path.resolve().is_relative_to(root) is False:
            raise ValueError('invalid task resource file')
        data = path.read_bytes()
        if digest(data) != sha:
            raise ValueError('task resource changed during export')
        entries[name] = data
    return _export(manifest, entries, archive)

def _export(manifest, entries, archive):
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('DESCRIPTOR.json', canonical(manifest))
        for name, data in sorted(entries.items()):
            z.writestr('files/'+safe_relative(name), data)
    return {'descriptor_id': manifest['id'], 'archive_sha256': digest(Path(archive).read_bytes())}

def materialize(manifest, archive, destination, *, archive_sha256, max_bytes=512*1024*1024):
    """Verify authenticated descriptor+archive before writing an isolated directory."""
    kind = manifest.get('kind');_descriptor(manifest, kind)
    if digest(Path(archive).read_bytes()) != archive_sha256:
        raise ValueError('resource archive hash mismatch')
    expected = manifest.get('files', {})
    if kind == 'dependencies':
        expected = {}
        for row in manifest['packages']:
            expected.update(row['files'])
            expected.update({n:digest(data) for n,data in _package_metadata(row).items()})
    elif kind != 'task-resources':
        raise ValueError('unknown resource kind')
    dest = Path(destination)
    if dest.exists():
        raise ValueError('refusing to overwrite resource root')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.trusted-env-', dir=dest.parent))
    try:
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist()) != len(set(z.namelist())):
                raise ValueError('duplicate archive member')
            wanted = {'DESCRIPTOR.json'}|{'files/'+safe_relative(n) for n in expected}
            if set(z.namelist()) != wanted or sum(i.file_size for i in z.infolist()) > max_bytes:
                raise ValueError('unexpected or oversized resource archive')
            if z.read('DESCRIPTOR.json') != canonical(manifest):
                raise ValueError('archive descriptor mismatch')
            for name, sha in expected.items():
                data = z.read('files/'+name)
                if digest(data) != sha:
                    raise ValueError('resource bytefile mismatch')
                path = temporary/name;path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data);path.chmod(0o400)
        os.rename(temporary, dest)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    return dest

def bind_task_resources(data, manifest, root):
    """Return mapped runtime data; canonical identity must be calculated BEFORE mapping.

    Next adapter must derive task identity from the shared symbolic resource ID,
    not upstream Task.hash() after resolving local filesystem paths.
    """
    _descriptor(manifest, 'task-resources')
    result = dict(data);base = Path(root).resolve()
    for field, name in manifest['bindings'].items():
        result[field] = str(base if name == '.' else base/safe_relative(name))
    return result

def canonical_task_identity(data, task_config, resource_manifest, *, environment_version):
    """Host-independent identity for the NEXT adapter; no absolute runtime paths.

    The containing signed spec separately binds provider code/checkpoint hashes.
    """
    _descriptor(resource_manifest, 'task-resources')
    symbolic = dict(data)
    for field, relative in resource_manifest['bindings'].items():
        suffix = '' if relative == '.' else safe_relative(relative)
        symbolic[field] = 'resource://'+resource_manifest['id']+'/'+suffix
    return digest(canonical({'schema':'portable-task-identity-v1',
                             'environment_version':environment_version,
                             'data':symbolic, 'task_config':task_config}))
