"""Authority-approved, bounded source bootstrap for direct-R2 miners.

Authenticating source authorizes execution of that code; it is not a sandbox.
No runtime, model or environment module is imported before source admission.
"""
import argparse
import base64
import hashlib
import gzip
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
import time
from urllib.parse import urlparse, parse_qs
import requests
from nacl.signing import VerifyKey

COMPRESSED_LIMIT = 32 * 1024**2
RAW_LIMIT = 256 * 1024**2
JSON_LIMIT = 32 * 1024**2
MAX_FILES = 20000
ROOT_FILES = {'AGENTS.md', 'GOAL.md', 'IMPLEMENTATION_PLAN.md', 'README.md', 'STATE.md', 'pyproject.toml', '.gitignore', 'LICENSE'}
PUBLIC_NAMESPACES = {'subnet', 'ops', 'tests', 'docs', 'dashboard', 'examples', 'prototype', 'systemd'}
TASK_ASSET = 'assets/original-math7496.tasks.json'

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def r2_url(url):
    if not isinstance(url, str): raise ValueError('direct R2 URL required')
    p = urlparse(url); q = parse_qs(p.query)
    if (p.scheme != 'https' or not (p.hostname or '').endswith('.r2.cloudflarestorage.com')
            or p.username or p.password or p.fragment or p.port not in (None, 443)
            or q.get('X-Amz-Algorithm') != ['AWS4-HMAC-SHA256']
            or len(q.get('X-Amz-Signature', [])) != 1 or not q['X-Amz-Signature'][0]):
        raise ValueError('invalid direct R2 URL')
    return url

def download(url, limit):
    r2_url(url)
    with requests.get(url, timeout=180, stream=True, allow_redirects=False,
                      headers={'Accept-Encoding': 'identity'}) as r:
        if 300 <= r.status_code < 400: raise ValueError('redirect refused')
        r.raise_for_status()
        if r.headers.get('Content-Encoding', 'identity') != 'identity': raise ValueError('HTTP encoding refused')
        length = r.headers.get('Content-Length')
        if length is not None and (not length.isdecimal() or int(length) > limit): raise ValueError('response size bound')
        chunks=[]; size=0
        for chunk in r.iter_content(1024*1024):
            size += len(chunk)
            if size > limit: raise ValueError('response size bound')
            chunks.append(chunk)
        if length is not None and size != int(length): raise ValueError('truncated response')
        return b''.join(chunks)

def signed(raw, authority):
    if not isinstance(authority, str) or not re.fullmatch('[0-9a-f]{64}', authority): raise ValueError('known authority required')
    document=json.loads(raw)
    if set(document) != {'payload', 'signature', 'signer'} or document['signer'] != authority: raise ValueError('wrong authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']), base64.b64decode(document['signature'], validate=True))
    if not isinstance(document['payload'], dict): raise ValueError('signed object required')
    return document['payload']

def manifest(current_url, authority, fetch=download):
    current=signed(fetch(r2_url(current_url), JSON_LIMIT), authority)
    if current.get('transport_policy') != 'direct-r2-v1': raise ValueError('transport downgrade')
    value=signed(fetch(r2_url(current.get('manifest_url')), JSON_LIMIT), authority)
    if value.get('transport_policy') != 'direct-r2-v1' or value.get('epoch') != current.get('epoch') or not isinstance(value.get('epoch'), str): raise ValueError('discovery epoch binding')
    if type(value.get('deadline')) not in (int,float) or value['deadline'] <= time.time(): raise ValueError('epoch closed')
    return value

def public_path(name):
    # A single conventional './' tar prefix is canonicalized; all other ambiguity refuses.
    if name.startswith('./'): name=name[2:]
    p=PurePosixPath(name)
    if not name or '\\' in name or '\x00' in name or p.is_absolute() or str(p)!=name or any(x in ('', '.', '..') for x in name.split('/')):
        raise ValueError('archive path')
    lowered=[x.lower() for x in p.parts]
    if any(x.startswith('.') and x != '.gitignore' for x in lowered) or any(x in {'state','wallets','hotkeys','coldkeys','__pycache__','credentials','node_modules'} or x.endswith(('.seed','.pem','.key','.pyc')) or x.startswith('.env') for x in lowered):
        raise ValueError('private archive path')
    if name not in ROOT_FILES and name != TASK_ASSET and p.parts[0] not in PUBLIC_NAMESPACES: raise ValueError('unapproved source path')
    return name

def admitted_files(body, descriptor):
    if (not isinstance(descriptor, dict) or not re.fullmatch('[0-9a-f]{64}', str(descriptor.get('sha256','')))
            or type(descriptor.get('size')) is not int or not 0 < descriptor['size'] <= COMPRESSED_LIMIT
            or len(body) != descriptor['size'] or hashlib.sha256(body).hexdigest() != descriptor['sha256']):
        raise ValueError('source archive integrity')
    files={}; total=0; directories=set()
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(body)) as compressed:
            raw=compressed.read(RAW_LIMIT+1)
        if len(raw)>RAW_LIMIT:raise ValueError('source raw budget')
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as archive:
            for member in archive:
                name=public_path(member.name)
                if not member.isfile() or member.issym() or member.islnk() or name in files or member.size < 0: raise ValueError('source regular membership')
                total += member.size
                if total > RAW_LIMIT or len(files) >= MAX_FILES: raise ValueError('source raw budget')
                # Reject ancestor conflicts before touching the filesystem.
                parents={str(x) for x in PurePosixPath(name).parents if str(x)!='.'}
                if name in directories or parents & files.keys(): raise ValueError('archive file/directory conflict')
                directories.update(parents)
                with archive.extractfile(member) as stream: data=stream.read(member.size+1)
                if len(data) != member.size: raise ValueError('truncated archive file')
                files[name]=data
    except (tarfile.TarError, EOFError, OSError) as exc: raise ValueError('malformed source archive') from exc
    if not {'subnet/__init__.py','subnet/cli.py',TASK_ASSET} <= files.keys(): raise ValueError('missing public miner/task source')
    return files

def check_parents(path):
    for parent in (path, *path.parents):
        if parent.is_symlink(): raise ValueError('cache symlink')

def verify_cache(destination, files):
    if destination.is_symlink() or not destination.is_dir(): raise ValueError('cache directory')
    observed=set(); observed_dirs=set()
    expected_dirs={str(parent) for name in files for parent in PurePosixPath(name).parents if str(parent)!='.'}
    for path in destination.rglob('*'):
        if path.is_symlink(): raise ValueError('cache symlink')
        if path.is_dir():
            observed_dirs.add(path.relative_to(destination).as_posix());continue
        if not path.is_file(): raise ValueError('cache nonregular file')
        name=path.relative_to(destination).as_posix(); observed.add(name)
        if name not in files or path.stat().st_size!=len(files[name]) or path.read_bytes()!=files[name]: raise ValueError('cache tampering')
    if observed != set(files) or observed_dirs!=expected_dirs: raise ValueError('cache membership')

def install(body, descriptor, cache):
    files=admitted_files(body,descriptor) # Complete bytes/membership admission before any write.
    cache=Path(os.path.abspath(cache)); check_parents(cache)
    destination=cache/descriptor['sha256']
    if destination.exists() or destination.is_symlink():
        verify_cache(destination,files); return destination
    cache.mkdir(parents=True, exist_ok=True, mode=0o700); check_parents(cache)
    temporary=Path(tempfile.mkdtemp(prefix='.source-',dir=cache))
    try:
        for name,data in files.items():
            path=temporary/name; path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream: stream.write(data)
            path.chmod(0o444)
        for path in sorted((p for p in temporary.rglob('*') if p.is_dir()), reverse=True): path.chmod(0o555)
        temporary.chmod(0o555)
        if destination.exists() or destination.is_symlink(): raise ValueError('cache concurrent collision')
        temporary.rename(destination)
        verify_cache(destination,files)
        return destination
    finally:
        if temporary.exists():
            temporary.chmod(0o700)
            for path in temporary.rglob('*'):
                if path.is_dir():path.chmod(0o700)
            shutil.rmtree(temporary)

def execute(source, arguments, executor=os.execve):
    # -I excludes host cwd/PYTHONPATH; only the approved source is explicitly inserted.
    loader="import os,runpy,sys;sys.path.insert(0,os.getcwd());sys.argv=['affine-miner']+sys.argv[1:];runpy.run_module('subnet.cli',run_name='__main__')"
    environment={k:v for k,v in os.environ.items() if not k.startswith('PYTHON')}
    os.chdir(source)
    executor(sys.executable,[sys.executable,'-I','-B','-c',loader,*arguments],environment)

def hydrate_task_assets(source,value,cache):
    """Admitted code hydrates separately signed data outside immutable source."""
    if not value.get('task_assets') and not any('math_corpus_asset' in r.get('spec',{}).get('config',{}) for r in value.get('environments',[])):
        return
    import subprocess
    root=Path(cache)/'task-assets';root.mkdir(parents=True,exist_ok=True);root.chmod(0o700)
    loader="import json,sys;sys.path.insert(0,sys.argv[1]);from subnet.task_assets import hydrate_manifest;hydrate_manifest(sys.argv[2],json.load(sys.stdin))"
    result=subprocess.run([sys.executable,'-I','-B','-c',loader,str(Path(source).resolve()),str(root.resolve())],
        input=canonical(value),capture_output=True,timeout=1800)
    if result.returncode:raise ValueError('signed task asset hydration failed')
    os.environ['AFFINE_MATH_CORPUS_ASSET_ROOT']=str(root.resolve())

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authority',required=True);p.add_argument('--current-url',required=True)
    p.add_argument('--source-cache',required=True);p.add_argument('--state',required=True)
    credential=p.add_mutually_exclusive_group(required=True)
    credential.add_argument('--key');credential.add_argument('--cap-file')
    p.add_argument('--gateway',default='https://unused.invalid');p.add_argument('--once',action='store_true');p.add_argument('--max-batches',type=int)
    p.add_argument('--compression-level',type=int,choices=range(10),help='assert signed epoch compression level')
    p.add_argument('--env-id');p.add_argument('--indices',nargs='+',type=int);p.add_argument('--search-budget',type=int)
    a=p.parse_args(argv)
    credential_flag='--cap-file' if a.cap_file else '--key'
    credential_path=Path(os.path.abspath(a.cap_file or a.key));state=Path(os.path.abspath(a.state));cache=Path(os.path.abspath(a.source_cache))
    if not credential_path.is_file() or credential_path.is_symlink():raise ValueError('explicit existing credential file required')
    # Forward only its path; never read, create or modify the key/capability.
    value=manifest(a.current_url,a.authority);descriptor=value.get('source_bundle')
    if not isinstance(descriptor,dict):raise ValueError('signed source bundle required')
    body=download(r2_url(descriptor.get('url')),COMPRESSED_LIMIT)
    source=install(body,descriptor,cache)
    hydrate_task_assets(source,value,cache)
    arguments=['--authority',a.authority,'--current-url',a.current_url,'--gateway',a.gateway,credential_flag,str(credential_path),'--state',str(state),'--source-bundle-sha256',descriptor['sha256']]
    if a.once:arguments+=['--once']
    if a.max_batches is not None:arguments+=['--max-batches',str(a.max_batches)]
    if a.compression_level is not None:arguments+=['--compression-level',str(a.compression_level)]
    if a.env_id is not None:arguments+=['--env-id',a.env_id]
    if a.indices is not None:arguments+=['--indices',*[str(i) for i in a.indices]]
    if a.search_budget is not None:arguments+=['--search-budget',str(a.search_budget)]
    execute(source,arguments)

if __name__=='__main__':main()
