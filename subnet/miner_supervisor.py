"""Continuous miner entrypoint with resumable downloads outside approved source.

Every epoch executes its untouched signed source in a fresh process. This
transport supervisor never imports model, sampler, grader or GPU modules.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse

import requests
from . import source_bootstrap as bootstrap
from . import checkpoint_transfer as hydration


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(bootstrap.canonical(value)); temporary.chmod(0o600)
    os.replace(temporary, path)


def opening(discovery_url, authority, *, now=None, get=requests.get, fetch=bootstrap.download):
    """Discovery is a hint; the fixed authority authenticates all contract data."""
    parsed = urlparse(discovery_url)
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.fragment:
        raise ValueError('HTTPS public discovery required')
    with get(discovery_url, timeout=30, stream=True, allow_redirects=False) as response:
        if response.status_code in (408, 429) or response.status_code >= 500:
            raise requests.HTTPError('transient public discovery response')
        if response.status_code != 200: raise ValueError('public discovery response')
        raw = response.raw.read(bootstrap.JSON_LIMIT + 1)
        if len(raw) > bootstrap.JSON_LIMIT: raise ValueError('discovery size bound')
        hint = json.loads(raw)
    if hint.get('accepting_submissions') is not True: return None
    if hint.get('authority') != authority: raise ValueError('discovery authority differs')
    pointer = bootstrap.signed(fetch(bootstrap.r2_url(hint['current_url']), bootstrap.JSON_LIMIT), authority)
    url = bootstrap.r2_url(pointer['manifest_url'])
    manifest = bootstrap.signed(fetch(url, bootstrap.JSON_LIMIT), authority)
    if (pointer.get('transport_policy') != 'direct-r2-v1' or manifest.get('transport_policy') != 'direct-r2-v1'
            or pointer.get('epoch') != manifest.get('epoch') or hint.get('epoch') != manifest.get('epoch')
            or not re.fullmatch('[A-Za-z0-9_-]{1,200}', manifest.get('epoch', ''))):
        raise ValueError('signed opening scope')
    now = time.time() if now is None else now
    start, deadline = manifest.get('start'), manifest.get('deadline')
    if (type(start) not in (int, float) or type(deadline) not in (int, float)
            or not start <= now < deadline): return None
    return dict(manifest=manifest, manifest_url=url)


def checkpoint_inventory(manifest):
    checkpoint = manifest['checkpoint']; identifier = checkpoint.get('id'); files = checkpoint.get('files')
    if (not re.fullmatch('[0-9a-f]{64}', str(identifier)) or not isinstance(files, dict)
            or not 1 <= len(files) <= 32 or hashlib.sha256(bootstrap.canonical(files)).hexdigest() != identifier
            or set(checkpoint.get('read_urls', {})) != set(files)):
        raise ValueError('signed checkpoint inventory')
    for name, digest in files.items():
        if (not re.fullmatch('[A-Za-z0-9_-][A-Za-z0-9_.-]*', name)
                or not re.fullmatch('[0-9a-f]{64}', str(digest))):
            raise ValueError('checkpoint file name/hash')
        hydration.read_url(checkpoint['read_urls'][name], identifier, name)
    return checkpoint


def object_size(session, url):
    with session.get(url, headers={'Range':'bytes=0-0', 'Accept-Encoding':'identity','Connection':'close'},
                     stream=True, timeout=(15,30), allow_redirects=False) as response:
        match = re.fullmatch(r'bytes 0-0/([0-9]+)', response.headers.get('Content-Range', ''))
        if (response.status_code != 206 or response.headers.get('Content-Encoding', 'identity') != 'identity'
                or not match or not 0 < int(match[1]) <= hydration.MAX_FILE):
            raise ValueError('bounded model object size')
        if len(response.raw.read(2)) != 1: raise ValueError('model size probe bytes')
        return int(match[1])


def prefetch_checkpoint(manifest, state, *, session=None, clock=time.time, download=hydration.download):
    checkpoint = checkpoint_inventory(manifest); identifier = checkpoint['id']
    state = Path(state).absolute(); bootstrap.check_parents(state)
    destination = state / identifier; stage = state / '.checkpoint-downloads' / identifier
    bootstrap.check_parents(destination); bootstrap.check_parents(stage)
    owner = state / '.miner-supervisor' / 'owned-checkpoints' / (identifier + '.json')
    bootstrap.check_parents(owner); owner.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not destination.exists() and not owner.exists():
        save(owner, dict(checkpoint=identifier, files=checkpoint['files']))
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage.mkdir(parents=True, exist_ok=True, mode=0o700)
    session = session or requests.Session()
    try:
        sizes = {name:object_size(session, checkpoint['read_urls'][name]) for name in checkpoint['files']}
        if sum(sizes.values()) > hydration.MAX_TOTAL: raise ValueError('total model size bound')
        for name, digest in checkpoint['files'].items():
            if clock() >= manifest['deadline']: raise ValueError('epoch closed during checkpoint preparation')
            final = destination / name; partial = stage / (name + '.partial')
            if final.is_symlink() or partial.is_symlink(): raise ValueError('model cache symlink')
            if final.exists():
                if not final.is_file() or final.stat().st_size != sizes[name] or hydration.sha(final) != digest:
                    raise ValueError('preserve mismatching cached checkpoint')
                continue
            remaining = sizes[name] - (partial.stat().st_size if partial.exists() else 0)
            if shutil.disk_usage(state).free < remaining + 1024**3:
                raise OSError('insufficient checkpoint disk capacity; partial retained')
            download(session, checkpoint['read_urls'][name], partial, sizes[name], digest, manifest['deadline'], clock=clock)
            if hydration.sha(partial) != digest: raise ValueError('model checksum before admission')
            os.replace(partial, final)
        retire_owned_checkpoints(state, identifier)
        return destination
    finally:
        session.close()


def retire_owned_checkpoints(state, current):
    """Remove only this supervisor's obsolete model bytes after new admission."""
    owners = state / '.miner-supervisor' / 'owned-checkpoints'
    for owner in owners.glob('*.json'):
        bootstrap.check_parents(owner)
        record = json.loads(owner.read_bytes()); identifier = record.get('checkpoint')
        files = record.get('files', {})
        if (not re.fullmatch('[0-9a-f]{64}', str(identifier)) or owner.stem != identifier
                or hashlib.sha256(bootstrap.canonical(files)).hexdigest() != identifier):
            raise ValueError('owned checkpoint inventory')
        if identifier == current: continue
        for root, suffix in ((state / identifier, ''), (state / '.checkpoint-downloads' / identifier, '.partial')):
            bootstrap.check_parents(root)
            if not root.exists(): continue
            allowed = {name + suffix for name in files}
            children = list(root.iterdir())
            if any(p.name not in allowed or not p.is_file() or p.is_symlink() for p in children):
                break
        else:
            for root in (state / identifier, state / '.checkpoint-downloads' / identifier):
                if root.exists():
                    for path in root.iterdir(): path.unlink()
                    root.rmdir()
            owner.unlink()


def launch_arguments(args, manifest, manifest_url):
    result = ['--authority',args.authority,'--manifest-url',manifest_url,'--gateway','https://unused.invalid',
              '--key',str(Path(args.key).absolute()),'--state',str(Path(args.state).absolute()),
              '--source-bundle-sha256',manifest['source_bundle']['sha256'],'--once']
    for flag in ('env_id','max_batches','search_budget'):
        value = getattr(args, flag, None)
        if value is not None: result += ['--'+flag.replace('_','-'),str(value)]
    return result


def launch_once(args, observed, source, record, *, clock=time.time, popen=subprocess.Popen):
    manifest = observed['manifest']
    if record.exists(): return False  # Never repeat an issued epoch, including interrupted launches.
    if clock() >= manifest['deadline']: return False
    if type(getattr(args, 'lock_fd', None)) is not int: raise ValueError('inherited supervisor lock required')
    entry = dict(lock_inherited=True,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],
                 source=manifest['source_bundle']['sha256'],status='launching',started_at=clock(),deadline=manifest['deadline'])
    save(record, entry)  # Fail closed across a crash between process launch and PID recording.
    loader="import os,runpy,sys;sys.path.insert(0,os.getcwd());sys.argv=['affine-miner']+sys.argv[1:];runpy.run_module('subnet.cli',run_name='__main__')"
    environment = {key:value for key,value in os.environ.items() if not key.startswith('PYTHON')}
    try:
        child = popen([sys.executable,'-I','-B','-c',loader,*launch_arguments(args,manifest,observed['manifest_url'])],
                      cwd=source,env=environment,start_new_session=True,pass_fds=(args.lock_fd,))
    except OSError:
        entry.update(status='launch-failed',finished_at=clock());save(record,entry);raise
    try: ticks=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError: ticks=None
    entry.update(status='running',pid=child.pid,pid_ticks=ticks); save(record, entry)
    try:
        code = child.wait(timeout=max(.01,manifest['deadline']-clock()))
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGTERM)
        try: code = child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL); code = child.wait()
        entry['deadline_stop'] = True
    entry.update(status='complete' if code==0 else 'failed',returncode=code,finished_at=clock()); save(record,entry)
    return True


def original_child_pending(meta, now=None, *, locked=True):
    """After supervisor restart, don't overlap an original detached child."""
    now=time.time()if now is None else now
    for path in meta.glob('*.json'):
        entry=json.loads(path.read_bytes())
        if entry.get('status')=='launching':
            if not locked: continue
            # Caller holds the lock inherited by every issued child. A surviving
            # child would prevent this supervisor from acquiring that lock.
            if entry.get('lock_inherited') is not True:
                raise ValueError('legacy launch intent lacks inherited lock evidence')
            entry.update(status='interrupted',finished_at=now);save(path,entry);continue
        if entry.get('status')!='running':continue
        pid=entry.get('pid');ticks=entry.get('pid_ticks')
        if type(pid)is not int or pid<=0:raise ValueError('original miner process identity')
        try: fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        except FileNotFoundError: fields=None
        if fields and fields[0]!='Z' and (ticks is None or fields[19]==ticks):
            if now>=entry['deadline']and ticks is not None and os.getpgid(pid)==pid:
                terminated=entry.get('termination_requested_at')
                os.killpg(pid,signal.SIGKILL if terminated is not None and now-terminated>=10 else signal.SIGTERM)
                if terminated is None:
                    entry['termination_requested_at']=now;save(path,entry)
            return True
        entry.update(status='interrupted',finished_at=now);save(path,entry)
    return False


def cycle(args, meta, *, read_opening=opening, prefetch=prefetch_checkpoint, launcher=launch_once):
    if original_child_pending(meta): return 'original-child-running'
    observed = read_opening(args.discovery_url,args.authority)
    if observed is None: return 'closed'
    manifest = observed['manifest']; record = meta/(manifest['epoch']+'.json')
    if record.exists(): return 'already-issued'
    descriptor = manifest['source_bundle']
    body = bootstrap.download(bootstrap.source_download_url(descriptor),bootstrap.COMPRESSED_LIMIT)
    source = bootstrap.install(body,descriptor,args.source_cache)
    bootstrap.hydrate_task_assets(source,manifest,args.source_cache)
    try: prefetch(manifest,args.state)
    except ValueError:
        if time.time() >= manifest['deadline']: return 'closed-during-download'
        raise
    latest = read_opening(args.discovery_url,args.authority)
    if latest is None or bootstrap.canonical(latest['manifest']) != bootstrap.canonical(manifest): return 'changed-or-closed'
    return 'issued' if launcher(args,observed,source,record) else 'closed'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--discovery-url',default='https://affine.io/mining.json')
    parser.add_argument('--authority',required=True);parser.add_argument('--key',required=True)
    parser.add_argument('--state',required=True);parser.add_argument('--source-cache',required=True)
    parser.add_argument('--env-id',default='affine_math');parser.add_argument('--max-batches',type=int,default=3)
    parser.add_argument('--search-budget',type=int,default=32);parser.add_argument('--poll-seconds',type=int,default=10)
    args=parser.parse_args(argv)
    if not re.fullmatch('[0-9a-f]{64}',args.authority) or not 1<=args.poll_seconds<=300:
        raise ValueError('known authority and bounded poll interval')
    key=Path(args.key).absolute();bootstrap.check_parents(key)
    if not key.is_file():raise ValueError('existing private miner key path required')
    state=Path(args.state).absolute();bootstrap.check_parents(state);state.mkdir(mode=0o700,parents=True,exist_ok=True)
    meta=state/'.miner-supervisor';bootstrap.check_parents(meta);meta.mkdir(mode=0o700,exist_ok=True)
    fd=os.open(meta/'lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'a+b') as lock:
        while True:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                original_child_pending(meta, locked=False)
                print(json.dumps(dict(status='waiting-for-original-miner',at=time.time())),flush=True)
                time.sleep(args.poll_seconds)
        args.lock_fd=lock.fileno()
        while True:
            try: status=cycle(args,meta)
            except (requests.RequestException,OSError) as error:status='retry-'+type(error).__name__
            print(json.dumps(dict(status=status,at=time.time())),flush=True)
            time.sleep(args.poll_seconds)


if __name__=='__main__':main()
