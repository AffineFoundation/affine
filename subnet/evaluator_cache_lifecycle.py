"""Owned evaluator checkpoint retention, independent of scientific results."""
import hashlib,json,shutil,time
from pathlib import Path
from .cache_lifecycle import CacheLifecycle,identifier
from .backend_jobs import signed
VERSION='owned-evaluator-cache-catalog-v1'

def retain(workspace,current,*,required_free_bytes=0):
    """Evict only authenticated old inventories; inherited locks protect readers."""
    lifecycle=CacheLifecycle(workspace)
    removed=lifecycle.evict_checkpoints(exclude=[identifier(current)],keep=0,
                                       required_free_bytes=required_free_bytes)
    return dict(removed=removed,free_bytes=shutil.disk_usage(workspace).free)

def adopt(envelope,authority,*,now=None):
    value=signed(envelope,authority);now=time.time()if now is None else now
    if (value.get('version')!=VERSION or value.get('quiescent_readers_confirmed')is not True or
        not value['created_at']<=now<value['expires_at']):
        raise ValueError('fresh explicit evaluator ownership catalog required')
    from .trainer_cache_lifecycle import live_original
    results=[]
    for entry in value['roots']:
        root=Path(entry['root'])
        if not root.is_absolute()or root!=root.resolve():raise ValueError('owned evaluator root')
        if any(live_original(json.loads(p.read_bytes()))for p in (root/'runner-status').glob('*.json')):
            raise ValueError('original evaluator still live')
        cache=CacheLifecycle(root)
        # Validate every inventory before registering or deleting any cache.
        reviewed=[]
        for row in entry['checkpoints']:
            cp=signed(row['checkpoint_document'],authority);cp=cp.get('checkpoint',cp)
            directory=root/row['relative_path']
            if directory.resolve()!=directory.absolute():raise ValueError('symlink evaluator cache')
            if set(p.name for p in directory.iterdir())!=set(cp['files']):raise ValueError('exact checkpoint inventory')
            for name,digest in cp['files'].items():
                path=directory/name
                from .cache_lifecycle import snapshot
                before=snapshot(path);h=hashlib.sha256()
                with path.open('rb')as stream:
                    for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
                if h.hexdigest()!=digest or snapshot(path)!=before:raise ValueError('authenticated evaluator cache bytes changed')
            reviewed.append((cp,directory,row['checkpoint_document']))
        for cp,directory,document in reviewed:
            with cache.lease_checkpoint(cp['id'],blocking=False):
                cache.adopt_checkpoint(cp['id'],directory,cp['files'],document)
        results.append(dict(root=str(root),**retain(root,entry['current_checkpoint'])))
    return results
