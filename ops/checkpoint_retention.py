"""Remove exact idle checkpoint replicas after complete archive verification.

The operator authenticates the checkpoint descriptor and hashes every archived
object before delegating this narrow plan. No current/protected checkpoint,
unarchived export, job, report, or arbitrary workspace directory can qualify.
"""
import hashlib
import json
import re
import secrets
import stat
import subprocess
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def processes():return Path('/proc').iterdir()


def gpu_processes():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()


def remove_verified_hardlink_alias(plan):
    """Remove exactly one known alias; keep the authenticated source intact.

    Unknown third links, an active/current checkpoint, open files, memory maps
    and GPU work all refuse retirement. The operator supplies a fresh complete
    archive readback and explicit source/alias paths, never recursive discovery.
    """
    checkpoint=plan['checkpoint'];files=plan['files']
    source=Path(plan['source_directory']);alias=Path(plan['directory'])
    protected=plan['protected_checkpoints'];active=plan['active_checkpoints']
    if (not re.fullmatch('[0-9a-f]{64}',checkpoint) or
            not isinstance(files,dict) or not 1<=len(files)<=32 or
            'config.json'not in files or not any(n.endswith('.safetensors')for n in files) or
            any(not re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.-]*',n) or
                Path(n).suffix not in {'.json','.safetensors','.txt','.model','.jinja','.tiktoken'}for n in files)):
        raise ValueError('approved safe checkpoint descriptor required')
    if any(not isinstance(v,dict) or not re.fullmatch('[0-9a-f]{64}',v.get('sha256','')) or
           type(v.get('size'))is not int or not 0<v['size']<=(32 if n.endswith('.safetensors')else 5)*1024**3
           for n,v in files.items()):raise ValueError('archived checkpoint object metadata required')
    if digest({n:v['sha256']for n,v in files.items()})!=checkpoint:raise ValueError('checkpoint descriptor changed')
    if (not isinstance(protected,list) or not 1<=len(protected)<=32 or
            not isinstance(active,list) or len(active)>512 or
            any(not isinstance(c,str)or not re.fullmatch('[0-9a-f]{64}',c)for c in protected+active)):
        raise ValueError('explicit current and active checkpoint protection required')
    if checkpoint in protected+active:raise ValueError('current or referenced checkpoint protected')
    if plan.get('archive_verified')is not True or plan.get('descriptor_authenticated')is not True:
        raise ValueError('authenticated complete archive readback required')
    if source==alias:raise ValueError('distinct source and alias paths required')
    for root in (source,alias):
        if (not root.is_absolute() or root.name!=checkpoint or root.parent.name not in ('checkpoint','checkpoints') or
                root.resolve()!=root or root.is_symlink() or not root.is_dir()):
            raise ValueError('canonical ordinary checkpoint cache path required')
        if {p.name for p in root.iterdir()}!=set(files):raise ValueError('exact checkpoint membership required')
    if gpu_processes():raise ValueError('checkpoint retirement requires idle GPU')
    before={}
    for name,expected in files.items():
        a=(alias/name).lstat();s=(source/name).lstat()
        if (not stat.S_ISREG(a.st_mode)or not stat.S_ISREG(s.st_mode)or
                a.st_nlink!=2 or s.st_nlink!=2 or
                (a.st_dev,a.st_ino)!=(s.st_dev,s.st_ino)or a.st_size!=expected['size']):
            raise ValueError('exact two known hardlinks required')
        h=hashlib.sha256()
        with (source/name).open('rb')as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=expected['sha256']:raise ValueError('checkpoint local bytes changed')
        before[name]=(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)
    for process in processes():
        if not process.name.isdecimal():continue
        try:descriptors=list((process/'fd').iterdir())
        except FileNotFoundError:continue
        for fd in descriptors:
            try:target=fd.readlink()
            except FileNotFoundError:continue
            if target.parent in (source,alias):raise ValueError('checkpoint object still open')
        try:maps=(process/'maps').read_text()
        except FileNotFoundError:continue
        for line in maps.splitlines():
            row=line.split(None,5)
            if len(row)==6 and any(row[5].startswith(str(root)+'/')for root in (source,alias)):
                raise ValueError('checkpoint object still memory mapped')
    for root in (source,alias):
        if {p.name for p in root.iterdir()}!=set(files):raise ValueError('checkpoint membership changed')
        for name,previous in before.items():
            s=(root/name).lstat()
            if s.st_nlink!=2 or (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=previous:
                raise ValueError('checkpoint changed during check')
    if gpu_processes():raise ValueError('checkpoint retirement state changed during check')
    retired=alias.with_name(checkpoint+'.retired-alias-'+secrets.token_hex(8));alias.rename(retired)
    for name in files:(retired/name).unlink()
    retired.rmdir()
    return dict(removed=True,alias_removed=True,physical_bytes_freed=0,
                source_preserved=True,checkpoint=checkpoint)


def remove_checkpoint_replica(plan):
    checkpoint=plan['checkpoint'];files=plan['files'];root=Path(plan['directory'])
    if (not re.fullmatch('[0-9a-f]{64}',checkpoint) or not isinstance(files,dict) or not 1<=len(files)<=32 or
            'config.json' not in files or not any(n.endswith('.safetensors') for n in files) or
            any(not re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.-]*',n) or Path(n).suffix not in
                {'.json','.safetensors','.txt','.model','.jinja','.tiktoken'} for n in files)):
        raise ValueError('approved safe checkpoint descriptor required')
    if any(not isinstance(v,dict) or not re.fullmatch('[0-9a-f]{64}',v.get('sha256','')) or
            type(v.get('size')) is not int or not 0<v['size']<=(32 if n.endswith('.safetensors') else 5)*1024**3
            for n,v in files.items()):
        raise ValueError('archived checkpoint object metadata required')
    if digest({name:row['sha256'] for name,row in files.items()})!=checkpoint:
        raise ValueError('checkpoint descriptor digest changed')
    protected=plan['protected_checkpoints'];active=plan['active_checkpoints']
    if (not isinstance(protected,list) or not 1<=len(protected)<=32 or not isinstance(active,list) or len(active)>512 or
            any(not isinstance(c,str) or not re.fullmatch('[0-9a-f]{64}',c) for c in protected+active)):
        raise ValueError('explicit current and active checkpoint protection required')
    if checkpoint in protected or checkpoint in active:
        raise ValueError('current or referenced checkpoint protected')
    if plan.get('archive_verified') is not True or plan.get('descriptor_authenticated') is not True:
        raise ValueError('authenticated complete archive readback required')
    # Only ordinary content-addressed cache directories can be retired here.
    # Job exports and intermediate optimizer snapshots need a separate policy.
    if not root.is_absolute() or root.name!=checkpoint or root.parent.name not in ('checkpoint','checkpoints') or root.resolve()!=root:
        raise ValueError('canonical checkpoint cache path required')
    if not root.exists():return dict(removed=False,already_absent=True,bytes=0,checkpoint=checkpoint)
    if not root.is_dir() or root.is_symlink():raise ValueError('regular checkpoint directory required')
    if gpu_processes():raise ValueError('checkpoint retirement requires idle GPU')
    if {p.name for p in root.iterdir()}!=set(files):raise ValueError('exact checkpoint membership required')
    before={};total=0
    for name,expected in files.items():
        path=root/name;before[name]=path.lstat()
        if not stat.S_ISREG(before[name].st_mode) or before[name].st_nlink!=1 or before[name].st_size!=expected['size']:
            raise ValueError('checkpoint local object type or size changed')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=expected['sha256']:raise ValueError('checkpoint local bytes changed')
        total+=before[name].st_size
    for process in processes():
        if not process.name.isdecimal():continue
        try:descriptors=list((process/'fd').iterdir())
        except FileNotFoundError:continue
        for fd in descriptors:
            try:target=fd.readlink()
            except FileNotFoundError:continue
            if target.parent==root:raise ValueError('checkpoint object still open')
        try:maps=(process/'maps').read_text()
        except FileNotFoundError:continue
        for line in maps.splitlines():
            row=line.split(None,5)
            if len(row)==6 and row[5].startswith(str(root)+'/'):
                raise ValueError('checkpoint object still memory mapped')
    for name,previous in before.items():
        current=(root/name).lstat()
        if (current.st_ino,current.st_size,current.st_mtime_ns)!=(previous.st_ino,previous.st_size,previous.st_mtime_ns):
            raise ValueError('checkpoint local object changed during check')
    if {p.name for p in root.iterdir()}!=set(files) or gpu_processes():
        raise ValueError('checkpoint retirement state changed during check')
    retired=root.with_name(checkpoint+'.retired-'+secrets.token_hex(8))
    if retired.exists():raise ValueError('retirement target already exists')
    root.rename(retired)
    for name in files:(retired/name).unlink()
    retired.rmdir()
    return dict(removed=True,already_absent=False,bytes=total,checkpoint=checkpoint,
                archives_and_protected_checkpoints_preserved=True)
