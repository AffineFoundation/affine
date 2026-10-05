"""Include approved native task dependencies before immutable source sealing."""
import hashlib,os,tarfile
from pathlib import Path,PurePosixPath

def snapshot_references(environments):
    refs=set()
    for row in environments:
        value=row.get('spec',row).get('config',{}).get('task_snapshot')
        if value is None:continue
        p=PurePosixPath(value)
        if not isinstance(value,str)or p.is_absolute()or '..'in p.parts or '\\'in value or not p.parts:raise ValueError('sealed source requires a bundled relative native snapshot')
        refs.add(value)
    return sorted(refs)
def file_hash(path):
    p=Path(path)
    if p.is_symlink()or not p.is_file():raise ValueError('regular approved snapshot/file required')
    return hashlib.sha256(p.read_bytes()).hexdigest()
def include_snapshots(candidate,environments,approved_root,approved_files):
    """Caller supplies a NEW candidate and authenticated original source inventory."""
    root=Path(candidate);approved=Path(approved_root);included={}
    for name in snapshot_references(environments):
        if name not in approved_files:raise ValueError('native snapshot absent from authenticated approved inventory')
        src=approved/name
        if file_hash(src)!=approved_files[name]:raise ValueError('approved original task data changed')
        dest=root/name
        if not dest.absolute().is_relative_to(root.absolute()):raise ValueError('snapshot candidate escape')
        for p in [dest,*dest.parents]:
            if p==root.parent:break
            if p.is_symlink():raise ValueError('snapshot directory symlink')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if file_hash(dest)!=approved_files[name]:raise ValueError('candidate snapshot differs; preserve evidence')
        else:
            fd=os.open(dest,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'wb')as f:f.write(src.read_bytes());f.flush();os.fsync(f.fileno())
        included[name]=file_hash(dest)
    return included

def validate_archive_snapshots(archive,environments,approved_files):
    """Pre-publication gate catches omissions even from a separate assembler."""
    required=snapshot_references(environments)
    with tarfile.open(archive,'r:gz')as tar:
        rows=tar.getmembers()
        for name in required:
            selected=[r for r in rows if r.name==name]
            if name not in approved_files or len(selected)!=1 or not selected[0].isfile():raise ValueError('sealed source missing exact native snapshot dependency')
            data=tar.extractfile(selected[0]).read()
            if hashlib.sha256(data).hexdigest()!=approved_files[name]:raise ValueError('sealed native snapshot integrity')
    return {n:approved_files[n]for n in required}

def seal_source_archive(candidate,archive,environments,approved_root,approved_files):
    """Fresh reviewable archive +complete inventory, including ignored task data."""
    root=Path(candidate);archive=Path(archive)
    if archive.exists()or archive.absolute().is_relative_to(root.absolute()):raise ValueError('fresh archive outside candidate required')
    include_snapshots(root,environments,approved_root,approved_files)
    inventory={}
    for path in sorted(root.rglob('*')):
        if '__pycache__'in path.parts or path.suffix=='.pyc':continue
        if path.is_symlink():raise ValueError('source symlink unsupported')
        if path.is_file():inventory[str(path.relative_to(root))]=file_hash(path)
    with tarfile.open(archive,'w:gz')as tar:
        for name in sorted(inventory):tar.add(root/name,arcname=name,recursive=False)
    validate_archive_snapshots(archive,environments,approved_files)
    return {'archive_sha256':file_hash(archive),'inventory':inventory,'native_snapshot_dependencies':{n:inventory[n]for n in snapshot_references(environments)}}
