"""Bounded continuous outbox I/O; signed requests retain their separate8MiB limit."""
import json,os,stat,tempfile
from pathlib import Path
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
OUTBOX_MAX_BYTES=64*1024**2
_FIELDS=('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
def _route(path):
 p=Path(path)
 if p.name!='outbox.json'or p!=p.absolute()or p.parent.resolve()!=p.parent:raise ValueError('exact canonical continuous outbox route')
 s=p.parent.lstat()
 if not stat.S_ISDIR(s.st_mode)or s.st_uid!=os.geteuid()or s.st_mode&0o077:raise ValueError('private owned continuous outbox directory')
 return p

def read_outbox(path):
 p=_route(path);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd)
  if not stat.S_ISREG(before.st_mode)or before.st_uid!=os.geteuid()or before.st_nlink!=1 or before.st_mode&0o077 or before.st_size>OUTBOX_MAX_BYTES:raise ValueError('bounded private owned continuous outbox')
  with os.fdopen(os.dup(fd),'rb')as stream:raw=stream.read(OUTBOX_MAX_BYTES+1)
  after=os.fstat(fd);current=p.lstat()
  if len(raw)!=before.st_size or any(getattr(before,k)!=getattr(after,k)or getattr(before,k)!=getattr(current,k)for k in _FIELDS):raise ValueError('continuous outbox changed during read')
  return json.loads(raw)
 finally:os.close(fd)

def save_outbox(path,value):
 p=_route(path);raw=canonical(value)
 # Refuse before creating a temporary file or replacing historical evidence.
 if len(raw)>OUTBOX_MAX_BYTES:raise ValueError('continuous outbox capacity exhausted; preserve originals')
 if p.exists()or p.is_symlink():read_outbox(p)
 fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='continuous-outbox-')
 try:
  with os.fdopen(fd,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
  os.replace(tmp,p)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
