"""Disposable, operator-owned cache lifetime; durable artifacts remain in R2.

Receipts are created only at authenticated hydration/readback or successful job
ACK. They bind the verified digest to inode metadata without hashing again.
Locks are inherited by backend children, so parent death cannot evict live input.
This module never discovers or adopts external directories implicitly.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time


def identifier(value):
    if not isinstance(value,str) or re.fullmatch(r'[A-Za-z0-9_-]{1,160}',value) is None:
        raise ValueError('cache identifier')
    return value


def snapshot(path):
    path=Path(path)
    if path.absolute()!=path.resolve():raise ValueError('cache symlink path')
    s=path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=os.getuid():
        raise ValueError('cache regular owned single-link file required')
    return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime=s.st_mtime_ns,
                ctime=s.st_ctime_ns,mode=s.st_mode,uid=s.st_uid)


class CacheLifecycle:
    def __init__(self,root):
        self.root=Path(root).absolute()
        self.root.mkdir(parents=True,exist_ok=True)
        if self.root!=self.root.resolve():raise ValueError('cache root symlink')
        self.meta=self.root/'.cache-lifecycle';self.meta.mkdir(exist_ok=True,mode=0o700)
        if self.meta!=self.meta.resolve():raise ValueError('metadata symlink')
        self.meta.chmod(0o700)

    def _path(self,relative):
        relative=Path(relative)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('outside owned cache')
        path=self.root/relative
        if path.absolute()!=path.resolve():raise ValueError('cache path symlink')
        return path

    @contextmanager
    def lease_checkpoint(self,cp,blocking=True):
        cp=identifier(cp);path=self.meta/(cp+'.lock')
        fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
            yield fd
        finally:os.close(fd)

    def _receipt(self,cp):return self.meta/(identifier(cp)+'.json')

    def _save(self,path,value):
        fd,tmp=tempfile.mkstemp(dir=self.meta,prefix='receipt-')
        try:
            with os.fdopen(fd,'w') as stream:
                json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)

    def record_checkpoint_member(self,cp,name,approved_files,verified_sha256,origin='authenticated-model-map'):
        """Caller has just checked these bytes against authenticated inventory.

        The caller must hold its checkpoint lease. External mapped caches are
        deliberately not representable through this API.
        """
        identifier(cp)
        if Path(name).name!=name or name in ('.','..') or approved_files.get(name)!=verified_sha256:
            raise ValueError('approved checkpoint receipt binding')
        if re.fullmatch('[0-9a-f]{64}',verified_sha256 or '') is None:raise ValueError('checkpoint digest')
        for member,sha in approved_files.items():
            if Path(member).name!=member or member in ('.','..') or re.fullmatch('[0-9a-f]{64}',sha or '') is None:
                raise ValueError('approved file inventory')
        path=self._path(Path('checkpoints')/cp/name);receipt=self._receipt(cp)
        value=json.loads(receipt.read_text()) if receipt.exists() else dict(cp=cp,files=approved_files,members={})
        if value['cp']!=cp or value['files']!=approved_files:raise ValueError('receipt inventory changed')
        value['members'][name]=dict(sha256=verified_sha256,stat=snapshot(path),origin=origin)
        value['touched']=time.time();self._save(receipt,value)

    def record_checkpoint(self,cp,approved_files,origin='authenticated-job-ACK'):
        """Record exact backend-verified inventory after successful report ACK."""
        path=self._path(Path('checkpoints')/identifier(cp))
        if not path.exists():return False
        if set(p.name for p in path.iterdir())!=set(approved_files):raise ValueError('checkpoint inventory')
        for name,sha in approved_files.items():self.record_checkpoint_member(cp,name,approved_files,sha,origin)
        return True

    def adopt_checkpoint(self,cp,path,approved_files,durability_ack):
        """Explicit verified export adoption after authenticated durable ACK.

        This is an operator API, not miner-supplied configuration. No arbitrary
        external path can be adopted. Caller holds the cp lease and has already
        authenticated the inventory and publication/readback acknowledgement.
        """
        identifier(cp)
        if not durability_ack:raise ValueError('durability ACK required')
        relative=Path(path).absolute().relative_to(self.root)
        ordinary=relative==Path('checkpoints')/cp
        export=(len(relative.parts)==3 and relative.parts[0]=='jobs' and
                relative.parts[2] in ('checkpoint-persistent-final','checkpoint-final'))
        if not ordinary and not export:raise ValueError('approved owned checkpoint path')
        if export:identifier(relative.parts[1])
        directory=self._path(relative)
        if set(p.name for p in directory.iterdir())!=set(approved_files):raise ValueError('checkpoint inventory')
        members={}
        for name,sha in approved_files.items():
            if Path(name).name!=name or name in ('.','..') or re.fullmatch('[0-9a-f]{64}',sha or '') is None:
                raise ValueError('approved inventory')
            members[name]=dict(sha256=sha,stat=snapshot(directory/name),origin='durable-publication-ACK')
        self._save(self._receipt(cp),dict(cp=cp,path=str(relative),files=approved_files,members=members,
                                        touched=time.time(),durability_ack=durability_ack))

    def evict_checkpoints(self,exclude=(),keep=1,required_free_bytes=0):
        if keep<0 or required_free_bytes<0:raise ValueError('retention budget')
        excluded=set(exclude);records=[];removed=[]
        for path in self.meta.glob('*.json'):
            value=json.loads(path.read_text());cp=value.get('cp')
            if cp:records.append((value.get('touched',0),cp,value))
        records.sort(reverse=True)
        retained={cp for _,cp,_ in records[:keep]}|excluded
        for _,cp,value in reversed(records):
            if cp in retained:continue
            if required_free_bytes and os.statvfs(self.root).f_bavail*os.statvfs(self.root).f_frsize>=required_free_bytes:break
            try:
                with self.lease_checkpoint(cp,blocking=False):
                    directory=self._path(value.get('path',str(Path('checkpoints')/identifier(cp))))
                    if not directory.exists():continue
                    members={p.name:p for p in directory.iterdir()}
                    if not members or not set(members)<=set(value['files']) or set(members)!=set(value['members']):continue
                    if any(snapshot(p)!=value['members'][name]['stat'] or value['members'][name]['sha256']!=value['files'][name] for name,p in members.items()):continue
                    for p in members.values():p.unlink()
                    directory.rmdir();self._receipt(cp).unlink();removed.append(cp)
            except (BlockingIOError,ValueError,FileNotFoundError):continue
        return removed

    def record_download(self,path,verified_sha256):
        path=Path(path).absolute();relative=path.relative_to(self.root)
        if len(relative.parts)!=3 or relative.parts[0]!='jobs' or not re.fullmatch(r'submission-[0-9]+\.(zip|json)',relative.name):
            raise ValueError('disposable submission path')
        identifier(relative.parts[1]);self._path(relative)
        if re.fullmatch('[0-9a-f]{64}',verified_sha256 or '') is None:raise ValueError('download digest')
        receipt=self.meta/('download-'+relative.parts[1]+'.json')
        value=json.loads(receipt.read_text()) if receipt.exists() else {}
        value[str(relative)]=dict(sha256=verified_sha256,stat=snapshot(path));self._save(receipt,value)

    def retire_downloads(self,job_id):
        """Call only after coordinator ACK; retain reports and all diagnostics."""
        receipt=self.meta/('download-'+identifier(job_id)+'.json')
        if not receipt.exists():return []
        value=json.loads(receipt.read_text());removed=[]
        for relative,record in list(value.items()):
            path=self._path(relative)
            try:
                if snapshot(path)!=record['stat']:continue
                path.unlink();removed.append(relative);del value[relative]
            except FileNotFoundError:del value[relative]
            except ValueError:continue
        if value:self._save(receipt,value)
        else:receipt.unlink()
        return removed
