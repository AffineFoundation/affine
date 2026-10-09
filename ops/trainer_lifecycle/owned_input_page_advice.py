"""Read-only advice for a model authenticated by the original backend.

Install in the same process after checkpoint authentication and before the
unchanged resource admission. Reinstall the StateCache hook after namespace
reloads. This operator never lowers reserves, deletes files or touches Adam.
"""
import importlib
import json
import os
from pathlib import Path
import stat


class OwnedInputPageAdvice:
    """Advise only unchanged files authenticated by the original checkpoint()."""
    fields=('st_dev','st_ino','st_mode','st_uid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
    def __init__(self):self.context=None
    @classmethod
    def snapshot(cls,value):return {name:getattr(value,name)for name in cls.fields}
    @staticmethod
    def observe():
        module=importlib.import_module('subnet.persistent_training_state')
        result={'available_ram_bytes':module.available_ram_bytes()}
        root=Path('/sys/fs/cgroup')
        if (root/'memory.current').exists():
            result['cgroup_current_bytes']=int((root/'memory.current').read_text())
            stats=dict(line.split()for line in(root/'memory.stat').read_text().splitlines())
            result['cgroup_bytes']={k:int(stats.get(k,0))for k in('anon','file','inactive_file','active_file','file_dirty','file_writeback','file_mapped','shmem','unevictable')}
        return result
    def authenticated(self,target,manifest,workspace,approved_cache=None):
        root=Path(workspace).absolute();target=Path(target).absolute();cp=manifest['checkpoint']
        expected=Path(approved_cache).absolute()if approved_cache is not None else root/'checkpoints'/cp['id']
        if root!=root.resolve()or target!=target.resolve()or target!=expected or target==root or not target.is_relative_to(root)or target.is_symlink()or any(p.startswith('.')for p in target.relative_to(root).parts):
            raise ValueError('post-load advice only exact owned input checkpoint')
        rows={}
        for name in cp['files']:
            member=target/name
            if Path(name).name!=name or member.is_symlink():raise ValueError('post-load checkpoint ordinary member')
            st=member.stat()
            if not stat.S_ISREG(st.st_mode)or st.st_uid!=os.geteuid()or st.st_nlink!=1:
                raise ValueError('post-load checkpoint owned unshared regular file')
            rows[name]=self.snapshot(st)
        self.context=dict(workspace=str(root),path=str(target),epoch=manifest['epoch'],checkpoint=cp['id'],files=dict(cp['files']),stats=rows)
    def install_checkpoint(self,backend):
        original=backend.checkpoint
        def checkpoint(manifest,workspace,cache=None):
            self.context=None
            target=original(manifest,workspace,cache)
            self.authenticated(target,manifest,workspace,cache)
            return target
        backend.checkpoint=checkpoint
    def install_cache(self,module):
        original=module.StateCache.admit;owner=self
        def admit(cache,plan,**kwargs):
            c=owner.context;cp=cache.manifest['checkpoint']
            if c is None or c['workspace']!=str(cache.workspace)or c['epoch']!=cache.manifest['epoch']or c['checkpoint']!=cp['id']or c['files']!=cp['files']:
                raise ValueError('post-load advice original authenticated model binding')
            before=owner.observe();advised=0;target=Path(c['path'])
            root=Path(c['workspace'])
            if target.resolve()!=target or not target.is_relative_to(root)or target==root:
                raise ValueError('post-load checkpoint root changed')
            for name,expected in c['stats'].items():
                member=target/name;fd=os.open(member,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
                try:
                    if owner.snapshot(os.fstat(fd))!=expected or owner.snapshot(member.lstat())!=expected:
                        raise ValueError('authenticated checkpoint changed before post-load page advice')
                    os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
                    if owner.snapshot(os.fstat(fd))!=expected or owner.snapshot(member.lstat())!=expected:
                        raise ValueError('checkpoint changed during post-load page advice')
                    advised+=expected['st_size']
                finally:os.close(fd)
            evidence=dict(version='owned-authenticated-input-postload-page-advice-v1',checkpoint=c['checkpoint'],files=len(c['stats']),advised_bytes=advised,bytes_deleted=0,optimizer_files_touched=False,before=before,after=owner.observe(),resource_guard_changed=False)
            print(json.dumps({'postload_input_page_advice':evidence},sort_keys=True),flush=True)
            result=original(cache,plan,**kwargs)
            return dict(result,postload_input_page_advice=evidence)
        module.StateCache.admit=admit
