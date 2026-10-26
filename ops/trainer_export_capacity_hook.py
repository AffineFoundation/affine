"""Prospective CPU-only storage reconciliation before the original disk gate.

Keep the original scientific capacity function and all its bounds unchanged.
A stale or consumed retirement catalog never creates a new epoch barrier.
"""
import hashlib
import json
import logging
import shlex
from pathlib import Path

DISK_ERROR = 'persistent trainer bounded stream/input/export/artifact disk reserve'
LOG = logging.getLogger(__name__)


def file_pin(row):
    path = Path(row['path'])
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('pinned trainer retention operator artifact')
    return path


def stage(remote, policy):
    helper = file_pin(policy['helper']); catalog = file_pin(policy['catalog'])
    if remote.workspace != policy['workspace'] or remote.code != policy['backend_code']:
        raise ValueError('original trainer endpoint workspace and backend binding')
    root = remote.workspace+'/private-trainer-export-retention/'+policy['helper']['sha256'][:16]+'-'+policy['catalog']['sha256'][:16]
    files = {root+'/helper.py':policy['helper']['sha256'],root+'/catalog.json':policy['catalog']['sha256']}
    code='import hashlib,json,os,stat;from pathlib import Path\np=Path('+repr(root)+');assert p.resolve()==p;p.mkdir(parents=True,mode=0o700,exist_ok=True);assert not p.stat().st_mode&0o077\nfiles='+repr(files)+'\nout={}\nfor name,digest in files.items():\n q=Path(name);assert not q.is_symlink();out[name]=hashlib.sha256(q.read_bytes()).hexdigest()if q.exists()else None\nprint(json.dumps(out))'
    observed=json.loads(remote.command(shlex.quote(remote.python)+' -I -B -c '+shlex.quote(code),timeout=30))
    for local,target in [(helper,root+'/helper.py'),(catalog,root+'/catalog.json')]:
        if observed[target] is None:remote.copy_to(local,target)
        elif observed[target] != files[target]:raise ValueError('immutable trainer retention staging collision')
    return root


def operation(remote, policy, authority, *, apply):
    """Stage immutable artifacts and run with the existing qualified lifecycle."""
    root=stage(remote,policy)
    data=dict(root=root,helper_sha256=policy['helper']['sha256'],catalog_sha256=policy['catalog']['sha256'],
              code=policy['backend_code'],overlay=policy['lifecycle_overlay'],authority=authority,
              workspace=policy['workspace'],apply=apply)
    code='DATA='+repr(data)+'\n'+'''import importlib.util,hashlib,json,sys,os,time
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(DATA['root']);hp=root/'helper.py';cp=root/'catalog.json'
assert hashlib.sha256(hp.read_bytes()).hexdigest()==DATA['helper_sha256']
assert hashlib.sha256(cp.read_bytes()).hexdigest()==DATA['catalog_sha256']
sys.path.insert(0,DATA['code']);import subnet
for name,digest in DATA['overlay']['files'].items():
 p=Path(DATA['overlay']['code'])/name
 assert p.resolve()==p and p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest
for name in ['cache_lifecycle','optimizer_state_cache','trainer_cache_lifecycle']:
 p=Path(DATA['overlay']['code'])/'subnet'/(name+'.py')
 spec=importlib.util.spec_from_file_location('subnet.'+name,p);m=importlib.util.module_from_spec(spec);sys.modules['subnet.'+name]=m;setattr(subnet,name,m);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('qualified_export_retention',hp);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
before=os.statvfs(DATA['workspace']).f_bavail*os.statvfs(DATA['workspace']).f_frsize
try:result=m.retire_catalog(json.loads(cp.read_bytes()),DATA['authority'],DATA['workspace'],apply=DATA['apply'])
except m.StaleCatalogError:result=dict(status='catalog_refresh_required',removed_checkpoints=[])
receipt=dict(at=time.time(),host=os.uname().nodename,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),result=result,free_before=before,apply=DATA['apply'],helper_sha256=DATA['helper_sha256'],catalog_sha256=DATA['catalog_sha256'])
p=root/('apply-'if DATA['apply']else'review-')
p=root/(p.name+str(time.time_ns())+'.json')
fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'w')as f:json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt))
'''
    return json.loads(remote.command(shlex.quote(remote.python)+' -I -B -c '+shlex.quote(code),timeout=180))


def retry_capacity(original, reconcile, *args, **kwargs):
    """Only exact disk exhaustion triggers retirement; always recheck the gate."""
    try:return original(*args,**kwargs)
    except ValueError as error:
        if str(error) != DISK_ERROR:raise
    try:reconcile()
    except Exception as error:
        # Storage reconciliation is optional. The unchanged original gate is
        # the authority on whether training can proceed after an ambiguous or
        # stale catalog; no guard or memory error is suppressed.
        LOG.warning('trainer export retention deferred: %s',type(error).__name__)
    return original(*args,**kwargs)


def install(router_class, policy, authority):
    """Call after installing the original capacity and hydration wrappers."""
    if getattr(router_class,'_authenticated_export_capacity_hook',False):
        raise ValueError('trainer export capacity hook already installed')
    for key in ['helper','catalog']:file_pin(policy[key])
    original=router_class.training_capacity
    def capacity(self,manifest,steps,submission_bytes=None):
        def reconcile():
            result=operation(self.roles['train'],policy,authority,apply=True)
            from subnet.remote_backend import save
            import time
            save(self.controller.state/'roles'/('trainer-retention-'+str(time.time_ns())+'.json'),result)
        return retry_capacity(lambda:original(self,manifest,steps,submission_bytes=submission_bytes),reconcile)
    router_class.training_capacity=capacity
    router_class._authenticated_export_capacity_hook=True
