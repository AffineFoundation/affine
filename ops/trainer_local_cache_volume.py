"""Own the trainer's local Adam working volume; no storage network access.

The inference model stays on disk. Adam's current and candidate states live in
alternating local disk/shared-memory volumes and retire through StateCache.
Losing the pod loses Adam; never manufacture a replacement optimizer.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys


def ordinary(path, *, directory=False):
    path=Path(path)
    st=path.lstat()
    if (path!=path.resolve() or st.st_uid!=os.geteuid() or st.st_mode&0o077 or
            (not stat.S_ISDIR(st.st_mode) if directory else
             not stat.S_ISREG(st.st_mode) or st.st_nlink!=1)):
        raise ValueError('private owned trainer volume member')
    return st


def memory_available():
    from subnet.persistent_training_state import available_ram_bytes
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return min(int(line.split()[1])*1024,available_ram_bytes())
    raise ValueError('actual trainer memory availability required')


def authenticated_current(workspace,root,authority):
    from subnet.backend_jobs import signed
    from subnet.storage import canonical
    from subnet.optimizer_state_cache import sha,verification_body,STAT_VERSION,candidate_directory
    from subnet.cache_lifecycle import snapshot
    ordinary(root/'current.json')
    current=json.loads((root/'current.json').read_bytes())
    ack=signed(current['ROOT_ack'],authority)
    job_id=current['job_id']
    from subnet.cache_lifecycle import identifier
    identifier(job_id)
    job_path=workspace/(job_id+'.json');report_path=workspace/'jobs'/job_id/'report.json'
    ordinary(job_path);ordinary(report_path)
    job=signed(json.loads(job_path.read_bytes()),authority)
    report=json.loads(report_path.read_bytes())
    if (ack['version']!='durable-original-trainer-cache-ACK-v1' or
            ack['authority_state_committed']is not True or ack['job_id']!=job_id or
            sha(job)!=ack['job_sha256'] or sha(report)!=ack['report_sha256'] or
            report['success']is not True or report['job_id']!=job_id or
            report['new_checkpoint']!=ack['new_checkpoint'] or
            current['descriptor_sha256']!=ack['trainer_state']['descriptor_sha256']):
        raise ValueError('actual acknowledged trainer-local state required')
    state=report['persistent_training_state'];descriptor=state['descriptor']
    if sha(descriptor)!=current['descriptor_sha256']:
        raise ValueError('acknowledged optimizer descriptor')
    shards={row['name']:row for row in descriptor['shards']}
    if set(current['files'])!=set(shards):raise ValueError('complete current local state')
    candidate=candidate_directory(workspace,root,job_id);ordinary(candidate,directory=True)
    for name,row in current['files'].items():
        if Path(name).name!=name or name in ('.','..'):raise ValueError('one original shard member')
        path=candidate/name
        if (snapshot(path)!=row['stat'] or
                (row['sha256'],row['size'])!=(shards[name]['sha256'],shards[name]['size'])):
            raise ValueError('unchanged acknowledged optimizer inode')
    if 'promotion_verification_sha256' in current and current['promotion_verification_sha256']!=sha(verification_body(current)):
        raise ValueError('original optimizer promotion verification')
    return current,descriptor


def prepare(workspace,authority,*,check=False):
    from subnet.optimizer_state_cache import VOLUME_VERSION,volume_policy
    from subnet.storage import canonical
    workspace=Path(workspace);ordinary(workspace,directory=True)
    root=workspace/'.optimizer-state-cache';ordinary(root,directory=True)
    current,descriptor=authenticated_current(workspace,root,authority)
    total=sum(row['size']for row in descriptor['shards'])
    volume=Path('/dev/shm')/('affine-optimizer-'+hashlib.sha256(str(workspace).encode()).hexdigest()[:20])
    selected=volume_policy(workspace,root)
    if selected is not None:
        return dict(ready=True,already_prepared=True,optimizer_bytes_uploaded=0,
                    optimizer_steps=descriptor['optimizer_steps'],local_bytes=total)
    if (root/'pending.json').exists():raise ValueError('finish original optimizer candidate before volume preparation')
    for path in (workspace/'runner-status').glob('*.json'):
        from subnet.trainer_cache_lifecycle import live_original
        if live_original(json.loads(path.read_bytes())):raise ValueError('idle trainer required for volume preparation')
    # At most one snapshot resides in shared memory. Current and candidate
    # alternate between disk and memory, leaving room for live CPU Adam.
    needed=2*total+40*1024**3
    if memory_available()<needed or shutil.disk_usage('/dev/shm').free<total+40*1024**3:
        raise ValueError('container memory capacity for one snapshot and CPU Adam')
    if check:return dict(ready=True,checked=True,preparation_required=True,local_bytes=total)
    lock=os.open(root/'lease',os.O_RDWR|os.O_NOFOLLOW)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        authenticated_current(workspace,root,authority)
        volume.mkdir(mode=0o700,exist_ok=True);ordinary(volume,directory=True)
        marker=root/'local-volume.json'
        payload=dict(version=VOLUME_VERSION,workspace=str(workspace),memory_root=str(volume))
        fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb')as stream:stream.write(canonical(payload));stream.flush();os.fsync(stream.fileno())
        if volume_policy(workspace,root) is None:raise ValueError('explicit original trainer volume policy')
        return dict(ready=True,already_prepared=False,optimizer_bytes_uploaded=0,
                    optimizer_steps=descriptor['optimizer_steps'],local_bytes=total)
    finally:os.close(lock)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',required=True)
    parser.add_argument('--authority',required=True);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    print(json.dumps(prepare(args.workspace,args.authority,check=args.check)))


if __name__=='__main__':main()
