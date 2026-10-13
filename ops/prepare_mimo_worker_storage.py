#!/usr/bin/env python3
"""Prepare bounded fast local worker storage; never move live state.

Provider /root may be encrypted FUSE storage. Optimizer staging belongs on a
qualified local volume. Private role credentials remain in their separate store.
"""
import argparse
import json
import os
from pathlib import Path
import shutil


def filesystem(path, mounts='/proc/mounts'):
    path=Path(path).absolute();matches=[]
    for line in Path(mounts).read_text().splitlines():
        fields=line.split()
        if len(fields)<3:continue
        mount=Path(fields[1].replace('\\040',' ').replace('\\134','\\'))
        if path==mount or mount in path.parents:matches.append((len(mount.parts),fields[2],str(mount)))
    if not matches:raise ValueError('local storage filesystem cannot be identified')
    _,kind,mount=max(matches)
    return kind,mount


def prepare(root, *, generation_bytes=126_300_000_000, cache_bytes=19_000_000_000,
            reserve_bytes=8*1024**3, mounts='/proc/mounts', disk_usage=shutil.disk_usage):
    """Fresh 9B trainer admission includes two generations and one read cache.

    Numbers are admission floors from the qualified 9B inventory, not dynamic
    garbage-collection permission. Runtime checks actual parameter bytes again
    before every update. Larger checkpoints require larger explicit bounds.
    """
    path=Path(root).absolute()
    if path.resolve()!=path:raise ValueError('ordinary local storage ancestry')
    if any(type(v)is not int or v<0 for v in (generation_bytes,cache_bytes,reserve_bytes)) or generation_bytes==0:
        raise ValueError('explicit positive generation and nonnegative resource bounds')
    parent=path
    while not parent.exists():parent=parent.parent
    kind,mount=filesystem(parent,mounts)
    if kind.startswith('fuse') or kind in ('nfs','nfs4','cifs','9p','tmpfs'):
        raise ValueError('trainer requires measured fast disk, not FUSE/network/tmpfs storage: '+kind)
    required=2*generation_bytes+cache_bytes+reserve_bytes;free=disk_usage(parent).free
    if free<required:raise ValueError(f'two-generation local storage admission requires {required} bytes; free={free}')
    path.mkdir(parents=True,exist_ok=True,mode=0o700)
    if path.stat().st_uid!=os.getuid() or path.stat().st_mode&0o077:
        raise ValueError('owner-only worker storage root')
    result=dict(version='mimo-fast-local-storage-v1',root=str(path),filesystem=kind,mount=mount,
        required_free_bytes=required,observed_free_bytes=free,
        retained_committed_generations=1,max_local_optimizer_generations=2,
        generation_bytes=generation_bytes,cache_bytes=cache_bytes,reserve_bytes=reserve_bytes)
    for name in ('workspace','trainer-state','checkpoint-cache'):
        target=path/name
        if target.is_symlink():raise ValueError('ordinary worker storage directory')
        target.mkdir(exist_ok=True,mode=0o700)
        if target.stat().st_uid!=os.getuid() or target.stat().st_mode&0o077:raise ValueError('owner-only worker directories')
        result[name.replace('-','_')]=str(target)
    result['live_state_moved']=False
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='/workspace/affine-mimo-trainer')
    parser.add_argument('--generation-bytes',type=int,default=126_300_000_000)
    parser.add_argument('--cache-bytes',type=int,default=19_000_000_000)
    parser.add_argument('--reserve-bytes',type=int,default=8*1024**3)
    args=parser.parse_args()
    print(json.dumps(prepare(args.root,generation_bytes=args.generation_bytes,
        cache_bytes=args.cache_bytes,reserve_bytes=args.reserve_bytes),sort_keys=True,indent=2))
