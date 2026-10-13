#!/usr/bin/env python3
"""Explicit owner-private boundary command; no implicit live paths or defaults.

ROOT prepares signed registry/transition/plans from authenticated originals.
Trainer receives those documents plus its own key, never the ROOT signing key.
This command neither rents machines nor edits services, workers, or queue jobs.
"""
import argparse,json
from pathlib import Path
from ops.mimo_lifecycle import private_json
from ops.mimo_training_rehearsal import identity
from subnet.mimo_release_policy import ReleasePolicy
from subnet.mimo_source_migration import (SourceReleaseRegistry,arm_boundary_hold,
    migrate_trainer_current,migrate_host_current,complete_boundary_hold)
from subnet.mimo_grpo.transaction import _write


def run(config, action, *, now):
    actor=identity(config['identity_seed_file'])
    registry=SourceReleaseRegistry(config['registry'],config['authority'],allow_public=config.get('allow_public',False))
    root=Path(config['state_root']).absolute()
    if root.resolve()!=root:raise ValueError('ordinary original state root')
    if action=='arm':
        policy=ReleasePolicy(config['from_policy'],config['authority'],allow_public=config.get('allow_public',False))
        result=arm_boundary_hold(root,config['hold'],policy,actor,now=now)
    elif action=='trainer':
        result=migrate_trainer_current(root,registry,config['transition'],config['planned'],actor)
    elif action=='host':
        result=migrate_host_current(root,registry,config['transition'],config['planned'],config['trainer_ack'],actor,now=now)
    elif action=='complete':
        result=complete_boundary_hold(root,registry,config['transition'],config['trainer_ack'],config['host_ack'],actor,now=now)
    else:raise ValueError('explicit migration action')
    if 'output' in config:
        path=Path(config['output']).absolute()
        if path.resolve()!=path:raise ValueError('ordinary private acknowledgement path')
        if path.exists():
            if path.is_symlink() or json.loads(path.read_bytes())!=result:raise ValueError('original acknowledgement output')
        else:_write(path,result)
    return {'action':action,'status':'held' if action=='arm' else 'acknowledged'}


def main():
    import time
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--action',required=True,choices=('arm','trainer','host','complete'))
    args=p.parse_args();print(json.dumps(run(private_json(args.config),args.action,now=int(time.time()))))


if __name__=='__main__':main()
