"""CPU dispatch facade; signing remains exclusively on the coordinator.

The training/retirement installer is a function, deliberately not a subprocess
command: it must execute in the actual worker or ACK interpreter.
"""
import argparse
import hashlib
import json
from pathlib import Path

HELPER_SHA256 = 'fb22acb5eebd04ca07b5a5a36528d448d83c068b8779c6b0fb80525673844abb'


def helper():
    import trainer_reset_lifecycle as module
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=HELPER_SHA256:
        raise ValueError('pinned CPU reset helper changed')
    return module


def install_in_child(envelope_path, authority, workspace, *, role, recovery_path=None):
    """Call after the fixed source path is installed, before backend execution."""
    module=helper();envelope=module._read(Path(envelope_path))
    # This separately pinned recovery entry also serves the existing ACK API.
    # Only its exact sibling ROOT grant is discoverable; never search paths or
    # infer a new job from mutable state. Train dispatch supplies it explicitly.
    if role=='retirement' and recovery_path is None:
        sibling=Path(envelope_path).with_name('retry.ROOT-SIGNED.private.json')
        if sibling.exists():recovery_path=sibling
    recovery=None if recovery_path is None else module._read(Path(recovery_path))
    if role=='train':return module.install_for_train(envelope,authority,workspace,recovery_envelope=recovery)
    if role=='retirement':return module.install_for_retirement(envelope,authority,workspace,recovery_envelope=recovery)
    raise ValueError('only train child or retirement ACK process')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--operation',choices=('plan','reset','check'),required=True)
    parser.add_argument('--authority',required=True)
    parser.add_argument('--workspace',required=True)
    parser.add_argument('--job')
    parser.add_argument('--receipt')
    parser.add_argument('--study-lease')
    parser.add_argument('--envelope')
    args=parser.parse_args();module=helper()
    if args.operation=='plan':
        if not args.job or not args.receipt or not args.study_lease:
            parser.error('plan needs exact signed job, durable receipt and study lease')
        result=module.plan_reset(module._read(Path(args.job)),args.authority,args.workspace,
            module._read(Path(args.receipt)),extra_lease_paths=[args.study_lease])
    else:
        if not args.envelope:parser.error('reset/check needs signed envelope')
        envelope=module._read(Path(args.envelope))
        if args.operation=='reset':result=module.execute_reset(envelope,args.authority,args.workspace)
        else:
            scope,_=module._ready(envelope,args.authority,args.workspace)
            result=dict(ready=True,first_job_id=scope['first_job_id'],genesis_sha256=scope['to_genesis_sha256'])
    print(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False))


if __name__=='__main__':main()
