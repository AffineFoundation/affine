"""Independent authenticated queue API, unaffected by learner startup/recovery.

Opt-in only. Uses the existing SQLite queue and protocol, never remote model
initialization. The operator must activate matching worker/lease configuration
in both services; this does not admit scientific source contracts.
"""
import argparse
import base64
import json
from pathlib import Path
from .distributed_roles import Coordinator, CoordinatorServer
from .storage import Identity, canonical


def queue_for_config(config, authority):
    remote=config['remote']; q=remote['verifier_queue']
    if q.get('external_api')is not True:
        raise ValueError('standalone queue requires explicit external_api=true')
    identities=[row['worker_identity']for row in remote['roles']['verify']]
    if not identities or len(set(identities))!=len(identities):
        raise ValueError('distinct verifier identities required')
    historical=set(config.get('historical_trusted_worker_identities',[]))
    inactive=set(config.get('inactive_claim_worker_identities',[]))
    if not inactive<=historical:
        raise ValueError('inactive verifier must retain historical signer trust')
    workers={identity:['verify']for identity in identities}
    for identity in historical:workers.setdefault(identity,[])
    for identity in inactive:workers[identity]=[]
    state=Path(config['state'])/'roles'
    state.mkdir(exist_ok=True)
    return Coordinator(state/'verifier-queue.sqlite3',authority,workers,
        lease_seconds=q.get('lease_seconds',300),max_attempts=q.get('max_attempts',3))


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--authority-seed',required=True)
    parser.add_argument('--expected-authority',required=True)
    args=parser.parse_args(argv)
    path=Path(args.authority_seed)
    if path.is_symlink()or not path.is_file()or path.stat().st_mode&0o077:
        raise ValueError('private regular authority seed required')
    authority=Identity(bytes.fromhex(path.read_text().strip()))
    if authority.id!=args.expected_authority:
        raise ValueError('expected queue authority mismatch')
    config=json.loads(Path(args.config).read_text())
    queue=queue_for_config(config,authority.id)
    q=config['remote']['verifier_queue']
    def sign(payload):
        return dict(payload=payload,signer=authority.id,
            signature=base64.b64encode(authority.key.sign(canonical(payload)).signature).decode())
    server=CoordinatorServer((q.get('host','127.0.0.1'),q['port']),queue,sign)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
