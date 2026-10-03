"""Recompute proposed weights from authenticated frozen audit reports, never chain writes."""
import argparse
import hashlib
import json
from pathlib import Path
from subnet.audit_policy import penalties
from subnet.distributed_roles import authenticate
from subnet.scoring import score
from subnet.storage import canonical


def proposed(ledger, authority, policy):
    # The ledger is the validator-signed frozen receipt/epoch registry. Individual
    # reports must be validator-authenticated exports of verified worker reports.
    body=authenticate(ledger,authority)
    if body.get('payable') is not False:raise ValueError('nonpayable ledger required')
    receipts=body['receipts'];reports={}
    if set(body['reports'])!=set(receipts):raise ValueError('complete frozen report registry')
    for miner,envelope in body['reports'].items():
        report=authenticate(envelope,authority)
        if report.get('epoch')!=body['epoch_id'] or report.get('submission_sha256')!=receipts[miner]['sha256']:
            raise ValueError('audit epoch/frozen submission binding')
        reports[miner]=report
    result=score(reports,penalties(policy))
    return dict(result,epoch_id=body['epoch_id'],payable=False,chain_transactions=False,
                authenticated_ledger_sha256=hashlib.sha256(canonical(ledger)).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger',required=True);p.add_argument('--authority',required=True)
    p.add_argument('--policy',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    policy=json.loads(Path(a.policy).read_text())
    result=proposed(json.loads(Path(a.ledger).read_text()),a.authority,policy.get('penalties',policy))
    path=Path(a.output)
    # Refuse replacement of historical output; use a new versioned destination.
    with path.open('xb') as stream:stream.write(canonical(result))
    path.chmod(0o600)
    print(json.dumps(dict(epoch_id=result['epoch_id'],miners=len(result['weights']),payable=False,chain_transactions=False)))

if __name__=='__main__':main()
