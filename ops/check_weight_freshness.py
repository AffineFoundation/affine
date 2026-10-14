"""Read-only assessment/receipt freshness check; never invokes a chain writer."""
import argparse
import base64
import json
import os
from pathlib import Path
import time

from nacl.signing import VerifyKey


def inspect_freshness(document, receipt, authority, *, now, grace_seconds=1200):
    if set(document) != {'payload', 'signer', 'signature'} or document['signer'] != authority:
        raise ValueError('explicit assessment authority required')
    payload = document['payload']
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    VerifyKey(bytes.fromhex(authority)).verify(raw, base64.b64decode(document['signature'], validate=True))
    cutoff = payload.get('evidence_cutoff')
    if (type(cutoff) is not int or cutoff % 3600 or cutoff > now or
            payload.get('assessment_stale') is not False or payload.get('cutoff') != cutoff):
        raise ValueError('authenticated fresh assessment required')
    if not 0 <= grace_seconds < 3600:
        raise ValueError('bounded hourly grace')
    expected = int(now - grace_seconds) // 3600 * 3600
    latest = receipt.get('latest', {})
    submitted = latest.get('window_end')
    submitted_ok = (type(submitted) is int and submitted % 3600 == 0 and
                    expected <= submitted <= now and latest.get('status') == 'submitted')
    reasons = []
    if cutoff < expected:
        reasons.append('assessment-evidence-stale')
    if not submitted_ok:
        reasons.append('local-submission-receipt-late-or-invalid')
    return dict(status='healthy' if not reasons else 'attention', reasons=reasons,
                checked_at=now, expected_hour=expected, evidence_cutoff=cutoff,
                local_submission_hour=submitted, assessment_age_seconds=now-cutoff,
                onchain_confirmation_performed=False, chain_calls=0, writer_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        assessment = json.loads((args.state/'current-assessment-never-burn-v1/last-valid-assessment.json').read_bytes())
        receipt = json.loads((args.state/'weights.json').read_bytes())
        report = inspect_freshness(assessment, receipt, args.authority, now=time.time())
    except Exception as error:
        report = dict(status='attention', reasons=['unreadable-or-invalid-freshness-evidence'],
                      error_type=type(error).__name__, checked_at=time.time(),
                      onchain_confirmation_performed=False, chain_calls=0, writer_calls=0)
    if args.output:
        temporary = args.output.with_name(args.output.name+'.'+str(os.getpid())+'.tmp')
        temporary.write_text(json.dumps(report, sort_keys=True)+'\n')
        temporary.replace(args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report['status'] == 'healthy' else 2


if __name__ == '__main__':
    raise SystemExit(main())
