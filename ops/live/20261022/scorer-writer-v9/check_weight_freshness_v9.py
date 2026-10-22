"""Read-only freshness check for the v9-only weight path (adapted from 20261014-community-weights-check-v1, same rules):
the hourly v9 assessment must be authenticated (authority key) and fresh, the local receipt must show the expected hour
submitted, and additionally the assessment must be the v9 rule.  The old audit-evidence assessment is no longer checked
(it is not on the weight path).  Never invokes a chain writer."""
import argparse, base64, json, os, time
from pathlib import Path
from nacl.signing import VerifyKey
V9 = 'sn120-v9-live-weights-v1'

def inspect_freshness(document, receipt, authority, *, now, grace_seconds=1200):
    if set(document) != {'payload', 'signer', 'signature'} or document['signer'] != authority:
        raise ValueError('explicit assessment authority required')
    payload = document['payload']
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    VerifyKey(bytes.fromhex(authority)).verify(raw, base64.b64decode(document['signature'], validate=True))
    cutoff = payload.get('evidence_cutoff')
    if (type(cutoff) is not int or cutoff % 3600 or cutoff > now or
            payload.get('assessment_stale') is not False or payload.get('cutoff') != cutoff or payload.get('version') != V9):
        raise ValueError('authenticated fresh v9 assessment required')
    expected = int(now - grace_seconds) // 3600 * 3600
    latest = receipt.get('latest', {}); submitted = latest.get('window_end')
    submitted_ok = (type(submitted) is int and submitted % 3600 == 0 and expected <= submitted <= now and latest.get('status') == 'submitted')
    reasons = []
    if cutoff < expected: reasons.append('assessment-evidence-stale')
    if not submitted_ok: reasons.append('local-submission-receipt-late-or-invalid')
    v9 = payload.get('v9') or {}
    return dict(status='healthy' if not reasons else 'attention', reasons=reasons, checked_at=now, expected_hour=expected,
                evidence_cutoff=cutoff, local_submission_hour=submitted, assessment_age_seconds=now - cutoff, rule=V9,
                v9_score_epoch=v9.get('score_epoch'), v9_latest_sealed=v9.get('latest_sealed'), v9_gated=v9.get('gated'),
                v9_fallback_reasons=v9.get('reasons'), onchain_confirmation_performed=False, chain_calls=0, writer_calls=0)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state', type=Path, required=True); ap.add_argument('--authority', required=True)
    ap.add_argument('--assessment-dir', default='current-assessment-v9-live-1'); ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    try:
        report = inspect_freshness(json.loads((a.state / a.assessment_dir / 'last-valid-assessment.json').read_bytes()),
                                   json.loads((a.state / 'weights.json').read_bytes()), a.authority, now=time.time())
    except Exception as error:
        report = dict(status='attention', reasons=['unreadable-or-invalid-freshness-evidence'], error_type=type(error).__name__,
                      checked_at=time.time(), onchain_confirmation_performed=False, chain_calls=0, writer_calls=0)
    if a.output:
        t = a.output.with_name(a.output.name + '.' + str(os.getpid()) + '.tmp'); t.write_text(json.dumps(report, sort_keys=True) + '\n'); t.replace(a.output)
    print(json.dumps(report, sort_keys=True)); return 0 if report['status'] == 'healthy' else 2

if __name__ == '__main__': raise SystemExit(main())
