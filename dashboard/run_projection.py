"""Select a new learning run without deleting any historical database rows."""
import base64
import json
import re
from nacl.signing import VerifyKey

AUTHORITY = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
VERSION = 'dashboard-training-run-boundary-v1'
RETIREMENT_VERSION = 'operator-protocol-regression-retirement-v1'


def authenticated(path):
    document = json.loads(path.read_bytes())
    if set(document) != {'payload', 'signer', 'signature'} or document['signer'] != AUTHORITY:
        raise ValueError('dashboard run authority')
    body = document['payload']
    raw = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    VerifyKey(bytes.fromhex(AUTHORITY)).verify(raw, base64.b64decode(document['signature'], validate=True))
    return body


def read_boundary(path):
    if not path.exists():
        return None
    body = authenticated(path)
    required = {'version', 'run_id', 'started_at', 'first_round', 'base_checkpoint', 'model', 'source_sha256'}
    if (set(body) != required or body['version'] != VERSION or type(body['first_round']) is not int
            or body['first_round'] < 0 or type(body['started_at']) not in (int, float)
            or not body['run_id'] or not body['model']):
        raise ValueError('dashboard run boundary')
    for key in ('base_checkpoint', 'source_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', body[key]):
            raise ValueError('dashboard run checkpoint/source binding')
    return body


def read_retirements(directory, boundary):
    """Only explicit ROOT retirement evidence removes a historical epoch."""
    if boundary is None:
        return set()
    retired = set()
    for path in sorted(directory.glob('*.ROOT-SIGNED.json')):
        body = authenticated(path)
        if (body.get('version') != RETIREMENT_VERSION or body.get('artifacts_preserved') is not True
                or body.get('miner_fault') is not False or body.get('penalty_evidence_eligible') is not False
                or body.get('training_completed') is not False or not isinstance(body.get('epochs'), list)):
            raise ValueError('explicit nonpenalizing protocol retirement')
        if body.get('source_sha256') != boundary['source_sha256']:
            continue
        for row in body['epochs']:
            match = re.fullmatch(r'nonpayable-live-reward-math-v1--[0-9]+-([0-9]+)', row.get('epoch', ''))
            if (set(row) != {'epoch', 'round', 'manifest_sha256'} or not match
                    or type(row['round']) is not int or row['round'] != int(match[1])
                    or not re.fullmatch('[0-9a-f]{64}', row.get('manifest_sha256', ''))):
                raise ValueError('exact retired epoch identity')
            retired.add(row['epoch'])
    return retired


def project(epochs, evaluations, boundary, retired=()):
    if boundary is None:
        return epochs, evaluations
    selected = []
    for row in epochs:
        match = re.fullmatch(r'nonpayable-live-reward-math-v1--[0-9]+-([0-9]+)', row['id'])
        if (row['id'] not in retired and row.get('source') == 'live-reward-math' and match
                and int(match[1]) >= boundary['first_round'] and row['start'] >= boundary['started_at']):
            selected.append(dict(row, display_epoch=int(match[1]) - boundary['first_round'] + 1,
                                 training_run_id=boundary['run_id']))
    identities = {row['id'] for row in selected}
    if retired:
        # Retired controller attempts are not learning epochs. Genuine failed
        # or pending epochs remain in the timeline, in their original state.
        for index, row in enumerate(sorted(selected, key=lambda r:(r['start'],r['id'])), 1):
            row['display_epoch'] = index
    completed = {}
    for row in sorted(selected, key=lambda r:(r['start'],r['id'])):
        training = row.get('training') or {}
        if row.get('phase') == 'trained' and training.get('weights_changed') is True and training.get('checkpoint'):
            completed.setdefault(training['checkpoint'], row['id'])
    results = []
    for row in evaluations:
        result = dict(row, training_run_id=boundary['run_id'])
        # Cached diagnostics have already authenticated the original report.
        # Associate its checkpoint with the training epoch that produced it,
        # not a later (possibly retired) opening that reused those weights.
        if (row.get('original_epoch_id') and row.get('original_report_sha256')
                and row.get('status') == 'complete' and row.get('checkpoint') in completed):
            result['epoch_id'] = completed[row['checkpoint']]
            result['display_epoch_association'] = 'completed-training-checkpoint'
        if result.get('epoch_id') in identities and result['timestamp'] >= boundary['started_at']:
            results.append(result)
    return selected, results
