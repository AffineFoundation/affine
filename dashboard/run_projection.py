"""Select a new learning run without deleting any historical database rows."""
import base64
import json
import re
from nacl.signing import VerifyKey

AUTHORITY = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
VERSION = 'dashboard-training-run-boundary-v1'


def read_boundary(path):
    if not path.exists():
        return None
    document = json.loads(path.read_bytes())
    if set(document) != {'payload', 'signer', 'signature'} or document['signer'] != AUTHORITY:
        raise ValueError('dashboard run authority')
    body = document['payload']
    raw = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    VerifyKey(bytes.fromhex(AUTHORITY)).verify(raw, base64.b64decode(document['signature'], validate=True))
    required = {'version', 'run_id', 'started_at', 'first_round', 'base_checkpoint', 'model', 'source_sha256'}
    if (set(body) != required or body['version'] != VERSION or type(body['first_round']) is not int
            or body['first_round'] < 0 or type(body['started_at']) not in (int, float)
            or not body['run_id'] or not body['model']):
        raise ValueError('dashboard run boundary')
    for key in ('base_checkpoint', 'source_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', body[key]):
            raise ValueError('dashboard run checkpoint/source binding')
    return body


def project(epochs, evaluations, boundary):
    if boundary is None:
        return epochs, evaluations
    selected = []
    for row in epochs:
        match = re.fullmatch(r'nonpayable-live-reward-math-v1--[0-9]+-([0-9]+)', row['id'])
        if (row.get('source') == 'live-reward-math' and match
                and int(match[1]) >= boundary['first_round'] and row['start'] >= boundary['started_at']):
            selected.append(dict(row, display_epoch=int(match[1]) - boundary['first_round'] + 1,
                                 training_run_id=boundary['run_id']))
    identities = {row['id'] for row in selected}
    results = [dict(row, training_run_id=boundary['run_id']) for row in evaluations
               if row.get('epoch_id') in identities and row['timestamp'] >= boundary['started_at']]
    return selected, results
