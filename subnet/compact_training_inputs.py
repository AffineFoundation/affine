"""Prospective compact transport; not selected by any existing v1 job.

Original operator admission remains nested and authenticated. Only the original
COMPLETE queue row's signed report supplies documents; no local ZIP is decoded.
Preparation can use the coordinator's existing signing/publication APIs only for
an explicitly selected future-v2 epoch. Importing it performs no operations.
"""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3

from .storage import canonical
from . import training_receipts as v1

VERSION = 'authenticated-verifier-compact-inputs-v2'
ARTIFACT_VERSION = 'canonical-accepted-batch-documents-v2'
# Original ZIP manifest decoder limits documents to 2 MB. This separate,
# prospective transport has the same conservative ceiling (including framing).
MAX_BYTES = 2_000_000
# Canonical bytes expand into Python containers/integers during admission.
DECODE_WORKING_BYTES = 64 * MAX_BYTES


def _original_object(receipt):
    if (not isinstance(receipt,dict) or not isinstance(receipt.get('payload'),dict)
            or not {'submission_sha256','submission_size','fully_audited_batches'} <= set(receipt['payload'])):
        raise ValueError('original signed v1 admission structure')
    value = receipt['payload']
    return dict(sha256=value['submission_sha256'], size=value['submission_size'],
        accepted_batch_sha256=sorted(t['batch_sha256'] for t in value['fully_audited_batches']),
        verifier_receipt=receipt)


def _row_evidence(row):
    if row is None or row['status'] != 'complete' or row['role'] != 'verify':
        raise ValueError('original verifier job not COMPLETE')
    job = json.loads(row['envelope']); report = json.loads(row['report'])
    request = json.loads(row['report_request'])
    if (job['payload'].get('job_id') != row['id']
            or v1.sha(job['payload']) != row['digest'] or v1.sha(report) != row['report_digest']
            or request['signer'] != row['worker'] or request['payload'].get('report') != report):
        raise ValueError('authoritative COMPLETE queue bytes')
    return job, report, request


def prepare_from_completed_row(row, authority, workers, manifest, miner, frozen,
                               local_audit, original_receipt):
    """Return canonical bytes and UNSIGNED prospective receipt payload.

    The caller must read ``row`` from its own authoritative queue, never a miner
    or worker supplied row. Signing/publication is a separate coordinator action.
    ``original_receipt`` is the unchanged signed v1 admission for these originals.
    """
    job, report, request = _row_evidence(row)
    original_manifest=v1.authenticate(v1.authenticate(job,authority)['manifest'],authority)
    if original_manifest.get('training_input_policy') != VERSION:
        raise ValueError('compact requires prospective original signed verifier manifest')
    expected = v1.receipt_payload(job, request, authority, workers, manifest,
                                  miner, frozen, local_audit)
    original = v1.validate_receipt(original_receipt, _original_object(original_receipt), manifest, authority)
    if original != expected:
        raise ValueError('original receipt differs from authoritative COMPLETE evidence')
    audits = [a for a in report['audits'] if a.get('submission_sha256') == frozen['sha256']]
    batches = {v1.sha(b): b for b in audits[0]['accepted']}
    documents = [dict(batch_number=t['batch_number'], batch=copy.deepcopy(batches[t['batch_sha256']]))
                 for t in original['fully_audited_batches']]
    artifact = dict(version=ARTIFACT_VERSION, epoch=original['epoch'],
        checkpoint=original['checkpoint'], original_verifier_receipt_sha256=v1.sha(original_receipt),
        submission_sha256=original['submission_sha256'], documents=documents)
    data = canonical(artifact)
    if not 0 < len(data) <= MAX_BYTES:
        raise ValueError('compact canonical document byte budget')
    payload = dict(version=VERSION, original_verifier_receipt=copy.deepcopy(original_receipt),
        compact_sha256=hashlib.sha256(data).hexdigest(), compact_size=len(data),
        artifact_version=ARTIFACT_VERSION)
    return data, payload


def prepare_from_queue(queue_path, authority, workers, manifest, miner, frozen,
                       local_audit, original_receipt):
    """Read-only original coordinator queue lookup; never accept an external row."""
    identifier = local_audit.get('remote_job_id')
    if identifier != original_receipt.get('payload', {}).get('original_verify_job_id'):
        raise ValueError('original queue identity binding')
    with sqlite3.connect('file:' + str(Path(queue_path).resolve()) + '?mode=ro', uri=True) as database:
        database.row_factory = sqlite3.Row
        row = database.execute('SELECT * FROM jobs WHERE id=?', (identifier,)).fetchone()
    return prepare_from_completed_row(row, authority, workers, manifest, miner,
                                      frozen, local_audit, original_receipt)


def validate_receipt(envelope, obj, manifest, authority):
    if manifest.get('training_input_policy') != VERSION:
        raise ValueError('prospective compact input policy required')
    value = v1.authenticate(envelope, authority)
    if not isinstance(value,dict) or set(value) != {'version','original_verifier_receipt','compact_sha256','compact_size','artifact_version'}:
        raise ValueError('exact compact receipt fields')
    if (value['version'] != VERSION or value['artifact_version'] != ARTIFACT_VERSION
            or type(value['compact_size']) is not int or not 0 < value['compact_size'] <= MAX_BYTES
            or obj.get('sha256') != value['compact_sha256'] or type(obj.get('size')) is not int
            or obj['size'] != value['compact_size']):
        raise ValueError('compact artifact signed digest/byte budget')
    v1.digest(value['compact_sha256'])
    receipt = value['original_verifier_receipt']
    original_obj = _original_object(receipt)
    original = v1.validate_receipt(receipt, original_obj, manifest, authority)
    if obj.get('accepted_batch_sha256') != original_obj['accepted_batch_sha256']:
        raise ValueError('compact original accepted inventory')
    return value, original


def _decode(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:raise ValueError('duplicate compact JSON key')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError('nonfinite compact JSON number')
    try:
        result = json.loads(data, object_pairs_hook=unique, parse_constant=invalid_constant)
        if canonical(result) != data:raise ValueError('exact canonical compact JSON required')
        return result
    except (UnicodeError, RecursionError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('compact JSON framing') from error


def admitted_submission(path, obj, manifest, authority, *, retire=False):
    """Authenticate compact bytes and exact documents; return unchanged pair data.

    No original ZIP/arrays/proofs are opened and no verifier computation is run.
    Reference log probabilities are still computed by the training objective.
    """
    value, original = validate_receipt(obj.get('verifier_receipt'), obj, manifest, authority)
    path = Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('compact artifact regular file required; symlink refused')
    with path.open('rb') as stream:
        # Bound the read even if a file changes after a caller's capacity check.
        data = stream.read(value['compact_size'] + 1)
    if len(data) != value['compact_size'] or hashlib.sha256(data).hexdigest() != value['compact_sha256']:
        raise ValueError('exact compact artifact bytes/size')
    artifact = _decode(data)
    if (not isinstance(artifact, dict) or set(artifact) != {'version','epoch','checkpoint',
            'original_verifier_receipt_sha256','submission_sha256','documents'}
            or artifact['version'] != ARTIFACT_VERSION or artifact['epoch'] != original['epoch']
            or artifact['checkpoint'] != original['checkpoint']
            or artifact['submission_sha256'] != original['submission_sha256']
            or artifact['original_verifier_receipt_sha256'] != v1.sha(value['original_verifier_receipt'])
            or not isinstance(artifact['documents'], list)
            or len(artifact['documents']) != len(original['fully_audited_batches'])):
        raise ValueError('compact artifact exact original context/population')
    from .protocol import entry, classification, sample_key
    pairs = []; accepted = []; seen = set()
    for document, target in zip(artifact['documents'], original['fully_audited_batches']):
        if (not isinstance(document,dict) or set(document) != {'batch_number','batch'}
                or type(document['batch_number']) is not int or document['batch_number'] != target['batch_number']):
            raise ValueError('compact original batch slot/order')
        batch = document['batch']
        if not isinstance(batch,dict) or v1.sha(batch) != target['batch_sha256']:
            raise ValueError('compact exact original accepted batch document')
        definition = entry(manifest,batch.get('env_id')); index = batch.get('index'); key = sample_key(batch)
        if (batch.get('schema') != 2 or batch.get('epoch') != original['epoch']
                or batch.get('checkpoint') != original['checkpoint'] or batch.get('env_id') != target['env_id']
                or type(index) is not int or index != target['index'] or index not in definition['indices']
                or key in seen or batch.get('sample_index') != index
                or batch.get('environment_version') != definition['spec']['version']
                or len(batch.get('rollouts',[])) != manifest['K'] + manifest['L']):
            raise ValueError('compact attested batch structure/task binding')
        seen.add(key)
        positives = [r for r in batch['rollouts'] if classification(r)=='positive']
        negatives = [r for r in batch['rollouts'] if classification(r)=='negative']
        if ([v1.sha(r) for r in positives] != target['positive_rollout_sha256']
                or [v1.sha(r) for r in negatives] != target['negative_rollout_sha256']):
            raise ValueError('compact exact authenticated rollout hashes')
        accepted.append(batch); pairs.extend((definition,p,n) for p,n in zip(positives,negatives))
    # Preserve original ZIP and verifier lineage fields used by source-bound audits.
    summary = dict(version=VERSION,epoch=manifest['epoch'],submission_sha256=original['submission_sha256'],
        submission_size=original['submission_size'],compact_sha256=obj['sha256'],compact_size=obj['size'],
        verifier_receipt_sha256=v1.sha(obj['verifier_receipt']),
        original_verifier_receipt_sha256=v1.sha(value['original_verifier_receipt']),
        accepted_batch_sha256=obj['accepted_batch_sha256'],accepted=accepted,
        original_verify_job_id=original['original_verify_job_id'],original_report_sha256=original['original_report_sha256'],
        trainer_verification_performed=False,verification_performed_by='registered-verifier')
    if retire:path.unlink()
    return summary,pairs


def validate_job(job, manifest, authority):
    """Prospective source admission; existing v1 execution amendments cannot opt in."""
    if (job.get('role') != 'train' or job.get('training_policy') not in v1.POLICIES
            or job.get('training_policy') != manifest.get('training_policy')
            or job.get('training_input_policy') != VERSION or manifest.get('training_input_policy') != VERSION
            or not {'subnet/compact_training_inputs.py','subnet/training_receipts.py'} <= set(job.get('source_files',{}))
            or manifest.get('training_execution_amendment') is not None):
        raise ValueError('prospective compact job/source/policy required; v1 amendments cannot opt in')
    submissions = job.get('submissions')
    if not isinstance(submissions,list) or not 1 <= len(submissions) <= 256:
        raise ValueError('bounded compact submission inventory')
    identities = set(); artifacts = set()
    for obj in submissions:
        _, original = validate_receipt(obj.get('verifier_receipt'),obj,manifest,authority)
        identity=(original['miner_identity'],original['submission_sha256'])if manifest.get('submission_transport_policy')else original['miner_identity']
        if identity in identities or obj['sha256'] in artifacts:
            raise ValueError('one compact admission per original miner/artifact')
        identities.add(identity); artifacts.add(obj['sha256'])


def validate_report(report, job, manifest, authority):
    """Preserve truthful admissions and ORIGINAL source-bound ZIP audit lineage."""
    validate_job(job, manifest, authority)
    rows = report.get('training_admissions')
    training = report.get('training',{})
    if (not isinstance(rows,list) or len(rows) != len(job['submissions']) or report.get('audits')
            or training.get('training_input_policy') != VERSION
            or training.get('trainer_verification_performed') is not False
            or training.get('all_pairs_authenticated_verifier_receipts') is not True
            or 'all_pairs_independently_reaudited' in training):
        raise ValueError('truthful compact training admissions without new audits')
    for row,obj in zip(rows,job['submissions']):
        value,original = validate_receipt(obj['verifier_receipt'],obj,manifest,authority)
        expected = dict(version=VERSION,epoch=manifest['epoch'],submission_sha256=original['submission_sha256'],
            submission_size=original['submission_size'],compact_sha256=obj['sha256'],compact_size=obj['size'],
            verifier_receipt_sha256=v1.sha(obj['verifier_receipt']),
            original_verifier_receipt_sha256=v1.sha(value['original_verifier_receipt']),
            accepted_batch_sha256=obj['accepted_batch_sha256'],original_verify_job_id=original['original_verify_job_id'],
            original_report_sha256=original['original_report_sha256'],trainer_verification_performed=False,
            verification_performed_by='registered-verifier')
        accepted = row.get('accepted') if isinstance(row,dict) else None
        if (not isinstance(row,dict) or set(row) != set(expected) | {'accepted'}
                or row.get('trainer_verification_performed') is not False
                or any(row.get(k) != v for k,v in expected.items()) or not isinstance(accepted,list)
                or sorted(v1.sha(batch) for batch in accepted) != obj['accepted_batch_sha256']):
            raise ValueError('compact admission report exact original lineage/population')


def selected(manifest):
    """Only an explicit signed prospective marker enables compact transport."""
    policy = manifest.get('training_input_policy')
    if policy not in (None,v1.VERSION,VERSION):raise ValueError('unapproved training input policy')
    return policy == VERSION


def original_submissions(submissions, manifest, authority):
    """Authenticate transport before mapping it back to original coverage inputs."""
    result = []
    for obj in submissions:
        value,_ = validate_receipt(obj.get('verifier_receipt'),obj,manifest,authority)
        result.append(_original_object(value['original_verifier_receipt']))
    return result


def receipt_inventory(submissions):
    return sorted([dict(submission_sha256=obj['verifier_receipt']['payload']['original_verifier_receipt']['payload']['submission_sha256'],
        compact_sha256=obj['sha256'],compact_size=obj['size'],
        verifier_receipt_sha256=v1.sha(obj['verifier_receipt']),
        original_verifier_receipt_sha256=v1.sha(obj['verifier_receipt']['payload']['original_verifier_receipt']),
        accepted_batch_sha256=obj['accepted_batch_sha256']) for obj in submissions],
        key=lambda row:(row['submission_sha256'],row['verifier_receipt_sha256']))


def prepare_submissions(controller, manifest, reports, receipts):
    """Future coordinator only: derive, durably read back, then sign v2 admission."""
    if not selected(manifest) or manifest.get('training_execution_amendment') is not None:
        raise ValueError('compact requires original prospective policy; no v1 amendment')
    queue = getattr(controller.jobs,'queue',None)
    if queue is None:raise ValueError('compact requires authoritative verifier queue')
    if not 1 <= sum(bool(audit.get('accepted')) for audit in reports.values()) <= 256:
        raise ValueError('bounded authenticated compact submission population')
    result = []
    rows=[]
    for miner,audit in reports.items():
        if not audit.get('accepted'):continue
        if manifest.get('submission_transport_policy'):
            for child in audit['artifact_audits']:
                if child.get('accepted'):
                    frozen=next(b for b in receipts[miner]['artifacts']if b['sha256']==child['submission_sha256']);rows.append((miner,child,frozen))
        else:rows.append((miner,audit,receipts[miner]))
    for miner,audit,frozen in rows:
        original = v1.issue(controller,manifest,miner,frozen,audit)
        data,payload = prepare_from_queue(queue.path,controller.authority.id,queue.workers,
            manifest,miner,frozen,audit,original)
        key = 'private/compact-training-inputs/' + payload['compact_sha256'] + '.json'
        controller.bucket.put(key,data,'application/json')
        response = controller.bucket.client.get_object(Bucket=controller.bucket.name,Key=key)
        body = response['Body']
        try:
            if response['ContentLength'] != len(data):raise ValueError('compact durable readback size')
            received = body.read(len(data)+1)
            if received != data:raise ValueError('compact durable exact byte readback')
        finally:body.close()
        receipt = controller.signed(payload)
        obj = dict(url=controller.bucket.presign(key),sha256=payload['compact_sha256'],size=len(data),
            accepted_batch_sha256=sorted(t['batch_sha256'] for t in original['payload']['fully_audited_batches']),
            verifier_receipt=receipt)
        validate_receipt(receipt,obj,manifest,controller.authority.id)
        result.append(obj)
    if not 1 <= len(result) <= 256:raise ValueError('bounded authenticated compact submissions')
    return result
