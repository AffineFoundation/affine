"""Read-only source-bound training input projection for independent ledger readers.

No archive code is executed, no object is fetched, and no training verification
is repeated. Caller authenticates the approved archive inventory separately and
provides ORIGINAL authoritative COMPLETE rows and their registered worker roster.
Historical v1 never imports the prospective compact module.
"""
import copy
import json

from subnet import training_receipts as v1

COMPACT_POLICY='authenticated-verifier-compact-inputs-v2'


def read_admissions(signed_job, report, authority, *, source_files, completed_rows, workers,
                    report_attestation=None):
    """Authenticate original lineage and distinguish transport from frozen ZIP.

    ``source_files`` is the runtime inventory independently authenticated from the
    job's authority-approved source archive. ``completed_rows`` maps original
    verifier job IDs to trusted coordinator COMPLETE rows (not worker input).
    The current or archived operator-authorized verifier roster is explicit.
    Caller must retain its existing archive/worker admission and ledger economic
    checks; this projection does not substitute for those independent contracts.
    ``report`` must be an authority-signed report envelope, or raw bytes/content
    committed by ``report_attestation`` with exact original job/report hashes.
    A raw remote report claiming a matching job hash is always refused alone.
    """
    job=v1.authenticate(signed_job,authority)
    # Matching a raw report's claimed job hash is consistency, not authenticity.
    # Remote SSH ownership is not cryptographic evidence available to this reader.
    if isinstance(report,dict) and set(report)=={'payload','signer','signature'}:
        if report_attestation is not None:raise ValueError('one original report authentication path required')
        report=v1.authenticate(report,authority)
    else:
        if report_attestation is None:raise ValueError('raw training report requires authority attestation')
        attestation=v1.authenticate(report_attestation,authority)
        if (not isinstance(attestation,dict) or set(attestation) != {'version','training_job_sha256','training_report_sha256'}
                or attestation['version'] != 'training-report-audit-attestation-v1'
                or attestation['training_job_sha256'] != v1.sha(job)
                or attestation['training_report_sha256'] != v1.sha(report)):
            raise ValueError('authority attestation exact original job/report hashes')
    manifest=v1.authenticate(job['manifest'],authority)
    if job.get('source_files') != source_files:
        raise ValueError('independently admitted original source inventory required')
    if (report.get('job_id') != job.get('job_id') or report.get('job_sha256') != v1.sha(job)
            or report.get('operator') != authority or report.get('success') is not True
            or report.get('role') != 'train' or report.get('epoch') != manifest['epoch']
            or report.get('checkpoint') != manifest['checkpoint']['id']
            or report.get('source_files') != job['source_files']
            or report.get('runtime_versions') != job['runtime_versions']
            or report.get('chain_transactions') is not False):
        raise ValueError('original training report/job/source binding')
    policy=job.get('training_input_policy')
    if policy==COMPACT_POLICY:
        if not {'subnet/compact_training_inputs.py','subnet/training_receipts.py'} <= set(source_files):
            raise ValueError('approved compact audit source pins required')
        from subnet.compact_training_inputs import validate_report,validate_receipt
    elif policy==v1.VERSION:
        validate_report=v1.validate_report;validate_receipt=v1.validate_receipt
    else:raise ValueError('explicit admitted training input policy required')
    validate_report(report,job,manifest,authority)
    result=[]
    for obj in job['submissions']:
        value=validate_receipt(obj['verifier_receipt'],obj,manifest,authority)
        if policy==COMPACT_POLICY:
            wrapper,original=value
            receipt=wrapper['original_verifier_receipt']
        else:
            original=value;receipt=obj['verifier_receipt']
        identifier=original['original_verify_job_id'];row=completed_rows.get(identifier)
        if row is None or row['status']!='complete' or row['role']!='verify' or row['id']!=identifier:
            raise ValueError('original authoritative COMPLETE verifier row required')
        verify_job=json.loads(row['envelope']);request=json.loads(row['report_request']);audit_report=json.loads(row['report'])
        if (v1.sha(verify_job['payload']) != row['digest'] or v1.sha(audit_report) != row['report_digest']
                or request['signer'] != row['worker'] or request['payload'].get('report') != audit_report):
            raise ValueError('exact original COMPLETE verifier row bytes')
        matches=[audit for audit in audit_report.get('audits',[])if audit.get('submission_sha256')==original['submission_sha256']]
        if len(matches)!=1:raise ValueError('original report frozen population')
        frozen=manifest['audit_frozen_receipts'][original['miner_identity']]
        reconstructed=v1.receipt_payload(verify_job,request,authority,workers,manifest,
            original['miner_identity'],frozen,matches[0])
        if reconstructed != original:raise ValueError('nested admission differs from original completed verifier evidence')
        if policy==COMPACT_POLICY:
            original_manifest=v1.authenticate(v1.authenticate(verify_job,authority)['manifest'],authority)
            if original_manifest.get('training_input_policy')!=COMPACT_POLICY:
                raise ValueError('historical original verifier manifest cannot authorize compact training')
        result.append(dict(miner_identity=original['miner_identity'],submission_sha256=original['submission_sha256'],
            submission_size=original['submission_size'],frozen_key=original['frozen_key'],
            original_verifier_receipt_sha256=v1.sha(receipt),training_transport_sha256=obj['sha256'],
            training_transport_size=obj['size'],training_input_policy=policy,
            original_verify_job_id=identifier,original_signed_job_sha256=original['original_signed_job_sha256'],
            original_signed_manifest_sha256=original['original_signed_manifest_sha256'],
            original_report_sha256=original['original_report_sha256'],
            original_worker_report_request_sha256=original['original_worker_report_request_sha256'],
            accepted_batch_sha256=copy.deepcopy(obj['accepted_batch_sha256'])))
    return result
