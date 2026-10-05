"""Training admission from authenticated verifier receipts, without re-verification.

The operator authenticates completed verifier jobs and registered workers' signed
report requests before issuing compact receipts. The trainer trusts that operator
admission, checks exact frozen bytes and batch/pair commitments, and performs no
inference, TOPLOC, sampler or environment verification. Training forward/backward
passes and current-checkpoint reference calculations remain optimizer work.
"""
import copy
import hashlib
import math
from pathlib import Path
import re
import sqlite3
from nacl.exceptions import BadSignatureError

from .storage import canonical
from .distributed_roles import authenticate as _authenticate

VERSION = 'authenticated-verifier-receipts-v1'
AMENDMENT_VERSION = 'private-training-execution-amendment-v1'
POLICIES = {'bf16-full-adamw-covered-fixed-reference-v3',
            'bf16-cpu-fp32-master-task-normalized-persistent-v4'}
COMPUTATION_FIELDS = ('epoch', 'checkpoint', 'source_bundle', 'start', 'deadline', 'training_policy',
    'K', 'L', 'max_batches', 'artifact_policy', 'environment', 'environment_revision',
    'indices', 'environments', 'sample_harness_registry', 'heldout_indices',
    'harness', 'harness_source_hash', 'model_id', 'model_runtime_revision', 'backend_profile',
    'numerical_policy', 'tokenizer_binding', 'sampling_contract', 'sampling_source_hash',
    'task_assets', 'persistent_publication_policy', 'optimizer_state_transport', 'submission_transport_policy', 'hourly_execution_policy', 'audit_exclusion_snapshot')


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('training receipt SHA256')
    return value


def authenticate(envelope, authority):
    if not isinstance(envelope,dict) or set(envelope)!={'payload','signer','signature'}:
        raise ValueError('exact signed training admission envelope required')
    try:return _authenticate(envelope,authority)
    except (ValueError,KeyError,TypeError,BadSignatureError) as error:
        raise ValueError('training admission signature authentication') from error


def require_execution_amendment(controller,manifest):
    """Must precede any capacity probe or first training request/signature."""
    required=getattr(controller,'training_execution_amendment_required_epochs',[])
    files=getattr(controller,'training_execution_amendment_files',{})
    if manifest['epoch'] in required and (manifest['epoch'] not in files or not Path(files[manifest['epoch']]).is_file()):
        raise ValueError('training held: approved private execution amendment is pending')


def computation_binding(manifest):
    result = {key:copy.deepcopy(manifest[key]) for key in COMPUTATION_FIELDS if key in manifest}
    result['checkpoint'].pop('read_urls', None)
    return result


def original_computation_manifest(manifest, authority):
    amendment = manifest.get('training_execution_amendment')
    if amendment is None:
        return manifest
    value = authenticate(amendment, authority)
    original = authenticate(value['original_signed_manifest'], authority)
    normalized = copy.deepcopy(manifest)
    normalized['source_bundle'] = original['source_bundle']
    if computation_binding(normalized) != computation_binding(original):
        raise ValueError('training amendment changed original computation contract')
    return original


def _targets(audit, manifest):
    from .protocol import classification
    accepted = audit.get('accepted')
    if (audit.get('epoch') != manifest['epoch'] or audit.get('training_eligibility') != 'fully-audited-only'
            or not isinstance(accepted, list) or not 1 <= len(accepted) <= manifest['max_batches']):
        raise ValueError('fully audited accepted verifier population')
    targets = []; seen = set()
    for batch in accepted:
        matches = [outcome for outcome in audit.get('outcomes', [])
                   if outcome.get('fully_audited') is True and outcome.get('valid') is True
                   and outcome.get('env_id') == batch.get('env_id') and outcome.get('index') == batch.get('index')]
        if (len(matches) != 1 or type(matches[0].get('batch')) is not int
                or not 0 <= matches[0]['batch'] < manifest['max_batches']
                or matches[0]['batch'] in seen or batch.get('epoch') != manifest['epoch']
                or batch.get('checkpoint') != manifest['checkpoint']['id']):
            raise ValueError('accepted batch lacks exact fully audited outcome')
        positives = [rollout for rollout in batch['rollouts'] if classification(rollout) == 'positive']
        negatives = [rollout for rollout in batch['rollouts'] if classification(rollout) == 'negative']
        if len(positives) != manifest['K'] or len(negatives) != manifest['L']:
            raise ValueError('authenticated verifier class quota')
        seen.add(matches[0]['batch'])
        targets.append(dict(batch_number=matches[0]['batch'], batch_sha256=sha(batch),
            env_id=batch['env_id'], index=batch['index'],
            positive_rollout_sha256=[sha(rollout) for rollout in positives],
            negative_rollout_sha256=[sha(rollout) for rollout in negatives]))
    return sorted(targets, key=lambda target:target['batch_number'])


def frozen_matches(manifest,miner,frozen):
    root=manifest.get('audit_frozen_receipts',{}).get(miner)
    if manifest.get('submission_transport_policy'):
        from .commitment_transport import VERSION
        return manifest['submission_transport_policy']==VERSION and isinstance(root,dict)and frozen in root.get('artifacts',[])
    return root==frozen


def receipt_payload(job_envelope, worker_request, authority, workers, manifest, miner,
                    frozen_receipt, local_audit):
    """Authenticate ORIGINAL evidence before minting an operator attestation.

    Caller must additionally bind these bytes to a COMPLETE authoritative queue
    row. A worker signature by itself is not proof of completed queue admission.
    """
    job = authenticate(job_envelope, authority)
    original = authenticate(job['manifest'], authority)
    worker = worker_request.get('signer')
    if worker not in workers or 'verify' not in workers[worker]:
        raise ValueError('registered verifier identity required')
    request = authenticate(worker_request, worker); report = request.get('report', {})
    data_manifest = original_computation_manifest(manifest, authority)
    if (job.get('role') != 'verify' or request.get('action') != 'report'
            or request.get('job_id') != job['job_id'] or computation_binding(original) != computation_binding(data_manifest)
            or report.get('job_id') != job['job_id'] or report.get('job_sha256') != sha(job)
            or report.get('operator') != authority or report.get('role') != 'verify'
            or report.get('epoch') != original['epoch'] or report.get('checkpoint') != original['checkpoint']['id']
            or report.get('source_files') != job['source_files']
            or report.get('runtime_versions') != job['runtime_versions']
            or report.get('backend_profile') != original['backend_profile']
            or report.get('numerical_policy') != original['numerical_policy']
            or report.get('success') is not True or report.get('chain_transactions') is not False):
        raise ValueError('original signed verifier job/report lineage')
    times = [job.get('created_at'), job.get('expires_at'), report.get('completed_at'), request.get('at')]
    if (any(type(t) not in (int, float) or not math.isfinite(t) for t in times)
            or not times[0] <= times[2] <= times[3] < times[1] or not 0 < times[1]-times[0] <= 86400):
        raise ValueError('original verifier signed report lifetime')
    matches = [audit for audit in report.get('audits', [])
               if audit.get('submission_sha256') == frozen_receipt['sha256']]
    if (len(matches) != 1 or not any(obj.get('sha256') == frozen_receipt['sha256'] for obj in job['submissions'])
            or not frozen_matches(original,miner,frozen_receipt)
            or any(local_audit.get(key) != value for key, value in matches[0].items())):
        raise ValueError('original authenticated frozen audit bytes')
    audit = matches[0]
    from .forced_sampling import require_report
    require_report(original, audit)
    if type(frozen_receipt.get('size')) is not int or frozen_receipt['size'] <= 0:
        raise ValueError('original frozen submission size')
    return dict(version=VERSION, epoch=original['epoch'], checkpoint=original['checkpoint']['id'],
        computation_binding_sha256=sha(computation_binding(original)),
        original_source_bundle_sha256=digest(original['source_bundle']['sha256']),
        original_signed_manifest_sha256=sha(job['manifest']), original_verify_job_id=job['job_id'],
        original_signed_job_sha256=sha(job_envelope), original_job_payload_sha256=sha(job),
        original_worker_report_request_sha256=sha(worker_request), original_report_sha256=sha(report),
        verifier_identity=worker, verifier_source_files_sha256=sha(job['source_files']),
        verifier_runtime_versions_sha256=sha(job['runtime_versions']), miner_identity=miner,
        submission_sha256=digest(frozen_receipt['sha256']), submission_size=frozen_receipt['size'],
        frozen_key=frozen_receipt['frozen_key'], original_report_completed_at=report['completed_at'],
        coordinator_accepted_report_at=request['at'], fully_audited_batches=_targets(audit, original),
        inference_TOPLOC_sampling_environment_verified_by='registered-verifier',
        trainer_verification_required=False)


def issue(controller, manifest, miner, frozen_receipt, local_audit):
    """Operator-only compact receipt from the accepted original SQLite row."""
    queue = getattr(controller.jobs, 'queue', None)
    if queue is None:
        raise ValueError('authenticated verifier report-request queue lineage required')
    identifier = local_audit.get('remote_job_id')
    if not isinstance(identifier, str) or re.fullmatch('[A-Za-z0-9_-]{1,100}', identifier) is None:
        raise ValueError('original verifier job identity')
    with sqlite3.connect('file:' + str(Path(queue.path).resolve()) + '?mode=ro', uri=True) as database:
        database.row_factory = sqlite3.Row
        row = database.execute('SELECT * FROM jobs WHERE id=?', (identifier,)).fetchone()
    if row is None or row['status'] != 'complete' or row['role'] != 'verify':
        raise ValueError('original verifier job not completed by coordinator')
    import json
    job = json.loads(row['envelope']); report = json.loads(row['report'])
    request = json.loads(row['report_request'])
    if (sha(job['payload']) != row['digest'] or sha(report) != row['report_digest']
            or request['signer'] != row['worker'] or request['payload'].get('report') != report):
        raise ValueError('authoritative completed queue bytes')
    payload = receipt_payload(job, request, controller.authority.id, queue.workers,
                              manifest, miner, frozen_receipt, local_audit)
    return controller.signed(payload)


def prepare_submissions(controller, manifest, reports, receipts):
    result = []
    for miner, report in reports.items():
        if not report.get('accepted'):
            continue
        receipt = issue(controller, manifest, miner, receipts[miner], report)
        targets = receipt['payload']['fully_audited_batches']
        result.append(dict(url=controller.bucket.presign(receipts[miner]['frozen_key']),
            sha256=receipts[miner]['sha256'], size=receipts[miner]['size'],
            accepted_batch_sha256=sorted(target['batch_sha256'] for target in targets),
            verifier_receipt=receipt))
    if not result:
        raise ValueError('no authenticated verifier-admitted training submissions')
    return result


def receipt_inventory(submissions):
    return sorted([dict(submission_sha256=obj['sha256'],
        verifier_receipt_sha256=sha(obj['verifier_receipt']),
        accepted_batch_sha256=obj['accepted_batch_sha256']) for obj in submissions],
        key=lambda row:(row['submission_sha256'], row['verifier_receipt_sha256']))


def amend_manifest(controller, manifest, submissions, steps):
    """Read a private approved amendment; never modify the original epoch file."""
    require_execution_amendment(controller,manifest)
    files = getattr(controller, 'training_execution_amendment_files', {})
    if manifest['epoch'] not in files:
        return dict(manifest, training_input_policy=VERSION)
    from .backend_jobs import signed
    import json
    envelope = json.loads(Path(files[manifest['epoch']]).read_bytes())
    amendment = signed(envelope, controller.authority.id)
    result = dict(manifest, source_bundle=copy.deepcopy(amendment['training_source_bundle']),
                  training_input_policy=VERSION, training_execution_amendment=envelope)
    validate_amendment(result, submissions, steps, controller.authority.id)
    if (controller.state / 'roles' / (manifest['epoch'] + '-train.json')).exists():
        # An unchanged original amendment is recovered through the signed job.
        # No new amendment may substitute its prior receipt/source/objective.
        record = json.loads((controller.state/'roles'/(manifest['epoch']+'-train.json')).read_bytes())
        job = signed(json.loads((controller.state/'roles'/(record['job_id']+'-job.json')).read_bytes()),controller.authority.id)
        if job['manifest']['payload'].get('training_execution_amendment') != envelope:
            raise ValueError('original training execution amendment is immutable')
    return result


def validate_amendment(manifest, submissions, steps, authority, job_created_at=None):
    envelope = manifest.get('training_execution_amendment')
    if envelope is None:
        return
    value = authenticate(envelope, authority)
    fields = {'version','epoch','original_signed_manifest','original_signed_manifest_sha256',
              'training_source_bundle','training_policy','training_input_policy','steps',
              'verifier_receipt_inventory','created_at','expires_at'}
    if set(value) != fields:
        raise ValueError('exact private training execution amendment fields')
    original = original_computation_manifest(manifest, authority)
    if (value['version'] != AMENDMENT_VERSION or value['epoch'] != manifest['epoch']
            or value['original_signed_manifest_sha256'] != sha(value['original_signed_manifest'])
            or value['training_source_bundle'] != manifest['source_bundle']
            or value['training_source_bundle'].get('sha256') == original['source_bundle'].get('sha256')
            or value['training_policy'] != manifest['training_policy']
            or value['training_policy'] != original['training_policy']
            or value['training_input_policy'] != VERSION or manifest.get('training_input_policy') != VERSION
            or type(value['steps']) is not int or value['steps'] != steps
            or value['verifier_receipt_inventory'] != receipt_inventory(submissions)):
        raise ValueError('private amendment original science/receipt/source/objective binding')
    digest(value['training_source_bundle'].get('sha256'))
    created, expires = value['created_at'], value['expires_at']
    if (type(created) not in (int,float) or type(expires) not in (int,float)
            or not math.isfinite(created) or not math.isfinite(expires) or not 0 < expires-created <= 86400
            or job_created_at is not None and not created <= job_created_at < expires):
        raise ValueError('training amendment original request authorization lifetime')


def validate_receipt(envelope, obj, manifest, authority):
    value = authenticate(envelope, authority)
    fields={'version','epoch','checkpoint','computation_binding_sha256','original_source_bundle_sha256',
        'original_signed_manifest_sha256','original_verify_job_id','original_signed_job_sha256',
        'original_job_payload_sha256','original_worker_report_request_sha256','original_report_sha256',
        'verifier_identity','verifier_source_files_sha256','verifier_runtime_versions_sha256','miner_identity',
        'submission_sha256','submission_size','frozen_key','original_report_completed_at',
        'coordinator_accepted_report_at','fully_audited_batches',
        'inference_TOPLOC_sampling_environment_verified_by','trainer_verification_required'}
    if not isinstance(value,dict) or set(value)!=fields:
        raise ValueError('exact authenticated training receipt fields')
    original = original_computation_manifest(manifest, authority)
    expected = dict(version=VERSION, epoch=original['epoch'], checkpoint=original['checkpoint']['id'],
        computation_binding_sha256=sha(computation_binding(original)),
        original_source_bundle_sha256=original['source_bundle']['sha256'],
        submission_sha256=obj['sha256'], submission_size=obj['size'],
        inference_TOPLOC_sampling_environment_verified_by='registered-verifier',
        trainer_verification_required=False)
    if any(value.get(key) != wanted for key,wanted in expected.items()):
            raise ValueError('authenticated training receipt epoch/checkpoint/source/input binding')
    for field in ('original_signed_manifest_sha256','original_signed_job_sha256','original_job_payload_sha256',
                  'original_worker_report_request_sha256','original_report_sha256','verifier_identity',
                  'verifier_source_files_sha256','verifier_runtime_versions_sha256','miner_identity'):
        digest(value.get(field))
    if (not isinstance(value['original_verify_job_id'],str)
            or re.fullmatch('[A-Za-z0-9_-]{1,100}',value['original_verify_job_id']) is None
            or any(type(value[k])not in(int,float)or not math.isfinite(value[k])
                   for k in ('original_report_completed_at','coordinator_accepted_report_at'))
            or value['original_report_completed_at']>value['coordinator_accepted_report_at']):
        raise ValueError('original verifier report identity/time binding')
    frozen = manifest.get('audit_frozen_receipts', {}).get(value['miner_identity'])
    if manifest.get('submission_transport_policy')and isinstance(frozen,dict):
        matches=[b for b in frozen['artifacts']if b['sha256']==obj['sha256']]
        frozen=matches[0]if len(matches)==1 else None
    if (not isinstance(frozen,dict) or frozen.get('sha256') != obj['sha256'] or frozen.get('size') != obj['size']
            or frozen.get('frozen_key') != value['frozen_key'] or type(obj['size']) is not int or obj['size'] <= 0):
        raise ValueError('exact original miner frozen receipt')
    targets = value.get('fully_audited_batches')
    if not isinstance(targets,list) or not 1 <= len(targets) <= manifest['max_batches']:
        raise ValueError('bounded fully audited receipt batches')
    seen = set()
    for target in targets:
        if (set(target) != {'batch_number','batch_sha256','env_id','index','positive_rollout_sha256','negative_rollout_sha256'}
                or type(target['batch_number']) is not int or not 0 <= target['batch_number'] < manifest['max_batches']
                or target['batch_number'] in seen or type(target['index']) is not int):
            raise ValueError('unique accepted batch slot commitment')
        seen.add(target['batch_number']); digest(target['batch_sha256'])
        for name, count in (('positive_rollout_sha256',manifest['K']),('negative_rollout_sha256',manifest['L'])):
            values = target[name]
            if not isinstance(values,list) or len(values) != count or len(set(values)) != count:
                raise ValueError('authenticated positive/negative pair commitment')
            for item in values:digest(item)
    if sorted(target['batch_sha256'] for target in targets) != obj.get('accepted_batch_sha256'):
        raise ValueError('exact signed accepted batch commitment')
    return value


def validate_job(job, manifest, authority):
    if (job.get('role') != 'train' or job.get('training_policy') not in POLICIES
            or job.get('training_input_policy') != VERSION or manifest.get('training_input_policy') != VERSION
            or 'subnet/training_receipts.py' not in job['source_files']):
        raise ValueError('authenticated verifier-receipt training admission required')
    if not isinstance(job.get('submissions'),list) or not 1<=len(job['submissions'])<=256:
        raise ValueError('bounded authenticated training submission inventory')
    validate_amendment(manifest,job['submissions'],job['steps'],authority,job['created_at'])
    identities = set()
    for obj in job['submissions']:
        value = validate_receipt(obj.get('verifier_receipt'),obj,manifest,authority)
        identity=(value['miner_identity'],value['submission_sha256'])if manifest.get('submission_transport_policy')else value['miner_identity']
        if identity in identities:
            raise ValueError('one frozen admission per miner')
        identities.add(identity)


def admitted_submission(path, obj, manifest, authority, *, retire=False):
    """Read authenticated bytes/structure and extract ONLY attested pairs."""
    from .backend_jobs import digest as file_digest
    from .batches import submission_records
    from .artifact_budget import for_manifest
    from .protocol import entry, classification, sample_key
    receipt = validate_receipt(obj.get('verifier_receipt'),obj,manifest,authority)
    path = Path(path)
    if (not path.is_file() or path.is_symlink() or path.stat().st_size != obj['size']
            or file_digest(path) != obj['sha256']):
        raise ValueError('exact frozen training ZIP bytes/size')
    records = submission_records(path.read_bytes(),budget=for_manifest(manifest),max_batches=manifest['max_batches'])
    accepted = []; pairs = []; seen = set()
    for target in receipt['fully_audited_batches']:
        number = target['batch_number']
        if number >= len(records):raise ValueError('attested batch slot missing from frozen ZIP')
        batch, arrays = records[number]
        definition = entry(manifest,batch.get('env_id')); index = batch.get('index'); key = sample_key(batch)
        if (sha(batch) != target['batch_sha256'] or batch.get('schema') != 2
                or batch.get('epoch') != manifest['epoch'] or batch.get('checkpoint') != manifest['checkpoint']['id']
                or batch.get('env_id') != target['env_id'] or index != target['index']
                or type(index) is not int or index not in definition['indices'] or key in seen
                or batch.get('sample_index') != index or batch.get('environment_version') != definition['spec']['version']
                or len(batch.get('rollouts',[])) != manifest['K']+manifest['L'] or len(arrays) != len(batch['rollouts'])):
            raise ValueError('attested batch bytes/structure/task binding')
        seen.add(key)
        positives=[r for r in batch['rollouts'] if classification(r)=='positive']
        negatives=[r for r in batch['rollouts'] if classification(r)=='negative']
        if ([sha(r)for r in positives] != target['positive_rollout_sha256']
                or [sha(r)for r in negatives] != target['negative_rollout_sha256']):
            raise ValueError('exact authenticated pair rollout digests')
        accepted.append(batch); pairs.extend((definition,p,n) for p,n in zip(positives,negatives))
    summary=dict(version=VERSION,epoch=manifest['epoch'],submission_sha256=obj['sha256'],
        verifier_receipt_sha256=sha(obj['verifier_receipt']),
        accepted_batch_sha256=obj['accepted_batch_sha256'],accepted=accepted,
        original_verify_job_id=receipt['original_verify_job_id'],
        original_report_sha256=receipt['original_report_sha256'],
        trainer_verification_performed=False, verification_performed_by='registered-verifier')
    if retire:path.unlink()
    return summary,pairs


def validate_report(report, job, manifest, authority):
    validate_job(job,manifest,authority)
    rows=report.get('training_admissions')
    if not isinstance(rows,list) or len(rows) != len(job['submissions']) or report.get('audits'):
        raise ValueError('trainer must report admissions without new audits')
    training=report.get('training',{})
    if (training.get('training_input_policy') != VERSION or training.get('trainer_verification_performed') is not False
            or training.get('all_pairs_authenticated_verifier_receipts') is not True
            or 'all_pairs_independently_reaudited' in training):
        raise ValueError('truthful verifier-receipt training report required')
    for row,obj in zip(rows,job['submissions']):
        receipt=validate_receipt(obj['verifier_receipt'],obj,manifest,authority)
        if (row.get('version') != VERSION or row.get('epoch') != manifest['epoch']
                or row.get('submission_sha256') != obj['sha256']
                or row.get('verifier_receipt_sha256') != sha(obj['verifier_receipt'])
                or row.get('accepted_batch_sha256') != obj['accepted_batch_sha256']
                or sorted(sha(batch)for batch in row.get('accepted',[])) != obj['accepted_batch_sha256']
                or row.get('original_verify_job_id') != receipt['original_verify_job_id']
                or row.get('original_report_sha256') != receipt['original_report_sha256']
                or row.get('trainer_verification_performed') is not False
                or row.get('verification_performed_by') != 'registered-verifier'):
            raise ValueError('training admission report exact signed receipt/population')
