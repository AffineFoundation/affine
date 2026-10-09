"""Drop authenticated training-only ancestry from inference-only upload requests.

The model inventory and every other publication field remain unchanged. A new
role label prevents an old terminal upload from being relaunched or overwritten.
"""
import copy
import hashlib
import json
from pathlib import Path

FIELD='trainer_local_state_original_manifest'
VERSION='ordinary-trainer-local-publication-projection-v1'
SUFFIX='-local-state-v1'

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def sha(value):return hashlib.sha256(canonical(value)).hexdigest()

def project(manifest,authority,signed,trainer_project):
    if FIELD not in manifest or 'training_startup_recovery' in manifest:
        return None
    document=manifest[FIELD]
    original=signed(document,authority)
    if FIELD in original:raise ValueError('ordinary publication projection cannot nest')
    expected=trainer_project(original,lambda _:document)
    # This is the sole change made by publication after successful training.
    expected['checkpoint']=copy.deepcopy(manifest['checkpoint'])
    if expected!=manifest:raise ValueError('publication changed original training context')
    checkpoint=manifest['checkpoint']
    if (not isinstance(checkpoint,dict) or not isinstance(checkpoint.get('files'),dict)
            or not checkpoint['files'] or not isinstance(checkpoint.get('id'),str)):
        raise ValueError('exact inference checkpoint inventory')
    result=copy.deepcopy(manifest)
    del result[FIELD]
    return result

def validate_recovery(declaration,authority,signed,trainer_project,read):
    if declaration.get('version')!=VERSION:raise ValueError('publication projection version')
    job=signed(read(declaration['original_job']['path']),authority)
    if sha(job)!=declaration['original_job']['payload_sha256'] or job['role']!='upload':
        raise ValueError('original ROOT upload request')
    manifest=signed(job['manifest'],authority)
    projected=project(manifest,authority,signed,trainer_project)
    if projected is None or sha(projected)!=declaration['projected_manifest_sha256']:
        raise ValueError('exact wrapper-only projection')
    failure=read(declaration['original_failure']['path'])
    if (sha(failure)!=declaration['original_failure']['sha256'] or failure.get('job_id')!=job['job_id']
            or failure.get('phase')!='failed' or failure.get('exit_code')!=1
            or not job['created_at']<=failure['started_at']<=failure['finished_at']<=declaration['created_at']):
        raise ValueError('original terminal upload failure')
    if Path(declaration['original_report_path']).exists():raise ValueError('publication already reported')
    report=read(declaration['successful_training_report']['path'])
    if (sha(report)!=declaration['successful_training_report']['sha256'] or report.get('role')!='train'
            or report.get('success')is not True or report.get('epoch')!=manifest['epoch']
            or {k:v for k,v in report.get('new_checkpoint',{}).items() if k!='path'}!=manifest['checkpoint']
            or report.get('new_checkpoint',{}).get('path')!=declaration['checkpoint_remote_path']):
        raise ValueError('same successful trained checkpoint')
    if declaration['original_label']!=manifest['epoch']+'-publish-'+manifest['checkpoint']['id'][:8]:
        raise ValueError('original publication label')
    if declaration['replacement_label']!=declaration['original_label']+SUFFIX:
        raise ValueError('distinct stable publication label')
    return dict(epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],projected_manifest_sha256=sha(projected))

def install(policy,guards):
    from subnet import remote_backend,trainer_local_state
    row=policy['publication_local_projection']
    if row['version']!=VERSION or guards.file_hash(__file__)!=row['sha256']:
        raise ValueError('pinned ordinary publication projection')
    recovery=guards.signed(guards.read(row['recovery']['path']))
    if sha(recovery)!=row['recovery']['payload_sha256']:raise ValueError('exact original publication recovery')
    validate_recovery(recovery,guards.AUTHORITY,remote_backend.signed,trainer_local_state.project,guards.read)
    original=remote_backend.publication_request
    def request(manifest,authority,configured=None):
        # Retain the original recovery projection and its authentication.
        if 'training_startup_recovery' in manifest or FIELD not in manifest:
            return original(manifest,authority,configured)
        if configured!={'version':remote_backend.PUBLICATION_PROJECTION_VERSION}:
            raise ValueError('explicit local publication projection policy')
        projected=project(manifest,authority,remote_backend.signed,trainer_local_state.project)
        if manifest['epoch']==recovery['epoch'] and sha(projected)!=recovery['projected_manifest_sha256']:
            raise ValueError('original upload recovery scope changed')
        label=manifest['epoch']+'-publish-'+manifest['checkpoint']['id'][:8]+SUFFIX
        return label,projected
    remote_backend.publication_request=request
    original_run=remote_backend.RemoteJobs.run
    def run(self,label,role,manifest,cache=None,dispatch_only=False,**fields):
        if role=='upload':
            import time
            now=time.time()
            payload=dict(schema=1,job_id=label+'-ffffffff',role=role,created_at=now,
                         expires_at=now+remote_backend.role_time_budget(self.config,role),
                         manifest=self.controller.signed(manifest),**self.metadata,**fields)
            enforce_size(self.controller.signed(payload))
        return original_run(self,label,role,manifest,cache,dispatch_only=dispatch_only,**fields)
    remote_backend.RemoteJobs.run=run

def enforce_size(envelope):
    # Match the unchanged upload parser; retain a tiny JSON float-length margin.
    if len(canonical(envelope))>3_999_936:
        raise ValueError('publication envelope exceeds original upload size budget')
    return len(canonical(envelope))
