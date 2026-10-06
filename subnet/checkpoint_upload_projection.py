"""Default-off operator routing for a fresh upload after prelaunch role failure.

The original scientific request, reports and failed upload stay immutable.
Only the train-only recovery declaration is omitted from the fresh upload.
"""
import copy
import hashlib
import math
import re
import time
from pathlib import Path
from .distributed_roles import authenticate, digest
from .storage import canonical
VERSION = 'recovery-checkpoint-upload-manifest-projection-v1'
RECOVERY = {'terminal-parent-restore-pre-update-recovery-v2',
            'terminal-parent-restore-pre-update-bootstrap-recovery-v3'}


def project_manifest(manifest):
    declaration = manifest.get('training_startup_recovery')
    if not isinstance(declaration, dict) or declaration.get('payload', {}).get('version') not in RECOVERY:
        raise ValueError('explicit recovery publication projection only')
    projected = copy.deepcopy(manifest)
    del projected['training_startup_recovery']
    return projected


def authorize(scope, authority, original_bytes, now=None):
    value = authenticate(scope, authority)
    fields = {'version', 'original_upload_file_sha256', 'original_job_sha256',
              'original_manifest_sha256', 'projected_manifest_sha256',
              'original_label', 'replacement_label', 'source_sha256', 'source_files',
              'runtime_versions', 'created_at', 'expires_at', 'helper_sha256',
              'prelaunch_witness_sha256', 'completed_training_report_sha256',
              'independent_reader_receipt_sha256', 'independent_reader_terminal_sha256'}
    if set(value) != fields or value['version'] != VERSION:
        raise ValueError('exact signed publication projection scope')
    import json
    envelope = json.loads(original_bytes)
    job = authenticate(envelope, authority)
    manifest = authenticate(job['manifest'], authority)
    declaration = authenticate(manifest['training_startup_recovery'], authority)
    if declaration['version'] not in RECOVERY or declaration['epoch'] != manifest['epoch']:
        raise ValueError('authentic original recovery scope')
    now = time.time() if now is None else now
    if any(type(value[k]) not in (int, float) or not math.isfinite(value[k]) for k in ('created_at', 'expires_at')) or not value['created_at'] <= now < value['expires_at'] or value['expires_at']-value['created_at'] > 3600 or value['expires_at'] > job['expires_at']:
        raise ValueError('fresh bounded projection authorization')
    projected = project_manifest(manifest)
    if job['role'] != 'upload' or len(job['source_files']) != 177 or value['original_upload_file_sha256'] != hashlib.sha256(original_bytes).hexdigest() or value['original_job_sha256'] != digest(job) or value['original_manifest_sha256'] != digest(manifest) or value['projected_manifest_sha256'] != digest(projected):
        raise ValueError('exact original and projected upload bindings')
    if value['source_sha256'] != manifest['source_bundle']['sha256'] or value['source_files'] != job['source_files'] or value['runtime_versions'] != job['runtime_versions'] or value['helper_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('unchanged sealed execution and pinned operator')
    if value['original_label'] != manifest['epoch']+'-publish-'+manifest['checkpoint']['id'][:8] or value['replacement_label'] == value['original_label'] or not re.fullmatch('[A-Za-z0-9_-]{1,90}', value['replacement_label']):
        raise ValueError('fresh distinct publication reservation')
    for k in ('prelaunch_witness_sha256', 'completed_training_report_sha256', 'independent_reader_receipt_sha256', 'independent_reader_terminal_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', value[k]):raise ValueError('actual original completion evidence required')
    return value, job, manifest, projected


class ProjectedUploadJobs:
    """Delegate every other operation unchanged; original stage retains fullGET."""
    def __init__(self, jobs, authorization, original_job, original_manifest, projected):
        self.original = jobs
        self.authorization = authorization
        self.job = original_job
        self.manifest = original_manifest
        self.projected = projected
    def __getattr__(self, name):return getattr(self.original, name)
    def run(self, label, role, manifest, cache=None, **fields):
        v = self.authorization
        if role != 'upload' or label != v['original_label']:
            return self.original.run(label, role, manifest, cache, **fields)
        if time.time() >= v['expires_at'] or canonical(manifest) != canonical(self.manifest) or set(fields) != {'put_urls'} or set(fields['put_urls']) != set(self.job['put_urls']):
            raise ValueError('exact unchanged checkpoint and original10 capabilities')
        return self.original.run(v['replacement_label'], role, self.projected, cache, put_urls=self.job['put_urls'])


def install_remote_jobs_projection(remote_jobs_class, authorization, original_job, original_manifest, projected):
    """Explicit operator patch; sealed role files and individual requests unchanged."""
    original_run = remote_jobs_class.run
    if getattr(original_run, '_publication_projection', False):
        raise ValueError('duplicate publication projection installation')
    def run(instance, label, role, manifest, cache=None, **fields):
        class BoundOriginal:
            def run(self, *args, **kwargs):return original_run(instance, *args, **kwargs)
        return ProjectedUploadJobs(BoundOriginal(), authorization, original_job,
                                   original_manifest, projected).run(label, role, manifest, cache, **fields)
    run._publication_projection = True
    remote_jobs_class.run = run
    return original_run
