"""CPU-only four-original factory for an approved continuous128 policy.

No GPU dispatch, clock extension on retries, task selection, or checkpoint
mutation occurs here. The caller durably saves the first returned packet.
"""
import copy
from pathlib import Path

from subnet.backend_jobs import signed
from ops.continuous_owned_heldout128 import digest
from ops.owned_cached_group_operator import validate_originals
from ops.owned_cached_group_retention import validate_scope
from ops.owned_cached_larger_cohort import SOURCE


def prepare_packet(policy, publication, identity, authority, signer, *, now, refresh_manifest):
    original=signed(policy['template_original_job'],authority)
    template=signed(original['manifest'],authority)
    if (original.get('role')!='evaluate' or original.get('checkpoint_cache') is not None or
        original.get('trusted_evaluation_policy') is not None or original.get('successor_calibration') is not None or
        original['source_files']!=policy['source_files'] or original['runtime_versions']!=policy['runtime_versions'] or
        template['source_bundle']['sha256']!=SOURCE):
        raise ValueError('exact ROOT-authenticated original cached evaluator template')
    descriptor=signed(publication['checkpoint_descriptor'],authority)
    cp=descriptor['id']
    if identity!=digest([digest(policy),cp,policy['cohort_sha256']]):
        raise ValueError('deterministic checkpoint/cohort original identity')
    root=Path(policy['remote_root'])/identity
    code=Path(policy['remote_source_path'])
    if not root.is_absolute() or root.resolve()!=root or not code.is_absolute() or code.resolve()!=code:
        raise ValueError('canonical isolated source and group namespace')
    expires=min(now+policy['group_lifetime_seconds'],policy['expires_at'])
    if expires<=now:raise ValueError('bounded new original lifetime')
    # Transport URLs are refreshed together; every chunk has exactly one
    # checkpoint map, scientific runtime and native taskset. No claimed model
    # state or optimizer counter is used to construct the baseline.
    manifest=copy.deepcopy(template)
    manifest.update(epoch='owned-heldout128-'+identity,checkpoint=copy.deepcopy(descriptor))
    before=copy.deepcopy(manifest)
    manifest=refresh_manifest(manifest,int(expires-now))
    def scientific(value):
        value=copy.deepcopy(value)
        for field in ('checkpoint','source_bundle'):
            for key in ('read_url','read_urls'):value[field].pop(key,None)
        return value
    if scientific(manifest)!=scientific(before):
        raise ValueError('transport refresh changed scientific template or checkpoint bytes')
    endpoint=copy.deepcopy(policy['endpoint']);endpoint.update(workspace=str(root),code=str(code))
    scope=copy.deepcopy(policy['group_scope_template'])
    scope.update(created_at=now,expires_at=expires,execute_allowed=True,workspace=str(root),source_path=str(code),
                 checkpoint=copy.deepcopy(manifest['checkpoint']),groups=copy.deepcopy(policy['groups']),
                 source_sha256=SOURCE,source_files=policy['source_files'],runtime_versions=policy['runtime_versions'],
                 endpoint=endpoint)
    scope.pop('original_jobs_PENDING_ROOT_SIGNATURES',None)
    jobs=[];bindings={}
    for group in policy['groups']:
        job=copy.deepcopy(original)
        job.update(job_id='owned128-'+identity+'-group-'+str(group['group']),created_at=now,expires_at=expires,
                   manifest=signer(copy.deepcopy(manifest)),heldout=[{k:v for k,v in group.items() if k!='group'}])
        envelope=signer(job)
        if signed(envelope,authority)!=job:raise ValueError('exact approved delegated CPU signer')
        jobs.append(envelope);bindings[job['job_id']]=dict(group=group['group'],job_sha256=digest(job))
    scope['original_jobs']=bindings
    envelope=signer(scope)
    validate_scope(envelope,authority,str(root),policy['source_files'],now=now)
    validate_originals(scope,jobs,authority)
    from ops.owned_cached_group_ack_relay import QualifiedGroupObserver
    QualifiedGroupObserver(scope,jobs,authority)  # Pure CPU route check before any dispatch.
    return dict(checkpoint=cp,cohort_sha256=policy['cohort_sha256'],identity=identity,
                workspace=str(root),expires_at=expires,scope=envelope,original_jobs=jobs,
                optimizer_step=signed(publication['optimizer_publication'],authority)['descriptor']['optimizer_steps'])
