"""Prospective signed MiMo mining/audit jobs on the existing role queue.

The operator signs source, runtime, manifest and original miner commitments.
This entry point bypasses legacy K/L and pair ZIP parsers explicitly; it never
changes an active service configuration or couples training to audit completion.
"""
from __future__ import annotations
import hashlib
from contextlib import contextmanager
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
from urllib.parse import urlparse, unquote

VERSION = 'mimo-complete-group-worker-v1'
REVISION = 'qwen3_5-mimo-v1'
from .mimo_runtime import SUPPORTED_VERSIONS
REPORT_POLICY = 'mimo-compact-group-audit-v1'
GROUP_REFERENCE = 'mimo-audited-group-reference-v1'


def selected(job, manifest):
    return (job.get('worker_policy') == VERSION or manifest.get('model_runtime_revision') in SUPPORTED_VERSIONS
            or manifest.get('sampling_contract', {}).get('version') == 'forced-inverse-cdf-complete-group-mimo-v1')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def source_inventory(root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    from .backend_jobs import digest
    result = {}
    for path in sorted((root/'subnet').rglob('*.py')):
        if '__pycache__' in path.parts: continue
        need(not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(root),
             'ordinary worker source file')
        result[path.relative_to(root).as_posix()] = digest(path)
    return result


def source_guard(job):
    """Pin all canonical package files, including already-loaded warm code.

    A warm worker cannot adopt another source closure mid-process. It must be
    restarted from that reviewed release. Only small code files are rehashed;
    model and optimizer state are never rehashed at a per-token/job boundary.
    """
    root = Path(__file__).resolve().parents[1]
    expected = job['source_files']
    observed = source_inventory(root)
    need(observed == expected, 'MiMo worker exact installed source closure differs')
    for name in expected:
        path = root/name
        need(not path.is_symlink() and path.resolve().is_relative_to(root), 'ordinary worker source file')
    for name, module in tuple(sys.modules.items()):
        if name == 'subnet' or name.startswith('subnet.'):
            filename = getattr(module, '__file__', None)
            if filename is not None:
                path = Path(filename).resolve()
                need(path.is_relative_to(root) and path.relative_to(root).as_posix() in expected,
                     'loaded worker module outside authenticated source closure')
    return hashlib.sha256(__import__('json').dumps(expected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_submissions(manifest, submissions, commitments):
    """Require exact signed miner slot evidence, not operator-supplied labels."""
    from .backend_jobs import canonical, r2_url
    from .complete_group_admission import validate_commitment, is_digest, sha
    from .complete_group_transport import paths
    need(type(submissions) is list and 1 <= len(submissions) <= 256, 'bounded complete-group audit selection')
    miners = {row.get('commitment_miner') for row in submissions if isinstance(row, dict)}
    need(type(commitments) is dict and set(commitments) == miners, 'exact selected miner commitments')
    parents = {}
    for miner, envelope in commitments.items():
        need(is_digest(miner), 'authenticated committed miner identity')
        raw = canonical(envelope)
        parents[miner] = (validate_commitment(raw, manifest, miner, registered_miners=miners), sha(raw))
    seen = set()
    for obj in submissions:
        need(type(obj) is dict and set(obj) == {'url','sha256','commitment_miner','commitment_ref'},
             'exact group audit object fields')
        ref = obj['commitment_ref']; miner = obj['commitment_miner']
        fields = {'miner','commitment_sha256','slot','env_id','index','batch_sha256','size','group_id','artifact_key'}
        need(type(ref) is dict and set(ref) == fields and ref['miner'] == miner,
             'exact group commitment reference')
        parent, parent_sha = parents[miner]
        need(ref['commitment_sha256'] == parent_sha and type(ref['slot']) is int
             and 0 <= ref['slot'] < len(parent['payload']['groups']), 'original committed group slot')
        claim = parent['payload']['groups'][ref['slot']]
        need(all(ref[k] == claim[k] for k in ('slot','env_id','index','batch_sha256','group_id'))
             and type(ref['index']) is int and type(ref['size']) is int
             and ref['size'] == claim['artifact_size'] and obj['sha256'] == claim['artifact_sha256'],
             'selected group must match original signed commitment')
        key = paths(manifest, miner, ref['slot'])[0]
        parsed = urlparse(r2_url(obj['url'], 'GET'))
        need(ref['artifact_key'] == key and unquote(parsed.path).endswith('/'+key),
             'immutable group artifact capability path')
        identity = (miner, ref['slot'])
        need(identity not in seen, 'duplicate selected group slot')
        seen.add(identity)
    return submissions


def submission_reference(manifest, miner, commitment, slot, read_url):
    """Build an unsigned selected slot; caller signs the whole immutable job."""
    from .complete_group_admission import validate_commitment, digest
    from .complete_group_transport import paths
    from .storage import canonical
    value = validate_commitment(canonical(commitment), manifest, miner, registered_miners={miner})
    need(type(slot) is int and 0 <= slot < len(value['payload']['groups']), 'selected original group slot')
    claim = value['payload']['groups'][slot]
    obj = dict(url=read_url, sha256=claim['artifact_sha256'], commitment_miner=miner,
        commitment_ref=dict(miner=miner, commitment_sha256=digest(value), slot=slot,
            env_id=claim['env_id'], index=claim['index'], batch_sha256=claim['batch_sha256'],
            size=claim['artifact_size'], group_id=claim['group_id'], artifact_key=paths(manifest,miner,slot)[0]))
    validate_submissions(manifest,[obj],{miner:value})
    return obj


def validate(envelope, authority, now=None, *, allow_public=False):
    from .backend_jobs import signed, r2_url, mining_window, mining_definitions
    from .runtime_factory import validate_backend
    from .batch_contract import for_manifest, REQUIRED_CLIENT_FEATURES
    from .complete_group_codec import limits
    from .complete_group_transport import validate_capability
    now = time.time() if now is None else now
    job = signed(envelope, authority)
    need(job.get('schema') == 1 and job.get('worker_policy') == VERSION and job.get('role') in ('mine','verify','train','evaluate'),
         'explicit MiMo complete-group worker policy')
    need(type(job.get('job_id')) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', job['job_id']), 'job ID')
    need(all(type(job.get(k)) in (int,float) and math.isfinite(job[k]) for k in ('created_at','expires_at'))
         and job['created_at'] <= now < job['expires_at'] and 0 < job['expires_at']-job['created_at'] <= 86400,
         'job expired/time budget')
    manifest = signed(job['manifest'], authority)
    from .mimo_release_policy import policy_for_manifest
    need(type(allow_public) is bool, 'explicit local MiMo public execution flag')
    policy_for_manifest(manifest, authority, allow_public=allow_public)
    need(manifest.get('model_runtime_revision') in SUPPORTED_VERSIONS, 'MiMo worker runtime version')
    validate_backend(manifest)
    contract = for_manifest(manifest, supported_features=REQUIRED_CLIENT_FEATURES)
    limits(manifest)
    cp = manifest['checkpoint']
    from .mimo_runtime import validate_descriptor
    validate_descriptor({key:cp[key] for key in ('id','files')})
    urls = cp.get('read_urls', {})
    need(not urls or set(urls) == set(cp['files']), 'checkpoint capability file binding')
    for url in urls.values(): r2_url(url, 'GET')
    refresh = job.get('checkpoint_read_urls')
    if refresh is not None:
        need(type(refresh) is dict and set(refresh) == set(cp['files']), 'refreshed checkpoint capability inventory')
        for url in refresh.values(): r2_url(url, 'GET')
    source = job.get('source_files')
    need(type(source) is dict and source and 'subnet/mimo_worker.py' in source, 'MiMo worker source closure required')
    for name, value in source.items():
        need(type(name) is str and name.startswith('subnet/') and '..' not in name
             and name.endswith('.py') and type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'worker source pin')
    need(set(job.get('runtime_versions', {})) == {'torch','transformers','toploc'}
         and all(type(x) is str and x for x in job['runtime_versions'].values()), 'runtime version pins')
    if job['role'] == 'verify':
        need('mining_subset' not in job, 'mining subset only for owned mining jobs')
        validate_submissions(manifest, job.get('submissions'), job.get('group_commitments'))
    elif job['role'] == 'train':
        from .complete_group_admission import is_digest
        need(is_digest(job.get('trainer_worker')), 'one authorized trainer worker identity')
        need('mining_subset' not in job, 'mining subset only for owned mining jobs')
        from .mimo_training_worker import validate_job
        validate_job(job, manifest, authority, allow_public=allow_public)
    elif job['role'] == 'evaluate':
        from .mimo_evaluation import validate_job
        validate_job(job, manifest)
    else:
        mining_window(manifest, now)
        from .complete_group_admission import is_digest
        need(is_digest(job.get('miner_id')), 'owned miner identity')
        budget = job.get('search_budget', contract.group_size)
        need(type(budget) is int and contract.group_size <= budget <= 1000,
             'local search budget must cover exact canonical N')
        need(job.get('seed_start', 0) == 0 and type(job.get('seed_start', 0)) is int,
             'canonical nonce group starts at zero')
        filename = job.get('miner_identity_file')
        need(type(filename) is str and Path(filename).is_absolute(), 'private owned miner signer path')
        validate_capability(job['capability'], manifest, job['miner_id'])
        mining_definitions(manifest, job)
    return job, manifest


def validate_report(report, job, manifest, authority, job_digest, now):
    from .backend_jobs import canonical
    from .complete_group_admission import digest
    expected = dict(job_id=job['job_id'], job_sha256=job_digest, operator=authority,
        worker_policy=VERSION, role=job['role'], epoch=manifest['epoch'], checkpoint=manifest['checkpoint']['id'],
        source_files=job['source_files'], runtime_versions=job['runtime_versions'],
        backend_profile=manifest['backend_profile'], numerical_policy=manifest['sampling_contract']['calibration'],
        chain_transactions=False, success=True)
    need(all(canonical(report.get(k)) == canonical(v) for k,v in expected.items()), 'MiMo report job/source/runtime binding')
    from .mimo_runtime import PRECISION_VERSION
    if manifest['mimo_runtime']['version'] == PRECISION_VERSION:
        from .mimo_precision import validate_provenance
        observed = validate_provenance(report.get('execution_provenance'),
            manifest['mimo_runtime']['precision_profile'], device=manifest['backend_profile']['device'])
        need(observed['build']['torch_distribution'] == job['runtime_versions']['torch'], 'observed worker Torch distribution')
    else:
        need('execution_provenance' not in report, 'legacy report cannot claim precision-v2 provenance')
    completed = report.get('completed_at')
    need(type(completed) in (int,float) and math.isfinite(completed)
         and job['created_at'] <= completed < job['expires_at'] and completed <= now+5, 'report signed deadline')
    if job['role'] == 'train':
        from .mimo_training_worker import validate_result
        return validate_result(report,job,manifest)
    if job['role'] == 'evaluate':
        from .mimo_evaluation import validate_result
        return validate_result(report.get('evaluation',{}),job,manifest)
    if job['role'] == 'mine':
        from .backend_jobs import mining_definitions
        need(report.get('miner_id') == job['miner_id']
             and type(report.get('submitted_groups')) is int
             and 0 <= report['submitted_groups'] <= manifest['max_batches']
             and report.get('exact_group_size') == manifest['group_contract']['group_size'],
             'owned complete-group mining report')
        allowed={(d['env_id'],index) for d in mining_definitions(manifest,job) for index in d['indices']}
        tasks=report.get('tasks')
        need(type(tasks) is list and len(tasks) <= len(allowed), 'bounded mining task report')
        seen=set()
        for row in tasks:
            need(type(row) is dict and set(row) == {'env_id','index','status'}
                 and (row['env_id'],row['index']) in allowed and (row['env_id'],row['index']) not in seen
                 and row['status'] in ('submitted','filtered-complete-group','native-indeterminate','epoch-closed'),
                 'authorized mining task disposition')
            seen.add((row['env_id'],row['index']))
        return
    validate_submissions(manifest, job['submissions'], job['group_commitments'])
    audits = report.get('audits')
    need(type(audits) is list and len(audits) == len(job['submissions']), 'complete group report coverage')
    for audit, obj in zip(audits, job['submissions'], strict=True):
        ref = obj['commitment_ref']; accepted = audit.get('accepted')
        need(audit.get('report_policy') == REPORT_POLICY
             and audit.get('epoch') == manifest['epoch'] and audit.get('miner') == obj['commitment_miner']
             and audit.get('submission_sha256') == obj['sha256'] and type(accepted) is list and len(accepted) <= 1,
             'exact selected group report identity')
        for batch in accepted:
            need(type(batch) is dict and set(batch) == {'schema','epoch','checkpoint','miner','env_id','index',
                 'batch_sha256','group_id','rollout_count','outcome_counts','generated_tokens'}
                 and batch['schema'] == GROUP_REFERENCE and batch['batch_sha256'] == ref['batch_sha256']
                 and batch['group_id'] == ref['group_id'] and batch['miner'] == obj['commitment_miner']
                 and batch['env_id'] == ref['env_id'] and type(batch['index']) is int and batch['index'] == ref['index']
                 and batch['epoch'] == manifest['epoch'] and batch['checkpoint'] == manifest['checkpoint']['id'],
                 'accepted original committed batch reference')
            n=manifest['group_contract']['group_size']; counts=batch['outcome_counts']
            need(type(batch['rollout_count']) is int and batch['rollout_count'] == n
                 and type(counts) is dict and set(counts) == {'positive','negative','neutral'}
                 and all(type(v) is int and v >= 0 for v in counts.values()) and sum(counts.values()) == n
                 and type(batch['generated_tokens']) is int and batch['generated_tokens'] >= n,
                 'complete group audit reference counts')
        outcomes = audit.get('outcomes')
        need(type(outcomes) is list and len(outcomes) == 1, 'one explicit group audit outcome')
        outcome = outcomes[0]
        if accepted:
            need(outcome.get('valid') is True and outcome.get('fully_audited') is True
                 and type(outcome.get('rollouts')) is list
                 and len(outcome['rollouts']) == manifest['group_contract']['group_size']
                 and [row.get('nonce') for row in outcome['rollouts']] == list(range(manifest['group_contract']['group_size'])),
                 'accepted group must have full audit')
        else:
            need(outcome.get('valid') is False and outcome.get('fully_audited') is False
                 and outcome.get('failure_kind') == 'invalid_sample', 'definite invalid group audit outcome')


def _preflight(data, manifest, obj):
    from .backend_jobs import canonical
    from .complete_group_codec import decode, limits
    from .complete_group_admission import local_task, structural_batch
    from .complete_group_distribution import eos_ids
    from .audit_policy import InvalidSample
    try:
        ref = obj['commitment_ref']
        need(len(data) == ref['size'] and hashlib.sha256(data).hexdigest() == obj['sha256'], 'committed group bytes')
        batch, arrays = decode(data, limits(manifest))[0]
        need(hashlib.sha256(canonical(batch)).hexdigest() == ref['batch_sha256'], 'committed batch document')
        geometry = manifest['group_transport_policy']; stops = eos_ids(geometry)
        # This is only cheap local consistency; verify() regenerates authority
        # from the actual native task and checkpoint tokenizer before acceptance.
        task = local_task(batch, manifest, eos_token_id=stops[0], vocab_size=geometry['vocab_size'], eos_token_ids=stops)
        structural_batch(batch, manifest, obj['commitment_miner'], task, arrays)
        return batch
    except (ValueError, KeyError, TypeError) as error:
        raise InvalidSample('committed complete-group structural mismatch') from error


def _compact_audit(result, manifest, obj):
    """No token/proof arrays on the queue API; immutable source bytes stay in R2."""
    from .complete_group_admission import digest
    need(result.get('accepted') and len(result['accepted']) == 1, 'one successfully audited complete group')
    batch=result['accepted'][0]; ref=obj['commitment_ref']; n=manifest['group_contract']['group_size']
    need(digest(batch) == ref['batch_sha256'], 'auditor returned different committed batch')
    details=result['outcomes'][0]['rollouts']
    need(len(details) == n, 'all N rollout audit diagnostics')
    allowed={'positions','all_intervals_verified','cdf_abs_error','sampler_distribution',
             'cached_reference_adjudication','fast_disagreement','sampling_seconds','full_verify_seconds'}
    diagnostics=[]
    for nonce, row in enumerate(details):
        summary={key:value for key,value in row.items() if key in allowed}
        need(summary.get('positions') == len(batch['rollouts'][nonce]['turns'][0]['output'])
             and summary.get('all_intervals_verified') is True, 'complete audit token coverage')
        # Every allowed field is scalar; an unbounded custom diagnostic cannot
        # silently fill the coordinator request with token arrays.
        need(all(type(v) in (bool,int,float,str) and (type(v) is not str or len(v) <= 100)
                 and (type(v) is not float or math.isfinite(v)) for v in summary.values()),
             'bounded scalar audit diagnostics')
        diagnostics.append(dict(nonce=nonce,**summary))
    summary=dict(schema=GROUP_REFERENCE,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],
        miner=obj['commitment_miner'],env_id=ref['env_id'],index=ref['index'],batch_sha256=ref['batch_sha256'],
        group_id=ref['group_id'],rollout_count=n,
        outcome_counts={kind:sum(r['classification']==kind for r in batch['rollouts']) for kind in ('positive','negative','neutral')},
        generated_tokens=sum(len(r['turns'][0]['output']) for r in batch['rollouts']))
    return dict(report_policy=REPORT_POLICY,epoch=manifest['epoch'],miner=obj['commitment_miner'],
        submission_sha256=obj['sha256'],accepted=[summary],sampling_assurance=result.get('sampling_assurance'),
        outcomes=[dict(batch=0,valid=True,fully_audited=True,rollouts=diagnostics)],
        training_eligibility='structurally-eligible-unaudited-permitted')


def _invalid(manifest, obj, error):
    return dict(report_policy=REPORT_POLICY,epoch=manifest['epoch'], miner=obj['commitment_miner'], submission_sha256=obj['sha256'],
        accepted=[], outcomes=[dict(batch=0, valid=False, fully_audited=False,
            failure_kind='invalid_sample', error_type=type(error).__name__)],
        training_eligibility='structurally-eligible-unaudited-permitted')


def execute(envelope, authority, workspace, cache=None, runtime_factory=None, *,
            assert_lease=None, trainer_state_root=None, training_context=None, allow_public=False,
            mimo_replicas=1):
    from .backend_jobs import canonical, digest, version, get_object, ArtifactRejected, checkpoint, mining_definitions
    job, manifest = validate(envelope, authority, allow_public=allow_public)
    need(type(mimo_replicas) is int and mimo_replicas in (1,2,4), 'local MiMo replica count')
    need(mimo_replicas == 1 or job['role'] == 'mine', 'local replicas apply only to mining')
    if job['role'] == 'train':
        need(callable(assert_lease), 'queue-owned trainer lease assertion required')
        need(trainer_state_root is not None and Path(trainer_state_root).is_absolute(),
             'explicit stable trainer state root required')
    def guard():
        need(time.time() < job['expires_at'], 'original signed worker deadline')
        if assert_lease is not None: assert_lease()
    guard()
    source_guard(job)
    for package, expected in job['runtime_versions'].items():
        need(version(package) == expected, 'runtime package mismatch')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUDA environment profile')
    from .task_assets import hydrate_manifest
    workspace = Path(workspace); out = workspace/'jobs'/job['job_id']
    if job['role'] == 'train':
        # Reopen only an ordinary owner-private result directory: the trainer's
        # durable original transaction decides whether to recover or step.
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        need(out.absolute() == out.resolve() and out.is_dir() and out.stat().st_uid == os.getuid()
             and not out.stat().st_mode & 0o077, 'ordinary owner-only training output directory')
    else:
        out.mkdir(parents=True, exist_ok=False); out.chmod(0o700)
    asset_root = workspace/'task-assets'
    if manifest.get('task_assets'):
        asset_root.mkdir(parents=True, exist_ok=True); asset_root.chmod(0o700)
        os.environ['AFFINE_MATH_CORPUS_ASSET_ROOT'] = str(asset_root.resolve())
    hydrate_manifest(asset_root, manifest)
    report = dict(schema=1, worker_policy=VERSION, job_id=job['job_id'], role=job['role'], operator=authority,
        job_sha256=hashlib.sha256(canonical(job)).hexdigest(), checkpoint=manifest['checkpoint']['id'],
        epoch=manifest['epoch'], backend_profile=manifest['backend_profile'],
        numerical_policy=manifest['sampling_contract']['calibration'], source_files=job['source_files'],
        runtime_versions=job['runtime_versions'], chain_transactions=False, full_model_finetune=False,
        execution_resources_enforced=False)
    if job['role'] == 'verify':
        from .complete_group_verification import verify
        from .audit_policy import InvalidSample
        owned = runtime_factory is None
        if owned:
            runtime_factory = persistent_factory(workspace/'mimo-cache')
        try:
            if callable(getattr(runtime_factory, 'bind_transport', None)):
                runtime_factory.bind_transport(manifest, job.get('checkpoint_read_urls', manifest['checkpoint'].get('read_urls', {})))
            reports = []
            for index, obj in enumerate(job['submissions']):
                guard()
                path = out/f'submission-{index}.zip'
                try:
                    get_object(obj['url'], obj['sha256'], path, obj['commitment_ref']['size'])
                    data = path.read_bytes(); _preflight(data, manifest, obj)
                    # Persistent factory hydrates once into its bounded cache;
                    # the legacy checkpoint() downloader is never invoked.
                    result = verify(data, manifest, cache, miner=obj['commitment_miner'], runtime_factory=runtime_factory)
                    reports.append(_compact_audit(result, manifest, obj))
                except (ArtifactRejected, InvalidSample) as error:
                    reports.append(_invalid(manifest, obj, error))
            report['audits'] = reports
        finally:
            if owned: runtime_factory.close()
    elif job['role'] == 'train':
        from .mimo_training_worker import execute as execute_training
        report = execute_training(job, manifest, authority, workspace, out, report,
                                  cache=cache, runtime_factory=runtime_factory, assert_lease=guard,
                                  trainer_state_root=trainer_state_root, context=training_context,
                                  allow_public=allow_public)
    elif job['role'] == 'evaluate':
        from .mimo_evaluation import validate_job, evaluate, validate_result
        selection = validate_job(job,manifest)
        definition = selection['definition']
        owned = runtime_factory is None
        if owned:
            runtime_factory = persistent_factory(workspace/'mimo-cache')
        try:
            if callable(getattr(runtime_factory, 'bind_transport', None)):
                runtime_factory.bind_transport(manifest, job.get('checkpoint_read_urls',manifest['checkpoint'].get('read_urls',{})))
            # Only its raw backend is used. The facade's mining authority and
            # canonical nonce sampler remain untouched and cannot sample the
            # eval-only indices. This lease keeps checkpoint bytes/model warm.
            evaluation_identity = hashlib.sha256(canonical(['private-evaluator',manifest['run_id']])).hexdigest()
            with runtime_factory.runtime_for_job(cache,manifest,definition['spec'],definition['harness'],
                                                  miner=evaluation_identity) as facade:
                report['evaluation'] = evaluate(manifest,selection,facade.backend,guard=guard)
            validate_result(report['evaluation'],job,manifest)
        finally:
            if owned: runtime_factory.close()
    else:
        from .storage import Identity
        from .miner import Miner, EpochClosed
        from .complete_group_mining import GroupFiltered, GroupIndeterminate
        filename = Path(job['miner_identity_file'])
        need(not filename.is_symlink() and stat.S_ISREG(filename.stat().st_mode)
             and not filename.stat().st_mode & 0o077, 'private local miner signing seed')
        identity = Identity(bytes.fromhex(filename.read_text().strip()))
        need(identity.id == job['miner_id'], 'local miner signer binding')
        from copy import deepcopy
        transport_manifest = deepcopy(manifest)
        if 'checkpoint_read_urls' in job:
            transport_manifest['checkpoint']['read_urls'] = job['checkpoint_read_urls']
        approved = checkpoint(transport_manifest, workspace, cache)
        actor = Miner(identity, manifest, approved, capability=job['capability'],
            state_path=out/'prepared.json', progress_path=out/'mining-progress.json',
            mimo_replicas=mimo_replicas)
        statuses = []
        try:
            for definition in mining_definitions(manifest, job):
                for index in definition['indices']:
                    if len(actor.batches) >= manifest['max_batches']: break
                    try:
                        actor.search(index, env_id=definition['env_id'], max_attempts=job.get('search_budget'))
                        actor.upload(); status = 'submitted'
                    except GroupFiltered: status = 'filtered-complete-group'
                    except GroupIndeterminate: status = 'native-indeterminate'
                    except EpochClosed: status = 'epoch-closed'
                    statuses.append(dict(env_id=definition['env_id'], index=index, status=status))
                    if status == 'epoch-closed': break
                if len(actor.batches) >= manifest['max_batches'] or statuses and statuses[-1]['status'] == 'epoch-closed': break
            report.update(miner_id=identity.id, submitted_groups=len(actor.batches), tasks=statuses,
                          exact_group_size=actor.batch_contract.group_size)
        finally:
            actor.close()
            backend = getattr(actor.runtime, "backend", None)
            if backend is not None and callable(getattr(backend, "close", None)): backend.close()
    guard()
    source_guard(job)
    from .mimo_runtime import PRECISION_VERSION
    if manifest['mimo_runtime']['version'] == PRECISION_VERSION:
        import torch
        from .mimo_precision import provenance
        # Captured locally, never copied from miner submissions. The ordinary
        # worker signs this entire report; the host checks it after signature
        # authentication. This observes the worker process, not external miners.
        report['execution_provenance'] = provenance(torch,
            manifest['mimo_runtime']['precision_profile'], device=manifest['backend_profile']['device'])
    report.update(success=True, completed_at=time.time())
    need(report['completed_at'] < job['expires_at'], 'job expired before report')
    encoded=canonical(report)
    need(job['role'] not in ('verify','evaluate') or len(encoded) <= 12_000_000, 'bounded audit/evaluation report')
    (out/'report.json').write_bytes(encoded)
    return report


class CheckpointTransportFactory:
    """Refresh job capabilities without changing the original sampling manifest.

    Worker execution is serialized; URLs are transport only and never enter the
    facade binding, model identity or original miner commitment.
    """
    def __init__(self,cache_root,*,max_bytes):
        import threading
        from .mimo_runtime import PersistentMimoRuntimeFactory
        self.checkpoint_id=None;self.urls={};self.local_directory=None
        self._job_lock=threading.Lock()
        self.factory=PersistentMimoRuntimeFactory(cache_root,max_bytes=max_bytes,fetch_file=self._fetch)

    def bind_transport(self,manifest,urls):
        from .backend_jobs import r2_url
        need(self._job_lock.acquire(blocking=False), 'checkpoint transport already in flight')
        try:
            need(type(urls) is dict and (not urls or set(urls)==set(manifest['checkpoint']['files'])),
                 'exact current checkpoint transport inventory')
            for url in urls.values():r2_url(url,'GET')
            self.checkpoint_id=manifest['checkpoint']['id'];self.urls=dict(urls)
        finally:
            self._job_lock.release()

    def _fetch(self,manifest,name,destination):
        from .backend_jobs import get_object
        cp=manifest['checkpoint']
        need(cp['id']==self.checkpoint_id and name in cp['files'], 'original checkpoint transport binding')
        if self.local_directory is not None:
            import shutil
            source=self.local_directory/name
            need(not source.is_symlink() and source.is_file() and 0<source.stat().st_size<=20_000_000_000,
                 'bounded ordinary local checkpoint staging file')
            # CheckpointCache authenticates every copied byte against cp.files
            # before load. Local metadata alone never establishes model trust.
            shutil.copyfile(source,destination)
            return
        need(name in self.urls, 'signed checkpoint read capability required')
        get_object(self.urls[name],cp['files'][name],destination,20_000_000_000)

    @contextmanager
    def runtime_for_job(self,checkpoint,manifest,environment,harness,*,miner):
        need(self._job_lock.acquire(blocking=False), 'checkpoint transport already in flight')
        try:
            need(manifest['checkpoint']['id']==self.checkpoint_id, 'active checkpoint transport binding')
            if checkpoint is not None:
                from .distributed_worker import checkpoint_cache_candidate
                need(checkpoint_cache_candidate(checkpoint,manifest['checkpoint']['files']),
                     'exact ordinary local checkpoint candidate inventory')
                path=Path(checkpoint)
                need(sum((path/name).stat().st_size for name in manifest['checkpoint']['files'])<=64*1024**3,
                     'bounded full local checkpoint candidate')
                self.local_directory=path
            with self.factory.runtime_for_job(checkpoint,manifest,environment,harness,miner=miner) as runtime:
                yield runtime
        finally:
            self.local_directory=None
            self._job_lock.release()

    def close(self):
        self.factory.close();self.urls.clear();self.checkpoint_id=None;self.local_directory=None


def persistent_factory(cache_root, *, max_bytes=128*1024**3):
    return CheckpointTransportFactory(cache_root,max_bytes=max_bytes)


def cleanup_acknowledged_inputs(out):
    """Only after the existing queue acknowledged this exact immutable report."""
    import shutil
    out = Path(out)
    need(out.absolute() == out.resolve() and out.is_dir() and out.stat().st_uid == os.getuid(),
         'ordinary owned completed job directory')
    for path in out.glob('submission-*.zip'):
        need(not path.is_symlink() and path.is_file() and path.stat().st_uid == os.getuid(),
             'ordinary completed audit input')
        path.unlink()
    inputs = out/'training-inputs'
    if inputs.exists() or inputs.is_symlink():
        need(not inputs.is_symlink() and inputs.is_dir() and inputs.stat().st_uid == os.getuid(),
             'ordinary owned completed training inputs')
        for path in inputs.rglob('*'):
            need(not path.is_symlink() and path.stat().st_uid == os.getuid()
                 and (path.is_dir() or path.is_file()), 'ordinary owned input members')
        shutil.rmtree(inputs)
