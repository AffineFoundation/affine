"""Prospective signed complete-group trainer on the existing role worker.

The common MiMo worker authenticates source/runtime/manifest before entering
here. The host signs native grading and exact compact inputs; a run-wide local
lease and original transaction prevent parallel updates and duplicate steps.
This module does not invoke the legacy pair trainer or change public discovery.
"""
from __future__ import annotations
from dataclasses import asdict
import fcntl
import gc
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time
from types import SimpleNamespace

from .mimo_grpo import TrainerConfig, InputBinding, execute_transaction, restore_transaction_state
from .mimo_grpo.data import digest, sha
from .mimo_grpo.optimizer import CPUAdamW
from .mimo_grpo.prepared import prepare_signed_dispatch
from .mimo_grpo.resources import admit_fresh_optimizer, admit_staging
from .mimo_grpo.publication import R2ModelPublisher
from .mimo_grpo.handover import initialize_private_current, private_current_callback
from .mimo_grpo.transaction import Job, RecoveryRequired, _write
from .mimo_grpo.service import training_job


def need(value, message):
    if not value:
        raise ValueError(message)


def validate_job(job, manifest, authority, *, allow_public=False):
    from .backend_jobs import r2_url
    from .mimo_release_policy import policy_for_manifest
    release_policy = policy_for_manifest(manifest, authority, allow_public=allow_public)
    t = job.get('training')
    need(type(t) is dict and set(t) == {'run_id','optimizer_step','config','dispatch','documents',
        'publication_namespace','publisher_id','role_config_file'}, 'explicit MiMo training job fields')
    need(t['run_id'] == manifest.get('run_id') and type(t['run_id']) is str and
         re.fullmatch(r'[A-Za-z0-9_-]{1,100}', t['run_id']), 'safe authenticated trainer run')
    need(type(t['optimizer_step']) is int and t['optimizer_step'] >= 0, 'original optimizer step')
    config = TrainerConfig(**t['config']['settings']).validate()
    need(config.document() == t['config'] and config.group_size == manifest['group_contract']['group_size'],
         'exact explicit MiMo objective and complete-group size')
    sampler = manifest['mimo_runtime']
    need((config.temperature, config.top_p, config.top_k) ==
         (sampler['temperature'], sampler['top_p'], sampler['top_k']), 'trainer reference sampling parameters')
    need(sha(t['publisher_id']) and type(t['publication_namespace']) is str,
         'scoped publisher identity/namespace')
    if release_policy is None:
        need(t['publication_namespace'].startswith('private/'), 'private scoped publication namespace')
    else:
        need(t['publication_namespace'] == release_policy.namespace(manifest),
             'publication namespace differs from qualified release policy')
    need(type(t['role_config_file']) is str and Path(t['role_config_file']).is_absolute(), 'private role configuration path')
    d = t['dispatch']
    need(type(d) is dict and set(d) == {'url','sha256','size'} and sha(d['sha256']) and
         type(d['size']) is int and 0 < d['size'] <= 256 * 1024**2, 'bounded signed native dispatch')
    r2_url(d['url'], 'GET')
    need(type(t['documents']) is dict and len(t['documents']) <= config.max_tasks, 'bounded original compact documents')
    for checksum, row in t['documents'].items():
        need(sha(checksum) and type(row) is dict and set(row) == {'url','size'} and
             type(row['size']) is int and 0 < row['size'] <= 128 * 1024**2, 'compact input hash/capability')
        r2_url(row['url'], 'GET')
    return config


def validate_result(report, job, manifest):
    """Host-side original trainer report and scoped signed receipt validation."""
    from .backend_jobs import signed
    from .mimo_grpo.publication import publication_verified
    from .mimo_runtime import validate_descriptor
    t = job['training']
    training = report.get('training')
    need(type(training) is dict and training.get('training_requires_sampling_audit') is False,
         'explicit independent trainer report')
    if training.get('status') == 'no-update':
        need(training.get('reason') == 'no-eligible-complete-groups' and
             not t['documents'] and
             training.get('group_count') == training.get('rollout_count') == 0 and
             training.get('input_checkpoint') == training.get('output_checkpoint') == manifest['checkpoint']['id'] and
             training.get('optimizer_step') == t['optimizer_step'] and 'publication_receipt' not in report,
             'empty epoch cannot advance model or optimizer')
        return
    need(training.get('status') == 'updated' and training.get('config') == t['config'] and
         training.get('input') == asdict(_input_binding(job, manifest)) and
         training.get('input_checkpoint') == manifest['checkpoint']['id'] and
         training.get('optimizer_step_before') == t['optimizer_step'] and
         training.get('optimizer_step') == t['optimizer_step'] + 1 and
         training.get('model_only_publication') is True and
         type(training.get('group_count')) is int and 1 <= training['group_count'] <= t['config']['settings']['max_tasks'] and
         training['group_count'] == len(t['documents']) and
         training.get('rollout_count') == training['group_count'] * manifest['group_contract']['group_size'],
         'actual original one-step complete-group training report')
    original = {k: v for k, v in training.items() if k not in
                ('output_checkpoint', 'training_requires_sampling_audit', 'model_only_publication')}
    envelope = report.get('publication_receipt')
    published = signed(envelope, t['publisher_id'])
    source = digest(job['source_files'])
    expected_job_id = 'mimo-' + digest(dict(input=training['input'],
        documents=training['input_documents_sha256'], config=training['config']))
    need(publication_verified(published) and published.get('job_id') == expected_job_id and
         published.get('epoch') == manifest['epoch'] and
         published.get('input_checkpoint') == manifest['checkpoint']['id'] and
         published.get('output_checkpoint') == training.get('output_checkpoint') == report.get('output_checkpoint') and
         published.get('optimizer_step') == t['optimizer_step'] + 1 and
         published.get('report_sha256') == digest(original) == report.get('transaction_report_sha256') and
         published.get('source_closure_sha256') == source and
         published.get('optimizer_state_uploaded') is False and published.get('authority_committed') is False,
         'signed model publication must bind the original trainer source/report/checkpoint')
    descriptor = validate_descriptor(published['descriptor'])
    need(descriptor['id'] == report['output_checkpoint'], 'actual output descriptor')
    objects = published['objects']
    need(type(objects) is dict and set(objects) == set(descriptor['files']), 'full published model inventory')
    for name, expected in descriptor['files'].items():
        row = objects[name]
        need(row['sha256'] == expected and type(row['size']) is int and row['size'] > 0 and
             row['key'] == t['publication_namespace'] + '/checkpoints/' + descriptor['id'] + '/model/' + name,
             'published model paths are within the authorized namespace')


class TrainingContext:
    """Optional warm actor/Adam state across jobs; durable journal remains truth."""
    def __init__(self):
        self.runtime = None
        self.optimizer = None
        self.run = None
        self.source = None
        self.reference = None
    def close(self):
        if self.reference is not None:
            close = getattr(self.reference, 'close', None)
            if callable(close):
                close()
            self.reference = None
        if self.runtime is not None:
            close = getattr(self.runtime, 'close', None)
            if callable(close):
                close()
        self.runtime = self.optimizer = None
        gc.collect()


def _private_json(path):
    path = Path(path)
    need(not path.is_symlink() and path.is_file() and path.stat().st_uid == os.getuid() and
         not path.stat().st_mode & 0o077 and path.stat().st_size <= 1024**2,
         'owner-only ordinary trainer configuration')
    return json.loads(path.read_bytes())


def _role_configuration(job):
    from .storage import Bucket, Identity
    value = _private_json(job['training']['role_config_file'])
    need(set(value) <= {'bucket','publisher_identity_file','initial_checkpoint_directory',
         'release_succession','source_succession'} and
         {'bucket','publisher_identity_file'} <= set(value), 'trainer role configuration fields')
    key = Path(value['publisher_identity_file'])
    need(key.is_absolute() and not key.is_symlink() and stat.S_ISREG(key.stat().st_mode) and
         key.stat().st_uid == os.getuid() and not key.stat().st_mode & 0o077,
         'private scoped publisher signing key')
    identity = Identity(bytes.fromhex(key.read_text().strip()))
    need(identity.id == job['training']['publisher_id'], 'configured publisher identity differs from signed job')
    return Bucket(value['bucket']), identity, value.get('initial_checkpoint_directory')


def _journal_job(value):
    row = dict(value['job'])
    row['prepared_documents_sha256'] = tuple(row['prepared_documents_sha256'])
    return Job(**row)


def _input_binding(job, manifest):
    return InputBinding(job['training']['run_id'], manifest['epoch'], manifest['checkpoint']['id'],
        digest(manifest['group_contract']), job['training']['optimizer_step'])


def _release_succession(job, authority, *, allow_public=False):
    value = _private_json(job['training']['role_config_file'])
    envelope = value.get('release_succession')
    if envelope is None:
        return None
    from .mimo_release_succession import ReleaseChain
    return ReleaseChain(envelope, authority, allow_public=allow_public)


SOURCE_SUCCESSION_VERSION = 'mimo-trainer-source-succession-v1'
SOURCE_SUCCESSION_FIELDS = {'version', 'run_id', 'release_policy_sha256', 'from_source_closure_sha256',
                            'to_source_closure_sha256', 'checkpoint', 'optimizer_step'}


def _source_succession(job, authority):
    value = _private_json(job['training']['role_config_file'])
    envelope = value.get('source_succession')
    if envelope is None:
        return None
    from .backend_jobs import signed
    document = signed(envelope, authority)
    need(type(document) is dict and set(document) == SOURCE_SUCCESSION_FIELDS and
         document['version'] == SOURCE_SUCCESSION_VERSION, 'exact signed trainer source succession')
    return document


def _adopt_source_succession(current_path, current, job, authority, active, source, binding):
    """Within one release, re-stamp only the trainer source closure when an
    operator-signed source succession in the private role config names this
    exact run, release, old/new closure and the job's input checkpoint/step.
    Fail-closed: without it, or on any mismatch, lineage checks reject as before.
    """
    if current.get('source_closure_sha256') == source:
        return current
    succession = _source_succession(job, authority)
    if succession is None:
        return current
    need(succession['run_id'] == binding.run and succession['release_policy_sha256'] == active and
         succession['from_source_closure_sha256'] == current.get('source_closure_sha256') and
         succession['to_source_closure_sha256'] == source and
         succession['checkpoint'] == binding.checkpoint and
         succession['optimizer_step'] == binding.optimizer_step and
         current.get('run') == binding.run and current.get('checkpoint') == binding.checkpoint and
         current.get('optimizer_step') == binding.optimizer_step,
         'signed source succession must bind this exact trainer lineage')
    _write(current_path, dict(current, source_closure_sha256=source))
    return json.loads(current_path.read_bytes())


def _adopt_release_succession(current_path, current, job, authority, release_policy,
                             source, binding, *, allow_public=False):
    """Re-stamp a trainer private CURRENT governed by a predecessor release onto
    the active successor, preserving the optimizer/model lineage, when an
    operator-signed release chain in the private role config authorizes the switch
    and the cursor sits exactly at this job's input checkpoint/step. Fail-closed:
    without a chain, or any mismatch, the lineage checks reject exactly as before.
    """
    bound = current.get('release_policy_sha256')
    active = release_policy.policy_sha256 if release_policy is not None else None
    if bound == active:
        return _adopt_source_succession(current_path, current, job, authority, active, source, binding)
    need(release_policy is not None, 'successor release requires a qualified release policy')
    chain = _release_succession(job, authority, allow_public=allow_public)
    need(chain is not None, 'release switch requires a signed release succession in the role config')
    need(chain.run_id == binding.run, 'release succession run differs from trainer job')
    need(chain.succeeds(bound, active),
         'trainer state release does not precede the active release in the chain')
    need(current.get('run') == binding.run and current.get('checkpoint') == binding.checkpoint
         and current.get('optimizer_step') == binding.optimizer_step,
         'release succession must extend the durable current optimizer/model lineage')
    adopted = dict(current, release_policy_sha256=active,
                   source_closure_sha256=source, execution_mode=release_policy.mode)
    _write(current_path, adopted)
    return json.loads(current_path.read_bytes())


def execute(job, manifest, authority, workspace, out, report, *, cache=None,
            runtime_factory=None, context=None, bucket=None, signer=None,
            load_runtime=None, download=None, resource_admission=None,
            initial_checkpoint_directory=None, assert_lease=None, trainer_state_root=None,
            allow_public=False, load_reference_runtime=None):
    """Execute a validated job, returning fields for the common signed report.

    Injection arguments support isolated integration tests and a warm worker.
    Production defaults use the actual MiMo runtime, scoped Bucket and signed
    compact-input capabilities. Credentials are never copied into report state.
    """
    from .backend_jobs import get_object, signed
    from .mimo_worker import source_guard
    from .mimo_runtime import MimoRuntime
    from .mimo_release_policy import policy_for_manifest
    config = validate_job(job, manifest, authority, allow_public=allow_public)
    release_policy = policy_for_manifest(manifest, authority, allow_public=allow_public)
    need(callable(assert_lease), 'queue-owned trainer lease assertion required')
    assert_lease()
    source = source_guard(job)
    binding = _input_binding(job, manifest)
    download = get_object if download is None else download
    load_runtime = (lambda path, desc: MimoRuntime.load(path, desc, manifest['mimo_runtime'],
                    device='cuda', trainable=True)) if load_runtime is None else load_runtime
    resource_admission = admit_fresh_optimizer if resource_admission is None else resource_admission
    owned_context = context is None
    context = TrainingContext() if context is None else context
    stable = Path(trainer_state_root).absolute() if trainer_state_root is not None else Path(workspace).absolute()/'mimo-training'
    root = stable/binding.run
    need(root.resolve() == root, 'ordinary isolated training run path')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    need(root.stat().st_uid == os.getuid() and not root.stat().st_mode & 0o077, 'owner-only trainer run state')
    lock = os.open(root/'.trainer.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(lock)
        raise
    def guard():
        assert_lease()
        os.fstat(lock)
        need(time.time() < job['expires_at'], 'original signed trainer deadline')
        if context.runtime is not None:
            context.runtime.guard()
    out = Path(out)
    try:
        guard()
        inputs = out/'training-inputs'
        inputs.mkdir(parents=True, exist_ok=True, mode=0o700)
        def fetch(obj, path):
            from .mimo_runtime import file_hash
            if path.exists():
                need(not path.is_symlink() and path.stat().st_size == obj['size'] and
                     file_hash(path) == obj['sha256'], 'existing original compact input changed')
            else:
                download(obj['url'], obj['sha256'], path, obj['size'])
                need(path.stat().st_size == obj['size'], 'exact committed input size')
        dispatch_path = inputs/'DISPATCH.json'
        fetch(job['training']['dispatch'], dispatch_path)
        dispatch_envelope = json.loads(dispatch_path.read_bytes())
        dispatch = signed(dispatch_envelope, authority)
        need(dispatch['epoch'] == manifest['epoch'] and dispatch['run_id'] == binding.run and
             dispatch['checkpoint'] == binding.checkpoint and dispatch['manifest_sha256'] == digest(manifest),
             'original signed dispatch opening binding')
        expected_documents = {row['document']['sha256'] for row in dispatch['selected']}
        need(set(job['training']['documents']) == expected_documents, 'exact signed selected document capability set')
        for checksum, obj in job['training']['documents'].items():
            fetch(dict(obj, sha256=checksum), inputs/(checksum + '.json'))
        shell = SimpleNamespace(checkpoint_id=binding.checkpoint,
            spec=SimpleNamespace(to_dict=lambda: manifest['mimo_runtime']),
            vocab_size=manifest['group_transport_policy']['vocab_size'],
            eos_token_ids=tuple(manifest['mimo_runtime']['eos_token_ids']))
        prepared = prepare_signed_dispatch(job['manifest'], dispatch_envelope, authority=authority,
            documents_root=inputs, runtime=shell, optimizer_step=binding.optimizer_step)
        current_path = root/'CURRENT.json'
        if not current_path.exists():
            need(binding.optimizer_step == 0, 'new run must start with a fresh optimizer')
            initialize_private_current(current_path, binding, source_closure_sha256=source,
                                       release_policy=release_policy, manifest=manifest)
        current = json.loads(current_path.read_bytes())
        current = _adopt_release_succession(current_path, current, job, authority,
                                            release_policy, source, binding, allow_public=allow_public)
        need(current.get('release_policy_sha256') ==
             (release_policy.policy_sha256 if release_policy is not None else None),
             'trainer state belongs to a different qualified release')
        transaction_root = root/'history'/('update-' + digest(dict(epoch=binding.epoch, checkpoint=binding.checkpoint)))
        original_path = transaction_root/'TRANSACTION.json'
        original = json.loads(original_path.read_bytes()) if original_path.exists() else None
        expected_job = None
        if prepared['groups']:
            expected_job = training_job(prepared['groups'], binding, config,
                manifest_sha256=digest(manifest), source_closure_sha256=source)
            if original:
                need(_journal_job(original) == expected_job, 'journal differs from authenticated original dispatch')
        # A retry may arrive after CURRENT already points at this job's output.
        same_update = bool(original and current.get('job_id') == original.get('job', {}).get('job_id'))
        need(current['run'] == binding.run and current['source_closure_sha256'] == source and
             (same_update or (current['checkpoint'] == binding.checkpoint and
                              current['optimizer_step'] == binding.optimizer_step)),
             'training job must extend durable current optimizer/model lineage')
        if not dispatch['selected']:
            # Authenticate the empty native dispatch without loading a model.
            return dict(report, full_model_finetune=True, training=dict(status='no-update',
                reason='no-eligible-complete-groups', group_count=0, rollout_count=0,
                input_checkpoint=binding.checkpoint, output_checkpoint=binding.checkpoint,
                optimizer_step=binding.optimizer_step, training_requires_sampling_audit=False))
        if bucket is None or signer is None:
            need(bucket is None and signer is None, 'provide both scoped bucket and publisher identity')
            bucket, signer, configured_input = _role_configuration(job)
            initial_checkpoint_directory = configured_input or initial_checkpoint_directory
        need(signer.id == job['training']['publisher_id'], 'publisher role key differs from authorized job')
        publisher = R2ModelPublisher(bucket, signer, namespace=job['training']['publication_namespace'],
            source_closure_sha256=source, assert_trainer_lease=guard)
        if context.run != binding.run or context.source != source:
            context.close()
            context.run, context.source = binding.run, source
        # Original complete staging wins over a stale CURRENT after a crash.
        restore_original = original and original['state'] in ('staged', 'published')
        if original and original['state'] == 'updating':
            raise RecoveryRequired('original update interrupted; restore original evidence explicitly')
        desired = original.get('output_checkpoint') if restore_original else current['checkpoint']
        if restore_original and desired is None:
            desired = digest({n.removeprefix('model/'): row['sha256'] for n, row in original['artifacts'].items()
                              if n.startswith('model/')})
        if context.runtime is not None and (context.runtime.checkpoint_id != desired or context.optimizer.failed):
            context.close()
        if context.runtime is None:
            if restore_original:
                context.runtime, context.optimizer, _ = restore_transaction_state(transaction_root,
                    expected_job=expected_job, load_runtime=load_runtime,
                    config=config, assert_trainer_lease=guard)
            elif current['optimizer_step']:
                matches = []
                for candidate in (root/'history').glob('*/TRANSACTION.json'):
                    row = json.loads(candidate.read_bytes())
                    if row.get('state') == 'published' and row.get('job', {}).get('job_id') == current.get('job_id'):
                        matches.append((candidate.parent, row))
                need(len(matches) == 1, 'one original current optimizer generation required')
                folder, journal = matches[0]
                context.runtime, context.optimizer, _ = restore_transaction_state(folder,
                    expected_job=_journal_job(journal), load_runtime=load_runtime,
                    config=config, assert_trainer_lease=guard)
            else:
                descriptor = {k: manifest['checkpoint'][k] for k in ('id','files')}
                if initial_checkpoint_directory:
                    approved = Path(initial_checkpoint_directory).resolve()
                else:
                    approved = root/'initial-checkpoint'/binding.checkpoint
                    approved.mkdir(parents=True, exist_ok=True, mode=0o700)
                    for name, checksum in descriptor['files'].items():
                        from .mimo_runtime import file_hash
                        path = approved/name
                        if path.exists():
                            need(not path.is_symlink() and file_hash(path) == checksum, 'original initial checkpoint bytes')
                        else:
                            read_urls = job.get('checkpoint_read_urls', manifest['checkpoint'].get('read_urls', {}))
                            need(name in read_urls, 'authenticated initial checkpoint read capability')
                            download(read_urls[name], checksum, path, 20_000_000_000)
                context.runtime = load_runtime(approved, descriptor)
                resource_admission(context.runtime, root)
                context.optimizer = CPUAdamW(context.runtime.model.named_parameters(), config,
                                               input_checkpoint=binding.checkpoint)
        runtime, optimizer = context.runtime, context.optimizer
        need(runtime.spec.to_dict() == manifest['mimo_runtime'] and
             runtime.eos_token_ids == shell.eos_token_ids,
             'actual trainer runtime differs from authenticated projection geometry')
        # Tokenizer support excludes the padded rows in the actual LM head.
        # Authenticate both with the shared mining/evaluation geometry check;
        # never require their sizes to be equal or expand admitted token IDs.
        from .complete_group_sampling import validate_geometry, token_geometry
        validate_geometry(runtime, {'token_geometry': token_geometry(manifest['group_transport_policy'])})
        head_config = getattr(runtime.model.config, 'text_config', runtime.model.config)
        need(runtime.vocab_size == head_config.vocab_size,
             'trainer scoring vocabulary differs from the actual pinned model head')
        if not restore_original:
            admit_staging(runtime, root)
        reference_runtime = None
        if config.kl_coefficient > 0:
            # Frozen KL anchor: the cutover base checkpoint named in the signed
            # config. Its bytes must be pre-staged at <run>/kl-reference/<id>;
            # the descriptor is rebuilt from file hashes and must match the id.
            if context.reference is None or context.reference.checkpoint_id != config.kl_reference_checkpoint:
                if context.reference is not None:
                    context.reference.close()
                    context.reference = None
                directory = root/'kl-reference'/config.kl_reference_checkpoint
                need(directory.is_dir() and not directory.is_symlink(), 'pre-staged KL reference checkpoint')
                from .mimo_runtime import file_hash
                files = {p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file()}
                need(digest(files) == config.kl_reference_checkpoint, 'KL reference bytes match signed id')
                loader = load_reference_runtime or (lambda path, desc: MimoRuntime.load(path, desc,
                    manifest['mimo_runtime'], device='cuda', trainable=False))
                context.reference = loader(directory, dict(id=config.kl_reference_checkpoint, files=files))
            reference_runtime = context.reference
        event_path = out/'training-events.jsonl'
        with event_path.open('a') as events:
            def record(row):
                events.write(json.dumps(dict(observed_at=time.time(), **row), sort_keys=True) + '\n')
                events.flush()
            result = execute_transaction(transaction_root, runtime, optimizer,
                prepared['groups'], binding, config=config, manifest_sha256=digest(manifest),
                source_closure_sha256=source, assert_trainer_lease=guard, publish_model=publisher,
                commit_current=private_current_callback(current_path, publisher, run=binding.run,
                    assert_trainer_lease=guard, release_policy=release_policy, manifest=manifest),
                retire_published_models=True, record=record, reference_runtime=reference_runtime)
        # An initial download owned by this run is obsolete only after CURRENT
        # points to a fully read-back export and runtime guard references it.
        initial_root = root/'initial-checkpoint'
        if initial_root.exists():
            need(not initial_root.is_symlink(), 'ordinary owned initial checkpoint cache')
            shutil.rmtree(initial_root)
        receipt = publisher.receipt(result['job']['job_id'])
        source_guard(job)
        return dict(report, full_model_finetune=True, execution_resources_enforced=True,
            training=dict(result['report'], output_checkpoint=result['output_checkpoint'],
                training_requires_sampling_audit=False, model_only_publication=True),
            checkpoint=manifest['checkpoint']['id'], output_checkpoint=result['output_checkpoint'],
            publication_receipt=receipt, transaction_report_sha256=result['report_sha256'])
    finally:
        if owned_context:
            context.close()
        os.close(lock)
