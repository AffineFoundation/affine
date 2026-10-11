"""Prospective coordinator integration for variable-dose positive SFT.

Operator-side only. Existing fixed-step execution releases and worker/evidence
paths stay exact. Install under the existing singleton learner lock, never from
unsigned miner or worker metadata. This is not an active deployment.
"""
import copy
import json
import time
from pathlib import Path

from ops import positive_sft_variable_dose as f
from subnet import unaudited_training_execution as execution
from subnet.committed_training_inputs import admitted_submission, receipt_inventory
from subnet.distributed_roles import authenticate
from subnet.persistent_training_protocol import validate_binding

BANK_VERSION = 'positive-sft-retained-variable-dose-release-bank-v1'
ROUTE_VERSION = 'positive-sft-selected-fixed-release-route-v1'


def bank(envelope, authority, *, now):
    value = authenticate(envelope, authority)
    f.require(set(value) == {'version', 'capacity', 'releases'}, 'exact release bank')
    f.require(value['version'] == BANK_VERSION, 'explicit variable-dose release bank')
    cap = f.policy(value['capacity'])
    releases = value['releases']
    f.require(type(releases) is list and len(releases) == cap['max_steps'],
              'one separately qualified release for each supported update count')
    baseline = None
    for k, signed in enumerate(releases, 1):
        r = execution.release(signed, authority, now=now)
        f.require(r['version'] == execution.SFT_RELEASE and r['method'] == execution.SFT_METHOD
                  and r['steps'] == k and r['minimum_optimizer_step'] >= 1,
                  'exact retained-parent fixed-step SFT release')
        # All releases share the same objective, runtime, source, lineage and
        # activation interval. Only their authorized step count may vary.
        invariant = {key: val for key, val in r.items() if key != 'steps'}
        if baseline is None:
            baseline = invariant
        else:
            f.require(invariant == baseline, 'release bank changes more than update count')
    return value, baseline


def pair_snapshot(controller, manifest, submissions, public, native_documents, document_paths):
    """Reconstruct actual selected documents after authenticated native grading.

    The existing native derivation validator proves final selected inventory.
    Missing/invalid files raise; they never create an empty training population.
    No outcomes, prompts, tokens or miner claims are rewritten or regraded here.
    """
    authority = controller.authority.id
    signed_manifest = controller.signed(manifest)
    binding = validate_binding(manifest.get('trainer_state_binding'), manifest)
    f.require(binding['parent'] is not None and binding['genesis'] is None
              and binding['global_step_before'] >= 1, 'retained SFT parent only')
    native = execution.preparation_scope(public, signed_manifest, submissions,
                                         native_documents, authority)
    f.require(submissions and len(document_paths) == len(submissions),
              'native nonempty complete input population')
    f.require(len({obj['sha256'] for obj in submissions}) == len(submissions),
              'each native-selected document appears once')
    pairs = []
    for path, obj in zip(document_paths, submissions):
        _, actual = admitted_submission(path, obj, manifest, authority, retire=False)
        pairs.extend(actual)
    snapshot = dict(version=f.ADMISSION_VERSION, epoch=manifest['epoch'],
        checkpoint=manifest['checkpoint']['id'], manifest_sha256=f.sha(signed_manifest),
        admitted_pairs_sha256=f.sha(pairs), pair_sha256=[f.pair_identity(p) for p in pairs],
        complete=True, pending_input_count=0, infrastructure_error_count=0,
        native_admission_sha256=f.sha(native), sampling_assurance='unaudited',
        proof_verification_performed=False)
    return pairs, signed_manifest, controller.signed(snapshot), native


def _original_documents(controller, manifest, submissions):
    """Read existing signed native derivation and owned selected input copies."""
    from ops.native_training_lifecycle import directory, document_path, verify_owned_document
    root = directory(controller, manifest['epoch'])
    if 'training_representative_policy' in manifest:
        from ops.native_task_representative_selection import read_documents
        documents = read_documents(root)
        candidates = sorted((root/'waves').iterdir())
    else:
        documents = {name: json.loads(f._read(root/(name+'.ROOT-SIGNED.json')))
                     for name in ('context', 'grades', 'subset')}
        candidates = [root]
    paths = []
    for obj in submissions:
        matches = [wave for wave in candidates if document_path(wave, obj['sha256']).exists()]
        f.require(len(matches) == 1, 'one original owned copy for every selected document')
        verify_owned_document(controller, matches[0], obj['sha256'])
        paths.append(document_path(matches[0], obj['sha256']))
    public = json.loads(f._read(Path(controller.state)/(manifest['epoch']+'-first-signed-manifest.json')))
    return public, documents, paths


def original_dispatch(controller, manifest, submissions):
    record_path = Path(controller.state)/'roles'/(manifest['epoch']+'-train.json')
    if not record_path.exists():
        return None
    record = json.loads(f._read(record_path))
    if record.get('prepared_only') is True:
        return None
    job_id = record.get('job_id')
    f.require(type(job_id) is str and Path(job_id).name == job_id
              and job_id not in ('.', '..'), 'original issued job identifier')
    envelope = json.loads(f._read(record_path.parent/(job_id+'-job.json')))
    job = authenticate(envelope, controller.authority.id)
    declared = execution.validate(envelope, controller.authority.id)
    f.require(job['role'] == 'train' and job['job_id'] == job_id
              and sha_job(job) == record.get('job_sha256')
              and authenticate(job['manifest'], controller.authority.id) == manifest
              and receipt_inventory(job['submissions']) == receipt_inventory(submissions)
              and declared['steps'] == job['steps'], 'original issued job exact inputs and parent')
    return dict(created_at=job['created_at'], steps=job['steps'],
                release_sha256=declared['execution_release_sha256'])


def sha_job(job):
    return f.sha(job)


def original_zero_closure(controller, manifest, submissions, bank_envelope):
    """Only a fully published original zero closure may outlive its release.

    A local prepared receipt alone is insufficient. Re-read its exact signed
    durable object, check the unchanged parent, and bind the original bank and
    input inventory. The normal route reconstruction still runs afterwards.
    """
    root = Path(controller.state)/(manifest['epoch']+'-positive-sft-fallback')
    receipt_path = root/'no-update.ROOT-SIGNED.json'
    if not receipt_path.exists():
        return None
    authority = controller.authority.id
    receipt = json.loads(f._read(receipt_path))
    value = authenticate(receipt, authority)
    dispatch = authenticate(json.loads(f._read(root/'dispatch.ROOT-SIGNED.json')), authority)
    signed_route = json.loads(f._read(root/'route.ROOT-SIGNED.json'))
    route = authenticate(signed_route, authority)
    status = json.loads(f._read(Path(controller.state)/'controller.json'))
    pointer_bytes = f._parent(controller, manifest, status)
    signed_manifest = controller.signed(manifest)
    f.require(dispatch['version'] == ROUTE_VERSION and dispatch['epoch'] == manifest['epoch']
              and dispatch['steps'] == 0 and dispatch['selected_release_sha256'] is None
              and dispatch['bank_sha256'] == f.sha(bank_envelope)
              and dispatch['manifest_sha256'] == f.sha(signed_manifest)
              and dispatch['input_inventory_sha256'] == f.sha(receipt_inventory(submissions))
              and dispatch['route'] == route, 'original committed zero dispatch bindings')
    f.require(value['version'] == f.NO_UPDATE_VERSION and value['epoch'] == manifest['epoch']
              and value['route_sha256'] == f.sha(signed_route)
              and value['manifest_sha256'] == f.sha(signed_manifest)
              and value['status'] == 'closed_positive_sft_no_update'
              and value['training_dispatched'] is False and value['training_succeeded'] is False
              and value['training_report'] is None and value['losses'] is None
              and value['optimizer_updates'] == 0
              and value['checkpoint'] == value['next_checkpoint'] == manifest['checkpoint']['id']
              and value['trainer_state'] == status['trainer_state']
              and value['optimizer_steps'] == manifest['trainer_state_binding']['global_step_before']
              and value['optimizer_pointer_sha256'] == __import__('hashlib').sha256(pointer_bytes).hexdigest()
              and value['training_steps_before'] == value['training_steps_after'] == status['training_steps'],
              'original zero closure keeps exact checkpoint and optimizer parent')
    data = f.canonical(receipt)
    try:
        stored = controller.bucket.get_bounded('public/'+manifest['epoch']+'/positive-sft-no-update.json',
                                               limit=len(data))
    except KeyError:
        return None
    except Exception as error:
        from botocore.exceptions import ClientError
        if isinstance(error, ClientError) and str(error.response.get('Error', {}).get('Code')) in ('NoSuchKey', '404', 'NotFound'):
            return None
        raise
    f.require(stored == data, 'original zero closure full durable readback')
    return dict(no_update_receipt_sha256=f.sha(receipt))


def prepare_route(controller, manifest, submissions, bank_envelope, *, now=None,
                  original_documents=_original_documents):
    now = time.time() if now is None else now
    authority = controller.authority.id
    issued = original_dispatch(controller, manifest, submissions)
    closed_zero = (original_zero_closure(controller, manifest, submissions, bank_envelope)
                   if issued is None else None)
    # A signed job already dispatched under an in-lifetime release must be
    # allowed to finish/reuse its original metrics after that release expires.
    # Unissued and prepare-only jobs retain current-time admission checks.
    admission_time = issued['created_at'] if issued is not None else (None if closed_zero else now)
    released, common = bank(bank_envelope, authority, now=admission_time)
    f.require(manifest.get('training_objective') == common['training_objective'],
              'manifest declares the independently qualified positive objective')
    f.require(manifest.get('K') == manifest.get('L'), 'unchanged complete K/L contract')
    public, documents, paths = original_documents(controller, manifest, submissions)
    pairs, signed_manifest, admission, native = pair_snapshot(controller, manifest,
        submissions, public, documents, paths)
    # This routing scope is narrow metadata. It does not qualify source/runtime
    # or authorize optimization; the selected existing v6 release does that.
    scope = controller.signed(dict(version=f.RELEASE_VERSION,
        manifest_sha256=f.sha(signed_manifest),
        training_source_sha256=manifest['trainer_state_binding']['source_sha256'],
        objective=common['training_objective'], seed=manifest['training_coverage']['seed'],
        eos_token_id=common['tokenizer_eos_token_id'], required_pairs_per_task=manifest['K'],
        capacity=released['capacity']))
    route = f.route_proposal(pairs, signed_manifest, admission, scope, authority)
    steps = route['decision']['steps']
    selected = released['releases'][steps-1] if steps else None
    if closed_zero is not None:
        f.require(steps == 0 and selected is None, 'committed zero closure cannot become an optimizer job')
    if issued is not None:
        f.require(steps == issued['steps'] and selected is not None
                  and f.sha(selected) == issued['release_sha256'],
                  'already-dispatched original release/count cannot be replaced')
    binding = validate_binding(manifest['trainer_state_binding'], manifest)
    common_release = common
    # Apply retained lineage restrictions even on EOS-empty closure. An empty
    # batch must not let an unrelated checkpoint/source/genesis enter the bank.
    import re
    prefix = common_release['epoch_prefix']
    suffix = manifest['epoch'][len(prefix):] if manifest['epoch'].startswith(prefix) else ''
    f.require(re.fullmatch(r'[0-9]+-[0-9]+', suffix) is not None
              and int(suffix.rsplit('-', 1)[1]) >= common_release['first_round']
              and binding['global_step_before'] >= common_release['minimum_optimizer_step']
              and binding['genesis_sha256'] == common_release['genesis_sha256']
              and f.sha(binding['hyperparameters']) == common_release['hyperparameters_sha256']
              and manifest['source_bundle']['sha256'] == common_release['original_source_bundle_sha256']
              and f.sha(manifest.get('training_runtime')) == common_release['training_runtime_sha256'],
              'complete retained source/runtime/lineage release scope')
    payload = dict(version=ROUTE_VERSION, epoch=manifest['epoch'],
        manifest_sha256=f.sha(signed_manifest), input_inventory_sha256=f.sha(receipt_inventory(submissions)),
        bank_sha256=f.sha(bank_envelope), route=route,
        selected_release_sha256=None if selected is None else f.sha(selected),
        steps=steps, scientific_fixed512_treatment_complete=route['decision']['task_count'] == 512,
        variable_dose=route['decision']['task_count'] != 512,
        no_historical_backfill=True, no_task_repetition=True,
        native_derivation_sha256=f.sha(native))
    root = Path(controller.state)/(manifest['epoch']+'-positive-sft-fallback')
    root.mkdir(mode=0o700, exist_ok=True)
    f.require(root.is_dir() and not root.is_symlink(), 'owned fallback route directory')
    for name, envelope in (('admission', admission), ('scope', scope),
                           ('dispatch', controller.signed(payload))):
        f._create(root/(name+'.ROOT-SIGNED.json'), envelope)
    return dict(payload=payload, pairs=pairs, manifest=signed_manifest,
                admission=admission, scope=scope, selected_release=selected)


def trainer_backend(controller):
    jobs = controller.jobs
    if hasattr(jobs, 'roles'):
        jobs = jobs.roles['train']
    return jobs


def install(controller, bank_envelope, *, original_documents=_original_documents):
    """Install on an isolated prospective learner under its normal singleton lock.

    Native-empty/indeterminate still takes the earlier existing handler. Only a
    finalized native nonempty population reaches this EOS-aware route.
    """
    # Installation validates signed structure historically; only prepare_route
    # admits new work at current time. This permits original-job completion
    # after its issuance release expires, including process reentry.
    _, common = bank(bank_envelope, controller.authority.id, now=None)
    trainer = trainer_backend(controller)
    f.require(trainer.metadata['source_files'] == common['execution_source_files']
              and trainer.metadata['runtime_versions'] == common['runtime_versions']
              and trainer.config.get('unaudited_training_execution') is None,
              'installed original dispatcher matches the qualified release source/runtime')
    f.require(not hasattr(controller, 'positive_sft_dispatch_route'), 'one SFT route installer')
    previous_completion = getattr(controller, 'native_completion_fields', lambda epoch: {})

    def route(manifest, submissions, requested_steps):
        f.require(type(requested_steps) is int and requested_steps > 0, 'ordinary configured step ceiling')
        prepared = prepare_route(controller, manifest, submissions, bank_envelope,
                                 original_documents=original_documents)
        p = prepared['payload']
        if p['steps'] == 0:
            status = json.loads(f._read(Path(controller.state)/'controller.json'))
            result = f.close_no_update(controller, prepared['pairs'], prepared['manifest'],
                prepared['admission'], prepared['scope'], status)
            return dict(no_update=result, steps=0)
        # The normal run/prepare path authenticates this exact release and
        # creates a job-bound LR grant for precisely the reconstructed steps.
        f.require(trainer_backend(controller) is trainer, 'original installed trainer backend')
        trainer.config['unaudited_training_execution_release'] = prepared['selected_release']
        f.require(trainer.config.get('unaudited_training_execution') is None,
                  'one original execution authority')
        return dict(no_update=None, steps=p['steps'])

    def completion(epoch):
        root = Path(controller.state)/(epoch+'-positive-sft-fallback')
        path = root/'dispatch.ROOT-SIGNED.json'
        if not path.exists():
            return previous_completion(epoch)
        p = authenticate(json.loads(f._read(path)), controller.authority.id)
        f.require(p['version'] == ROUTE_VERSION and p['epoch'] == epoch,
                  'exact original SFT dispatch closure')
        result = f.completion_fields(controller, epoch) if p['steps'] == 0 else previous_completion(epoch)
        return dict(result, positive_sft_dispatch=p,
                    positive_sft_dispatch_sha256=f.sha(json.loads(f._read(path))))

    controller.positive_sft_dispatch_route = route
    controller.native_completion_fields = completion
    controller.training_completion_fields = completion
    return controller
