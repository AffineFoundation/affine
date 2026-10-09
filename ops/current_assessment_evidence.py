"""Read-only original audit evidence for the independent assessment writer.

Registry/config integrity or an unavailable authoritative queue raises so the
writer can distinguish integrity errors from temporary outages. Individual malformed populations,
jobs, failures and adjudications are refused without suppressing valid peers.
This module neither signs nor publishes snapshots and never reads learner or
opening state. It does not construct a Coordinator (which initializes SQLite).
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from nacl.exceptions import BadSignatureError

from subnet.audit_queue_snapshot import queue_rows
from subnet.continuous_audit_policy import (
    admit_artifact_failures, digest, finite, observations, snapshot, valid_digest, policy, RESOLUTION_VERSION,
    historical_report_workers,
)
from subnet.continuous_audit_service import (
    admitted_service_config, admit_completed_reports, register_population,
)
from subnet.distributed_roles import authenticate
from subnet.storage import canonical


def _read(path, maximum=256 * 1024**2):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('assessment input must be an original regular file')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('bounded assessment input exceeded')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _issue(kind, identifier, error):
    # No signed payloads, queue tokens, URLs or credentials enter diagnostics.
    return dict(kind=kind, identifier=str(identifier), error_type=type(error).__name__,
                reason=str(error)[:240])


def _population(document, epoch, authority, sources):
    p = authenticate(document, authority)
    manifest = authenticate(p['manifest_document'], authority)
    if manifest['epoch'] != epoch:
        raise ValueError('original population epoch identity')
    source = manifest['source_bundle']['sha256']
    pins = sources['approved_sources'].get(source)
    metadata = sources['job_metadata'].get(source)
    profile = sources.get('execution_evidence_policy', {}).get('sources', {}).get(source)
    if not isinstance(pins, dict) or not isinstance(metadata, dict) or not isinstance(profile, dict):
        raise ValueError('population source/native profile not admitted')
    if metadata.get('source_files') != pins:
        raise ValueError('population exact source metadata')
    for field in ('model_runtime_revision', 'backend_profile', 'numerical_policy'):
        if manifest.get(field) != profile.get(field):
            raise ValueError('population source native profile binding: ' + field)
    if profile.get('runtime_versions') != metadata.get('runtime_versions'):
        raise ValueError('population source native runtime binding')
    # The harness digest is composite, not the individual harness.py file pin.
    # This is archived metadata authenticated by the original ROOT manifest;
    # source/profile admission above authorizes that exact original source.
    if not valid_digest(manifest.get('harness_source_hash')) or not valid_digest(pins.get('subnet/harness.py')):
        raise ValueError('population signed native harness/source binding')
    if manifest.get('sampling_contract') is not None and manifest.get('sampling_source_hash') != pins.get('subnet/forced_sampling.py'):
        raise ValueError('population admitted sampling source')
    from subnet.protocol import read_only_archived_entries
    definitions = read_only_archived_entries(manifest, manifest['harness_source_hash'])
    if not definitions or any(not valid_digest(row['spec'].get('source_hash')) for row in definitions):
        raise ValueError('population versioned native environment binding')
    ids = p['eligible_evidence_ids']
    if type(ids) is not list or any(not valid_digest(i) for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('explicit original eligible evidence set')
    selected = [row for row in p['records'] if digest(row) in ids]
    pairs = [{key: row[key] for key in ('miner', 'commitment_sha256', 'batch_sha256', 'proof_sha256')} for row in selected]
    reconstructed = register_population(p['manifest_document'], p['receipts'], p['round'],
        p['committed_at'], authority, eligible_pairs=pairs, version=p['version'])
    if canonical(reconstructed) != canonical(p):
        raise ValueError('exact original signed population reconstruction')
    return p, manifest


def load_evidence(audit_config_path, *, authority, cutoff, verifiers,
                  expected_source_admission_sha256=None, expected_numerical_resolution_policy_sha256=None):
    """Return unsigned snapshots plus original population times and diagnostics.

    ``verifiers`` and the optional canonical envelope digest come from the
    writer's independently authenticated ROOT policy, not this mutable config.
    ``cutoff`` bounds original commitment and actual report completion times.
    """
    finite(cutoff, 0, 2**53, 'assessment cutoff')
    if not valid_digest(authority) or type(verifiers) is not list or not verifiers or len(verifiers) != len(set(verifiers)) or any(not valid_digest(v) for v in verifiers):
        raise ValueError('explicit assessment authority/verifier policy')
    config, config_sha = _read(audit_config_path)
    service = config['continuous_audit_service']
    numerical = {}
    if expected_numerical_resolution_policy_sha256 is not None:
        settings = service.get('numerical_resolution')
        if not valid_digest(expected_numerical_resolution_policy_sha256) or type(settings) is not dict or set(settings) != {'policy_document', 'reference_archives'}:
            raise ValueError('explicit ROOT pinned numerical resolution configuration')
        document = settings['policy_document']
        authenticate(document, authority)
        if digest(document) != expected_numerical_resolution_policy_sha256:
            raise ValueError('ROOT numerical resolution policy pin changed')
        archive_inputs = []
        if type(settings['reference_archives']) is not list or len(settings['reference_archives']) > 100:
            raise ValueError('bounded numerical reference archives')
        for entry in settings['reference_archives']:
            if type(entry) is not dict or set(entry) != {'ack_path', 'archive_path'}:
                raise ValueError('exact numerical reference file inputs')
            ack, _ = _read(entry['ack_path'])
            path = Path(entry['archive_path'])
            if path.is_symlink(): raise ValueError('original numerical reference archive regular file')
            with path.open('rb') as stream: raw = stream.read(64 * 1024**2 + 1)
            if len(raw) > 64 * 1024**2: raise ValueError('bounded numerical reference archive')
            archive_inputs.append(dict(ack=ack, archive=raw))
        numerical = dict(numerical_resolution_policy=document,
            expected_numerical_resolution_policy_sha256=expected_numerical_resolution_policy_sha256,
            numerical_reference_archives=archive_inputs)
    try:
        sources = admitted_service_config(service, authority)
    except BadSignatureError as error:
        raise ValueError("source admission signature integrity") from error
    source_sha = digest(service['source_admission'])
    if expected_source_admission_sha256 is not None and source_sha != expected_source_admission_sha256:
        raise ValueError('assessment policy exact source admission changed')
    state_root = Path(config['state'])
    directory = state_root / 'continuous-audit'
    state, state_sha = _read(directory / 'audit-state.json')
    if any(type(state.get(key)) is not dict for key in ('populations', 'jobs', 'draws')):
        raise ValueError('original audit state registry shape')
    workers = {worker: ['verify'] for worker in verifiers}
    # Retired identities may observe only their exact ROOT-admitted originals.
    # The current claim roster remains the independently signed seven workers.
    historical = sources.get('historical_report_admission')
    retired = historical_report_workers(historical)
    observers = dict(workers, **{worker: ['verify'] for worker in retired},
                     **{authority: ['operator-artifact-capture']})
    excluded, refused, deferred, populations = [], [], [], {}
    if numerical:
        numerical['numerical_unavailable_execution_deferrals'] = deferred
    hashes = dict(audit_config_file_sha256=config_sha, audit_state_file_sha256=state_sha,
                  source_admission_sha256=source_sha, population_documents={},
                  original_queue_view_sha256=None, adjudication_files={}, artifact_failures={})
    for epoch, document in sorted(state['populations'].items()):
        try:
            p, manifest = _population(document, epoch, authority, sources)
            if p['committed_at'] > cutoff:
                excluded.append(dict(kind='population', identifier=epoch, reason='original commitment after cutoff'))
                continue
            populations[epoch] = (p, manifest)
            hashes['population_documents'][epoch] = digest(document)
        except (ValueError, KeyError, TypeError, AttributeError, BadSignatureError) as error:
            refused.append(_issue('population', epoch, error))
    records = [row for p, _ in populations.values() for row in p['records']]
    record_by_id = {digest(row): row for row in records}
    record_by_batch = {(r['epoch'], r['miner'], r['batch_sha256']): r for r in records}
    identifiers = []
    for identifier, entry in sorted(state['jobs'].items()):
        try:
            ids = entry['row_sha256s'] if 'row_sha256s' in entry else [entry['row_sha256']]
            if type(ids) is not list or not ids or len(ids) != len(set(ids)):
                raise ValueError('original selected audit row identifiers')
            if any(key not in record_by_id for key in ids):
                excluded.append(dict(kind='job', identifier=identifier, reason='original committed population unavailable or after cutoff'))
                continue
            if any(state['draws'][key]['row'] != record_by_id[key] for key in ids):
                raise ValueError('original selected audit draw population binding')
            identifiers.append(identifier)
        except (ValueError, KeyError, TypeError, AttributeError, BadSignatureError) as error:
            refused.append(_issue('job', identifier, error))
    queue_path = state_root / 'roles' / 'verifier-queue.sqlite3'
    actuals = queue_rows(SimpleNamespace(path=queue_path), identifiers, complete=True)
    hashes['original_queue_view_sha256'] = digest(actuals)
    if queue_path.exists():
        stat = queue_path.stat(); hashes['queue_identity'] = dict(dev=stat.st_dev, ino=stat.st_ino)
    candidates = []
    for identifier in identifiers:
        try:
            actual = actuals.get(identifier)
            if actual is None:
                raise ValueError('original queue job absent')
            if actual['status'] != 'complete':
                excluded.append(dict(kind='job', identifier=identifier, reason='original queue job not completed'))
                continue
            report = json.loads(actual['report']) if type(actual['report']) is str else actual['report']
            completed = finite(report['completed_at'], 0, 2**53, 'original report completion')
            if completed > cutoff:
                excluded.append(dict(kind='job', identifier=identifier, reason='original report completed after cutoff'))
                continue
            expected = state['jobs'][identifier].get('job_sha256')
            if expected is not None and expected != actual['digest']:
                raise ValueError('original audit job journal digest')
            candidates.append((completed, identifier, actual))
        except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
            refused.append(_issue('job', identifier, error))
    adjudications = []
    for path in sorted(directory.glob('*-adjudication.json')):
        try:
            doc, file_sha = _read(path); p = authenticate(doc, authority)
            if set(p) != {'version', 'evidence_id', 'original_job_sha256', 'reference_job_sha256', 'outcome'} or p['version'] != 'continuous-audit-adjudication-v1' or any(not valid_digest(p[k]) for k in ('evidence_id', 'original_job_sha256', 'reference_job_sha256')) or p['outcome'] not in ('verified_valid', 'confirmed_invalid'):
                raise ValueError('exact original reference adjudication')
            adjudications.append(doc); hashes['adjudication_files'][path.name] = file_sha
        except (ValueError, KeyError, TypeError, AttributeError, BadSignatureError) as error:
            refused.append(_issue('adjudication', path.name, error))
    ep = sources.get('execution_evidence_policy')
    execution = ep if ep is not None and cutoff >= ep['effective_cutoff'] else None
    admitted_candidates = []
    for completed, identifier, actual in sorted(candidates):
        try:
            entry = state['jobs'][identifier]
            selected_ids = entry['row_sha256s'] if 'row_sha256s' in entry else [entry['row_sha256']]
            selected_records = [record_by_id[i] for i in selected_ids]
            admitted, delayed = admit_completed_reports([actual], selected_records, authority, workers,
                sources['approved_sources'], execution_evidence_policy=execution, cutoff=cutoff,
                deferral_policy=sources.get('backend_evidence_deferral_policy'),
                historical_report_admission=historical)
            deferred.extend(delayed)
            for key, value in admitted.items():
                candidates_entry = (completed, identifier, key, value)
                # Separate list preserves temporal ordering across GPU and ROOT artifact observations.
                admitted_candidates.append(candidates_entry)
        except (ValueError, KeyError, TypeError, AttributeError, BadSignatureError) as error:
            refused.append(_issue('job', identifier, error))
    for identifier, failure in sorted(state.get('capture_failures', {}).items()):
        try:
            if failure.get('kind') != 'confirmed_invalid_artifact': continue
            document = failure['document']; p = authenticate(document, authority)
            if p['completed_at'] > cutoff:
                excluded.append(dict(kind='artifact_failure', identifier=identifier, reason='original failure completed after cutoff')); continue
            if digest(p['row']) not in record_by_id:
                excluded.append(dict(kind='artifact_failure', identifier=identifier, reason='original committed population unavailable')); continue
            admitted = admit_artifact_failures([document], [record_by_id[digest(p['row'])]], authority)
            hashes['artifact_failures'][identifier] = digest(document)
            admitted_candidates.extend((p['completed_at'], identifier, key, value) for key, value in admitted.items())
        except (ValueError, KeyError, TypeError, AttributeError, BadSignatureError) as error:
            refused.append(_issue('artifact_failure', identifier, error))
    admissions, resolved = {}, {}
    resolutions = [authenticate(doc, authority) for doc in adjudications]
    for _, identifier, key, value in sorted(admitted_candidates, key=lambda row: (row[0], row[1], row[2])):
        try:
            if key in admissions and admissions[key] != value:
                raise ValueError('conflicting original admitted job')
            selected = {(o['epoch'], o['miner'], o['batch_sha256']) for o in value['observations']}
            selected_records = [record_by_batch[k] for k in sorted(selected)]
            candidate = observations([dict(admitted_queue_job_sha256=key)], selected_records,
                observers, cutoff, admitted_jobs={key: value}, authority=authority)
            updates = candidate_updates(resolved, candidate, resolutions)
            # Whole grouped original job commits only after all children pass.
            resolved.update(updates)
            admissions[key] = value
        except (ValueError, KeyError, TypeError, AttributeError, BadSignatureError) as error:
            refused.append(_issue('observation', identifier, error))
    snapshots, timings = [], []
    for epoch, (p, manifest) in sorted(populations.items(), key=lambda item: (item[1][0]['round'], item[0])):
        try:
            previous = [r for r in records if r['round'] <= p['round']]
            allowed = {(r['epoch'], r['miner'], r['batch_sha256']) for r in previous}
            joined = {key: value for key, value in admissions.items() if all((o['epoch'], o['miner'], o['batch_sha256']) in allowed for o in value['observations'])}
            snap = snapshot(previous, [dict(admitted_queue_job_sha256=k) for k in joined],
                observers, epoch=epoch, round=p['round'], checkpoint=manifest['checkpoint']['id'],
                cutoff=cutoff, audit_policy=service['policy'], admitted_jobs=joined,
                eligible_evidence_ids=p['eligible_evidence_ids'], adjudications=adjudications, authority=authority, **numerical)
            if execution is not None:
                snap.update(execution_evidence_policy_sha256=digest(execution), execution_evidence_policy_version=execution['version'], os_resource_enforcement_claimed=False, historical_execution_proven=False)
            snapshots.append(snap)
            timings.append(dict(epoch=epoch, round=p['round'], checkpoint=manifest['checkpoint']['id'], committed_at=p['committed_at'], cutoff=cutoff))
        except (ValueError, KeyError, TypeError, AttributeError, BadSignatureError) as error:
            refused.append(_issue('snapshot', epoch, error))
    audit_observations = observations([dict(admitted_queue_job_sha256=k) for k in admissions],
        records, observers, cutoff, admitted_jobs=admissions,
        adjudications=adjudications, authority=authority, **numerical)
    if numerical:
        hashes['numerical_resolution_policy_sha256'] = expected_numerical_resolution_policy_sha256
        hashes['numerical_resolution_archive_ACK_sha256'] = sorted(digest(x['ack']) for x in numerical['numerical_reference_archives'])
        hashes['numerical_resolution_effective_observations'] = [dict(evidence_id=o['evidence_id'], original_job_sha256=o['job_sha256'], original_outcome=o['original_outcome'], outcome=o['outcome'], original_observation_sha256=o['original_observation_sha256'], review_sha256=o['numerical_resolution_review_sha256'], sampler_and_grader_completion_claimed=False) for o in audit_observations if 'numerical_resolution_review_sha256' in o]
    global_round = max((p['round'] for p, _ in populations.values()), default=0)
    global_details = current_estimates(audit_observations,
        {r['miner'] for r in records}, global_round, service['policy'])
    for snap in snapshots:
        snap['cohort_miner_details'] = {m: dict(d) for m, d in snap['miners'].items()}
        snap['current_global_round'] = global_round
        for miner, detail in snap['miners'].items():
            detail.update(global_details[miner])
        snap['points'] = {m: d['unique_eligible_batches'] * d['validity_probability'] *
                         d['reward_multiplier'] * d['resolution_coverage_factor']
                         for m, d in snap['miners'].items()}
        total = sum(snap['points'].values())
        snap['weights'] = {m: v / total if total else 0. for m, v in snap['points'].items()}
    return dict(snapshots=snapshots, committed_at_by_epoch={r['epoch']: r['committed_at'] for r in timings},
                population_timings=timings, evidence_hashes=hashes, excluded=excluded,
                refused=refused, deferred=deferred)


def current_estimates(audits, miners, round, audit_policy):
    """Global estimates from admitted, deduplicated original evidence.

    New unaudited cohorts cannot reset confidence, ambiguity coverage or recent
    penalties. UNKNOWN changes resolution coverage only under the signed v3
    policy; infrastructure changes neither confidence nor coverage.
    """
    p = policy(audit_policy)
    result = {}
    for miner in sorted(miners):
        recent = [o for o in audits if o['miner'] == miner and
                  0 <= round - o['round'] < p['recent_epochs']]
        alpha, beta = float(p['prior_alpha']), float(p['prior_beta'])
        invalid, latest_bad, resolved, unknown = 0, None, 0., 0.
        for o in recent:
            weight = p['decay'] ** (round - o['round'])
            if o['outcome'] == 'verified_valid':
                alpha += weight; resolved += weight
            elif o['outcome'] == 'confirmed_invalid':
                beta += weight; resolved += weight; invalid += 1
                latest_bad = max(o['round'], latest_bad if latest_bad is not None else o['round'])
            elif o['outcome'] == 'numerical_ambiguous':
                unknown += weight
        blacklisted = bool(p['blacklist_after'] and invalid >= p['blacklist_after'] and
            latest_bad is not None and round - latest_bad < p['blacklist_epochs'])
        zero = bool(p['zero_epoch_after'] and invalid >= p['zero_epoch_after'])
        coverage = resolved / (resolved + unknown) if resolved + unknown else 1.
        if p['version'] != RESOLUTION_VERSION: coverage = 1.
        result[miner] = dict(validity_probability=alpha / (alpha + beta),
            recent_posterior_mean=alpha / (alpha + beta),
            confirmed_invalid_recent=invalid, latest_bad_round=latest_bad,
            reward_multiplier=0. if blacklisted or zero else p['invalid_multiplier'] ** invalid,
            blacklisted=blacklisted, resolution_coverage_factor=coverage,
            recent_resolution_coverage=coverage, resolved_recent_weight=resolved,
            numerical_ambiguous_recent_weight=unknown, unresolved_is_fraud=False,
            infrastructure_counted_in_coverage=False, current_estimate_round=round)
    return result


def candidate_updates(resolved, candidate, resolutions):
    """Atomic per-original-job conflict admission, matching observations()."""
    updates = {}
    for new in candidate:
        key = new['evidence_id']
        old = updates.get(key, resolved.get(key))
        if old is not None:
            if old['outcome'] == new['outcome'] or new['outcome'] == 'infrastructure_error':
                continue
            if old['outcome'] != 'infrastructure_error':
                expected = dict(version='continuous-audit-adjudication-v1',
                    evidence_id=key, original_job_sha256=old['job_sha256'],
                    reference_job_sha256=new['job_sha256'], outcome=new['outcome'])
                if not (expected in resolutions and old['outcome'] == 'numerical_ambiguous'
                        and new['outcome'] in ('verified_valid', 'confirmed_invalid')):
                    raise ValueError('conflicting authenticated audits require explicit reference adjudication')
        updates[key] = new
    return updates
