"""Default-off ROOT review overlays. Originals remain immutable; UNKNOWN is not VALID."""
import hashlib
import io
import json
import math
import tarfile

from .distributed_roles import authenticate

VERSION = 'reviewed-toploc-numerical-resolution-v1'
canonical = lambda v: json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
digest = lambda v: hashlib.sha256(canonical(v)).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def valid_hash(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def reference(document, raw, authority):
    """Recheck the archived original result, rather than trusting a summary label."""
    ack = authenticate(document, authority)
    need(ack.get('version') == 'research-original-archive-full-readback-ack-v1'
         and ack.get('full_readback_verified') is True and ack.get('production_changes') is False,
         'ROOT full reference archive ACK')
    need(type(raw) is bytes and len(raw) <= 64 * 1024**2
         and hashlib.sha256(raw).hexdigest() == ack.get('sha256'), 'exact bounded reference archive')
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        members = archive.getmembers()
        need(len(members) <= 100 and len({m.name for m in members}) == len(members)
             and all(m.isfile() and m.size <= 2 * 1024**2 for m in members), 'bounded unique archive members')
        def read(name):
            return archive.extractfile(name).read()
        scope_names = [m.name for m in members if m.name.startswith('original/scope.') and m.name.endswith('.ROOT-SIGNED.private.json')]
        terminal_names = [m.name for m in members if m.name.startswith('original/original-execute.') and m.name.endswith('.terminal.private.json')]
        need(len(scope_names) == len(terminal_names) == 1, 'one original reference scope and terminal')
        scope_bytes = read(scope_names[0])
        scope = authenticate(json.loads(scope_bytes), authority)
        scope_sha = hashlib.sha256(scope_bytes).hexdigest()
        terminal = json.loads(read(terminal_names[0]))
        need(scope_sha == ack.get('scope_file_sha256') == terminal.get('scope_sha256')
             and terminal.get('exit_code') == 0 and terminal.get('timed_out') is False,
             'genuine original reference terminal/scope')
        need(scope.get('production_evidence_or_rewards_modified') is False
             and scope.get('ROOT_reviewed_genuine_qualification_passed') is True
             and scope.get('ROOT_reviewed_honest_control_passed') is True
             and scope.get('ROOT_reviewed_mutated_proof_rejected') is True,
             'reviewed genuine reference qualification')
        need(scope.get('version') in ('four-toploc-reference-root-dispatch-v1', 'five-toploc-reference-root-dispatch-v1'), 'reviewed bounded reference runner version')
        tool_names = [m.name for m in members if m.name == 'original/toploc_reference_adjudication.py']
        runner_names = [m.name for m in members if m.name in ('original/run_four_TOPLOC_references.REVIEW-ONLY.py', 'original/run_five_TOPLOC_references.REVIEW-ONLY.py')]
        need(len(tool_names) == len(runner_names) == 1
             and hashlib.sha256(read(tool_names[0])).hexdigest() == scope.get('diagnostic_sha256')
             and hashlib.sha256(read(runner_names[0])).hexdigest() == scope.get('runner_sha256'), 'exact ROOT pinned reference execution bytes')
        results = {}
        for member in members:
            if member.name.startswith('outputs/case-') and member.name.endswith('-research.json'):
                raw_result = read(member.name)
                result = json.loads(raw_result)
                results[hashlib.sha256(raw_result).hexdigest()] = result
        return ack, scope, results


def apply(observations, records, admitted_jobs, *, authority, cutoff,
          policy_document=None, expected_policy_sha256=None, reference_archives=()):
    """Only explicit signed policy pins can change the effective scoring category."""
    if policy_document is None and expected_policy_sha256 is None and not reference_archives:
        return observations
    need(valid_hash(authority) and valid_hash(expected_policy_sha256)
         and digest(policy_document) == expected_policy_sha256, 'present ROOT resolution policy pin')
    policy = authenticate(policy_document, authority)
    need(set(policy) == {'version', 'effective_cutoff', 'entries'} and policy['version'] == VERSION,
         'exact reviewed numerical resolution policy')
    need(type(policy['effective_cutoff']) is int and 0 <= policy['effective_cutoff'] <= cutoff,
         'prospective resolution cutoff')
    need(type(policy['entries']) is list and len(policy['entries']) <= 1000, 'bounded reviewed resolutions')
    archives = {}
    for item in reference_archives:
        need(set(item) == {'ack', 'archive'}, 'exact reference input')
        key = digest(item['ack'])
        need(key not in archives, 'duplicate reference archive')
        archives[key] = reference(item['ack'], item['archive'], authority)
    by_evidence = {digest(row): row for row in records}
    by_key = {(o['evidence_id'], o['job_sha256']): o for o in observations}
    replacements, seen = {}, set()
    fields = {'evidence_id', 'original_job_sha256', 'original_observation_sha256',
              'original_report_sha256', 'original_report_request_sha256', 'artifact_sha256',
              'epoch', 'checkpoint', 'source_sha256', 'reference_result_sha256',
              'reference_archive_ack_sha256', 'reviewed_at', 'outcome'}
    for entry in policy['entries']:
        need(type(entry) is dict and set(entry) == fields and entry['outcome'] == 'numerical_ambiguous',
             'only exact ROOT numerical UNKNOWN resolution')
        need(all(valid_hash(entry[k]) for k in fields - {'epoch', 'reviewed_at', 'outcome'}),
             'resolution immutable digests')
        need(type(entry['reviewed_at']) is int and 0 <= entry['reviewed_at'] <= cutoff,
             'resolution operator review at or before cutoff')
        key = (entry['evidence_id'], entry['original_job_sha256'])
        need(key not in seen, 'duplicate historical resolution')
        seen.add(key)
        row = by_evidence.get(entry['evidence_id'])
        # Earlier cohort snapshots intentionally lack later committed populations.
        if row is None:
            need(not any(r['epoch'] == entry['epoch'] for r in records), 'resolution exact original evidence id')
            continue
        need(row['epoch'] == entry['epoch'] and row['checkpoint'] == entry['checkpoint']
             and row['proof_sha256'] == entry['artifact_sha256'], 'resolution exact original population')
        admission = admitted_jobs.get(entry['original_job_sha256'])
        need(admission is not None, 'resolution original admitted execution')
        original = [o for o in admission['observations'] if digest(o) == entry['original_observation_sha256']]
        need(len(original) == 1 and original[0]['outcome'] == 'confirmed_invalid'
             and original[0]['epoch'] == row['epoch'] and original[0]['checkpoint'] == row['checkpoint']
             and original[0]['miner'] == row['miner'] and original[0]['batch_sha256'] == row['batch_sha256']
             and original[0]['completed_at'] <= entry['reviewed_at'], 'resolution exact original observation')
        native = admission.get('native_observations', {}).get(entry['original_observation_sha256'])
        need(native is not None and native['reason'] == 'InvalidSample: TOPLOC'
             and native['failure_kind'] == 'confirmed_invalid' and native['fully_audited'] is True
             and native['artifact_sha256'] == entry['artifact_sha256']
             and admission['source_sha256'] == entry['source_sha256']
             and admission['original_report_sha256'] == entry['original_report_sha256']
             and admission['original_report_request_sha256'] == entry['original_report_request_sha256'],
             'only original fully audited TOPLOC reason with exact report/source')
        need(entry['reference_archive_ack_sha256'] in archives, 'required authenticated reference archive')
        ack, scope, results = archives[entry['reference_archive_ack_sha256']]
        result = results.get(entry['reference_result_sha256'])
        need(result is not None and result.get('version') == 'toploc-reference-research-v1' and result.get('production_evidence') is False
             and result.get('rewards_or_original_reports_modified') is False
             and result.get('source_bundle_sha256') == entry['source_sha256']
             and result.get('checkpoint') == entry['checkpoint'] and result.get('epoch') == entry['epoch']
             and result.get('artifact_sha256') == entry['artifact_sha256']
             and result.get('original_job_sha256') == entry['original_job_sha256']
             and result.get('original_report_request_sha256') == entry['original_report_request_sha256'],
             'exact original/reference result linkage')
        need(ack['at'] <= entry['reviewed_at'] and ack['at'] >= original[0]['completed_at'], 'reference precedes reviewed resolution')
        need(type(result.get('results')) is list and 1 <= len(result['results']) <= 64, 'bounded complete reference rollouts')
        rejected = False
        for rollout in result['results']:
            need(rollout['classification'] in ('reference_valid', 'reference_rejected'), 'reference infrastructure is UNKNOWN only')
            if rollout['classification'] == 'reference_rejected':
                need(rollout.get('error_type') == 'InvalidSample' and rollout.get('reason') == 'TOPLOC', 'supported reference reason')
                rejected = True
            mismatches = 0
            need(type(rollout.get('toploc_calls')) is list and rollout['toploc_calls'], 'native mismatch metrics actually measured')
            for call in rollout['toploc_calls']:
                need(call['expected_segments'] == call['returned_segments'] == len(call['segments']) and call['segments'], 'complete native metric population')
                for metric in call['segments']:
                    exp, mean, median = metric['exp_mismatches'], metric['mant_err_mean'], metric['mant_err_median']
                    need(type(exp) is int and 0 <= exp <= 1 and type(mean) in (int, float)
                         and math.isfinite(mean) and 0 <= mean <= 1 / 128
                         and type(median) in (int, float) and median == 0,
                         'bounded small TOPLOC uncertainty only')
                    mismatches += bool(exp or mean)
            need(mismatches <= 6, 'bounded per-rollout mismatch count')
        need(rejected, 'reference actually reproduced TOPLOC rejection')
        # An absent original row may belong to another snapshot. Never fabricate it.
        if key in by_key:
            need(by_key[key]['outcome'] == 'confirmed_invalid', 'effective original category unchanged before overlay')
            replacements[key] = dict(by_key[key], outcome='numerical_ambiguous',
                original_outcome='confirmed_invalid', original_observation_sha256=entry['original_observation_sha256'],
                numerical_resolution_policy_sha256=expected_policy_sha256,
                numerical_resolution_review_sha256=digest(entry), sampler_and_grader_completion_claimed=False)
    return [replacements.get((o['evidence_id'], o['job_sha256']), o) for o in observations]
