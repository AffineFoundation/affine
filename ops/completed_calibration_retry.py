"""Read back timely completed calibration without extending an execution deadline."""
import json
import math

VERSION = 'completed-calibration-readback-v1'


def read_original(jobs, label, manifest, request):
    """Authenticate retained local evidence. Never dispatch, copy, or launch a job."""
    from subnet.backend_jobs import signed
    from subnet.successor_calibration import digest
    prior = json.loads((jobs.state / (label + '.json')).read_bytes())
    job = signed(json.loads((jobs.state / (prior['job_id'] + '-job.json')).read_bytes()), jobs.controller.authority.id)
    if (job.get('role') != 'evaluate' or job.get('job_id') != prior['job_id']
            or job.get('successor_calibration') != request
            or signed(job['manifest'], jobs.controller.authority.id) != manifest
            or digest(job) != prior['job_sha256']):
        raise ValueError('completed calibration original request binding')
    report = json.loads((jobs.state / (prior['job_id'] + '-report.json')).read_bytes())
    return jobs.checked(report, prior, manifest)


def completed(original, controller, config, opening, record, path, key, manifest, req, report, jobs, cache, opt):
    from subnet import successor_calibration as c
    from subnet.fast_prefill_audit import policy_from_executed_controls
    fields = ('confirmation_sha256', 'confirmation_original_job_id', 'report_sha256', 'original_job_id', 'calibration')
    if not all(field in record for field in fields):
        return original(controller, config, opening, record, path, key, manifest, req, report, jobs, cache, opt)
    journal = record.get('bounded_confirmations', {})
    created, deadline = journal.get('created_at'), journal.get('deadline')
    if (journal.get('version') != c.RECALIBRATION_VERSION
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in (created, deadline))
            or not 0 < deadline - created <= opt['deadline_seconds'] + 1
            or type(journal.get('rounds')) is not list
            or not 1 <= len(journal['rounds']) <= opt['max_confirmations']):
        raise ValueError('completed calibration immutable journal')
    original_report = read_original(jobs, 'successor-calibration-' + key[:32], manifest, req)
    if (c.canonical(original_report) != c.canonical(report)
            or c.digest(report) != record['report_sha256'] or report['job_id'] != record['original_job_id']
            or not report['completed_at'] < deadline):
        raise ValueError('completed calibration original seed changed or late')
    c.admitted_policy(report['successor_calibration'], manifest, req)
    reports = list(report['successor_calibration']['reports'])
    token_only = opening.get('token_artifact_policy') is not None
    if token_only:
        from subnet.token_only_protocol import for_manifest
        for_manifest(opening)
    for index, row in enumerate(journal['rounds']):
        policy = policy_from_executed_controls(reports, checkpoint=manifest['checkpoint']['id'],
            model_runtime_revision=manifest['model_runtime_revision'], backend_profile=manifest['backend_profile'],
            harness=req['harness'], safety_factor=4.)
        if row['policy'] != policy or row['label'] != 'successor-reconfirm-' + key[:24] + '-' + str(index):
            raise ValueError('completed calibration proposal changed')
        actual = read_original(jobs, row['label'], manifest, row['request'])
        if (c.canonical(row.get('report')) != c.canonical(actual)
                or not created <= actual['completed_at'] < deadline):
            raise ValueError('completed calibration confirmation changed or late')
        confirmed = actual['successor_calibration']
        c.admitted_policy(confirmed, manifest, row['request'])
        passed = all(r['measured_cdf_abs_error'] <= policy['cdf_abs_error']
            and (token_only or r['measured_logprob_abs_error'] <= policy['logprob_atol']) for r in confirmed['reports'])
        if passed:
            if (index != len(journal['rounds']) - 1 or record['confirmation_sha256'] != c.digest(actual)
                    or record['confirmation_original_job_id'] != actual['job_id'] or record['calibration'] != policy):
                raise ValueError('completed calibration final evidence changed')
            result = dict(opening)
            result['sampling_policy'] = dict(config['sampling_policy'], calibration=policy)
            return result
        reports.extend(confirmed['reports'])
    raise ValueError('completed calibration never confirmed')


def install():
    from subnet import successor_calibration
    original = successor_calibration._bounded_confirmation
    def observed(*args, **kwargs):
        return completed(original, *args, **kwargs)
    successor_calibration._bounded_confirmation = observed
