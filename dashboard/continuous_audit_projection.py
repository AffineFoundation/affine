"""Read-only display of coordinator-authenticated continuous audit outcomes.

This is a projection, not inference verification or a reward calculation.
Only exact captured children are joined. A signed submission remains unaudited
until a completed, authenticated verifier report establishes its outcome.
"""
import json
import math
import sqlite3
import time
from pathlib import Path
from nacl.exceptions import BadSignatureError

from dashboard.learner_projection import AUTHORITY, authenticated, digest


def captured_batches(document, manifest, authority=AUTHORITY):
    """Return the immutable batch identities already admitted by learner projection."""
    epoch = manifest['epoch']; checkpoint = manifest['checkpoint']['id']
    source = manifest['source_bundle']['sha256']; population = document['population']
    if (document['version'] != 'committed-unaudited-training-v1'
            or population['epoch'] != epoch or population['checkpoint'] != checkpoint
            or population['assurance'] != 'unaudited'):
        raise ValueError('same committed learner population')
    result = {}
    for row in population['committed_inventory']:
        miner = row['miner']; commitment = authenticated(row['commitment_document'], miner)
        if (commitment['epoch'] != epoch or commitment['checkpoint'] != checkpoint
                or commitment['source'] != source or commitment['miner'] != miner
                or commitment['version'] != 'small-commitment-pairs-v2'
                or digest(row['commitment_document']) != row['commitment_sha256']):
            raise ValueError('exact signed committed population child')
        captures = None
        if 'training_documents' in row:
            captures = {r['slot']: r for r in row['training_documents']}
            if len(captures) != len(row['training_documents']):
                raise ValueError('distinct captured training slots')
        for child in commitment['batches']:
            slot = child['slot']
            if type(slot) is not int or slot < 0:
                raise ValueError('integer committed slot')
            if captures is not None:
                if slot not in captures: continue
                if (captures[slot]['sha256'] != child['training_sha256']
                        or captures[slot]['size'] != child['training_size']):
                    raise ValueError('exact captured training document')
            identity = (miner, row['commitment_sha256'], slot, child['batch_sha256'], child['sha256'])
            if identity in result: raise ValueError('duplicate captured child')
            result[identity] = (child['env_id'], child['index'])
    return result


def admitted_execution_resources(job, manifest, report, source_admission):
    """Preserve the auditor's explicit, ROOT-approved standard-backend scope."""
    expected_enforced = True
    if source_admission is not None:
        source = manifest['source_bundle']['sha256']
        pins = source_admission['approved_sources'].get(source)
        if not isinstance(pins, dict) or any(pins.get(k) != v for k, v in job['source_files'].items()):
            raise ValueError('ROOT-admitted executed source pins')
        policy = source_admission.get('execution_evidence_policy')
        if policy is not None:
            if (set(policy) != {'version', 'effective_cutoff', 'sources'}
                    or policy['version'] not in ('explicit-backend-execution-evidence-v1','explicit-backend-execution-evidence-v2')
                    or type(policy['effective_cutoff']) not in (int, float)
                    or not math.isfinite(policy['effective_cutoff'])
                    or not 0 <= policy['effective_cutoff'] <= time.time()):
                raise ValueError('effective ROOT execution-evidence scope')
            entry = policy['sources'].get(source)
            if entry is not None:
                fields = {'backend','backend_module_sha256','model_runtime_revision',
                          'backend_profile','numerical_policy','runtime_versions','execution_resources_enforced'}
                if policy['version'] == 'explicit-backend-execution-evidence-v2': fields.add('effective_cutoff')
                if (set(entry) != fields
                        or entry['backend'] != 'standard-backend-no-os-resource-enforcement-v1'
                        or entry['execution_resources_enforced'] is not False
                        or pins.get('subnet/backend_jobs.py') != entry['backend_module_sha256']
                        or job['source_files'] != pins
                        or manifest['model_runtime_revision'] != entry['model_runtime_revision']
                        or manifest['backend_profile'] != entry['backend_profile']
                        or manifest['numerical_policy'] != entry['numerical_policy']
                        or job['runtime_versions'] != entry['runtime_versions']):
                    raise ValueError('exact ROOT standard-backend admission')
                cutoff = entry.get('effective_cutoff', policy['effective_cutoff'])
                if type(cutoff) not in (int, float) or not math.isfinite(cutoff) or not policy['effective_cutoff'] <= cutoff <= 2**53:
                    raise ValueError('prospective per-source backend admission cutoff')
                if time.time() >= cutoff: expected_enforced = False
    if report.get('execution_resources_enforced') is not expected_enforced:
        raise ValueError('original admitted execution-resource contract')


def observations(row, manifest, authority=AUTHORITY, source_admission=None):
    """Authenticate the exact terminal request stored by the coordinator.

    The queue's completion transition already requires an admitted worker and
    its live lease. Recheck its immutable signed evidence and ROOT job binding;
    never accept a free-standing report merely because it claims success.
    """
    if row['status'] != 'complete' or row['role'] != 'verify':
        raise ValueError('completed coordinator verifier job')
    parse = lambda value: json.loads(value) if isinstance(value, str) else value
    job = authenticated(parse(row['envelope']), authority)
    assigned = authenticated(job['manifest'], authority)
    report = parse(row['report'])
    request = authenticated(parse(row['report_request']), row['worker'])
    if (job['job_id'] != row['id'] or job['role'] != 'verify'
            or digest(job) != row['digest'] or digest(report) != row['report_digest']
            or request.get('action') != 'report' or request.get('job_id') != row['id']
            or request.get('token') != row['token'] or request.get('report') != report):
        raise ValueError('original coordinator job and terminal request binding')
    if (assigned['epoch'] != manifest['epoch']
            or assigned['checkpoint']['id'] != manifest['checkpoint']['id']
            or assigned['source_bundle']['sha256'] != manifest['source_bundle']['sha256']
            or report.get('success') is not True or report.get('role') != 'verify'
            or report.get('job_id') != row['id'] or report.get('job_sha256') != row['digest']
            or report.get('operator') != authority or report.get('epoch') != manifest['epoch']
            or report.get('checkpoint') != manifest['checkpoint']['id']
            or report.get('source_files') != job.get('source_files') or not job.get('source_files')
            or any(report.get('runtime_versions', {}).get(k) != v for k, v in job['runtime_versions'].items())
            or report.get('backend_profile') != assigned['backend_profile']
            or report.get('numerical_policy') != assigned['numerical_policy']):
        raise ValueError('original audited checkpoint/source/runtime scope')
    admitted_execution_resources(job, assigned, report, source_admission)
    audits = report.get('audits')
    if not isinstance(audits, list) or len(audits) != len(job['submissions']):
        raise ValueError('one outcome per queued immutable child')
    result = []
    for submission, audit in zip(job['submissions'], audits):
        ref = submission['commitment_ref']; outcomes = audit.get('outcomes')
        if (audit.get('epoch') != manifest['epoch']
                or audit.get('submission_sha256') != submission['sha256']
                or not isinstance(outcomes, list) or len(outcomes) != 1):
            raise ValueError('exact audited proof and outcome binding')
        outcome = outcomes[0]
        if (outcome.get('env_id') != ref['env_id'] or outcome.get('index') != ref['index']
                or type(ref['slot']) is not int or ref['slot'] < 0):
            raise ValueError('same immutable task and submission slot')
        classification = None
        if outcome.get('valid') is True and outcome.get('fully_audited') is True:
            accepted = audit.get('accepted')
            if (not isinstance(accepted, list) or len(accepted) != 1
                    or digest(accepted[0]) != ref['batch_sha256']):
                raise ValueError('exact accepted batch bytes')
            classification = 'accepted'
        elif (outcome.get('valid') is False and
              (outcome.get('failure_kind') == 'structural_invalid'
               or outcome.get('fully_audited') is True and outcome.get('failure_kind') == 'confirmed_invalid')):
            classification = 'rejected'
        # Infrastructure/numerical ambiguity and pending work are not failures.
        if classification is not None:
            identity = (ref['miner'], ref['commitment_sha256'], ref['slot'],
                        ref['batch_sha256'], submission['sha256'])
            result.append((identity, (ref['env_id'], ref['index']), classification))
    return result


class Projection:
    """Incremental event-cursor cache; web requests only read the dashboard DB.

    At refresh, each current epoch examines at most 256 new completed events.
    Full report authentication is performed once per new relevant job. Cache
    values contain identities/counts only, never URLs, report tokens, or traces.
    """
    def __init__(self, authority=AUTHORITY, max_events=256):
        if type(max_events) is not int or not 1 <= max_events <= 256:
            raise ValueError('bounded incremental audit projection budget')
        self.authority = authority
        self.max_events = max_events
        self.epochs = {}
        self.source_admission = None
        self.source_admission_sha256 = None

    def configure_source_admission(self, document):
        checksum = digest(document) if document is not None else None
        if checksum == self.source_admission_sha256: return
        payload = authenticated(document, self.authority) if document is not None else None
        if payload is not None and (payload.get('version') != 'continuous-audit-service-sources-v1'
                                    or not isinstance(payload.get('approved_sources'), dict)):
            raise ValueError('existing signed continuous-auditor source admission')
        self.source_admission = payload
        self.source_admission_sha256 = checksum
        self.epochs.clear()  # Reconsider prior unknown evidence under the exact new admission.

    def project(self, queue, document, manifest):
        queue = Path(queue)
        if not queue.is_file() or queue.is_symlink(): return None
        captured = captured_batches(document, manifest, self.authority)
        stat = queue.stat()
        cache_key = (str(queue.resolve()), stat.st_dev, stat.st_ino,
                     manifest['epoch'], manifest['checkpoint']['id'], manifest['source_bundle']['sha256'])
        cached = self.epochs.setdefault(cache_key, dict(cursor=0, outcomes={}, rejected_records=0))
        db = sqlite3.connect(queue.resolve().as_uri() + '?mode=ro', uri=True, timeout=1)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA query_only=ON'); db.execute('BEGIN')
            watermark = db.execute('SELECT COALESCE(MAX(sequence),0) FROM events').fetchone()[0]
            if watermark < cached['cursor']:
                cached.update(cursor=0, outcomes={}, rejected_records=0)
            events = db.execute('''SELECT e.sequence, e.job, json_extract(j.report,'$.epoch') AS epoch
                FROM events e JOIN jobs j ON j.id=e.job
                WHERE e.sequence>? AND e.sequence<=? AND e.kind='completed' AND e.at>=?
                    AND j.status='complete' AND j.role='verify'
                ORDER BY e.sequence LIMIT ?''',
                (cached['cursor'], watermark, manifest.get('start', 0), self.max_events)).fetchall()
            # Commit cache progress only after this consistent read succeeds.
            additions = []; invalid = 0
            for event in events:
                if event['epoch'] != manifest['epoch']: continue
                row = db.execute('SELECT * FROM jobs WHERE id=?', (event['job'],)).fetchone()
                try: additions.extend(observations(dict(row), manifest, self.authority, self.source_admission))
                except (BadSignatureError, ValueError, KeyError, TypeError): invalid += 1
            for identity, task, outcome in additions:
                cached['outcomes'].setdefault((identity, task), set()).add(outcome)
            cached['rejected_records'] += invalid
            cached['cursor'] = events[-1]['sequence'] if len(events) == self.max_events else watermark
        finally:
            db.close()
        joined = [value for (identity, task), value in cached['outcomes'].items()
                  if captured.get(identity) == task]
        accepted = sum(value == {'accepted'} for value in joined)
        rejected = sum(value == {'rejected'} for value in joined)
        conflicts = sum(len(value) > 1 for value in joined)
        return dict(accepted=accepted, rejected=rejected, unchecked=len(captured)-accepted-rejected,
                    captured=len(captured), conclusive=accepted+rejected, conflicts=conflicts,
                    source='authenticated-continuous-coordinator-reports',
                    backlog=cached['cursor'] < watermark)
