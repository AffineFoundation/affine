"""Single hourly SN120 writer, independent of training and opening completion."""
import argparse
import hashlib
import json
import math
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from nacl.signing import SigningKey
from subnet.chain import ChainAdapter, OWNER, weights_lock
from subnet.weight_submission_transaction import SubmissionJournal, sdk_seam_paths
from subnet.current_assessment import calculate, fallback, recipients, HALF_LIFE_HOURS, HISTORY_HOURS, VERSION as ASSESSMENT_VERSION, number
from subnet.live_reward_bridge import signed, sha, writer_gate
from ops.live_reward_writer import (authenticate_cutover, global_lock, guard_files,
                                    observe_units, process_identity, read, file_hash)
from ops.live_reward_exporter import atomic, sign

VERSION = 'hourly-current-assessment-writer-v1'
NUMERICAL_RESOLUTION_VERSION = 'hourly-current-assessment-writer-reviewed-numerical-v2'
NEVER_BURN_VERSION = 'hourly-current-assessment-writer-never-burn-v3'
EVIDENCE_TIMEOUT_SECONDS = 360
ASSESSMENT_DIRECTORY = 'current-assessment-never-burn-v1'


def validate_policy(document, authority, cutover_document, anchor_document):
    p = signed(document, authority)
    fields = {'version','half_life_hours','first_window','netuid','owner_hotkey',
              'audit_config','source_admission_sha256','verifiers','module_hashes',
              'cutover_sha256','anchor_sha256','execute_enabled','zero_total_policy',
              'registration_change_policy'}
    expected_fields = fields | {'numerical_resolution_policy_sha256', 'fallback_assessments'}
    if 'authenticated_assessment_sources' in p: expected_fields.add('authenticated_assessment_sources')
    if set(p) != expected_fields or p['version'] != NEVER_BURN_VERSION or p['half_life_hours'] != HALF_LIFE_HOURS:
        raise ValueError('exact never-burn hourly assessment policy')
    import importlib.util
    from subnet.continuous_audit_policy import valid_digest
    if not valid_digest(p['numerical_resolution_policy_sha256']): raise ValueError('explicit ROOT numerical resolution digest')
    required = {str(Path(__file__).resolve())} | {str(Path(importlib.util.find_spec(name).origin).resolve()) for name in ('subnet.numerical_resolution', 'subnet.continuous_audit_policy', 'ops.current_assessment_evidence', 'subnet.current_assessment', 'subnet.chain', 'ops.live_reward_writer', 'subnet.weight_submission_transaction', 'subnet.weight_submission_reconciliation')}
    required |= sdk_seam_paths()
    if not required <= {str(Path(path).resolve()) for path in p['module_hashes']}:
        raise ValueError('present ROOT pins for never-burn execution modules')
    if not isinstance(p['fallback_assessments'], list): raise ValueError('explicit fallback assessment pins')
    for row in p['fallback_assessments']:
        if set(row) != {'path', 'sha256', 'writer_policy_sha256'} or not Path(row['path']).is_absolute() or not valid_digest(row['sha256']) or not valid_digest(row['writer_policy_sha256']):
            raise ValueError('exact immutable fallback assessment binding')
    if not isinstance(p.get('authenticated_assessment_sources', []), list): raise ValueError('explicit authenticated assessment sources')
    for row in p.get('authenticated_assessment_sources', []):
        if set(row) != {'path', 'writer_policy_sha256'} or not Path(row['path']).is_absolute() or not valid_digest(row['writer_policy_sha256']):
            raise ValueError('exact authenticated assessment producer binding')
    if p['netuid'] != 120 or p['owner_hotkey'] != OWNER or type(p['execute_enabled']) is not bool:
        raise ValueError('owner/netuid/execution policy')
    if p['zero_total_policy'] != 'no-owner-retain-v1':
        raise ValueError('explicit zero-total policy')
    if p['registration_change_policy'] != 'current-hotkey-snapshot-v1':
        raise ValueError('explicit current-registration policy')
    if p['cutover_sha256'] != sha(cutover_document) or p['anchor_sha256'] != sha(anchor_document):
        raise ValueError('single writer cutover binding')
    if type(p['first_window']) is not int or p['first_window'] % 3600:
        raise ValueError('first assessment UTC hour')
    if not p['verifiers'] or len(p['verifiers']) != len(set(p['verifiers'])):
        raise ValueError('exact admitted verifier identities')
    for file, digest in p['module_hashes'].items():
        if file_hash(file) != digest:
            raise ValueError('reviewed assessment writer bytes')
    return p


@contextmanager
def evidence_budget(seconds=EVIDENCE_TIMEOUT_SECONDS):
    """Reserve chain-finalization time; preserve any enclosing process deadline."""
    old_handler = signal.getsignal(signal.SIGALRM)
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    def expired(*_): raise TimeoutError('bounded current-assessment evidence')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, remaining) if remaining else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if remaining:
            signal.setitimer(signal.ITIMER_REAL, max(.001, remaining - (time.monotonic() - started)), interval)


def validate_assessment(a, p, cutoff):
    if (a.get('version') != ASSESSMENT_VERSION or type(a.get('cutoff')) is not int or a['cutoff'] > cutoff
        or a['cutoff'] % 3600 or a.get('half_life_hours') != HALF_LIFE_HOURS or a.get('history_hours') != HISTORY_HOURS):
        raise ValueError('compatible authenticated historical assessment')
    if (a.get('smoothing_basis') != 'estimated-valid-contribution-before-penalty'
        or a.get('penalties_applied_after_smoothing') is not True
        or a.get('training_completion_required') is not False
        or a.get('unaudited_samples_claimed_verified') is not False
        or not math.isclose(number(a.get('hourly_alpha')), 1 - 2 ** (-1 / HALF_LIFE_HOURS), rel_tol=1e-12)
        or type(a.get('evidence_cutoff', a['cutoff'])) is not int
        or not 0 <= a.get('evidence_cutoff', a['cutoff']) <= a['cutoff']):
        raise ValueError('exact authenticated assessment EMA and evidence semantics')
    hashes = a.get('evidence_hashes', {})
    for field in ('source_admission_sha256', 'numerical_resolution_policy_sha256'):
        if hashes.get(field) != p[field]: raise ValueError('historical assessment evidence policy binding')
    for miner, value in a['points'].items():
        number(value)
        details = a['miner_estimates'][miner]
        for field in ('validity_probability','reward_multiplier','resolution_coverage_factor'):
            if number(details.get(field, 1.)) > 1: raise ValueError('historical assessment factor')
        if (details.get('blacklisted') or details['reward_multiplier'] == 0) and value != 0:
            raise ValueError('historical assessment resurrects excluded miner')
    return a


def historical_assessments(directory, p, policy_document, authority, cutoff):
    candidates, refusals = [], []
    paths = [(directory/name, None) for name in ('last-valid-assessment.json','last-positive-assessment.json')]
    paths += [(Path(row['path']), row) for row in p['fallback_assessments']]
    for path, pin in paths:
        if pin is None and not path.exists(): continue
        try:
            if path.is_symlink() or not path.is_file(): raise ValueError('regular assessment file')
            if pin and file_hash(path) != pin['sha256']: raise ValueError('fallback bytes binding')
            a = signed(read(path), authority)
            expected = pin['writer_policy_sha256'] if pin else sha(policy_document)
            if a.get('writer_policy_sha256') != expected: raise ValueError('fallback writer policy binding')
            candidates.append(validate_assessment(a, p, cutoff))
        except Exception as error:
            refusals.append(dict(path=str(path), error_type=type(error).__name__))
    return candidates, refusals


def authenticated_assessment_sources(p, authority, cutoff):
    """Optional signed producer cache, never a training-completion dependency."""
    candidates, refusals = [], []
    for row in p.get('authenticated_assessment_sources', []):
        path = Path(row['path'])
        try:
            if path.is_symlink() or not path.is_file(): raise ValueError('regular assessment source')
            document = read(path)
            a = signed(document, authority)
            if a.get('writer_policy_sha256') != row['writer_policy_sha256']:
                raise ValueError('authenticated assessment producer policy binding')
            validate_assessment(a, p, cutoff)
            a = dict(a, authenticated_source=dict(path=str(path),
                writer_policy_sha256=row['writer_policy_sha256'],
                original_envelope_sha256=sha(document), cutoff=a['cutoff']))
            candidates.append(a)
        except Exception as error:
            refusals.append(dict(path=str(path), error_type=type(error).__name__))
    return candidates, refusals


def best_fallback(history, cutoff, reason, current=None):
    positive = [a for a in history if any(v > 0 for v in a['points'].values())]
    if not positive: return None
    prior = max(positive, key=lambda a: a['cutoff'])
    constraints = list(history) + ([current] if current is not None else [])
    return fallback(prior, cutoff, reason, constraints=constraints)



def existing_hourly_assessment(target, p, policy_document, authority, cutoff):
    """Adopt only exact ROOT-pinned same-window bytes across a policy handoff."""
    if target.is_symlink() or not target.is_file():
        raise ValueError('regular immutable hourly assessment')
    raw=target.read_bytes();document=json.loads(raw);assessment=signed(document,authority)
    if assessment['cutoff'] != cutoff:
        raise ValueError('immutable hourly assessment binding')
    if assessment['writer_policy_sha256'] != sha(policy_document):
        matches=[row for row in p['fallback_assessments']
                 if row['writer_policy_sha256']==assessment['writer_policy_sha256']
                 and row['sha256']==hashlib.sha256(raw).hexdigest()]
        if len(matches)!=1:
            raise ValueError('explicit exact same-window assessment adoption required')
        snapshot=Path(matches[0]['path'])
        if snapshot.is_symlink() or not snapshot.is_file() or snapshot.read_bytes()!=raw:
            raise ValueError('same-window assessment snapshot bytes binding')
        validate_assessment(assessment,p,cutoff)
    return document,assessment


def run_once(policy_document, cutover_document, anchor_document, authority, *, execute=False,
             adapter_factory=ChainAdapter, evidence_loader=None):
    p = validate_policy(policy_document, authority, cutover_document, anchor_document)
    c, _ = authenticate_cutover(cutover_document, anchor_document, authority)
    if execute and not p['execute_enabled']:
        raise ValueError('assessment writer execution not admitted')
    if evidence_loader is None:
        from ops.current_assessment_evidence import load_evidence
        evidence_loader = load_evidence
    with global_lock(c['global_lock_path']):
        guard_files(c); observe_units()
        now = time.time(); cutoff = int(now) // 3600 * 3600
        if cutoff < p['first_window']:
            return dict(status='waiting_for_hour', chain_executed=False)
        directory = Path(c['reward_state']) / ASSESSMENT_DIRECTORY
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = SigningKey(bytes.fromhex(Path(c['authority_seed_file']).read_text().strip()))
        if key.verify_key.encode().hex() != authority:
            raise ValueError('assessment signing authority')
        chain_status = read(Path(c['chain_state'])/'weights.json') if (Path(c['chain_state'])/'weights.json').exists() else {}
        if chain_status.get('last_submitted_window', -1) >= cutoff:
            return dict(status='already_submitted', window_end=cutoff, chain_executed=False)
        # Never escape an uncertain old submission by changing output directories.
        cursor_path = Path(c['reward_state'])/'current-assessment-v1'/'submission.json'
        cursor_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        cursor = read(cursor_path) if cursor_path.exists() else {}
        if cursor.get('status') == 'submitting':
            if not execute or not cursor.get('attempt_directory'):
                raise RuntimeError('uncertain chain outcome requires actual-chain reconciliation')
            adapter = adapter_factory(c['chain_state'], netuid=120, expected_owner=OWNER)
            try:
                # Same order as submission: global writer lock, then weights.lock.
                with weights_lock(Path(c['chain_state']) / 'weights.lock'):
                    recovery = SubmissionJournal(cursor_path, c['chain_state']).recover(adapter.chain)
                if recovery.get('preserve_fence', True):
                    return dict(status='reconciliation_pending', window_end=cutoff,
                                reason=recovery.get('reason'), retained_onchain_weights=True,
                                chain_executed=False)
            finally:
                chain = getattr(adapter, 'chain', None)
                if chain is not None and hasattr(chain, 'close'): chain.close()
            cutoff = int(time.time()) // 3600 * 3600
            # Only the observed committed hour is advanced. Skip old windows;
            # if the current hour is still due, compute its own latest assessment.
            chain_status = read(Path(c['chain_state'])/'weights.json') if (Path(c['chain_state'])/'weights.json').exists() else {}
            if chain_status.get('last_submitted_window', -1) >= cutoff:
                return dict(status='already_submitted', window_end=cutoff, chain_executed=False,
                            recovered=True)
        target = directory/('assessment-'+str(cutoff)+'.json')
        if target.exists():
            document, assessment = existing_hourly_assessment(target,p,policy_document,authority,cutoff)
        else:
            history, history_refusals = historical_assessments(directory, p, policy_document, authority, cutoff)
            cached, cache_refusals = authenticated_assessment_sources(p, authority, cutoff)
            history += cached
            history_refusals += cache_refusals
            current_cache = [a for a in cached if a['cutoff'] == cutoff and a.get('evidence_cutoff') == cutoff and a.get('assessment_stale') is False]
            fresh = None
            try:
                if current_cache:
                    fresh = current_cache[-1]
                else:
                    with evidence_budget():
                        evidence = evidence_loader(p['audit_config'], authority=authority, cutoff=cutoff,
                            verifiers=p['verifiers'], expected_source_admission_sha256=p['source_admission_sha256'],
                            expected_numerical_resolution_policy_sha256=p['numerical_resolution_policy_sha256'])
                        fresh = calculate(evidence['snapshots'], evidence['committed_at_by_epoch'], cutoff)
                        fresh.update(evidence_cutoff=cutoff, assessment_stale=False,
                            evidence_hashes=evidence['evidence_hashes'], evidence_refusals=evidence.get('refused', []),
                            evidence_exclusions=evidence.get('excluded', []), evidence_deferrals=evidence.get('deferred', []))
                        validate_assessment(fresh, p, cutoff)
                assessment = fresh
                if not any(v > 0 for v in fresh['points'].values()):
                    assessment = best_fallback(history, cutoff, 'no_positive_current_evidence', fresh) or fresh
            except Exception as error:
                # Discard all unvalidated new values, including malformed JSON,
                # signature failures, bounded-input errors and network outages.
                fresh = None
                assessment = best_fallback(history, cutoff, type(error).__name__)
                if assessment is None:
                    health = dict(status='no_valid_assessment', error_type=type(error).__name__, at=now,
                                  retained_onchain_weights=True, chain_executed=False, history_refusals=history_refusals)
                    atomic(directory/'health.json', health)
                    return health
            assessment['writer_policy_sha256'] = sha(policy_document)
            assessment['fallback_history_refusals'] = history_refusals
            document = sign(assessment, key); atomic(target, document)
            if fresh is not None:
                fresh['writer_policy_sha256'] = sha(policy_document)
                fresh_document = sign(fresh, key)
                atomic(directory/'last-valid-assessment.json', fresh_document)
                if any(v > 0 for v in fresh['points'].values()):
                    atomic(directory/'last-positive-assessment.json', fresh_document)
        adapter = adapter_factory(c['chain_state'], netuid=120, expected_owner=OWNER)
        try:
            registrations = adapter.registrations()
            points, selected, excluded = recipients(assessment, registrations)
            if not points:
                result = dict(status='no_valid_registered_recipients', window_end=cutoff,
                              retained_onchain_weights=True, chain_executed=False)
                atomic(directory/'health.json', dict(at=time.time(), **result))
                atomic(directory/'last-run.json', dict(at=time.time(), result=result,
                       assessment_sha256=sha(document), positive_identities=0,
                       excluded_unregistered=excluded, assessment_stale=assessment['assessment_stale'], chain_executed=False))
                return result
            identity = process_identity(); now = time.time()
            guard_files(c); old_units = observe_units()
            receipt = sign(dict(version='single-live-reward-writer-v1', netuid=120, observed_at=now,
                                global_writer_lock_held=True, legacy_validator_guard_verified=True,
                                old_writers=old_units, **identity), key)
            if execute:
                writer_gate(receipt, authority, now=now, boot_id=identity['boot_id'],
                            writer_pid=identity['writer_pid'], writer_ticks=identity['writer_start_ticks'])
            journal = SubmissionJournal(cursor_path, c['chain_state'],
                assessment_sha256=sha(document), policy_sha256=sha(policy_document)) if execute else None
            result = adapter.submit_hour(points, selected, cutoff, execute=execute,
                                         zero_total_policy=p['zero_total_policy'],
                                         registration_change_policy=p['registration_change_policy'],
                                         submission_journal=journal)
            # Rate-limited/planned responses are safe to retry the SAME assessment.
            atomic(directory/'last-run.json', dict(at=now, result=result, assessment_sha256=sha(document),
                   positive_identities=len(points), excluded_unregistered=excluded,
                   assessment_stale=assessment['assessment_stale'], chain_executed=result['status']=='submitted'))
            if execute:
                atomic(cursor_path, dict(status=result['status'], window_end=cutoff, assessment_sha256=sha(document)))
            return result
        finally:
            chain = getattr(adapter, 'chain', None)
            if chain is not None and hasattr(chain, 'close'):
                chain.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('policy','cutover','anchor','authority'):p.add_argument('--'+field, required=True)
    p.add_argument('--execute', action='store_true'); args = p.parse_args()
    def timeout(*_):raise TimeoutError('bounded current assessment writer')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(720)
    try:
        result = run_once(read(args.policy), read(args.cutover), read(args.anchor), args.authority, execute=args.execute)
        print(json.dumps({k: result[k] for k in ('status','window_end','remaining_blocks') if k in result}))
    finally:signal.alarm(0)


if __name__ == '__main__':main()
